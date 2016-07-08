
'''
'''
import sys, time
import cv2
import dlib
import numpy as np
import random
import math
import pickle

class StrongRegressor:
    baseFunction = []
    weakRegressors = []
    def __init__(self, base):
        self.baseFunction = base
    def add(self, weakRegressor):
        self.weakRegressors.append(weakRegressor)
    def eval(self, image, shapeEstimate, shapeTransform):
        res = self.baseFunction
        for weakRegressor in self.weakRegressors:
            res += lr * weakRegressor.eval(image, shapeEstimate, shapeTransform) # TODO is it self.baseFunction? or is it shapeEstimate?
        return res
class RegressionTree:
    def __init__(self, node, depth=1, leftTree=None, rightTree=None):
        self.node = node
        self.depth = depth
        self.leftTree = leftTree
        self.rightTree = rightTree
    def eval(self, image, shapeEstimate, shapeTransform): # warp based on shapeEstimate which is based off result from StrongRegressor
        if self.depth == 1: # leaf
            return self.node # need to transform???? No, right? because these are the residuals, which we did not transform when computing
        if split(image, self.node[0], self.node[1], self.node[2], shapeEstimate, shapeTransform) == 1:
            return self.leftTree.eval(image, shapeEstimate, shapeTransform)
        else:
            return self.rightTree.eval(image, shapeEstimate, shapeTransform)
basePath = '/Users/frieda/Downloads/'
loadPath = 'faceDetector.pkl'
savePath = loadPath
saveTestPath = 'test'
# lr = 0.1
# T = 10
# K = 500
# F = 5
# P = 400
# S = 20
# n = 2000
# R = 20 # Use 1 initialization instead? >:(
# N = n*R
# # M = 10 What is this??
lr = 0.1
T = 5
K = 5
F = 5
P = 20
S = 20
n = 10
R = 5 # Use 1 initialization instead? >:(
N = n*R
# M = 10
lmbda = 0.1
strongRegressors = [[] for i in range(T)]
shapeDeltas = [[] for i in range(N)]
pi = []
shapes = [[] for i in range(n)]
shapeEstimates = [[] for i in range(N)]
I = [[] for i in range(n)]
residuals = [[] for i in range(N)]
meanShape = []
meanWidth = 0
meanHeight = 0
samplePoints = []
samplePairs = []
priorWeights = []
random.seed()
similarityTransforms = []
def prior(u,v):
    return math.exp(-lmbda*np.linalg.norm(np.subtract(u,v)))
def calculateSimilarityTransforms():
    global similarityTransforms
    similarityTransforms = [calculateSimilarityTransform(meanShape, shapeEstimates[i]) for i in range(N)]
def calculateSimilarityTransform(w, v):
    ''' Calculate similarity transform for a given face estimate '''
    center_w = np.sum(w, 0)*1./len(w)
    center_v = np.sum(v, 0)*1./len(v)
    B = np.dot(np.transpose(w - center_w), v - center_v) *1./len(w)
    U, s, V1 = np.linalg.svd(B)
    m = np.shape(U)[0]
    n = np.shape(V1)[1]
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    M = np.zeros((m, n))
    if np.linalg.det(B) >= 0:
        M = np.identity(n)
    else:
        M[:n, :n] = np.diag(np.append(np.ones(n - 1), 1))
    R = np.dot(U, np.dot(M, V1))
    var = 1./len(v) * np.sum(np.linalg.norm((v - center_v), axis=1)**2)
    varw = 1./len(w) * np.sum(np.linalg.norm((w - center_w), axis=1)**2)
    c = 1./var*np.trace(np.dot(S, M))
    t = np.transpose(np.transpose(center_w) - c * np.dot(R, np.transpose(center_v)))
    return c, R, t
def applyTransform(transform, points):
    S, R, t = transform
    return S * np.transpose(np.dot(R, np.transpose(points)))
def applyRotation(transform, points):
    S, R, t = transform
    return np.transpose(np.dot(R, np.transpose(points)))
def closest(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)
def warpPoint(u, X, Y, similarityTransform): # TODO check
    S, R, t = similarityTransform # global indexing
    k_u = closest(u, X) # local indexing
    delta_x_u = u - X[k_u]
    u1 = Y[k_u] + 1./S * np.dot(np.transpose(R), delta_x_u)
    return u1
def split(image, tau, u, v, shapeEstimate, similarityTransform):
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    # print image[u1[0]][u1[1]]
    # print image[v1[0]][v1[1]] # TODO were the same
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    if im_u - im_v > tau:
    # if int(image[u1[1],u1[0]]) - int(image[v1[1],v1[0]]) > tau: # doesn't matter
    # if int(image[u1[0]][u1[1]]) - int(image[v1[0]][v1[1]]) > tau:
        return 1
    else:
        return 0
def splitPoints(Q, theta):
    # print theta
    tau, u, v = theta
    left, right = [], []
    for i in Q:
        left.append(i) if split(I[pi[i]], tau, u, v, shapeEstimates[pi[i]], similarityTransforms[pi[i]]) == 1 else right.append(i)
    return left, right
def tryNodeSplit(Q, mu, theta):
    maxval = 0
    Q_l, Q_r = splitPoints(Q, theta)
    if len(Q_l) == 0:
        mu_theta_l = 0
        mu_theta_r = np.mean([residuals[i] for i in Q_r], 0)
    else:
        mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)
        if len(Q_r) == 0:
            mu_theta_r = 0
        else:
            mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)
    val = len(Q_l) * np.linalg.norm(mu_theta_l) + len(Q_r) * np.linalg.norm(mu_theta_r)
    if val > maxval:
        maxval = val
        argmax = theta
    return val, Q_l, Q_r, mu_theta_l, mu_theta_r
def groundEstimate(shapes):
    return np.mean(shapeEstimates, axis=0)
def samplePixels():
    points = [(random.randint(0, meanWidth), random.randint(0, meanHeight)) for i in range(P)]
    pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
    priorWeights = [prior(p[0], p[1]) for p in pairs]
    priorWeights = [x / sum(priorWeights) for x in priorWeights]
    return points, pairs, priorWeights
def samplePair():
    return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]
def generateCandidateSplit():
    pair = samplePair()
    threshold = random.randint(76, 178) # random.randint(0, 255) # TODO placeholder
    return threshold, pair[0], pair[1] # TODO made in haste
def fitRegressionTree():
    mu = np.mean(residuals, 0)
    tree = fitNode(range(N), mu, F)
    return tree
def fitNode(Q, mu, depth):
    if depth == 1 or len(Q) == 0: # TODO check if should be 0 instead
        return RegressionTree(mu) # Leaf node
    maxval = 0
    for i in range(S):
        candidateSplit = generateCandidateSplit()
        val, q_l, q_r, mu_l0, mu_r0 = tryNodeSplit(Q, mu, candidateSplit)
        if val > maxval:
            maxval = val
            split = candidateSplit
            Q_l = q_l
            Q_r = q_r
            mu_l = mu_l0
            mu_r = mu_r0
    tree = RegressionTree(split, depth)
    if len(Q) == N:
        print "------------------------------------------"
        print Q_l
        print Q_r
    if depth > 0:
        tree.leftTree = fitNode(Q_l, mu_l, depth - 1)
        tree.rightTree = fitNode(Q_r, mu_r, depth - 1)
    return tree
def loadData():
    ''' Load images?!?!?! and shapes '''
    for i in range(n):
        filePath = basePath + 'annotation/' + str(i+1) + '.txt'
        imagePath = ""
        with open(filePath, 'r') as f:
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = np.array([[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()])
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg', cv2.IMREAD_GRAYSCALE)
def calculateMeanShape():
    ''' Calculate mean shape and bounding box shape of all faces '''
    # TODO placeholder implementation right now
    global meanShape, meanWidth, meanHeight
    meanShape = np.mean(shapes, 0)
    meanWidth, meanHeight = np.max(meanShape - np.min(meanShape, 0), 0).astype(int)
def generateTrainingData():
    # pi = np.random.permutation(np.repeat(np.arange(N), R)) # why does it even need to be random? order never matters
    global shapeEstimates, shapeDeltas, shapes, pi
    pi = np.repeat(np.arange(N), R)
    for i in range(n):
        sample = random.sample(range(n), R) # array of length 20 containing indices
        for j in range(R):
            shapeEstimates[i*R+j] = shapes[sample[j]] # TODO these need to be separate objects and not the same reference
    for i in range(N):
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
def updateShapes(t):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms
    for i in range(N):
        shapeEstimates[i] += strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i])
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
def learnFaceDetector(save=True, test=True):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms, residuals, samplePoints, samplePairs, priorWeights
    loadData()
    calculateMeanShape()
    # print meanWidth
    # print meanHeight
    generateTrainingData()
    for t in range(T):
        samplePoints, samplePairs, priorWeights = samplePixels()
        ''' Get mean shape '''
        print "Learning strong regressor ", str(t+1)
        strongRegressors[t] = StrongRegressor(groundEstimate(shapeDeltas))
        print "Calculating similarity transforms"
        calculateSimilarityTransforms()
        ''' Calculate similarity transforms for each shape estimate '''
        print "Computing residuals"
        for k in range(K):
            for i in range(N):
                ''' Evaluate on each image to calculate residuals '''
                # residuals[i]
                # shapeDeltas[i]
                # strongRegressors[t]
                # I[pi[i]]
                # similarityTransforms[i]
                residuals[i] = shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
            print "Fitting weak regression tree ", str(k+1)
            tree = fitRegressionTree()
            strongRegressors[t].add(tree)
        print "Updating shape estimates"
        updateShapes(t)
        if test:
            predictedShape = detectFace(strongRegressors, I[0])
            image = I[0].copy()
            width, height = np.shape(image)
            s = 5
            for a,b in predictedShape:
                a = int(a)
                b = int(b)
                for i in range(a-s, a+s):
                    for j in range(b-s,b+s):
                        if i < height and j < width and i >= 0 and j >= 0:
                            image[j,i] = 255
            cv2.imwrite(saveTestPath + '_' + str(t+1) + '.jpg', image)
    faceDetector = strongRegressors
    if(save):
        saveDetector(faceDetector, savePath)
    return faceDetector
def loadDetector(path=loadPath):
    f = open(path, 'r')
    faceDetector = pickle.load(f)
    f.close()
    return faceDetector
def saveDetector(detector, path=savePath):
    f = open(path, 'w')
    pickle.dump(detector, f)
    f.close()
def detectFace(faceDetector, image):
    transform = (1, np.identity(2), 0) # identity transform
    predictedShape = meanShape
    x = 0
    s = 5
    width, height = np.shape(image)
    for strongRegressor in faceDetector:
        if strongRegressor:
            predictedShape += strongRegressor.eval(image, predictedShape, transform)
            transform = calculateSimilarityTransform(meanShape, predictedShape)
            for a,b in predictedShape:
                a = int(a)
                b = int(b)
                for i in range(a-s, a+s):
                    for j in range(b-s,b+s):
                        if i < height and j < width and i >= 0 and j >= 0:
                            image[j,i] = 255
            cv2.imwrite(saveTestPath + '_temp_' + str(x) + '.jpg', image)
            x += 1
    return predictedShape
if __name__ == '__main__':
    # detector = learnFaceDetector()
    # global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms, residuals, samplePoints, samplePairs, priorWeights
    loadData()
    calculateMeanShape()
    generateTrainingData()
    for t in range(1):
        samplePoints, samplePairs, priorWeights = samplePixels()
        ''' Get mean shape '''
        print "Learning strong regressor ", str(t+1)
        strongRegressors[t] = StrongRegressor(groundEstimate(shapeDeltas))
        print "Calculating similarity transforms"
        calculateSimilarityTransforms()
        ''' Calculate similarity transforms for each shape estimate '''
        print "Computing residuals"
        for k in range(5):
            for i in range(N):
                ''' Evaluate on each image to calculate residuals '''
                # residuals[i]
                # shapeDeltas[i]
                # strongRegressors[t]
                # I[pi[i]]
                # similarityTransforms[i]
                residuals[i] = shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
            print residuals
            print "Fitting weak regression tree ", str(k+1)
            tree = fitRegressionTree()
            strongRegressors[t].add(tree)
            # print strongRegressors[t].eval(I[0], shapeEstimates[0], similarityTransforms[0])
        print "Updating shape estimates"
        updateShapes(t)

def test():
    loadData()
    calculateMeanShape()
    detector = loadDetector()
    predictedShape = detectFace(detector, I[0])
    print predictedShape
    image = I[0].copy()
    width, height = np.shape(image)
    s = 5
    for a,b in predictedShape:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 255
    cv2.imwrite(saveTestPath + '.jpg', image)
