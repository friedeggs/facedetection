'''
# training time: about 1 hour with a single CPU on the HELEN dataset
# runtime: about 1 millisecond per image
'''

# Yet the improvement
# displayed may not be saturated because we know that the
# underlying dimension of the shape parameters are much
# lower than the dimension of the landmarks (194×2). There
# is, therefore, potential for a more significant improvement
# with partial labels by taking explicit advantage of the correlation
# between the position of landmarks. Note that the gradient boosting procedure described in this paper does not
# take advantage of the correlation between landmarks. This
# issue could be addressed in a future work.

import sys, time # time.time() eg 1465655128.768409
import cv2
import dlib
import numpy as np
import random
import math
import pickle

class StrongRegressor:
    baseFunction # vector in R^2p
    weakRegressors # array of RegressionTree

    def __init__(self, base):
        self.baseFunction = base

    def add(weakRegressor):
        weakRegressors.append(weakRegressor)

    def eval(image, shape): # apply the strong regressor tree function
        # fk = f_(k-1) + lr * g_k # f_k = lr * g_k + lr * g_(k-1) + ... + lr * g_1 + f_0
        # r_t = f_K
        res = baseFunction # vector in R^2p
        for k in range(K):
            res += lr * weakRegressor[k].eval(image, shape)
        return res

class RegressionTree:

    def __init__(self, node, leftTree=null, rightTree=null):
        self.node = node
        self.leftTree = leftTree
        self.rightTree = rightTree

    def eval(i):
        # images, functions not defined

annotationPath = 'Users/frieda/Downloads/annotation/' # TODO use ~
loadPath = 'faceDetector.pkl'
savePath = loadPath

lr = 0.1 # learning rate (v)
T = 10 # number of strong regressors, r_t
# each r is composed of K weak regressors, g_k
K = 500 # number of weak regressors
F = 5 # depth of trees used to represent g_k
P = 400 # number of pixel locations sampled from the image at each level of the cascade

S = 20 # number of random potential splits
n = 2000 # number of training images (placeholder number)
R = 20 # number of initializations for each training example
N = n*R # training examples # N = nR where R is the number of initializations per face

# averaging predictions of multiple regression trees as alternative to learning rate:
M = 10 # fit multiple trees to the residuals in each iteration of the gradient boosting algorithm and average the result
# lr = 1 # lr * M, or 0.1 * 10

lmbda = 0.1 # exponential prior parameter
# untried extension: use cross validation when learning each strong regressor in the cascade to select this parameter

strongRegressors = [[] for i in range(T)] # array of StrongRegressor
# weakRegressors = [[] for i in range(T)] # array of RegressionTree
# weakRegressorBase = [[] for i in range(T)] # array of vectors in R^2p
shapeDeltas = []
pi = []
shapes = [] # len = 2000
shapeEstimates = [] # len = N = 2000 * 20
I = [] # face images
residuals = [[] for i in range(N)]

meanShape = []
meanWidth = 0
meanHeight = 0

samplePoints = []
samplePairs = []
priorWeights = [] # [[] for i in range(P)] # adjacency matrix of weights for each possible pair of points
random.seed()

similarityTransforms = [] # triples of t, S, R

# to train weak regressors, we randomly sample a pair of these P pixel locations according to our prior and choose a random threshold to create a potential split as described in equation 9 (the h thresholding split equation)

# def find_best_split():
#     # sample a point based on our prior and a random threshold to create our potential split
#     # take the best one that optimizes our objective


def prior(u,v):
    return math.exp(-lmbda*np.linalg.norm(u-v)) # np.linalg.norm(u-v) calculates the euclidean distance between two points u, v

# only once at each level of the cascade
# "In practice the assignments and local translations are determined
# during the training phase."

def calculateSimilarityTransforms():
    similarityTransforms = [calculateSimilarityTransform(meanShape, shapeEstimates[i]) for i in range(N)]

# Compute least squares transform as described in
# https://en.wikipedia.org/wiki/Wahba%27s_problem
# http://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_Umeyama.pdf
# http://eigen.tuxfamily.org/dox-devel/Umeyama_8h_source.html
def calculateSimilarityTransform(w, v):
    # s, R = where sum of square x, s_i R_i x_i + t_i is minimum
    # return s, R

    # shape 1
    # shape 2
    # v = meanShape
    # w = shapeEstimates[i]

    center_w = np.sum(w, 0)*1./len(w)
    center_v = np.sum(v, 0)*1./len(v)

    B = np.dot(np.transpose(w), v)
    U, s, V^ = np.linalg.svd(B)
    m = np.shape(U)[0]
    n = np.shape(V^)[1]
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)

    M = np.zeros((m, n))
    M[:n, :n] = np.diag(np.append(np.ones(n - 1), np.linalg.det(U) * np.linalg.det(V^)))

    R = np.dot(U, np.dot(M, V^))
    # var = np.var(v) # np.var(np.sum(w, 0))
    var = 1./len(v) * np.sum(np.linalg.norm((v - center_v), axis=1)**2)
    c = 1./var*np.trace(np.dot(S, M))
    t = np.transpose(np.transpose(center_w) - c * np.dot(R, np.transpose(center_v)))

    return c, R, t

# http://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest(node, nodes):
    nodes = np.asarray(nodes) # TODO already np array?
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def warpPoint(u, similarityTransform):
    S, R, t = similarityTransform
    # u is a point in coordinate system of mean shape
    k_u = closest(u, meanShape)
    delta_x_u = u - k_u # offset from u
    # s, R = calculate_similarity_transform(x, t)
    u^ = = k_u + 1./S * np.dot(np.transpose(R), delta_x_u) # TODO check
    return u^

def split(i, tau, u1, v1):
    if I[pi[i]][u1] - I[pi[i]][v1] > tau: # compare intensities
        return 1 # left
    else
        return 0 # right

# returns Q_theta, l
def splitPoints(Q, theta):
    # Q is the set of indices of the training examples at a node
    tau, u, v = theta
    left, right = [], []
    for i in Q:
        u1 = warpPoint(u, similarityTransforms[pi[i]])
        v1 = warpPoint(v, similarityTransforms[pi[i]])
        # return [i for i in Q if split(i, tau, u1, v1) == 1] http://stackoverflow.com/questions/949098/python-split-a-list-based-on-a-condition
        left.append(i) if split(i, tau, u1, v1) == 1 else right.append(i)
    return left, right

def tryNodeSplit(Q, mu, theta): # theta is a node split (tau, u, v)
    Q_l, Q_r = splitPoints(Q, theta)
    # our function is approximated as a piecewise constant function fit to each leaf node:
    # left node = one constant
    # right node = another constant
    # we split based on a theta

    # to choose the theta, we generate a set of random thetas and then take the best candidate of these based on mimimizing the sum of square error

    # Q is the set of indices of the training examples at a node

    # Q_theta, l is the indices of the examples that are sent to the left node due to the decision induced by theta

    # r_i is the vector of all the residuals computed for image i in the gradient boosting algorithm

    # we define:
    mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)

    # mu_theta, r can be calculated from mu_theta, l by:
    mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)

    # choose theta to minimize
        # argmin E(Q,theta) = sum {l,r} sum {i in Q} ||r_i - mu_theta, {l | r}||^2

    # in fact this can be done efficiently because
    # argmin E(Q, theta) = argmax sum {l,r} |Q_theta, s| * transpose (mu_theta, s) * mu_theta, s # squaring mu_theta, s

    # so:
    # return argmax sum {l,r} |Q_theta, s| * transpose (mu_theta, s) * mu_theta, s
    val = len(Q_l) * np.dot(mu_theta_l, mu_theta_l) + len(Q_r) * np.dot(mu_theta_r, mu_theta_r)
    if val > maxval:
        maxval = val
        argmax = theta
    return val, Q_l, Q_r, mu_l, mu_r

# # https://en.wikipedia.org/wiki/Least_squares#Linear_least_squares
# https://en.wikipedia.org/wiki/Geometric_median
def groundEstimate(shapes):
    # shapes is an array of face shapes [[], [], [], ...]
    # return np.median(shapeEstimates, axis=0) # returns the geometric median
    return np.mean(shapeEstimates, axis=0) # TODO please check


def samplePixels():
    # return P = 400 pixel locations sampled from the image
    # run for each level of the cascade (T = 10)
    points = [(random.randint(0, meanWidth), random.randint(0, meanHeight)) for i in range(P)]
    pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
    priorWeights = [prior(p) for p in pairs]
    return points, pairs, priorWeights

def samplePair():
    # sample two points from the sampled pixels based on our prior
    # order matters
    return np.random.choice(samplePairs, p=priorWeights) # choose a pair given probability distribution priorWeights

def generateCandidateSplit():
    pair = samplePair()
    threshold = random() # placeholder line
    return threshold, pair

def fitRegressionTree():
    # mu = 1 / len(Q) * sum([r[i] for i in Q])
    mu = np.mean(residuals, 0)
    tree = fitNode(range(N), mu, F) # F = 5; depth of tree

def fitNode(Q, depth):
    maxval = 0
    for i in range(S): # S = 20
        candidateSplit = generateCandidateSplit()
        val, q_l, q_r, mu_l0, mu_r0 = tryNodeSplit(Q, candidateSplit)
        if val > maxval:
            maxval = val
            split = candidateSplit
            Q_l = q_l
            Q_r = q_r
            mu_l = mu_l0
            mu_r = mu_r0
    tree = RegressionTree(split, depth)
    if level > 0:
        tree.leftTree = fitNode(Q_l, depth - 1)
        tree.rightTree = fitNode(Q_r, depth - 1)
    return tree

def loadData():
    for i in range(1, N+1):
        filePath = annotationPath + str(i) + '.txt'
        imagePath = "" # TODO check scope
        with open(filePath, 'r') as f:
            # don't need to call f.close() this way http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list-with-python
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = [[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()]

def initializeShapes():
    pass

def calculateMeanShape():
    meanShape =
    meanWidth =
    meanHeight =

def generateTrainingData():
    pi = np.random.permutation(np.repeat(np.arange(N), R)) # TODO change to no fixed point
    initializeShapes() # # use dlib to generate images for each face # use training data annotations
    for i in range(N):
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]

def updateShapes(t):
    # S[i] += r(i) # r(I[pi[i]], S[i])
    for i in range(N):
        shapeEstimates[i] += strongRegressor[t].eval(I[pi[i]], shapeEstimates[i])

def loadDetector(path=loadPath):
    f = open(path, 'r')
    faceDetector = pickle.load(f)
    f.close()
    return faceDetector

def saveDetector(detector, path=savePath):
    f = open(path, 'w')
    pickle.dump(detector, f)
    f.close()

def learnFaceDetector(save=true):
    loadData()
    calculateMeanShape()
    generateTrainingData()
    for t in range(T):
        samplePoints, samplePairs, priorWeights = samplePixels()
        # learn the regression function using gradient boosting and a sum of square error loss
        # f_i are the weak regressors
        # f0 is the point at which the distance from that point to all the other points in the delta shape is at a minimum
        strongRegressor[t] = new StrongRegressor(groundEstimate(shapeDeltas)) # f_0 is a median face shape, gamma, belonging to R^2p where p is the number of facial landmarks

        calculateSimilarityTransforms()
        for k in range(K):
            # f_k is f_(k-1) of a guess ^S(t) plus lr * g_k of a guess ^S(t)
            for i in range(N):
                residuals[i][k] = shapeDelta[i] - strongRegressor[t].eval(I[pi[i]], shapeEstimates[i]) # is the delta shape - f of k-1 # compute residuals
                # r_ik is a delta delta shape; r_ik also belongs to R^2p

            # regression
            # fit a regression tree to the 2p-dimensional targets r_ik giving a weak regression function g_k(I, S^(t))
            # there are N targets r_ik (k is fixed; i = 1..N)

            tree = fitRegressionTree(residuals)
            strongRegressor[t].add(tree)

            # fk = f_(k-1) + lr * g_k # f_k = lr * g_k + lr * g_(k-1) + ... + lr * g_1 + f_0
        # r_t = f_K # done

        # update the training set
        # S_(t+1) = S_t + r_t()
        # delta_S_(t+1) = actual shape S - S_(t+1)
        updateShapes(t)
        # S[i] += r(i) # r(I[pi[i]], S[i])

    faceDetector = strongRegressors
    if(save):
        saveDetector(faceDetector, savePath)
    return faceDetector
