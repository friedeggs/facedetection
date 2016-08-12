from MathFunctions import calculateSimilarityTransform, adjustPoints, normalize, renormalize, setLambda, warpPoint
from HelperFunctions import *
from Sampler import Sampler
import numpy as np
import cv2
from Regression import StrongRegressor, RegressionTree

cascadePath = 'data/lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
cascadePaths = [
    'data/lbpcascade_frontalface.xml',
    'data/lbpcascade_profileface.xml',
    'data/haarcascade_frontalface_default.xml',
    'data/haarcascade_frontalface_alt.xml',
    'data/haarcascade_frontalface_alt2.xml',
    'data/haarcascade_profileface.xml',
    'data/haarcascade_frontalface_alt_tree.xml',
    'data/haarcascade_frontalcatface.xml',
    'data/haarcascade_frontalcatface_extended.xml'
    ]
faceCascades = [cv2.CascadeClassifier(path) for path in cascadePaths]

def detectFaceRectangle(image, ind=0): # TODO test
    width, height = np.shape(image)
    try:
        im = image
        faces = faceCascades[0].detectMultiScale(
                    image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(width/4, height/4))
                    # minSize = (10,10))
        # print faces
        index = 0
        while len(faces) == 0 and index+1 < len(faceCascades):
            index += 1
            faces = faceCascades[index].detectMultiScale(
                        image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(width/3, height/3))
                        # minSize = (10,10))
            # print faces
        if len(faces) == 0:
            return meanRectangle
        index = np.argmax(faces[:,2]) # argmax of width # and height
        faceRect = faces[index]
        faceRect = adjustRect(faceRect) # does not affect original shapeRectangle
        # y,x,h,w = faceRect
        # faceRect = (x,y,w,h)
        return faceRect # TODO or return largest one?
    except():
        e = sys.exc_info()[0]
        print(e)
        return FaceDetector.meanRectangle

def adjustToFit(shape, shapeRectangle, adapterOnly=False): # shapeRectangle is given as (x,y,w,h)
    x,y,w,h = shapeRectangle
    x = np.array([x,y])
    off = np.array([w,h])
    X1 = np.min(shape, 0)# .astype(int) TODO why astype int?
    Y1 = np.max(shape, 0).astype(int)
    scale = 1.*off/(Y1 - X1)
    offset = X1 * scale - x
    shape = shape * scale - offset
    transform = 1./scale
    if adapterOnly:
        return (scale, offset)
    return shape

def adjustRect(rect): # TODO extremely arbitrary even if it depends on opencv's output
    x,y,w,h = rect
    cx = x + w/2
    cy = y + h/2
    w = 2*w/3
    h = 3*h/4
    x = cx - w/2
    y = cy - h/3
    return (x,y,w,h)

def rectangle(shape):
    x1,y1 = np.min(shape, 0).astype(int)
    x2,y2 = np.max(shape, 0).astype(int)
    return (x1,y1,x2-x1,y2-y1) # TODO

def loadDataSet(n, basePath):
    ''' Load images and shapes '''
    I = [[] for i in range(n)]
    shapes = [[] for i in range(n)]
    for i in range(n):
        filePath = basePath + 'annotation/' + str(i+1) + '.txt'
        imagePath = ""
        with open(filePath, 'r') as f:
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = np.array([[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()])
            # shapes[i] = coarsenShape(shapes[i])
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg', 0)# cv2.IMREAD_GRAYSCALE
    return I, shapes

class FaceDetectorFactory:
    def __init__(self, settings):
        for attr, val in settings.iteritems():
            setattr(self, attr, val)
        self.strongRegressors = []
        self.shapeDeltas = [[] for i in range(self.N)]
        self.shapeEstimates = [[] for i in range(self.N)]
        self.shapes = [[] for i in range(self.n)]
        self.I = [[] for i in range(self.n)]
        self.rectangles = [[] for i in range(self.n)]
        self.imageAdapters = [[] for i in range(self.N)]
        self.similarityTransforms = [[] for i in range(self.N)]
        setPrintOptions(self.PRINT_TIME_STATS)
        setLambda(self.lmbda)

    def generateTrainingData(self):
        global meanRectangle
        self.pi = np.repeat(np.arange(self.N), self.R)
        self.meanShape = np.mean(self.shapes, 0)
        # x1,y1 = np.min(self.meanShape, 0).astype(int)
        # x2,y2 = np.max(self.meanShape, 0).astype(int)
        # self.meanRectangle = (x1,y1,x2-x1,y2-y1) # TODO
        self.meanRectangle = rectangle(self.meanShape) # TODO
        meanRectangle = self.meanRectangle
        for i in range(self.n):
            self.rectangles[i] = detectFaceRectangle(self.I[i])
            self.imageAdapters[i] = adjustToFit(self.meanShape, self.rectangles[i], adapterOnly=True) # TODO same code is run in here and shapeEstimates[i*R+j] line
        for i in range(self.n):
            sample = random.sample(range(i) + range(i+1, self.n), self.R)
            for j in range(self.R):
                self.shapeEstimates[i*self.R+j] = adjustToFit(self.shapes[sample[j]], self.rectangles[i])
        for i in range(self.N):
            self.shapeDeltas[i] = self.shapes[self.pi[i]] - self.shapeEstimates[i]

    def applyRegressionTree(self, strongRegressor, tree, i):
        strongRegressor += self.lr * normalize(tree.eval(self.I[self.pi[i]], (self.meanShape, self.shapeEstimates[i], self.similarityTransforms[i])), self.imageAdapters[self.pi[i]])

    def fitRegressionTree(self, residuals):
        tree = self.fitNode(self.F, range(self.N), np.mean(residuals, 0), residuals)
        return tree

    def fitNode(self, depth, Q, mu, residuals):
        if depth == 1 or len(Q) == 1:
            return RegressionTree(mu, residuals=Q) # Leaf node
        maxval = 0
        split = None
        maxresult = ()
        for i in range(self.S):
            split = Node(self.sampler.samplePair())
            val, result = self.tryNodeSplit(split, (Q, mu, residuals))
            if val > maxval:
                maxval = val
                maxresult = result
        Q_l, Q_r, mu_theta_l, mu_theta_r, theta = result
        tau,u,v = theta.tup # TODO clean this part
        split.setThreshold(tau)
        tree = RegressionTree(split)
        if depth > 1:
            tree.leftTree = self.fitNode(depth-1, Q_l, mu_theta_l, residuals)
            tree.rightTree = self.fitNode(depth-1, Q_r, mu_theta_r, residuals)
        return tree

    # class NodeBuilder:
    #     def __init__(factory)
    #         self.factory = factory

    def splitPoints(self, Q, theta):
        tau, u, v = theta.tup
        thresholds = [theta.eval(self.I[self.pi[i]], (self.meanShape, self.shapeEstimates[i], self.similarityTransforms[i])) for i in Q] # TODO apparently don't need self.imageAdapters[self.pi[i]], don't remember
        total = np.array(sorted(zip(thresholds,Q))) # increasing # TODO rough, possibly inefficient. could just get the two threshold elements
        cutoff = random.randint(1, len(Q)-1) # includes len(Q)-1; cutoff includes that element and up, so each side always has at least one element
        # self.tau = (total[cutoff-1][0] + total[cutoff][0])/2 # TODO should be changed
        tau = (total[cutoff-1][0] + total[cutoff][0])/2 # TODO should be changed
        theta.setThreshold(tau)
        while cutoff < len(Q) and total[cutoff-1][0] == total[cutoff][0]: # TODO how should random cutoff be picked in the case of duplicated thresholds?
            cutoff += 1 # could potentially get 0 in this case!
        left = total[cutoff:][:,1] # includes cutoff
        right = total[:cutoff][:,1]
        return left, right, theta

    def tryNodeSplit(self, theta, trainingData):
        Q, mu, residuals = trainingData
        Q_l, Q_r, theta = self.splitPoints(Q, theta)
        if len(Q_l) == 0:
            mu_theta_l = 0
            mu_theta_r = np.mean([residuals[i] for i in Q_r], 0)
            # assert(1==0) # throw error
        else:
            mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)
            if len(Q_r) == 0:
                mu_theta_r = 0
                # assert(1==0) # throw error
            else:
                mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)
        val = len(Q_l) * np.linalg.norm(mu_theta_l) + len(Q_r) * np.linalg.norm(mu_theta_r)
        return val, (Q_l, Q_r, mu_theta_l, mu_theta_r, theta)

    def manufacture(self, fd):
        self.I, self.shapes = loadDataSet(self.n, self.basePath)
        self.generateTrainingData()
        strongRegressors = []
        evaluatedRegressor = [[] for i in range(self.N)]
        residuals = [[] for i in range(self.N)]
        for t in range(self.T):
            strongRegressors.append([])
            self.sampler = Sampler(K=self.K, S=self.S, P=self.P, F=self.F) # TODO
            x,y,w,h = self.meanRectangle
            # self.sampler.samplePixels(*self.meanRectangle)
            self.sampler.samplePixels(x,y,x+w,y+h)
            meanDelta = np.mean(self.shapeDeltas, axis=0)
            strongRegressors[t] = StrongRegressor(meanDelta)
            # calculateSimilarityTransforms()
            for i in range(self.N):
                self.similarityTransforms[i] = calculateSimilarityTransform(self.meanShape, self.shapeEstimates[i])
            updatedShape = [[] for i in range(self.N)]
            for k in range(self.K):
                for i in range(self.N):
                    if k == 0:
                        evaluatedRegressor[i] = np.copy(meanDelta)
                    else:
                        self.applyRegressionTree(evaluatedRegressor[i], strongRegressors[t].weakRegressors[k-1], i)
                        # evaluatedRegressor[i].applyRegressionTree(strongRegressors[t].weakRegressors[k-1])
                    residuals[i] = renormalize(self.shapeDeltas[i] - evaluatedRegressor[i], self.imageAdapters[self.pi[i]])
                tree = self.fitRegressionTree(residuals)
                strongRegressors[t].add(tree)
            if t < self.T:
                # updateShapes(t)
                for i in range(self.N):
                    self.shapeEstimates[i] += evaluatedRegressor[i]
                    self.shapeDeltas[i] = self.shapes[self.pi[i]] - self.shapeEstimates[i]
            # save(strongRegressors[t], self.tempPath + 'strong_regressor_' + str(t+1))
        fd.create(self.meanShape, strongRegressors)
        save(fd, self.tempPath + 'face_detector')

class FaceDetector:
    meanRectangle = []
    def create(self, meanShape, strongRegressors):
        self.meanShape = meanShape
        self.strongRegressors = strongRegressors
        self.meanRectangle = rectangle(meanShape)
    def load(self, path):
        fd = load(path)
        self.create(fd.meanShape, fd.strongRegressors)
    def detectFace(self, image, meanShape=None):
        transform = (1, np.identity(2), 0) # identity transform
        shapeRectangle, im = detectFaceRectangle(image)
        adjustment = adjustToFit(self.meanShape, shapeRectangle, adapterOnly=True)
        predictedShape = adjustToFit(self.meanShape, shapeRectangle)
        for strongRegressor in self.strongRegressors:
            if strongRegressor:
                delta = strongRegressor.eval(image, self.meanShape, predictedShape, transform, adjustment)
                predictedShape += delta # normalize(delta, adjustment)
                transform = calculateSimilarityTransform(self.meanShape, predictedShape)
        return predictedShape # adjustPoints(predictedShape, adjustment)
    def train(self, settings):
        faceDetectorFactory = FaceDetectorFactory(settings)
        faceDetectorFactory.manufacture(self)
    def visualize(self, basePath):
        I, shapes = loadDataSet(1, basePath)
        rect = detectFaceRectangle(I[0])
        im = np.zeros(np.shape(I[0]))
        shape = adjustToFit(self.meanShape, rect)
        adjustment = adjustToFit(self.meanShape, rect, adapterOnly=True)
        splits = []
        for weakRegressor in self.strongRegressors[0].weakRegressors: # TODO all strongRegressors
            print weakRegressor.node.tup
            splits += weakRegressor.splits()
        count = len(splits)
        for i in range(count):
            split = splits[i]
            threshold, p0, p1 = split.tup
            pair = np.array([p0,p1])
            adjustedPair = adjustPoints(pair, adjustment)
            adjustedPair = [[int(s) for s in p] for p in adjustedPair]
            adjustedPair = tuple(map(tuple,adjustedPair))
            cv2.line(im, adjustedPair[0], adjustedPair[1], color=255) #+ 255./(count-1)*i, thickness=1)
            im = markImage(im, adjustPoints(pair, adjustment), markSize=4)
        displayImage(im)
        cv2.imwrite('results/regression_trees_visualization.jpg', im)
class Node:
    def __init__(self, pair):
        self.u, self.v = pair
        self.tup = (0, self.u, self.v)

    def setThreshold(self, tau):
        self.tau = tau
        self.tup = (tau, self.u, self.v)

    def split(self, image):
        return 1 if applySplit(node, image) > tau else 0

    def eval(self, image, transformationData):
        meanShape, shapeEstimate, similarityTransform = transformationData
        u = self.u
        v = self.v
        u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
        v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
        w, h = np.shape(image)
        im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
        im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
        return im_u - im_v

if __name__ == '__main__':
    n = 5
    R = 2
    basePath = '/Users/frieda/Downloads/'
    settings = {
        "lr": 0.4,
        "T": 1,
        "K": 2,
        "F": 5,
        "P": 400,
        "S": 20,
        "n": n,
        "R": R,
        "N": n*R,
        "basePath": basePath, # TODO '~' should work instead
        "tempPath": 'temp_',
        "testPath": 'test_', # not used
        "lmbda": 0.05,
        "PRINT_TIME_STATS": True
    } # a dict of parameters
    fd = FaceDetector()
    fd.train(settings)
    fd.load('temp_face_detector')
    fd.visualize(basePath)
    # fd.test()
    # fd.predict('https://link_to_image')
