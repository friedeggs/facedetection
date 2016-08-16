import sys
sys.path.append('..')

from FaceDetection import FaceDetector, loadDataSet, overlap, detectFaceRectangle, rectangle
from HelperFunctions import markImage, displayImage, drawRect
import numpy as np

import unittest, nose

basePath = '/Users/frieda/Downloads/'
n = 10
settings = {
    "lr": 1,
    "T": 1,
    "K": 1,
    "F": 20,
    "P": 400,
    "S": 20,
    "n": n,
    "R": 1,
    "basePath": basePath, # TODO '~' should work instead
    "tempPath": 'temp_',
    "testPath": 'test_', # not used
    "lmbda": 0.05,
    "PRINT_TIME_STATS": True
} # a dict of parameters

def testOverlap():
    rect1 = (1, 5, 8, 10)
    rect2 = (2, 6, 4, 5)
    overlap1 = overlap(rect1, rect1)
    overlap2 = overlap(rect1, rect2)
    nose.tools.assert_equal(1, overlap1)
    nose.tools.assert_equal(0.25, overlap2)

def testFaceDetection():
    n = 20
    I, shapes = loadDataSet(n, basePath)
    for i in range(n):
        print i
        rect = detectFaceRectangle(I[i], shapes[i])
        rect3 = rectangle(shapes[i])
        rect2 = overlap(rect3, rect, test=True)
        x,y,W,H = rect2
        x1,y1,w1,h1 = rect3
        val2 = 1.*W*H/(w1*h1)
        print val2
        val = overlap(rect3, rect)
        np.testing.assert_almost_equal(val, val2)
        im = drawRect(I[i], rect, color=0)
        im = drawRect(im, rect3, color=155)
        im = drawRect(im, rect2)

        displayImage(im)

def testChooseSplit():
    from FaceDetection import FaceDetectorFactory
    faceDetectorFactory = FaceDetectorFactory(settings)
    mark("Loading data set")
    self.I, self.shapes = loadDataSet(self.n, self.basePath)
    mark("Generating training data")
    self.generateTrainingData()
    strongRegressors = []
    evaluatedRegressor = [[] for i in range(self.N)]
    for t in range(self.T):
        strongRegressors.append([])
        mark("Sampling pixels")
        self.sampler = Sampler(K=self.K, S=self.S, P=self.P, F=self.F) # TODO
        x,y,w,h = self.meanRectangle
        # self.sampler.samplePixels(*self.meanRectangle)
        self.sampler.samplePixels(x,y,x+w,y+h)
        meanDelta = np.mean(self.shapeDeltas, axis=0)
        strongRegressors[t] = StrongRegressor(meanDelta)
        strongRegressors[t].setLearningRate(self.lr)
        mark("Calculating similarity transforms")
        # calculateSimilarityTransforms()
        for i in range(self.N):
            self.similarityTransforms[i] = calculateSimilarityTransform(self.meanShape, self.shapeEstimates[i])
        for k in range(self.K):
            mark("Fitting weak regressor %d of %d" % ((k+1), (self.K+1)))
            for i in range(self.N):
                if k == 0:
                    evaluatedRegressor[i] = np.copy(meanDelta)
                else:
                    self.applyRegressionTree(evaluatedRegressor[i], strongRegressors[t].weakRegressors[k-1], i)
                    # evaluatedRegressor[i].applyRegressionTree(strongRegressors[t].weakRegressors[k-1])
                self.residuals[i] = renormalize(self.shapeDeltas[i] - evaluatedRegressor[i], self.imageAdapters[self.pi[i]])

            tree = self.fitRegressionTree()

if __name__ == '__main__':
    # fd = FaceDetector()
    # fd.train(settings)
    # I, shapes = loadDataSet(n, basePath)
    # for i in range(n):
    #     prediction = fd.predict(I[i])
    #     im = markImage(I[i], prediction)
    #     im = markImage(im, shapes[i])
    #     displayImage(im)
    #     # np.testing.assert_almost_equal(fd.predict(I[i]), shapes[i])
    # testOverlap()
    testFaceDetection()


