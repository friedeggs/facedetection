
'''
'''
from profilestats import profile
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf) # for testing
from FaceDetector import * # TODO change to follow best practices
from StrongRegressor import StrongRegressor
from WeakRegressor import *
from MathFunctions import calculateSimilarityTransform, renormalize, normalize
from HelperFunctions import *
from CommonFunctions import *
from Settings import *
import unittest, nose
random.seed()
strongRegressors = [[] for i in range(T)]
shapeDeltas = [[] for i in range(N)]
pi = []
# shapes = [[] for i in range(n)]
I = [[] for i in range(n)]
residuals = [[] for i in range(N)]
rectangles = [[] for i in range(n)]
def calculateSimilarityTransforms():
    global similarityTransforms
    for i in range(N):
        similarityTransforms[i] = calculateSimilarityTransform(meanShape, shapeEstimates[i]) # IMPORTANT do not use list comprehension
def groundEstimate(shapes):
    return np.mean(shapes, axis=0)
def loadData(): # [CHECKED]
    ''' Load images and shapes '''
    for i in range(n):
        filePath = basePath + 'annotation/' + str(i+1) + '.txt'
        imagePath = ""
        with open(filePath, 'r') as f:
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = np.array([[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()])
            # shapes[i] = coarsenShape(shapes[i])
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg', 0)# cv2.IMREAD_GRAYSCALE
    return I, shapes
def coarsenShape(shape):
    shape = np.concatenate((shape[:41][0::4], # face shape
                           shape[41:58][0::2], # nose
                           shape[58:72][0::2], # mouth
                           shape[72:88][0::3], # bottom part of mouth
                           shape[87:114][0::4], # inner lips of mouth
                           shape[114:154][0::3], # eyes
                           shape[154:][0::4])) # eyebrows
    return shape
def calculateMeanShape(shapes): # [CHECKED]
    ''' Calculate mean shape and bounding box shape of all faces '''
    # TODO placeholder implementation right now
    global meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY
    meanShape = np.mean(shapes, 0)
    meanWidthX, meanHeightX = np.min(meanShape, 0).astype(int)
    meanWidthY, meanHeightY = np.max(meanShape, 0).astype(int)
    return meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY
    # meanWidth, meanHeight = np.max(meanShape - np.min(meanShape, 0), 0).astype(int)
def generateTrainingData(I, shapes): # [CHECKED]
    # pi = np.random.permutation(np.repeat(np.arange(N), R)) # why does it even need to be random? order never matters
    global shapeEstimates, shapeDeltas, pi, imageAdapters
    pi = np.repeat(np.arange(N), R)
    FaceDetector.meanRectangle = (
                meanWidthX,
                meanHeightX,
                meanWidthY - meanWidthX,
                meanHeightY - meanHeightY)
    for i in range(n):
        result = detectFaceRectangle(I[i])
        if result is None:
            x,y = shapes[i].min(0)
            X,Y = shapes[i].max(0)
            rectangles[i] = (x,y,X-x,Y-y)
        else:
            rectangles[i], im = result
        # im = I[i].copy()
        # im = drawRect(im, rectangles[i])
        # displayImage(im)
        imageAdapters[i] = adjustToFit(meanShape, rectangles[i], adapterOnly=True) # TODO same code is run in here and shapeEstimates[i*R+j] line
    for i in range(n):
        sample = random.sample(range(i) + range(i+1, n), R) # array of length 20 containing indices
        for j in range(R): # for efficiency don't call detectFaceRectangle R times
            # x,y = shapes[i].min(0)
            # X,Y = shapes[i].max(0)

            shapeEstimates[i*R+j] = adjustToFit(shapes[sample[j]], rectangles[i])

            # shapeEstimates[i*R+j] = np.copy(shapes[sample[j]])


            # im = I[i].copy()
            # im = markImage(im, shapeEstimates[i*R+j])
            # displayImage(im)
            # shapeEstimates[i*R+j] = adjustToFit(shapes[sample[j]], (x,y,X-x,Y-y))
        shapeEstimates[i*R] = adjustToFit(meanShape, rectangles[i]) # testing purposes
    for i in range(N):
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
    return pi, shapeEstimates, shapeDeltas, imageAdapters
def updateShapes(t):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms
    for i in range(N):
        shapeEstimates[i] += strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]])
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
# class TestFaceDetectorFactory(unittest.TestCase):
@profile(print_stats=20, dump_stats=True)
def test_learnFaceDetector(saveDetector=True, test=True, saveIntermediates=True, debug=True):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms, residuals, samplePoints, samplePairs, priorWeights
    try:
        print "Loading data"
        I, shapes = loadData()
        markTime()
        print "Calculating mean shape"
        meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY = calculateMeanShape(shapes)
        markTime()
        print "Generating training data"
        pi, shapeEstimates, shapeDeltas, imageAdapters = generateTrainingData(I, shapes)
        markTime()
        for t in range(T):
            print "Sampling pixels"
            samplePixels(meanWidthX, meanHeightX, meanWidthY, meanHeightY)
            markTime()
            ''' Get mean shape '''
            print "Learning strong regressor ", str(t+1)
            meanDelta = groundEstimate(shapeDeltas)
            strongRegressors[t] = StrongRegressor(groundEstimate(shapeDeltas))
            print "Calculating similarity transforms"
            calculateSimilarityTransforms()
            markTime()
            ''' Calculate similarity transforms for each shape estimate '''
            updatedShape = [[] for i in range(N)]
            for k in range(K):
                for i in range(N):
                    ''' Evaluate on each image to calculate residuals '''
                    residuals[i] = renormalize(shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]), imageAdapters[pi[i]]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
                    # np.testing.assert_array_equal(residuals[i],
                    #         renormalize(shapeDeltas[i] - meanDelta, imageAdapters[pi[i]])) # self.assertEqual .tolist()
                    # np.testing.assert_almost_equal(shapeEstimates[i] + meanDelta + normalize(residuals[i], imageAdapters[pi[i]]),
                    #         shapes[pi[i]]) # self.assertEqual .tolist()
                print "Fitting weak regression tree ", str(k+1)
                tree = fitRegressionTree(I, pi, meanShape, residuals)
                markTime()
                strongRegressors[t].add(tree)
                print tree.leaves()
                if saveIntermediates and (k+1) % 20 == 0:
                    save(strongRegressors[t], 'weak_regressors_' + str(t) + '-' + str(k+1))
                if debug and (k+1) % 10 == 0:
                    for j in range(20):
                        predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[j])
                        image = markImage(I[j], predictedShape)
                        width, height = np.shape(image)
                        cv2.imwrite(resultsPath + 'debug_' + str(t+1) + '_' + str(j) + '.jpg', image)
                        markTime()
                    raw_input()
                if False:
                    for j in range(0, 1*R, R):
                        im = I[pi[j]].copy()
                        # shapeEstimate = adjustToFit(shapeEstimates[j], rectangles[pi[j]])

                        # im = markImage(im, shapeEstimates[j],color=100)

                        val, residual_index = strongRegressors[0].weakRegressors[0].special_eval(I[pi[j]], shapeEstimates[j], similarityTransforms[j], imageAdapters[pi[j]])
                        # output((residual_index, j))
                        # nose.tools.assert_equal(residual_index, j)
                        # np.testing.assert_almost_equal(val[0], residuals[residual_index][0])
                        # val = strongRegressors[0].weakRegressors[0].eval(I[pi[j]], shapeEstimates[j], similarityTransforms[j], imageAdapters[pi[j]])
                        res = np.array(residual_index)
                        print residual_index
                        # np.testing.assert_almost_equal(val, residuals[j])
                        # im = markImage(im, updatedShape[j],color=0)

                        # # residual = shapeDeltas[j] - strongRegressors[t].eval(I[pi[j]], shapeEstimates[j], similarityTransforms[j], imageAdapters[pi[j]])
                        # residual = residuals[j]
                        # # updatedShape[j] = normalize(residual, imageAdapters[pi[j]])
                        # updatedShape[j] = shapeEstimates[j] + normalize(strongRegressors[t].baseFunction, imageAdapters[pi[j]]) + normalize(residual, imageAdapters[pi[j]])
                        # im = markImage(im, updatedShape[j])

                        updatedShape[j] = shapeEstimates[j] + normalize(strongRegressors[t].baseFunction, imageAdapters[pi[j]])
                        im = markImage(im, updatedShape[j], color=60)

                        # updatedShape[j] = shapeEstimates[j] + shapeDeltas[j]
                        # im = markImage(im, updatedShape[j], color=0)

                        updatedShape[j] = shapeEstimates[j] + normalize(strongRegressors[t].eval(I[pi[j]], shapeEstimates[j], similarityTransforms[j], imageAdapters[pi[j]]), imageAdapters[pi[j]])
                        im = markImage(im, updatedShape[j])

                        displayImage(im)
            # nose.tools.assert_equal(1, 0) # deliberately fail to produce output
            print "Updating shape estimates"
            updateShapes(t)
            markTime()
            save(strongRegressors[t], tempPath + 'strong_regressor_' + str(t+1))
            if test:
                for i in range(20):
                    # predictedShape = detectFace(strongRegressors, I[0])
                    predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[i])
                    image = markImage(I[i], predictedShape)
                    # width, height = np.shape(image)
                    cv2.imwrite(resultsPath + tempPath + str(t+1) + '_' + str(i) + '.jpg', image)
                    markTime()
    finally:
        save(FaceDetector(meanShape, strongRegressors), tempPath + 'strong_regressor_saved')
        markTime()
    faceDetector = FaceDetector(meanShape, strongRegressors)
    if(saveDetector):
        save(faceDetector, 'face_detector')
    return faceDetector
def showFaceDetector():
    I, shapes = loadData()
    meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY = calculateMeanShape(shapes)
    strongRegressors = load('temp_strong_regressor_saved').strongRegressors
    # window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Rectangle', 1000, 800)
    # width, height = 1000, 800
    rect, im = detectFaceRectangle(I[0])
    im = im.copy()
    # im = drawRect(im, rect)
    shape = adjustToFit(meanShape, rect)
    adjustment = adjustToFit(meanShape, rect, adapterOnly=True)
    splits = []
    for i in range(10):
        splits += strongRegressors[0].weakRegressors[i].splits()
    for split in splits:
        threshold, p0, p1 = split
        pair = np.array([p0,p1])
        adjustedPair = adjustPoints(pair, adjustment)
        adjustedPair = [[int(s) for s in p] for p in adjustedPair]
        adjustedPair = tuple(map(tuple,adjustedPair))
        # print pair, adjustPoints(pair,adjustment)
        cv2.line(im, adjustedPair[0], adjustedPair[1], color=255, thickness=1)
        im = markImage(im, adjustPoints(pair, adjustment), markSize=4)
    # im = markImage(im, shape)
    displayImage(im)
    # im = cv2.resize(im,(width, height))
    # cv2.imshow('Rectangle', im)
    # cv2.waitKey()
def test():
    I, shapes = loadData()
    meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY = calculateMeanShape(shapes)
    detector = load('temp_strong_regressor_saved')
    # strongRegressors[0] = detector
    # strongRegressors = load('temp_strong_regressor_saved')
    # detector = FaceDetector(meanShape, strongRegressors)
    # window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Rectangle', 1000, 800)
    for i in range(0,20):
        predictedShape = detector.detectFace(I[i])
        index = random.randint(0,n-1)
        startingShape = shapes[index]
        rectangle, im = detectFaceRectangle(I[i])
        startingShapeAdjusted = adjustToFit(startingShape, rectangle)
        # predictedShape = FaceDetector.detectFace(FaceDetector(startingShape, strongRegressors), I[i]) # used to be meanShape
        image = markImage(I[i], startingShapeAdjusted, color=0)
        image = markImage(image, predictedShape)
        displayImage(image)
        # res = cv2.resize(image, (1000, 800))
        # cv2.imshow('Rectangle', res)
        # cv2.waitKey()
        # cv2.imwrite(resultsPath + testPath + str(i) + '.jpg', image)
if __name__ == '__main__':
    # test_learnFaceDetector(debug=True, saveDetector=False)
    # testFaceDetector()
    # test()
    showFaceDetector()
    # unittest.main()
    markTime()
