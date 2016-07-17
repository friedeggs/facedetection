
'''
'''
import sys
import cv2
import numpy as np
from FaceDetector import * # TODO change to follow best practices
from StrongRegressor import StrongRegressor
from WeakRegressor import *
from MathFunctions import calculateSimilarityTransform
from HelperFunctions import *
from CommonFunctions import *
from Settings import *
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
    # similarityTransforms = [calculateSimilarityTransform(meanShape, shapeEstimates[i]) for i in range(N)]
    for i in range(N):
        similarityTransforms[i] = calculateSimilarityTransform(meanShape, shapeEstimates[i]) # IMPORTANT do not use list comprehension
def groundEstimate(shapes):
    return np.mean(shapes, axis=0)
def loadData(): # [CHECKED]
    global shapes, I
    ''' Load images?!?!?! and shapes '''
    for i in range(n):
        filePath = basePath + 'annotation/' + str(i+1) + '.txt'
        imagePath = ""
        with open(filePath, 'r') as f:
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = np.array([[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()])
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg', 0)# cv2.IMREAD_GRAYSCALE)
def calculateMeanShape(): # [CHECKED]
    ''' Calculate mean shape and bounding box shape of all faces '''
    # TODO placeholder implementation right now
    global meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY
    meanShape = np.mean(shapes, 0)
    meanWidthX, meanHeightX = np.min(meanShape, 0).astype(int)
    meanWidthY, meanHeightY = np.max(meanShape, 0).astype(int)
    # meanWidth, meanHeight = np.max(meanShape - np.min(meanShape, 0), 0).astype(int)
def generateTrainingData(): # [CHECKED]
    # pi = np.random.permutation(np.repeat(np.arange(N), R)) # why does it even need to be random? order never matters
    global shapeEstimates, shapeDeltas, shapes, pi, imageAdapters
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
        imageAdapters[i] = adjustToFit(meanShape, rectangles[i], adapterOnly=True) # TODO same code is run in here and shapeEstimates[i*R+j] line
    for i in range(n):
        sample = random.sample(range(n), R) # array of length 20 containing indices
        for j in range(R): # for efficiency don't call detectFaceRectangle R times
            # x,y = shapes[i].min(0)
            # X,Y = shapes[i].max(0)
            # # shapeEstimates[i*R+j] = adjustToFit(shapes[i], detectFaceRectangle(I[i])) # TODO not a problem
            # # shapeEstimates[i*R+j] = adjustToFit(shapes[sample[j]], (x,y,X-x,Y-y)) # good but not what it's supposed to be
            # shapeEstimates[i*R+j] = adjustToFit(meanShape, (x,y,X-x,Y-y))
            shapeEstimates[i*R+j] = adjustToFit(meanShape, rectangles[i])
            res = markImage(I[i].copy(), shapeEstimates[i*R+j])
            # res = cv2.resize(res, (1000, 800))
            # cv2.imshow('Rectangle', res)
            # cv2.waitKey()
    for i in range(N):
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
    return pi
def updateShapes(t):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms
    for i in range(N):
        shapeEstimates[i] += strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]])
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
def learnFaceDetector(saveDetector=True, test=True, saveIntermediates=False, debug=True):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms, residuals, samplePoints, samplePairs, priorWeights
    try:
        print "Loading data"
        loadData()
        markTime()
        print "Calculating mean shape"
        calculateMeanShape()
        markTime()
        print "Generating training data"
        generateTrainingData()
        markTime()
        for t in range(T):
            print "Sampling pixels"
            samplePixels(meanWidthX, meanHeightX, meanWidthY, meanHeightY)
            markTime()
            ''' Get mean shape '''
            print "Learning strong regressor ", str(t+1)
            strongRegressors[t] = StrongRegressor(groundEstimate(shapeDeltas))
            print "Calculating similarity transforms"
            calculateSimilarityTransforms()
            markTime()
            ''' Calculate similarity transforms for each shape estimate '''
            # print "Computing residuals"
            # for j in range(10):
            #     # predictedShape = detectFace(strongRegressors, I[0])
            #     predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[j])
            #     image = markImage(I[j], predictedShape)
            #     width, height = np.shape(image)
            #     cv2.imwrite(resultsPath + 'debug_' + str(t+1) + '_' + str(j) + '.jpg', image)
            #     markTime()
            # raw_input()
            for k in range(K):
                for i in range(N):
                    ''' Evaluate on each image to calculate residuals '''
                    residuals[i] = shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
                print "Fitting weak regression tree ", str(k+1)
                tree = fitRegressionTree(I, pi, meanShape, residuals)
                markTime()
                strongRegressors[t].add(tree)
                print tree.leaves()
                if saveIntermediates and (k+1) % 20 == 0:
                    save(strongRegressors[t], 'weak_regressors_' + str(t) + '-' + str(k+1))
                if debug and (k+1) % 10 == 0:
                    for j in range(20):
                        # predictedShape = detectFace(strongRegressors, I[0])
                        predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[j])
                        image = markImage(I[j], predictedShape)
                        width, height = np.shape(image)
                        cv2.imwrite(resultsPath + 'debug_' + str(t+1) + '_' + str(j) + '.jpg', image)
                        markTime()
                    raw_input()
            print "Updating shape estimates"
            updateShapes(t)
            markTime()
            save(strongRegressors[t], tempPath + 'strong_regressor_' + str(t+1))
            if test:
                for i in range(20):
                    # predictedShape = detectFace(strongRegressors, I[0])
                    predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[i])
                    image = markImage(I[i], predictedShape)
                    width, height = np.shape(image)
                    cv2.imwrite(resultsPath + tempPath + str(t+1) + '_' + str(i) + '.jpg', image)
                    markTime()
    finally:
        save(strongRegressors, tempPath + 'strong_regressor_saved')
        markTime()
    faceDetector = FaceDetector(meanShape, strongRegressors)
    if(saveDetector):
        save(faceDetector, 'face_detector_4')
    return faceDetector
def test():
    loadData()
    # calculateMeanShape()
    detector = load('face_detector_4')
    # strongRegressors = load('temp_strong_regressor_saved')
    # detector = FaceDetector(meanShape, strongRegressors)
    for i in range(0,20):
        predictedShape = detector.detectFace(I[i])
        # print predictedShape[:2]
        imageCenter = np.array(np.shape(I[i]))/2
        shapeCenter = np.mean(predictedShape, 0)
        predictedShape -= (shapeCenter - imageCenter)
        image = markImage(I[i], predictedShape)
        cv2.imwrite(resultsPath + testPath + str(i) + '.jpg', image)
def testFaceDetector():
    loadData()
    calculateMeanShape()
    global shapeEstimates, shapeDeltas, shapes, pi
    FaceDetector.meanRectangle = (
                meanWidthX,
                meanHeightX,
                meanWidthY - meanWidthX,
                meanHeightY - meanHeightY)
    window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    width = 1000
    height = 800
    cv2.resizeWindow('Rectangle', 1000, 800)
    for i in range(n):
        im = I[i].copy()
        rect, im2 = detectFaceRectangle(I[i]) # TODO not a problem
        # rect2 = adjustRect(rect)
        x,y,w,h = rect
        # im = I[i] #.copy()
        thickness = 5
        cv2.line(im, (x,y), (x,y+h), thickness)
        cv2.line(im, (x,y+h), (x+w,y+h), thickness)
        cv2.line(im, (x+w,y+h), (x+w,y), thickness)
        cv2.line(im, (x+w,y), (x,y), thickness)
        im = markImage(im, adjustToFit(meanShape, rect))
        res = cv2.resize(im,(width, height))
        cv2.imshow('Rectangle', res)
        cv2.waitKey()
if __name__ == '__main__':
    detector = learnFaceDetector(saveIntermediates=True)
    # testFaceDetector()
    # test()
    # strongRegressors = load('temp_strong_regressor_saved_')
    # print strongRegressors[0].weakRegressors[0].node[:5]
    markTime()
def displayPrediction(im, predictedShape, show=False, savePath=None):
    image = im.copy()
    width, height = np.shape(image)
    s = 5
    for a,b in predictedShape:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 255
    for a,b in meanShape:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 0
    for k in range(1):
        for i in range(N):
            residuals[i] = shapeDeltas[i] - strongRegressors[0].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i])
    mu = np.mean(residuals, 0)
    for a,b in mu:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 0
    if show:
        cv2.imshow('Prediction', image)
        cv2.waitKey()
    if savePath:
        cv2.imwrite(savePath + '.jpg', image)
