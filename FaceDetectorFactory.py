
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
shapes = [[] for i in range(n)]
I = [[] for i in range(n)]
residuals = [[] for i in range(N)]
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
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg', cv2.IMREAD_GRAYSCALE)
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
    global shapeEstimates, shapeDeltas, shapes, pi
    pi = np.repeat(np.arange(N), R)
    for i in range(n):
        sample = random.sample(range(n), R) # array of length 20 containing indices
        for j in range(R):
            shapeEstimates[i*R+j] = shapes[sample[j]] # TODO not a problem
    for i in range(N):
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
    return pi
def updateShapes(t):
    global shapeEstimates, shapeDeltas, strongRegressors, shapes, similarityTransforms
    for i in range(N):
        shapeEstimates[i] += strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i])
        shapeDeltas[i] = shapes[pi[i]] - shapeEstimates[i]
def learnFaceDetector(saveDetector=True, test=True, saveIntermediates=False):
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
            print "Computing residuals"
            for k in range(K):
                for i in range(N):
                    ''' Evaluate on each image to calculate residuals '''
                    residuals[i] = shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
                print "Fitting weak regression tree ", str(k+1)
                tree = fitRegressionTree(I, pi, meanShape, residuals)
                markTime()
                strongRegressors[t].add(tree)
                print tree.leaves()
                if saveIntermediates and k % 20 == 0:
                    save(strongRegressors[t], 'weak_regressors_' + str(t) + '-' + str(k))
            print "Updating shape estimates"
            updateShapes(t)
            markTime()
            save(strongRegressors[t], tempPath + 'strong_regressor_' + str(t+1))
            if test:
                # predictedShape = detectFace(strongRegressors, I[0])
                predictedShape = FaceDetector.detectFace(FaceDetector(meanShape, strongRegressors), I[0])
                image = markImage(I[0], predictedShape)
                width, height = np.shape(image)
                cv2.imwrite(tempPath + str(t+1) + '.jpg', image)
    finally:
        save(strongRegressors, tempPath + 'strong_regressor_saved')
        markTime()
    faceDetector = FaceDetector(meanShape, strongRegressors)
    if(saveDetector):
        save(faceDetector, 'face_detector')
    return faceDetector
def test():
    loadData()
    calculateMeanShape()
    # detector = load('face_detector')
    strongRegressors = load('temp_strong_regressor_saved')
    detector = FaceDetector(meanShape, strongRegressors)
    for i in range(10,20):
        predictedShape = detector.detectFace(I[i])
        # print predictedShape[:2]
        imageCenter = np.array(np.shape(I[i]))/2
        shapeCenter = np.mean(predictedShape, 0)
        # predictedShape -= (shapeCenter - imageCenter)
        image = markImage(I[i], predictedShape)
        cv2.imwrite(resultsPath + testPath + str(i) + '.jpg', image)
if __name__ == '__main__':
    # detector = learnFaceDetector(saveIntermediates=True)
    test()
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
