
'''
'''
import sys, time
import cv2
import dlib
import numpy as np
import random
import math
import pickle
from FaceDetector import * # TODO change to follow best practices
from StrongRegressor import *
from WeakRegressor import *
from MathFunctions import *
from HelperFunctions import *
from CommonFunctions import *
from Settings import *
random.seed()
def calculateSimilarityTransforms():
    global similarityTransforms
    similarityTransforms = [calculateSimilarityTransform(meanShape, shapeEstimates[i]) for i in range(N)]
    return similarityTransforms
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
                residuals[i] = shapeDeltas[i] - strongRegressors[t].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i]) # strongRegressor[t] is the current collection of weak regressors g_1..g_k_1 that make up f_k_1
            print "Fitting weak regression tree ", str(k+1)
            tree = fitRegressionTree()
            strongRegressors[t].add(tree)
            print tree.leaves()
            if k % 20 == 0:
                saveDetector(strongRegressors[t], 'weak_regressors_' + str(k+1) + '.pkl')
        print "Updating shape estimates"
        updateShapes(t)
        saveDetector(strongRegressors[t], 'strong_regressor_' + str(t+1) + '.pkl')
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
            cv2.imwrite(tempPath + '_' + str(t+1) + '.jpg', image)
    faceDetector = FaceDetector(strongRegressors)
    if(save):
        save(faceDetector, resultsPath)
    return faceDetector
if __name__ == '__main__':
    detector = learnFaceDetector()
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
    cv2.imwrite(testPath + '.jpg', image)
