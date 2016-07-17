import numpy as np
import random
from Settings import *
from MathFunctions import prior, adjustPoints
import cv2
from HelperFunctions import markImage
from FaceDetector import detectFaceRectangle, adjustToFit
def samplePixels(meanWidthX, meanHeightX, meanWidthY, meanHeightY):
    global samplePairs, priorWeights
    points = [(random.randint(meanWidthX, meanWidthY), random.randint(meanHeightX, meanHeightY)) for i in range(P)]
    pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
    priorWeights = [prior(p[0], p[1]) for p in pairs]
    total = sum(priorWeights)
    priorWeights = [x / total for x in priorWeights]
    samplePairs = pairs
    return points, pairs, priorWeights
def samplePair():
    return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]
def generateCandidateSplit():
    pair = samplePair()
    threshold = random.randint(76, 178) # random.randint(0, 255) # TODO placeholder
    return threshold, pair[0], pair[1] # TODO made in haste
if __name__ == '__main__':
    from FaceDetectorFactory import loadData, calculateMeanShape
    loadData()
    meanShape, meanWidthX, meanHeightX, meanWidthY, meanHeightY = calculateMeanShape()
    points, pairs, priorWeights = samplePixels(meanWidthX, meanHeightX, meanWidthY, meanHeightY)
    im = cv2.imread('/Users/frieda/Downloads/images/2351450794_1.jpg', 0)
    rect, im2 = detectFaceRectangle(im) # TODO not a problem
    # rect2 = adjustRect(rect)
    x,y,w,h = rect
    # im = I[i] #.copy()
    thickness = 5
    cv2.line(im, (x,y), (x,y+h), thickness)
    cv2.line(im, (x,y+h), (x+w,y+h), thickness)
    cv2.line(im, (x+w,y+h), (x+w,y), thickness)
    cv2.line(im, (x+w,y), (x,y), thickness)
    im = markImage(im, adjustToFit(meanShape, rect))
    adjustment = adjustToFit(meanShape, rect, adapterOnly=True)
    adjustedPoints = adjustPoints(points, adjustment)
    im = markImage(im, adjustedPoints, color=0)
    res = cv2.resize(im,(1000, 800))
    window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    cv2.imshow('Rectangle', res)
    cv2.waitKey()
    for i in range(20):
        pair = samplePair()
        adjustedPair = adjustPoints(pair, adjustment)
        adjustedPair = [[int(s) for s in p] for p in adjustedPair]
        adjustedPair = tuple(map(tuple,adjustedPair))
        # print pair, adjustPoints(pair,adjustment)
        cv2.line(im, adjustedPair[0], adjustedPair[1], color=255, thickness=10)
        im = markImage(im, adjustPoints(pair, adjustment), markSize=10)
    width = 1000
    height = 800
    cv2.resizeWindow('Rectangle', 1000, 800)
    res = cv2.resize(im, (1000, 800))
    cv2.imshow('Rectangle', res)
    cv2.waitKey()
    cv2.imwrite(resultsPath + 'prior_test.jpg', im)
