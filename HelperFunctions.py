import sys, time
import cv2
import dlib
import numpy as np
import random
import math
import pickle
import os
curdir=os.path.dirname(__file__)
random.seed()
startTime = time.time()
lastTime = startTime
resultsPath = 'results/'
window = cv2.namedWindow('Face Detector', cv2.WINDOW_NORMAL)
PRINT_TIME_STATS = True
def setPrintOptions(printTimeStats):
    global PRINT_TIME_STATS
    PRINT_TIME_STATS = printTimeStats
def load(filename):
    f = open(os.path.join(curdir, resultsPath) + filename + '.pkl', 'r')
    obj = pickle.load(f)
    f.close()
    return obj
def save(obj, filename):
    f = open(os.path.join(curdir, resultsPath) + filename + '.pkl', 'w')
    pickle.dump(obj, f)
    f.close()
def output(s):
    print s
def mark(log):
    global lastTime
    thisTime = time.time()
    if PRINT_TIME_STATS:
        print '{:<{}s}'.format(log, 55), " -- Total time elapsed: %9.2fs, Time since last: %9.2f" % ((thisTime - startTime), (thisTime - lastTime))
    lastTime = thisTime
def markImage(im, predictedShape, markSize=3, color=255):
    image = im.copy()
    width, height = np.shape(image)
    for a,b in predictedShape:
        a = int(a)
        b = int(b)
        for i in range(a-markSize, a+markSize):
            for j in range(b-markSize,b+markSize):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = color
    if len(predictedShape) == 30: # keypoints
        points = [
            0, 10, 20, 30, 40, # face
            45, 49, 54, # nose l-to-r
            58, 70, # mouth corners l-to-r
            65, 93, 106, 78, # mouth middles top to bottom
            119, 124, 129, 133, # left eye
            139, 144, 149, 153, # right eye
            160, 164, 168, # right eyebrow
            180, 184, 188 # left eyebrow
        ]
        image = drawPolygon(image, predictedShape[:5], markSize, color, False)
        image = drawPolygon(image, predictedShape[5:8], markSize, color, False)
        image = drawPolygon(image, [predictedShape[i] for i in [8, 10, 9, 13, 8, 11, 9, 12]], markSize, color) # sort mouth points
        image = drawPolygon(image, predictedShape[14:18], markSize, color)
        image = drawPolygon(image, predictedShape[18:22], markSize, color)
        image = drawPolygon(image, predictedShape[22:26], markSize, color)
        image = drawPolygon(image, predictedShape[26:30], markSize, color)
    return image

def drawPolygon(im, points, markSize=3, color=255, connect=True):
    numpoints = len(points)
    for i in range(numpoints-1):
        cv2.line(im, tuple(map(int, points[i])), tuple(map(int, points[i+1])), color, (markSize+1)/2)
    if connect:
        cv2.line(im, tuple(map(int, points[numpoints-1])), tuple(map(int, points[0])), color, (markSize+1)/2)
    return im

def drawRect(im, rect, color=255, thickness=5):
    x,y,w,h = rect
    cv2.line(im, (x,y), (x,y+h), color, thickness)
    cv2.line(im, (x,y+h), (x+w,y+h), color, thickness)
    cv2.line(im, (x+w,y+h), (x+w,y), color, thickness)
    cv2.line(im, (x+w,y), (x,y), color, thickness)
    return im
def saveImage(image, path=resultsPath):
    cv2.imwrite(os.path.join(curdir, path) + '_temp_' + str(x) + '.jpg', image) # TODO
def displayImage(image, width=1000, height=800):
    cv2.resizeWindow('Face Detector', width, height)
    image = cv2.resize(image, (width, height))
    cv2.imshow('Face Detector', image)
    cv2.waitKey()
