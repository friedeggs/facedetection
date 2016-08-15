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
        print '{:<{}s}'.format(log, 40), "\t\t -- Total time elapsed: %9.2fs, Time since last: %9.2f" % ((thisTime - startTime), (thisTime - lastTime))
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
    return image
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
