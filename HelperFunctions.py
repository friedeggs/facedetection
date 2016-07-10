import sys, time
import cv2
import dlib
import numpy as np
import random
import math
import pickle
from Settings import *
random.seed()
startTime = time.time()/100
lastTime = startTime
def load(filename):
    f = open(resultsPath + filename + '.pkl', 'r')
    obj = pickle.load(f)
    f.close()
    return obj
def save(obj, filename):
    f = open(resultsPath + filename + '.pkl', 'w')
    pickle.dump(obj, f)
    f.close()
def markTime():
    global lastTime
    thisTime = time.time()/100
    print "\t\t -- Total time elapsed: %7.2fs, Time since last: %7.2f" % ((thisTime - startTime), (thisTime - lastTime))
    lastTime = thisTime
def markImage(im, predictedShape, markSize=5):
    image = im.copy()
    width, height = np.shape(image)
    for a,b in predictedShape:
        a = int(a)
        b = int(b)
        for i in range(a-markSize, a+markSize):
            for j in range(b-markSize,b+markSize):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 255
    return image
def saveImage(image, path=resultsPath):
    cv2.imwrite(path + '_temp_' + str(x) + '.jpg', image) # TODO
