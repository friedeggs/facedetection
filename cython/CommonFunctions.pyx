import numpy as np
import random
from Settings import *
from MathFunctions import prior
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
