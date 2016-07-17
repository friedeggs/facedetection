import numpy as np
cimport numpy as np
import random
from Settings import *
from MathFunctions import prior
def samplePixels(int meanWidthX, int meanHeightX, int meanWidthY, int meanHeightY):
    cdef np.ndarray[DTYPE_t, ndim=2] samplePairs, priorWeights, pairs
    global samplePairs, priorWeights
    cdef np.ndarray[DTYPE_t, ndim=2] points # TODO check dimension
    cdef int i, j
    points = np.empty((len(P),1)
    pairs = np.zeros((len(P),len(P))
    priorWeights = np.zeros((len(P),len(P))
    for i in range(P):
        points[i] = (random.randint(meanWidthX, meanWidthY), random.randint(meanHeightX, meanHeightY))
    cdef double total = 0
    for i in range(P):
        for j in range(P):
            if i != j:
                pairs[i][j] = (points[i], points[j])
                if i < j:
                    priorWeights[i][j] = prior(p[0], p[1])
                    total += priorWeights[i][j]
                else:
                    priorWeights[i][j] = priorWeights[j][i]
    # cdef double total = sum(priorWeights)
    total *= 2
    for i in range(P):
        for j in range(P):
            if i != j:
                priorWeights[i][j] /= total
    samplePairs = pairs
    return points, pairs, priorWeights
def samplePair():
    return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]
def generateCandidateSplit():
    pair = samplePair()
    threshold = random.randint(76, 178) # random.randint(0, 255) # TODO placeholder
    return threshold, pair[0], pair[1] # TODO made in haste
