import numpy as np
import random
from MathFunctions import prior, adjustPoints

class Sampler:
    def __init__(self, numPoints, numSamples):
        self.numPoints = numPoints
        self.numSamples = numSamples

    def samplePixels(self, meanWidthX, meanHeightX, meanWidthY, meanHeightY):
        random.seed()
        height = meanHeightY-meanHeightX
        points = [(random.randint(meanWidthX, meanWidthY), random.randint(meanHeightX-height/3, meanHeightY+height/7)) for i in range(self.numPoints)] # TODO so arbitrary
        pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
        priorWeights = [prior(p[0], p[1]) for p in pairs]
        total = sum(priorWeights)
        priorWeights = [x / total for x in priorWeights]
        self.samplePairs = pairs
        self.presampledPairs = np.random.choice(len(self.samplePairs), self.numSamples, p=priorWeights)
        self.counter = 0

    def samplePair(self):
        self.counter += 1
        return self.samplePairs[self.presampledPairs[self.counter-1]]
        # return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]