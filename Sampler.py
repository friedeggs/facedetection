import numpy as np
import random
from MathFunctions import prior, adjustPoints

class Sampler:
    def __init__(self, K, S, P, F):
        self.K = K
        self.S = S
        self.P = P
        self.F = F

    def samplePixels(self, meanWidthX, meanHeightX, meanWidthY, meanHeightY):
        height = meanHeightY-meanHeightX
        points = [(random.randint(meanWidthX, meanWidthY), random.randint(meanHeightX-height/3, meanHeightY+height/7)) for i in range(self.P)] # TODO confirm this adjustment
        pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
        priorWeights = [prior(p[0], p[1]) for p in pairs]
        total = sum(priorWeights)
        priorWeights = [x / total for x in priorWeights]
        self.samplePairs = pairs
        self.presampledPairs = np.random.choice(len(self.samplePairs), self.K*self.S*(2**self.F), p=priorWeights)
        # presampledPairs = np.random.choice(len(samplePairs), 20*20*10, p=priorWeights)
        self.counter = 0

    def samplePair(self):
        self.counter += 1
        return self.samplePairs[self.presampledPairs[self.counter-1]]
        # return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]