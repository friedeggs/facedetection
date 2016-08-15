import numpy as np
from MathFunctions import normalize

class StrongRegressor:
    def __init__(self, base):
        self.baseFunction = np.copy(base)
        self.weakRegressors = []
        self.lr = 0.1
    def setLearningRate(self, learningRate):
        self.lr = learningRate
    def add(self, weakRegressor):
        self.weakRegressors.append(weakRegressor)
    def eval(self, image, estimate, adjustment):
        res = np.copy(self.baseFunction)
        for weakRegressor in self.weakRegressors:
            res += self.lr * normalize(weakRegressor.eval(image, estimate), adjustment)
        return res

class RegressionTree:
    def __init__(self, node, leftTree=None, rightTree=None):
        self.node = node
        self.leftTree = leftTree
        self.rightTree = rightTree
    def eval(self, image, estimate): # warp based on shapeEstimate which is based off result from StrongRegressor
        if self.isLeaf(): # leaf
            return self.node
        if self.node.eval(image, estimate) == 1:
            return self.leftTree.eval(image, estimate)
        else:
            return self.rightTree.eval(image, estimate)
    def leaves(self):
        if self.isLeaf():
            return self.node[:5]
        return self.leftTree.leaves() #, self.rightTree.leaves()
    def splits(self):
        if self.isLeaf():
            return []
        leftSplits = self.leftTree.splits() if self.leftTree else []
        rightSplits = self.rightTree.splits() if self.rightTree else []
        return [self.node] + leftSplits + rightSplits
    def isLeaf(self):
        return self.leftTree is None and self.rightTree is None