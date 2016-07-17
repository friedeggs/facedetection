# [CHECKED]
import numpy as np
import sys
from Settings import *
from CommonFunctions import generateCandidateSplit
from Node import tryNodeSplit, split
class RegressionTree:
    def __init__(self, node, meanShape=None, depth=1, leftTree=None, rightTree=None):
        self.node = node
        self.depth = depth
        self.leftTree = leftTree
        self.rightTree = rightTree
        self.meanShape = meanShape
    def eval(self, image, shapeEstimate, shapeTransform, adjustment): # warp based on shapeEstimate which is based off result from StrongRegressor
        if self.depth == 1: # leaf
            # print self.node[:1]
            return self.node # need to transform???? No, right? because these are the residuals, which we did not transform when computing
        # sys.stdout.write(str(split(image, self.node, self.meanShape, shapeEstimate, shapeTransform)))
        if split(image, self.node, self.meanShape, shapeEstimate, shapeTransform, adjustment) == 1:
            return self.leftTree.eval(image, shapeEstimate, shapeTransform, adjustment)
        else:
            return self.rightTree.eval(image, shapeEstimate, shapeTransform, adjustment)
    def leaves(self):
        if self.depth == 1: # leaf
            return self.node[:5]
        return self.leftTree.leaves() #, self.rightTree.leaves()
def fitRegressionTree(I, pi, meanShape, residuals):
    mu = np.mean(residuals, 0)
    tree = fitNode(I, pi, meanShape, range(N), mu, F, residuals)
    return tree
def fitNode(I, pi, meanShape, Q, mu, depth, residuals):
    if depth == 1 or len(Q) == 1: # TODO check if should be 0 instead
        return RegressionTree(mu) # Leaf node
    maxval = 0
    for i in range(S):
        candidateSplit = generateCandidateSplit()
        # print "candidate split: ", candidateSplit
        val, q_l, q_r, mu_l0, mu_r0, candidateSplit = tryNodeSplit(I, pi, meanShape, Q, mu, candidateSplit, residuals)
        if val > maxval:
            maxval = val
            split = candidateSplit
            # print "fitnode: ", split
            Q_l = q_l
            Q_r = q_r
            mu_l = mu_l0
            mu_r = mu_r0
    tree = RegressionTree(split, meanShape, depth)
    if depth > 1:
        tree.leftTree = fitNode(I, pi, meanShape, Q_l, mu_l, depth - 1, residuals)
        tree.rightTree = fitNode(I, pi, meanShape, Q_r, mu_r, depth - 1, residuals)
    return tree
