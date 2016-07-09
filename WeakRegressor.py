from Settings import *
class RegressionTree:
    def __init__(self, node, depth=1, leftTree=None, rightTree=None):
        self.node = node
        self.depth = depth
        self.leftTree = leftTree
        self.rightTree = rightTree
    def eval(self, image, shapeEstimate, shapeTransform): # warp based on shapeEstimate which is based off result from StrongRegressor
        if self.depth == 1: # leaf
            return self.node # need to transform???? No, right? because these are the residuals, which we did not transform when computing
        if split(image, self.node[0], self.node[1], self.node[2], shapeEstimate, shapeTransform) == 1:
            return self.leftTree.eval(image, shapeEstimate, shapeTransform)
        else:
            return self.rightTree.eval(image, shapeEstimate, shapeTransform)
    def leaves(self):
        if self.depth == 1: # leaf
            return self.node
        return self.leftTree.leaves(), self.rightTree.leaves()
def fitRegressionTree():
    mu = np.mean(residuals, 0)
    tree = fitNode(range(N), mu, F)
    return tree
def fitNode(Q, mu, depth):
    if depth == 1 or len(Q) == 0: # TODO check if should be 0 instead
        return RegressionTree(mu) # Leaf node
    maxval = 0
    for i in range(S):
        candidateSplit = generateCandidateSplit()
        val, q_l, q_r, mu_l0, mu_r0 = tryNodeSplit(Q, mu, candidateSplit)
        if val > maxval:
            maxval = val
            split = candidateSplit
            Q_l = q_l
            Q_r = q_r
            mu_l = mu_l0
            mu_r = mu_r0
    tree = RegressionTree(split, depth)
    # if len(Q) == N:
    #     print "------------------------------------------"
    #     print Q_l
    #     print Q_r
    if depth > 1:
        tree.leftTree = fitNode(Q_l, mu_l, depth - 1)
        tree.rightTree = fitNode(Q_r, mu_r, depth - 1)
    return tree
