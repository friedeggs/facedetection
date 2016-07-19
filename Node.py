# [CHECKED]
from profilestats import profile
import numpy as np
import random
from Settings import *
from MathFunctions import warpPoint, adjustPoints
def __init__(self, tau, u, v):
    self.tau = tau
    self.u = u
    self.v = v
def split_diff(image, node, meanShape, shapeEstimate, similarityTransform, adjustment):
    tau, u, v = node
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    u1 = adjustPoints(u1, adjustment)
    v1 = adjustPoints(v1, adjustment)
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    return im_u - im_v
def splitPoints2(I, pi, meanShape, Q, theta): # [CHECKED]
    tau, u, v = theta
    thresholds = [split_diff(I[pi[i]], theta, meanShape, shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]) for i in Q]
    total = np.array(sorted(zip(thresholds,Q))) # increasing # TODO rough, possibly inefficient. could just get the two threshold elements
    cutoff = random.randint(1, len(Q)-1) # includes len(Q)-1; cutoff includes that element and up, so each side always has at least one element
    left = total[cutoff:][:,1] # includes cutoff
    right = total[:cutoff][:,1]
    tau = (total[cutoff-1][0] + total[cutoff][0])/2 # TODO should be changed
    # print "tau:", tau
    return left, right, (tau, u, v)
def splitPoints(I, pi, meanShape, Q, theta):
    tau, u, v = theta
    left, right = [], []
    for i in Q:
        left.append(i) if split(I[pi[i]], theta, meanShape, shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]) == 1 else right.append(i)
    return left, right, theta
def tryNodeSplit(I, pi, meanShape, Q, mu, theta, residuals):
    Q_l, Q_r, theta = splitPoints2(I, pi, meanShape, Q, theta)
    if len(Q_l) == 0:
        mu_theta_l = 0
        mu_theta_r = np.mean([residuals[i] for i in Q_r], 0)
        assert(1==0) # throw error
    else:
        mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)
        if len(Q_r) == 0:
            mu_theta_r = 0
            assert(1==0) # throw error
        else:
            mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)
    np.testing.assert_almost_equal(mu_theta_r, np.mean([residuals[i] for i in Q_r], 0))
    val = len(Q_l) * np.linalg.norm(mu_theta_l) + len(Q_r) * np.linalg.norm(mu_theta_r)
    return val, Q_l, Q_r, mu_theta_l, mu_theta_r, theta
def split(image, node, meanShape, shapeEstimate, similarityTransform, adjustment):
    tau, u, v = node
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    u1 = adjustPoints(u1, adjustment)
    v1 = adjustPoints(v1, adjustment)
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    if im_u - im_v > tau:
        return 1
    else:
        return 0
