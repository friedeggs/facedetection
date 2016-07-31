# [CHECKED]
from profilestats import profile
import numpy as np
import random
from Settings import *
from MathFunctions import warpPoint, adjustPoints, closest, normalize
from HelperFunctions import markImage, displayImage, drawRect
import cv2
# random.seed() # also should be here
class Node:
    meanDelta = []
def __init__(self, tau, u, v):
    self.tau = tau
    self.u = u
    self.v = v
def split_diff(image, node, meanShape, shapeEstimate, similarityTransform, adjustment):
    tau, u, v = node
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    return im_u - im_v
def splitPoints2(I, pi, meanShape, Q, theta): # [CHECKED]
    tau, u, v = theta
    thresholds = [split_diff(I[pi[i]], theta, meanShape, shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]) for i in Q]
    total = np.array(sorted(zip(thresholds,Q))) # increasing # TODO rough, possibly inefficient. could just get the two threshold elements
    cutoff = random.randint(1, len(Q)-1) # includes len(Q)-1; cutoff includes that element and up, so each side always has at least one element
    tau = (total[cutoff-1][0] + total[cutoff][0])/2 # TODO should be changed
    while cutoff < len(Q) and total[cutoff-1][0] == total[cutoff][0]: # TODO how should random cutoff be picked in the case of duplicated thresholds?
        cutoff += 1 # could potentially get 0 in this case!
    left = total[cutoff:][:,1] # includes cutoff
    right = total[:cutoff][:,1]
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
    # Q_l_test, Q_r_test, _ = splitPoints(I, pi, meanShape, Q, theta)
    # np.testing.assert_equal(sorted(Q_l), sorted(Q_l_test), err_msg=str(sorted(Q_l)) + " " + str(sorted(Q_l_test)) + " " + str(sorted(Q_r)) + " " + str(sorted(Q_r_test)))
    # np.testing.assert_equal(sorted(Q_r), sorted(Q_r_test), err_msg=str(sorted(Q_l)) + " " + str(sorted(Q_l_test)) + " " + str(sorted(Q_r)) + " " + str(sorted(Q_r_test)))
    if len(Q_l) == 0:
        mu_theta_l = 0
        mu_theta_r = np.mean([residuals[i] for i in Q_r], 0)
        # assert(1==0) # throw error
    else:
        mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)
        if len(Q_r) == 0:
            mu_theta_r = 0
            # assert(1==0) # throw error
        else:
            mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)
    # np.testing.assert_almost_equal(mu_theta_r, np.mean([residuals[i] for i in Q_r], 0))
    val = len(Q_l) * np.linalg.norm(mu_theta_l) + len(Q_r) * np.linalg.norm(mu_theta_r)
    return val, Q_l, Q_r, mu_theta_l, mu_theta_r, theta
def split(image, node, meanShape, shapeEstimate, similarityTransform, adjustment, show=False):
    tau, u, v = node
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    if show:
        adjustedMeanShape = adjustPoints(meanShape, adjustment)
        u0 = adjustPoints(u, adjustment)
        v0 = adjustPoints(v, adjustment)
        im = markImage(image, np.array([u1, v1, u0, v0]), markSize=5)
        im = markImage(im, adjustedMeanShape, color=0)
        im = markImage(im, shapeEstimate, color=255)
        thickness = 5
        # im = markImage(im, shapeEstimate + normalize(residuals[i] + Node.meanDelta, adjustment), color=255)
        cv2.line(im, tuple(map(int, u0)), tuple(map(int, v0)), 0, thickness+3)
        cv2.line(im, tuple(map(int, u1)), tuple(map(int, v1)), 255, thickness)
        displayImage(im)
    if im_u - im_v > tau:
        return 1
    else:
        return 0
def showSplits(I, pi, node, meanShape, Q, residuals):
    tau, u, v = node
    thickness = 5

    for i in Q:
        image = I[pi[i]]
        shapeEstimate, similarityTransform, adjustment = shapeEstimates[i], similarityTransforms[i], imageAdapters[pi[i]]

        u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
        v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
        # u1 = adjustPoints(u1, adjustment) # CONDEMNED
        # v1 = adjustPoints(v1, adjustment)
        w, h = np.shape(image)
        # u0 = shapeEstimate[closest(u, meanShape)]
        # v0 = shapeEstimate[closest(v, meanShape)]
        adjustedMeanShape = adjustPoints(meanShape, adjustment)
        # u0 = adjustedMeanShape[closest(u, meanShape)]
        # v0 = adjustedMeanShape[closest(v, meanShape)]
        u0 = adjustPoints(u, adjustment)
        v0 = adjustPoints(v, adjustment)
        im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
        im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0

        im = markImage(image, np.array([u1, v1, u0, v0]), markSize=5)
        im = markImage(im, adjustedMeanShape, color=0)
        im = markImage(im, shapeEstimate, color=255)
        # im = markImage(im, shapeEstimate + normalize(residuals[i] + Node.meanDelta, adjustment), color=255)
        cv2.line(im, tuple(map(int, u0)), tuple(map(int, v0)), 0, thickness+3)
        cv2.line(im, tuple(map(int, u1)), tuple(map(int, v1)), 255, thickness)
        # cv2.line(im, tuple(map(int, u1)), tuple(map(int, v1)), thickness)
        if im_u - im_v > tau:
            im = drawRect(im, (0,0,w,h), color=255)
        else:
            im = drawRect(im, (0,0,w,h), color=0)
        displayImage(im)
