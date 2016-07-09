def splitPoints(Q, theta):
    # print theta
    tau, u, v = theta
    left, right = [], []
    for i in Q:
        left.append(i) if split(I[pi[i]], tau, u, v, shapeEstimates[i], similarityTransforms[i]) == 1 else right.append(i)
    return left, right
def tryNodeSplit(Q, mu, theta):
    # maxval = 0
    Q_l, Q_r = splitPoints(Q, theta)
    if len(Q_l) == 0:
        mu_theta_l = 0
        mu_theta_r = np.mean([residuals[i] for i in Q_r], 0)
    else:
        mu_theta_l = np.mean([residuals[i] for i in Q_l], 0)
        if len(Q_r) == 0:
            mu_theta_r = 0
        else:
            mu_theta_r = (len(Q)*mu - len(Q_l) * mu_theta_l) / len(Q_r)
    val = len(Q_l) * np.linalg.norm(mu_theta_l) + len(Q_r) * np.linalg.norm(mu_theta_r)
    # if val > maxval:
    #     maxval = val
    #     argmax = theta
    return val, Q_l, Q_r, mu_theta_l, mu_theta_r
def split(image, tau, u, v, shapeEstimate, similarityTransform):
    u1 = warpPoint(u, meanShape, shapeEstimate, similarityTransform)
    v1 = warpPoint(v, meanShape, shapeEstimate, similarityTransform)
    # print image[u1[0]][u1[1]]
    # print image[v1[0]][v1[1]] # TODO were the same
    w, h = np.shape(image)
    im_u = int(image[u1[1],u1[0]]) if u1[1] >= 0 and u1[1] < w and u1[0] >= 0 and u1[0] < h else 0 # TODO is this logically valid?
    im_v = int(image[v1[1],v1[0]]) if v1[1] >= 0 and v1[1] < w and v1[0] >= 0 and v1[0] < h else 0
    if im_u - im_v > tau:
    # if int(image[u1[1],u1[0]]) - int(image[v1[1],v1[0]]) > tau: # doesn't matter
    # if int(image[u1[0]][u1[1]]) - int(image[v1[0]][v1[1]]) > tau:
        return 1
    else:
        return 0
