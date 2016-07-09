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
