'''
# training time: about 1 hour with a single CPU on the HELEN dataset
# runtime: about 1 millisecond per image
'''

# Yet the improvement
# displayed may not be saturated because we know that the
# underlying dimension of the shape parameters are much
# lower than the dimension of the landmarks (194×2). There
# is, therefore, potential for a more significant improvement
# with partial labels by taking explicit advantage of the correlation
# between the position of landmarks. Note that the gradient boosting procedure described in this paper does not
# take advantage of the correlation between landmarks. This
# issue could be addressed in a future work.

import sys, time # time.time() eg 1465655128.768409
import cv2
import dlib
import numpy as np

lr = 0.1 # learning rate (v)
T = 10 # number of strong regressors, r_t
# each r is composed of K weak regressors, g_k
K = 500 # number of weak regressors
F = 5 # depth of trees used to represent g_k
P = 400 # number of pixel locations sampled from the image at each level of the cascade

S = 20 # number of random potential splits
n = 2000 # number of training images (placeholder number)
R = 20 # number of initializations for each training example
N = n*R # training examples # N = nR where R is the number of initializations per face

# averaging predictions of multiple regression trees as alternative to learning rate:
M = 10 # fit multiple trees to the residuals in each iteration of the gradient boosting algorithm and average the result
# lr = 1 # lr * M, or 0.1 * 10

lmbda = 0.1 # exponential prior parameter
# untried extension: use cross validation when learning each strong regressor in the cascade to select this parameter

# to train weak regressors, we randomly sample a pair of these P pixel locations according to our prior and choose a random threshold to create a potential split as described in equation 9 (the h thresholding split equation)

def find_best_split():
    # sample a point based on our prior and a random threshold to create our potential split
    # take the best one that optimizes our objective


def prior(u,v):
    return exp(-lmbda||u-v||) # what is lambda

# only once at each level of the cascade
# "In practice the assignments and local translations are determined
# during the training phase."
def calculate_similarity_transform(x, t):
    s, R = where sum of square x, s_i R_i x_i + t_i is minimum
    return s, R

def warp_points(u, v):
    # u,v are points in coordinate system of mean shape
    k_u is argmin dist(mean shape_k - u)
    delta_x_u = u - x_k_u # offset from u
    s, R = calculate_similarity_transform(x, t)
    u^ = = x_i,k_u + 1/s_i * R_i^T * delta_x_u

    is minimum

    # same for v^

def split(tau, u^, v^):
    if I_pi_i(u^) - I_pi_i(v^) > tau:
        return 1
    else
        return 0

def choose_node_split():
    # our function is approximated as a piecewise constant function fit to each leaf node:
    # left node = one constant
    # right node = another constant
    # we split based on a theta

    # how do we split by theta? AHHHH....wait.........
    # theta is the three parameter vector in the previous section (tau, u, v)
    # u is the left
    # v is the right
    # tau is ? ...??

    # to choose the theta, we generate a set of random thetas and then take the best candidate of these based on mimimizing the sum of square error

    Q is the set of indices of the training examples at a node

    # Q_theta, l is the indices of the examples that are sent to the left node due to the decision induced by theta

    # r_i is the vector of all the residuals computed for image i in the gradient boosting algorithm

    # we define:
    mu_theta, l = 1 / |Q_theta, l| * sum r_i over i in Q_theta, l

    # mu_theta, r can be calculated from mu_theta, l by:
    mu_theta, r = (|Q|*mu - |Q_theta, l| * mu_theta, l) / Q_theta, r

    # choose theta to minimize
        # argmin E(Q,theta) = sum {l,r} sum {i in Q} ||r_i - mu_theta, {l | r}||^2

    # in fact this can be done efficiently because
    # argmin E(Q, theta) = argmax sum {l,r} |Q_theta, s| * transpose (mu_theta, s) * mu_theta, s # squaring mu_theta, s

    # so:
    return argmax sum {l,r} |Q_theta, s| * transpose (mu_theta, s) * mu_theta, s



for t in range(1, T):
    # learn the regression function using gradient boosting and a sum of square error loss
    f0 is the point at which the distance from that point to all the other points in the delta shape is at a minimum

    calculate_similarity_transform() # think this goes here? carelessly placed for now
    for k in range(1, K):
        for i in range(1, N):
            r_ik is the delta shape - f of k-1 # compute residuals

        # regression
        # fit a regression tree to the targets r_ik giving a weak regression function g_k(I, S^(t))

        node_split = choose_node_split()
        split(node_split)
        # ~~choose closer pixel pairs by using an exponential prior~~ updated

        fk = f_(k-1) + lr * g_k
    r_t = f_K # done

    # update the training set
    S_(t+1) = S_t + r_t()
    delta_S_(t+1) = actual shape S - S_(t+1)
