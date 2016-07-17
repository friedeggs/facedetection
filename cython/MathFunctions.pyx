import math
import numpy as np
cimport numpy as np
from Settings import *
def prior(u,v):
    return math.exp(-lmbda*np.linalg.norm(np.subtract(u,v)))
def calculateSimilarityTransform(np.ndarray[DTYPE_t, ndim=2] w, np.ndarray[DTYPE_t, ndim=2] v):
    ''' Calculate similarity transform for a given face estimate '''
    cdef np.ndarray[DTYPE_t, ndim=2] center_w, center_v, B, U, s, V1, S, M, R, t
    cdef int m, n
    cdef double var, varw
    center_w = np.sum(w, 0)*1./len(w)
    center_v = np.sum(v, 0)*1./len(v)
    B = np.dot(np.transpose(w - center_w), v - center_v) *1./len(w)
    U, s, V1 = np.linalg.svd(B)
    m = np.shape(U)[0]
    n = np.shape(V1)[1]
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    M = np.zeros((m, n))
    if np.linalg.det(B) >= 0:
        M = np.identity(n)
    else:
        M[:n, :n] = np.diag(np.append(np.ones(n - 1), 1))
    R = np.dot(U, np.dot(M, V1))
    var = 1./len(v) * np.sum(np.linalg.norm((v - center_v), axis=1)**2)
    varw = 1./len(w) * np.sum(np.linalg.norm((w - center_w), axis=1)**2)
    c = 1./var*np.trace(np.dot(S, M))
    t = np.transpose(np.transpose(center_w) - c * np.dot(R, np.transpose(center_v)))
    return c, R, t
def applyInverseTransform(transform, np.ndarray[DTYPE_t, ndim=2] points):
    cdef R, t
    cdef double S
    S, R, t = transform
    return 1./S * np.transpose(np.dot(np.transpose(R), np.transpose(points))) - 1./S * np.dot(np.transpose(R), t)
def applyTransform(transform, np.ndarray[DTYPE_t, ndim=2] points):
    cdef R, t
    cdef double S
    S, R, t = transform
    return S * np.transpose(np.dot(R, np.transpose(points)))
def applyRotation(transform, np.ndarray[DTYPE_t, ndim=2] points):
    cdef R, t
    cdef double S
    S, R, t = transform
    return np.transpose(np.dot(R, np.transpose(points)))
def closest(node, nodes):
    # nodes = np.asarray(nodes) # shouldn't be needed
    cdef np.ndarray[DTYPE_t, ndim=2] deltas, dist_2
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)
def warpPoint(u, np.ndarray[DTYPE_t, ndim=2] X, np.ndarray[DTYPE_t, ndim=2] Y, similarityTransform): # TODO check
    cdef R, t, u1, delta_x_u
    cdef int k_u
    cdef double S
    S, R, t = similarityTransform # global indexing
    k_u = closest(u, X) # local indexing
    delta_x_u = u - X[k_u]
    u1 = Y[k_u] + 1./S * np.dot(np.transpose(R), delta_x_u)
    return u1
