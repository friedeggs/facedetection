import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
basePath = '/Users/frieda/Downloads/'
resultsPath = 'results/'
tempPath = 'temp_'
testPath = 'test_'
cdef double lr, lmbda
cdef int T, K, F, P, S, n, R, N, meanWidthX, meanHeightX, meanWidthY, meanHeightY
lr = 0.1
T = 3
K = 20
F = 5
P = 400
S = 20
n = 2000
R = 2 # Use 1 initialization instead? >:(
N = n*R
lmbda = 0.1
# strongRegressors = [[] for i in range(T)]
# shapeDeltas = [[] for i in range(N)]
# pi = []
cdef np.ndarray shapes = np.array((n,1))
cdef np.ndarray shapeEstimates = np.array((N,1))
# I = [[] for i in range(n)]
# residuals = [[] for i in range(N)]
cdef np.ndarray meanShape
meanWidthX = 0
meanHeightX = 0
meanWidthY = 0
meanHeightY = 0
# samplePoints = []
# samplePairs = []
# priorWeights = []
cdef np.ndarray similarityTransforms = np.array((N,1))
