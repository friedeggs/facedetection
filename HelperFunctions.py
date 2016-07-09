from Settings import *
def load(path=resultsPath):
    f = open(path, 'r')
    obj = pickle.load(f)
    f.close()
    return obj
def save(obj, path=resultsPath):
    f = open(path, 'w')
    pickle.dump(obj, f)
    f.close()
def samplePixels():
    global samplePairs, priorWeights
    points = [(random.randint(meanWidthX, meanWidthY), random.randint(meanHeightX, meanHeightY)) for i in range(P)]
    pairs = [(points[i], points[j]) for i in range(len(points)) for j in range(len(points)) if i != j]
    priorWeights = [prior(p[0], p[1]) for p in pairs]
    priorWeights = [x / sum(priorWeights) for x in priorWeights]
    samplePairs = pairs
    return points, pairs, priorWeights
def samplePair():
    # print samplePairs
    return samplePairs[np.random.choice(len(samplePairs), p=priorWeights)]
def generateCandidateSplit():
    pair = samplePair()
    threshold = random.randint(76, 178) # random.randint(0, 255) # TODO placeholder
    return threshold, pair[0], pair[1] # TODO made in haste
def displayPrediction(im, predictedShape, show=False, savePath=None):
    image = im.copy()
    width, height = np.shape(image)
    s = 5
    for a,b in predictedShape:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 255
    for a,b in meanShape:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 0
    for k in range(1):
        for i in range(N):
            residuals[i] = shapeDeltas[i] - strongRegressors[0].eval(I[pi[i]], shapeEstimates[i], similarityTransforms[i])
    mu = np.mean(residuals, 0)
    for a,b in mu:
        a = int(a)
        b = int(b)
        for i in range(a-s, a+s):
            for j in range(b-s,b+s):
                if i < height and j < width and i >= 0 and j >= 0:
                    image[j,i] = 0
    # if show:
    #     cv2.imshow('Prediction', image)
    #     cv2.waitKey()
    if savePath:
        cv2.imwrite(savePath + '.jpg', image)
