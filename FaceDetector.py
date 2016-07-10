import numpy as np
from Settings import *
from MathFunctions import calculateSimilarityTransform
class FaceDetector:
    def __init__(self, meanShape, strongRegressors):
        self.meanShape = meanShape
        self.strongRegressors = strongRegressors
    def detectFace(self, image):
        transform = (1, np.identity(2), 0) # identity transform
        predictedShape = np.copy(self.meanShape)
        for strongRegressor in self.strongRegressors:
            if strongRegressor:
                # print "predicting"
                delta = strongRegressor.eval(image, predictedShape, transform)
                predictedShape += delta
                transform = calculateSimilarityTransform(self.meanShape, predictedShape)
                # print delta[:5]
        return predictedShape
