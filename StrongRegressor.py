# [CHECKED]
import numpy as np
from Settings import lr
from MathFunctions import normalize
class StrongRegressor:
    def __init__(self, base):
        self.baseFunction = np.copy(base)
        self.weakRegressors = []
    def add(self, weakRegressor):
        self.weakRegressors.append(weakRegressor)
    def eval(self, image, shapeEstimate, shapeTransform, adjustment):
        # res = applyInverseTransform(shapeTransform, self.baseFunction)
        res = np.copy(self.baseFunction)
        for weakRegressor in self.weakRegressors:
            res += lr * normalize(weakRegressor.eval(image, shapeEstimate, shapeTransform, adjustment), adjustment) # TODO is it self.baseFunction? or is it shapeEstimate?
            # print res[0][0]
        return res
