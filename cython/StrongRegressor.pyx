# [CHECKED]
import numpy as np
cimport numpy as np
from Settings import lr
class StrongRegressor:
    def __init__(self, base):
        self.baseFunction = np.copy(base)
        self.weakRegressors = []
    def add(self, weakRegressor):
        self.weakRegressors.append(weakRegressor)
    def eval(self, image, shapeEstimate, shapeTransform):
        # res = applyInverseTransform(shapeTransform, self.baseFunction)
        res = np.copy(self.baseFunction)
        for weakRegressor in self.weakRegressors:
            res += lr * weakRegressor.eval(image, shapeEstimate, shapeTransform) # TODO is it self.baseFunction? or is it shapeEstimate?
        return res
