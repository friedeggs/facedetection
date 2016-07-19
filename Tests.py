import sys, time
import cv2
import dlib
import numpy as np
import random
import math
import pickle
import unittest
import faceDetection as fd
import cv2

class Functions(unittest.TestCase):
    # def test_generate_training_data(self):
    #     ans = array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    #     self.assertEqual(np.repeat(np.arange(5), 2), ans)
    pass

if __name__ == '__main__':
    unittest.main()

    loadData()
    calculateMeanShape()
    image = markImage(I[0], shapes[0])
    image = markImage(image, shapes[1])
    image = markImage(image, adjustToFit(shapes[0], shapes[1]))
    cv2.imwrite('adjust_to_fit_test.jpg', image)
