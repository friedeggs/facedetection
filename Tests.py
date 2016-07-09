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
    def test_generate_training_data(self):
        ans = array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(np.repeat(np.arange(5), 2), ans)

if __name__ == '__main__':
    unittest.main()
