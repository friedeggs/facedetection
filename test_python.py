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

class RegressionTreeTest(unittest.TestCase):
    '''Tests for Regression Tree class.'''

    def setUp(self):
        fd.meanWidth = 30
        fd.meanHeight = 30
        fd.P = 5
        fd.samplePoints, fd.samplePairs, fd.priorWeights = fd.samplePixels()

    def fitNode(self, depth):
        split = fd.generateCandidateSplit()
        tree = fd.RegressionTree(split, depth)
        if depth > 0:
            tree.leftTree = self.fitNode(depth - 1)
            tree.rightTree = self.fitNode(depth - 1)
        return tree

    def traverse(self, tree):
        if tree is None:
            return
        sys.stdout.write("(")
        if tree.depth > 0:
            self.traverse(tree.leftTree)
        sys.stdout.write(str(tree.node))
        if tree.depth > 0:
            self.traverse(tree.rightTree)
        sys.stdout.write(")")

    # def test_regression_tree(self):
    #     '''Create a tree and evaluate it on an image.'''
    #     split = fd.generateCandidateSplit()
    #     tree = fd.RegressionTree(split)
    #     self.assertIsNotNone(tree)
    #     tree = self.fitNode(2)
    #     self.traverse(tree)

N = 1000
I = [[] for i in range(N)]
shapes = [[] for i in range(N)]
savePath = '/Users/frieda/Documents/Coding/Learning/FaceDetection/shapes_test.pkl'
loadPath = savePath
def loadImages():
    basePath = '/Users/frieda/Downloads/'
    for i in range(N):
        filePath = basePath + 'annotation/' + str(i+1) + '.txt'
        imagePath = ""
        with open(filePath, 'r') as f:
            imagePath = f.readline().rstrip('\n').rstrip('\r')
            shapes[i] = [[float(s) for s in line.rstrip('\n').rstrip('\r').split(',')] for line in f.readlines()]
            I[i] = cv2.imread(basePath + 'images/' + imagePath + '.jpg')
    saveDetector(shapes)

def saveDetector(detector, path=savePath):
    f = open(path, 'w')
    pickle.dump(detector, f)
    f.close()

def loadDetector(path=loadPath):
    f = open(path, 'r')
    faceDetector = pickle.load(f)
    f.close()
    return faceDetector

class SimilarityTransformTest(unittest.TestCase):
    def setUp(self):
        self.shape1 = np.array([[84, 238], [87, 270], [94, 300], [101, 330], [110, 358], [126, 383], [148, 404], [172, 420], [199, 426], [226, 420], [251, 405], [273, 385], [290, 360], [299, 332], [306, 301], [312, 270], [314, 239], [102, 221], [118, 209], [138, 207], [158, 211], [177, 220], [224, 219], [243, 211], [263, 207], [282, 210], [297, 221], [200, 245], [200, 266], [200, 287], [200, 308], [179, 321], [189, 325], [200, 328], [211, 325], [221, 321], [126, 246], [139, 238], [155, 239], [169, 252], [154, 254], [137, 254], [232, 252], [245, 239], [261, 238], [274, 246], [263, 254], [247, 254], [157, 360], [174, 354], [189, 350], [200, 353], [211, 350], [226, 354], [243, 360], [226, 372], [212, 378], [200, 379], [188, 378], [174, 372], [165, 361], [189, 362], [200, 363], [211, 362], [236, 361], [211, 361], [200, 362], [189, 361]])
        self.shape2 = np.array([[162, 248], [164, 283], [167, 319], [174, 352], [190, 382], [218, 404], [251, 421], [282, 434], [311, 437], [337, 429], [356, 406], [374, 381], [389, 354], [397, 325], [401, 295], [401, 265], [399, 236], [197, 221], [217, 205], [242, 201], [267, 206], [291, 215], [323, 214], [342, 206], [362, 200], [381, 201], [393, 216], [309, 234], [311, 254], [314, 275], [316, 297], [285, 311], [298, 315], [313, 318], [325, 314], [335, 310], [227, 238], [241, 230], [258, 230], [272, 241], [257, 243], [240, 243], [331, 239], [343, 227], [358, 227], [370, 234], [360, 241], [345, 241], [250, 346], [275, 342], [298, 338], [311, 340], [324, 337], [341, 341], [358, 342], [342, 367], [326, 380], [311, 383], [296, 382], [274, 373], [257, 349], [297, 344], [311, 345], [324, 343], [351, 344], [325, 364], [311, 367], [297, 366]])

    # def test_calculation(self):
    #     fd.calculateSimilarityTransform()

if __name__ == '__main__':
    # unittest.main()
    loadImages()
    # shapes = loadDetector()
    print shapes[0]
