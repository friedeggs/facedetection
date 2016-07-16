import numpy as np
import cv2
from Settings import *
from MathFunctions import calculateSimilarityTransform
cascadePath = 'data/lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
cascadePaths = [[
    'data/lbpcascade_frontalface.xml',
    'data/lbpcascade_profileface.xml',
    'data/haarcascade_frontalface_default.xml',
    'data/haarcascade_frontalface_alt.xml',
    'data/haarcascade_frontalface_alt2.xml',
    'data/haarcascade_profileface.xml',
    'data/haarcascade_frontalface_alt_tree.xml',
    'data/haarcascade_frontalcatface.xml',
    'data/haarcascade_frontalcatface_extended.xml'
    ],
    [
    'data/haarcascade_eye.xml',
    'data/haarcascade_eye_tree_eyeglasses.xml'
    ],
    [
    'data/haarcascade_lefteye_2splits.xml',
    'data/haarcascade_righteye_2splits.xml'
    ],
    [
    'data/haarcascade_smile.xml'
    ]
    ]
faceCascades = [[cv2.CascadeClassifier(path) for path in cascades] for cascades in cascadePaths]
window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
width = 1000
height = 800
cv2.resizeWindow('Rectangle', 1000, 800)
class FaceDetector:
    meanRectangle = []
    def __init__(self, meanShape, strongRegressors):
        self.meanShape = meanShape
        self.strongRegressors = strongRegressors
        x,y = meanShape.min(0) # just different way to get min as opposed to np.min(array,0)
        X,Y = meanShape.max(0)
        self.meanRectangle = (x,y,X-x,Y-y) # meanRectangle # or compute meanRectangle from meanShape
    def detectFace(self, image):
        transform = (1, np.identity(2), 0) # identity transform
        shapeRectangle, im = detectFaceRectangle(image)
        predictedShape = adjustToFit(self.meanShape, shapeRectangle)
        for strongRegressor in self.strongRegressors:
            if strongRegressor:
                # print "predicting"
                delta = strongRegressor.eval(image, predictedShape, transform)
                predictedShape += delta
                transform = calculateSimilarityTransform(self.meanShape, predictedShape)
                # print delta[:5]
        return predictedShape
def detectFaceRectangle(image, ind=0): # TODO test
    width, height = np.shape(image)
    try:
        im = image
        faces = faceCascades[ind][0].detectMultiScale(
                    image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    scaleFactor=1.1,
                    minNeighbors=2,
                    # minSize=(width/4, height/4))
                    minSize = (10,10))
        print faces
        index = 0
        while len(faces) == 0 and index+1 < len(faceCascades[ind]):
            index += 1
            faces = faceCascades[ind][index].detectMultiScale(
                        image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        scaleFactor=1.05,
                        minNeighbors=3,
                        # minSize=(width/3, height/3))
                        minSize = (10,10))
            print faces
        # if len(faces) == 0:
        #     im = image.copy()
        #     src = im
        #     borderw = 3*width/8
        #     borderh = 3*height/8
        #     im = cv2.resize(im, (width/4, height/4))
        #     im = cv2.copyMakeBorder(im, borderh, borderh, borderw, borderw, cv2.BORDER_WRAP)
        #     res = cv2.resize(im,(width, height))
        #     cv2.imshow('Rectangle', res)
        #     cv2.waitKey()
        #     print("trying second scale factor")
        #     faces = faceCascade.detectMultiScale(
        #             image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #             scaleFactor=1.1,
        #             minNeighbors=3,
        #             minSize=(30, 30))
        #     print faces
        if len(faces) == 0:
            return (0,0,0,0), im
        index = np.argmax(faces[:,2]) # argmax of width # and height
        return faces[index], im # TODO or return largest one?
    except():
        e = sys.exc_info()[0]
        print(e)
        # cv2.imshow('Image', image)
        # cv2.waitKey()
        return FaceDetector.meanRectangle, im # TODO check if this is right in Python
def adjustToFit(shape, shapeRectangle): # shapeRectangle is given as (x,y,w,h)
    x,y,w,h = shapeRectangle
    x = np.array([x,y])
    off = np.array([w,h])
    X1 = np.min(shape, 0)# .astype(int) TODO why astype int?
    Y1 = np.max(shape, 0).astype(int)
    scale = 1.*off/(Y1 - X1)
    offset = X1 * scale - x
    shape = shape * scale - offset
    return shape
