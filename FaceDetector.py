import numpy as np
import cv2
from Settings import *
from MathFunctions import calculateSimilarityTransform, adjustPoints
cascadePath = 'data/lbpcascade_frontalface.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
cascadePaths = [
    'data/lbpcascade_frontalface.xml',
    'data/lbpcascade_profileface.xml',
    'data/haarcascade_frontalface_default.xml',
    'data/haarcascade_frontalface_alt.xml',
    'data/haarcascade_frontalface_alt2.xml',
    'data/haarcascade_profileface.xml',
    'data/haarcascade_frontalface_alt_tree.xml',
    'data/haarcascade_frontalcatface.xml',
    'data/haarcascade_frontalcatface_extended.xml'
    ]
# faceCascades = [[cv2.CascadeClassifier(path) for path in cascades] for cascades in cascadePaths]
faceCascades = [cv2.CascadeClassifier(path) for path in cascadePaths]
# window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
# width = 1000
# height = 800
# cv2.resizeWindow('Rectangle', 1000, 800)
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
        adjustment = adjustToFit(self.meanShape, shapeRectangle, adapterOnly=True)
        predictedShape = np.copy(self.meanShape)
        for strongRegressor in self.strongRegressors:
            if strongRegressor:
                # print "predicting"
                delta = strongRegressor.eval(image, predictedShape, transform, adjustment)
                predictedShape += delta
                transform = calculateSimilarityTransform(self.meanShape, predictedShape)
                # print delta[:5]
        return adjustPoints(predictedShape, adjustment)
def detectFaceRectangle(image, ind=0): # TODO test
    width, height = np.shape(image)
    try:
        im = image
        faces = faceCascades[0].detectMultiScale(
                    image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    scaleFactor=1.1,
                    minNeighbors=2,
                    minSize=(width/4, height/4))
                    # minSize = (10,10))
        # print faces
        index = 0
        while len(faces) == 0 and index+1 < len(faceCascades):
            index += 1
            faces = faceCascades[index].detectMultiScale(
                        image, # should be grayscale - gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(width/3, height/3))
                        # minSize = (10,10))
            # print faces
        if len(faces) == 0:
            return None
        index = np.argmax(faces[:,2]) # argmax of width # and height
        faceRect = faces[index]
        faceRect = adjustRect(faceRect) # does not affect original shapeRectangle
        # y,x,h,w = faceRect
        # faceRect = (x,y,w,h)
        return faceRect, im # TODO or return largest one?
    except():
        e = sys.exc_info()[0]
        print(e)
        # cv2.imshow('Image', image)
        # cv2.waitKey()
        return FaceDetector.meanRectangle, im # TODO check if this is right in Python
def adjustToFit(shape, shapeRectangle, adapterOnly=False): # shapeRectangle is given as (x,y,w,h)
    x,y,w,h = shapeRectangle
    x = np.array([x,y])
    off = np.array([w,h])
    X1 = np.min(shape, 0)# .astype(int) TODO why astype int?
    Y1 = np.max(shape, 0).astype(int)
    scale = 1.*off/(Y1 - X1)
    offset = X1 * scale - x
    shape = shape * scale - offset
    transform = 1./scale
    if adapterOnly:
        return (scale, offset)
    return shape
def adjustRect(rect): # TODO extremely arbitrary even if it depends on opencv's output
    x,y,w,h = rect
    cx = x + w/2
    cy = y + h/2
    w = 2*w/3
    h = 3*h/4
    x = cx - w/2
    y = cy - h/3
    return (x,y,w,h)
