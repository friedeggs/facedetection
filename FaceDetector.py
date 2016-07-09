from Settings import *
class FaceDetector:

    def markFace(image, shape, markSize=5):
        width, height = np.shape(image)
        for a,b in predictedShape:
            a = int(a)
            b = int(b)
            for i in range(a-s, a+s):
                for j in range(b-s,b+s):
                    if i < height and j < width and i >= 0 and j >= 0:
                        image[j,i] = 255
        return image
    def saveImage(image, path=resultsPath):
        cv2.imwrite(path + '_temp_' + str(x) + '.jpg', image) # TODO
    def detectFace(self, image):
        transform = (1, np.identity(2), 0) # identity transform
        predictedShape = meanShape
        x = 0
        for strongRegressor in faceDetector:
            if strongRegressor:
                predictedShape += strongRegressor.eval(image, predictedShape, transform)
                transform = calculateSimilarityTransform(meanShape, predictedShape)

                x += 1
        return predictedShape
