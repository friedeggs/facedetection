def testFaceDetector():
    loadData()
    calculateMeanShape()
    global shapeEstimates, shapeDeltas, shapes, pi
    FaceDetector.meanRectangle = (
                meanWidthX,
                meanHeightX,
                meanWidthY - meanWidthX,
                meanHeightY - meanHeightY)
    window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    width = 1000
    height = 800
    cv2.resizeWindow('Rectangle', 1000, 800)
    for i in range(n):
        im = I[i].copy()
        rect, im2 = detectFaceRectangle(I[i]) # TODO not a problem
        # rect2 = adjustRect(rect)
        x,y,w,h = rect
        # im = I[i] #.copy()
        thickness = 5
        cv2.line(im, (x,y), (x,y+h), thickness)
        cv2.line(im, (x,y+h), (x+w,y+h), thickness)
        cv2.line(im, (x+w,y+h), (x+w,y), thickness)
        cv2.line(im, (x+w,y), (x,y), thickness)
        im = markImage(im, adjustToFit(meanShape, rect))
        res = cv2.resize(im,(width, height))
        cv2.imshow('Rectangle', res)
        cv2.waitKey()
def showFaceDetector():
    loadData()
    calculateMeanShape()
    strongRegressors[0] = load('weak_regressors_0-20')
    window = cv2.namedWindow('Rectangle', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Rectangle', 1000, 800)

    rect, im = detectFaceRectangle(I[0])
    im = im.copy()
    shape = adjustToFit(meanShape, rect)
    adjustment = adjustToFit(meanShape, rect, adapterOnly=True)
    splits = []
    for i in range(20):
        splits += strongRegressors[0].weakRegressors[i].splits()
    for split in splits:
        threshold, p0, p1 = split
        pair = np.array([p0,p1])
        adjustedPair = adjustPoints(pair, adjustment)
        adjustedPair = [[int(s) for s in p] for p in adjustedPair]
        adjustedPair = tuple(map(tuple,adjustedPair))
        # print pair, adjustPoints(pair,adjustment)
        cv2.line(im, adjustedPair[0], adjustedPair[1], color=255, thickness=1)
        im = markImage(im, adjustPoints(pair, adjustment), markSize=4)
    im = markImage(im, shape)
    im = cv2.resize(im,(width, height))
    cv2.imshow('Rectangle', im)
    cv2.waitKey()
