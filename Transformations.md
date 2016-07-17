# Functions

```py
def detectFaceRectangle:
    predict the face rectangle
def adjustRect:
    the rectangle opencv predicts
def adjustToFit:
    takes a shape and a rectangle
```

groundEstimate

# Coordinate Systems

image meanShape
   ↑↓ ------------ imageAdapter/adjustment - interface between image and algorithm
meanShape, current shape estimate
   ↑↓ ------------ similarityTransform - shape invariant indexing in algorithm
current shape estimate

# Pathway of parameters
FaceDetector <- residuals
↓
StrongRegressor
↓
WeakRegressor
↓
split()
