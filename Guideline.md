# CHECK:

[] any variables that are set in the function are listed in a `global` line
[] for loops iterate on the right variable (eg. `n` vs `N`)
[] check if python objects by references

# Pass by Reference?
- if so, then shapeEstimates points to a shape
- shapeEstimates += eval then adds R evals to each _shape_
- this would cause an iterative increase across each StrongRegressor

- observations point to increases per WeakRegressor
- could be similar problem?

- this is in generateTrainingData

## Conclusion:
http://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference
- Python (arguments) are passed by assignment

- check list, variable assignments etc

# Class variables [warning!]
http://stackoverflow.com/questions/11040438/class-variables-is-shared-across-all-instances-in-python

# Bizarre - Decorator
http://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function

# Global variables across classes - import one file
http://stackoverflow.com/questions/13034496/using-global-variables-between-files-in-python

# Local scoping, dynamic use of scoping - careful of tripping up
http://stackoverflow.com/questions/4693120/use-of-global-keyword-in-python
In this answer: http://stackoverflow.com/a/4694310

# Python docs - Classes
https://docs.python.org/2/tutorial/classes.html#tut-object

# Module reload
If working interactively, you can use `reload(modulename`).

# Relative imports
`from ..filters import equalizer`

# Warning
`my_numpy_array += 2` maintains the original reference
same for python lists
```py
a = [0,1,2]
b = a
a += [3, 4] # a = [0, 1, 2, 3, 4]
            # b = [0, 1, 2, 3, 4]
a = [5, 6]  # a = [5, 6]
            # b = [0, 1, 2, 3, 4]
```

# Printing Formatting
```py
x = 1./7
print "%s" % x # 0.142857142857
print "%f" % x # 0.142857
print "%d" % x # 0
print "%.2f" % x # 0.14
print "%.2d" % x # 00
x += 1234
print "%.2f" % x # 1234.14
print "%.2s" % x # 12
print "%5.2s" % x #    12 # pad up to 5 characters
```

# Common Mistake to Avoid - Declaring Methods
http://stackoverflow.com/questions/1132941/least-astonishment-in-python-the-mutable-default-argument

# Tuples vs Lists
http://stackoverflow.com/questions/8900166/whats-the-difference-between-lists-enclosed-by-square-brackets-and-parentheses
Adding to list changes the original list unlike tuples, which are immutable

# OpenCV Face Detectors
/usr/local/Cellar/opencv3/3.1.0/share/OpenCV/haarcascades

# Directions
- Go through all TODO comments
- Step by step in depth run
- Speed up / Rewrite in C++
- Cut out huge inefficiency in reevaluating strongRegressors each and every time!! this contributes to almost an additional 30s by the 100th tree

# Adjustments
I shouldn't have adjustments at all
The mean shape needs to be scaled and offset to the image
All of this is contained in the similarity transform and warp point
None of the adjustment nonsense is necessary
