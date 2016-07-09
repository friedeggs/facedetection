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
