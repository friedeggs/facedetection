from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name = "My hello app",
    ext_modules = cythonize('FaceDetectorFactory.pyx'),  # accepts a glob pattern
    include_dirs = [np.get_include()]
)
