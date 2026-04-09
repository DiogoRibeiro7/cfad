"""
Build script for Cython C extensions.
Run: pip install -e . --no-build-isolation
Or:  python setup.py build_ext --inplace
"""
from setuptools import setup, Extension
import numpy as np
from Cython.Build import cythonize

NUMPY_INCLUDE = np.get_include()

extensions = [
    Extension(
        "cfad._ext.rolling_ecf",
        sources=["cfad/_ext/rolling_ecf.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
    ),
    Extension(
        "cfad._ext.contour_quad",
        sources=["cfad/_ext/contour_quad.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=["-O3", "-march=native"],
    ),
    Extension(
        "cfad._ext.cusum",
        sources=["cfad/_ext/cusum.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=True,
    )
)
