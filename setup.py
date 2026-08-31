"""Build configuration for CFAD's optional Cython extensions."""

from __future__ import annotations

import platform

from Cython.Build import cythonize
import numpy as np
from setuptools import Extension, setup

NUMPY_INCLUDE = np.get_include()

if platform.system() == "Windows":
    EXTRA_COMPILE_ARGS = ["/O2"]
else:
    # Avoid -march=native in distributable wheels: binaries must remain portable
    # across compatible CPUs rather than target the build host specifically.
    EXTRA_COMPILE_ARGS = ["-O3"]

EXTENSIONS = [
    Extension(
        "cfad._ext.rolling_ecf",
        sources=["cfad/_ext/rolling_ecf.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    ),
    Extension(
        "cfad._ext.cusum",
        sources=["cfad/_ext/cusum.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    ),
]

setup(
    ext_modules=cythonize(
        EXTENSIONS,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=False,
    )
)
