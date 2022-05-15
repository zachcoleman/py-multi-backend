from setuptools import setup
from Cython.Build import cythonize

setup(
    name="mask to rle",
    ext_modules=cythonize("./cython/cython_rle.pyx"),
    zip_safe=False,
)
