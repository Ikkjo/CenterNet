"""CPU extensions used by the Pascal VOC evaluator; no CUDA toolkit needed."""
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

setup(
    name='centernet-voc-eval',
    ext_modules=cythonize([
        Extension('utils.cython_bbox', ['utils/bbox.pyx']),
        Extension('nms.cpu_nms', ['nms/cpu_nms.pyx']),
    ], compiler_directives={'language_level': 3}),
    include_dirs=[np.get_include()],
)
