from Cython.Build import cythonize
from setuptools import Extension

import setuptools
from pascal_voc_tools._version import version

import numpy as np

_NP_INCLUDE_DIRS = np.get_include()


# Extension modules
ext_modules = [
    Extension(
        name='pascal_voc_tools.utils.cython_nms',
        sources=[
            'pascal_voc_tools/utils/cython_nms.pyx'
        ],
        extra_compile_args=[
            '-Wno-cpp'
        ],
        include_dirs=[
            _NP_INCLUDE_DIRS
        ]
    ),
]


with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pascal_voc_tools',
    version=version,
    author='ternence wang',
    author_email='ternencewang2015@outlook.com',
    description='some tools about pascal voc format dataset',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wang-tf/pascal_voc_tools',
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=['jinja2', 'opencv-python', 'tqdm'],
    ext_modules=cythonize(ext_modules),
)
