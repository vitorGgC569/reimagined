import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Helper class to build pybind11 extension
class get_pybind_include(object):
    def __str__(self):
        import pybind11
        return pybind11.get_include()

ext_modules = [
    Extension(
        'uhk.uhk_core',
        ['sdk/bindings.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            'src'
        ],
        language='c++'
    ),
]

setup(
    name='uhk',
    version='0.1.0',
    author='UHK Team',
    description='Universal Heterogeneous Kernel Python SDK',
    packages=['uhk'],
    package_dir={'uhk': 'sdk/uhk'},
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0', 'numpy'],
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
