#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name='onnxoptim',
    version='0.0.1',
    install_requires=[
        'numpy',
        'onnx',
        'onnxoptimizer',
        'onnxsim',
    ],
    packages=find_packages(),
)
