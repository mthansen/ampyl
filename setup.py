#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: mthansen
"""

from setuptools import setup

VERSION = (0, 0, 0)


def version():
    """Version method."""
    v = ".".join(str(v) for v in VERSION)
    cnt = f'__version__ = "{v}" \n__version_full__ = __version__'
    with open('ampyl/version.py', 'w') as f:
        f.write(cnt)
    return v


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ampyl",
    version=version(),
    author="Maxwell T. Hansen",
    author_email="maxwell.hansen@ed.ac.uk",
    description="package for relating finite-volume data with amplitudes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mthansen/ampyl.git",
    packages=['ampyl'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT license",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "scipy",
        "quaternionic",
        "spherical"
    ],
)
