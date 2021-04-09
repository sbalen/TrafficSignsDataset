#!/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import os
import re

from setuptools import find_namespace_packages, setup

from setuptools import setup

setup(name='trafficsigns-dataset',
      version='0.1',
      description='This is a set of clearly cropped images of (Belgian) traffic signs sorted in to one of 62 classes. ',
      url='',
      author='Stijn van Balen',
      author_email='sbalen@gmail.com',
      license='',
      zip_safe=False,
      install_requires = [
                       "tensorflow",
                       "tensorflow_datasets",
                       "opencv-python"
       ],
)
