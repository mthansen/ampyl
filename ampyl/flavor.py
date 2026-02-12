#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# flavor.py
#
# MIT License
# Copyright (c) 2022 Maxwell T. Hansen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

import numpy as np
from copy import deepcopy
from inspect import signature
from . import flavor_utils
from .functions import QCFunctions
from .constants import bcolors
from .constants import G_TEMPLATE_DICT
import warnings
warnings.simplefilter("once")


class Particle:
    """
    Class used to represent a particle.

    :param mass: mass of the particle (default is ``1.``)
    :type mass: float
    :param spin: spin of the particle (default is ``0.``)
    :type spin: float
    :param flavor: flavor of the particle (default is ``'pi'``)
    :type flavor: str
    :param isospin_multiplet: specifies whether this is an isospin multiplet\
        (default is ``False``)
    :type isospin_multiplet: bool
    :param isospin: isospin of the particle (default is ``None``)
    :type isospin: float

    :raises ValueError: If `isospin_multiplet` is ``True`` but `isospin` is
        ``None``.
    """
