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
    def __init__(self, mass=1., spin=0., flavor='pi',
                 isospin_multiplet=False, isospin=None, verbosity=0):

        self.mass = mass
        self.spin = spin
        self.flavor = flavor

        self._isospin_multiplet = isospin_multiplet
        self._isospin = isospin
        self.isospin_multiplet = self._isospin_multiplet
        self.isospin = self._isospin

        self.verbosity = verbosity
        if self._verbosity >= 2:
            self.print_summary()

    @property
    def mass(self):
        """Mass of the particle."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Set the mass of the particle."""
        flavor_utils.check_type(mass, 'mass', float, 'float')
        self._mass = mass

    @property
    def spin(self):
        """Spin of the particle."""
        return self._spin

    @spin.setter
    def spin(self, spin):
        """Set the spin of the particle."""
        flavor_utils.check_type(spin, 'spin', float, 'float')
        self._spin = spin

    @property
    def flavor(self):
        """Flavor of the particle."""
        return self._flavor

    @flavor.setter
    def flavor(self, flavor):
        """Set the flavor of the particle."""
        flavor_utils.check_type(flavor, 'flavor', str, 'str')
        self._flavor = flavor

    @property
    def isospin_multiplet(self):
        """Whether the particle is an isospin multiplet."""
        return self._isospin_multiplet

    @isospin_multiplet.setter
    def isospin_multiplet(self, isospin_multiplet):
        """Set whether the particle is an isospin multiplet."""
        flavor_utils.check_type(isospin_multiplet, 'isospin_multiplet',
                                bool, 'bool')

        if not isospin_multiplet and self._isospin is not None:
            isospin_multiplet = True
        if isospin_multiplet and self._isospin is None:
            raise ValueError("isospin cannot be None when isospin_multiplet "
                             "is True")

        self._isospin_multiplet = isospin_multiplet

    @property
    def isospin(self):
        """Isospin of the particle."""
        return self._isospin

    @isospin.setter
    def isospin(self, isospin):
        """Set the isospin of the particle."""
        flavor_utils.check_type(isospin, 'isospin', float, 'float')

        if not self._isospin_multiplet and isospin is not None:
            self._isospin_multiplet = True
        if isospin is None and self._isospin_multiplet:
            raise ValueError("isospin cannot be None when isospin_multiplet "
                             "is True")

        self._isospin = isospin

    @property
    def verbosity(self):
        """Verbosity of the particle."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the particle."""
        flavor_utils.check_type(verbosity, 'verbosity', int, 'int')
        self._verbosity = verbosity

    def __eq__(self, other):
