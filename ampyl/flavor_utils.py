#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# flavor_utils.py
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
from .constants import EPSILON4


def check_type(variable, variable_string, variable_type,
               variable_type_str):
    """Check that a variable is of the correct type."""
    if variable is not None:
        if not isinstance(variable, variable_type):
            raise TypeError(f"{variable_string} must be of type "
                            f"{variable_type_str}")


def particle_to_string(particle):
    """Convert a particle to a string."""
    particle_str = "Particle with the following properties:\n"
    particle_str += f"    mass: {particle.mass},\n"
    particle_str += f"    spin: {particle.spin},\n"
    particle_str += f"    flavor: {particle.flavor},\n"
    particle_str += f"    isospin_multiplet: {particle.isospin_multiplet},\n"
    if particle.isospin_multiplet:
        particle_str += f"    isospin: {particle.isospin},\n"
    return particle_str[:-2]+"."


def particles_equal(particle1, particle2):
    """Check if two particles are equivalent."""
    return (particle1.mass == particle2.mass and
            particle1.spin == particle2.spin and
            particle1.flavor == particle2.flavor and
            particle1.isospin_multiplet == particle2.isospin_multiplet and
            particle1.isospin == particle2.isospin)


