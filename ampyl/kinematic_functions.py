#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# kinematic_functions.py
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
from scipy.special import sph_harm
from sympy.physics.quantum.cg import CG

from .constants import EPSILON15
from .constants import PI
from .constants import QC_IMPL_DEFAULTS
from .constants import R4PI


def J_slow(z=0.5):
    r"""Return the slow implementation of the cutoff function ``J(z)``.

    The function smoothly interpolates between ``0`` for ``z < 0`` and
    ``1`` for ``z > 1``.

    Parameters
    ----------
    z : float or numpy.ndarray, optional
        Cutoff-function argument.

    Returns
    -------
    float or numpy.ndarray
        Value of the cutoff function evaluated at ``z``.

    Raises
    ------
    ValueError
        If ``z`` is neither a float nor a NumPy array.
    """
    if isinstance(z, np.ndarray):
        J_array = np.array([])
        for z_val in z:
            if z_val <= 0.0:
                J_array = np.append(J_array, 0.0)
            elif z_val >= 1.0:
                J_array = np.append(J_array, 1.0)
            else:
                J_array = np.append(
                    J_array,
                    np.exp(-1.0/z_val*np.exp(-1.0/(1.0-z_val)))
                    )
        return J_array
    if isinstance(z, float):
        if z <= 0:
            return 0.0
        if z >= 1.0:
            return 1.0
        return np.exp(-1.0/z*np.exp(-1.0/(1.0-z)))
    raise ValueError("z must be a float or np.ndarray")
