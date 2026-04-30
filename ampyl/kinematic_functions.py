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

def J(z=np.array([0.5])):
    r"""Return the vectorized cutoff function ``J(z)``.

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
        J_array = np.zeros_like(z)
        mask1 = 0.0 < z
        mask2 = z < 1.0
        mask = (
            np.concatenate((mask1, mask2)).reshape((2, len(mask1))).T
            ).all(axis=1)
        mask_one = 1.0 <= z
        J_array[mask_one] = 1.0
        J_array[mask] = np.exp(-1.0/z[mask]*np.exp(-1.0/(1.0-z[mask])))
        return J_array
    if isinstance(z, float):
        if z <= 0:
            return 0.0
        if z >= 1.0:
            return 1.0
        return np.exp(-1.0/z*np.exp(-1.0/(1.0-z)))
    raise ValueError("z must be a float or np.ndarray")

def H(E2CMSQ=9.0, threshold=2.0, alpha=-1.0, beta=0.0, J_slow=False):
    r"""Return the kinematic cutoff function ``H``.

    This function is built from ``J(z)`` and ranges from ``0`` below a
    chosen two-particle CMF energy to ``1`` above threshold.

    Parameters
    ----------
    E2CMSQ : float or numpy.ndarray, optional
        Squared two-particle center-of-mass energy.
    threshold : float, optional
        Two-particle threshold value.
    alpha : float, optional
        Width parameter. The standard choice is ``-1.0``.
    beta : float, optional
        Shift parameter. The standard choice is ``0.0``.
    J_slow : bool, optional
        If ``True``, use :meth:`J_slow` instead of :meth:`J`.

    Returns
    -------
    float or numpy.ndarray
        Value of the cutoff function.
    """
    z = (E2CMSQ-(1.0+alpha)*threshold**2/4.0)\
        / ((3.0-alpha)*threshold**2/4.0)+beta
    if J_slow:
        return globals()['J_slow'](z)
    return J(z)

def phase_space(E2CMSQ=9.0, omk=1.0):
    r"""Return the two-body phase space including ``2\omega_k``.

    Parameters
    ----------
    E2CMSQ : float or numpy.ndarray, optional
        Squared two-particle center-of-mass energy.
    omk : float or numpy.ndarray, optional
        Time component of the four-vector ``k``.

    Returns
    -------
    float or numpy.ndarray
        Two-body phase-space factor.
    """
    return 1.0/(16.0*PI*np.sqrt(E2CMSQ))/(2.0*omk)

def phase_space_alt(omk=1.0, m=1.0):
    r"""Return the alternate definition of the two-body phase space.

    Parameters
    ----------
    omk : float or numpy.ndarray, optional
        Time component of the four-vector ``k``.
    m : float or numpy.ndarray, optional
        Mass entering the alternate normalization.

    Returns
    -------
    float or numpy.ndarray
        Two-body phase-space factor.
    """
    return 1.0/(32.0*PI*m)/(2.0*omk)

