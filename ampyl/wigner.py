#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# spaces.py
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

import functools
import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:
    sph_harm_y = None
    from scipy.special import sph_harm

from .constants import EPSILON8
from .constants import EPSILON15


def _sample_vectors(ell):
    n_points = max(2*ell+5, 12)
    inds = np.arange(n_points, dtype=float)
    golden_angle = np.pi*(3.0-np.sqrt(5.0))
    z_vals = 1.0 - 2.0*(inds+0.5)/n_points
    radii = np.sqrt(1.0-z_vals**2)
    phis = golden_angle*inds
    return np.column_stack((radii*np.cos(phis),
                            radii*np.sin(phis),
                            z_vals))


def _cart_sph_harm(ell, mazi, nvec_arr):
    nxs = (nvec_arr.T)[0]
    nys = (nvec_arr.T)[1]
    nzs = (nvec_arr.T)[2]
    nmags = np.sqrt((nvec_arr**2).sum(1))
    thetas = np.arccos(nzs/(nmags+EPSILON15))
    phis = np.arctan2(nys, nxs)
    if sph_harm_y is None:
        ylm = sph_harm(mazi, ell, phis, thetas)
    else:
        ylm = sph_harm_y(ell, mazi, thetas, phis)
    return np.sqrt(4.0*np.pi)*(nmags**ell)*ylm


def _cart_sph_harm_real(ell, mazi, nvec_arr):
    if mazi == 0:
        return _cart_sph_harm(ell, mazi, nvec_arr).real
    if mazi < 0:
        return (np.sqrt(2.0)*(-1.0)**mazi)\
            * _cart_sph_harm(ell, np.abs(mazi), nvec_arr).imag
    return (np.sqrt(2.0)*(-1.0)**mazi)\
        * _cart_sph_harm(ell, mazi, nvec_arr).real


def _basis_values(ell, nvec_arr, real_harmonics):
    harmonics = []
    for mazi in range(-ell, ell+1):
        if real_harmonics:
            harmonics.append(_cart_sph_harm_real(ell, mazi, nvec_arr))
        else:
            harmonics.append(_cart_sph_harm(ell, mazi, nvec_arr))
    return np.column_stack(harmonics)


def _matrix_key(g_elem):
    g_arr = np.array(g_elem, dtype=float)
    return tuple(np.rint(g_arr.reshape(-1)).astype(int))


@functools.lru_cache(maxsize=None)
def _generate_wigner_d_cached(ell, matrix_key, real_harmonics):
    g_elem = np.array(matrix_key, dtype=float).reshape((3, 3))
    nvec_arr = _sample_vectors(ell)
    basis = _basis_values(ell, nvec_arr, real_harmonics)
    rotated_basis = _basis_values(ell, (g_elem@nvec_arr.T).T, real_harmonics)
    wigner_d = np.linalg.lstsq(basis, rotated_basis, rcond=None)[0].T
    if real_harmonics:
        if not (np.abs(wigner_d.imag) < EPSILON8).all():
            raise ValueError("real Wigner-D is complex")
        wigner_d = wigner_d.real
    return wigner_d


def generate_wigner_d(ell, g_elem, real_harmonics=True):
    """Generate a Wigner-D matrix."""
    return _generate_wigner_d_cached(ell, _matrix_key(g_elem), real_harmonics)
