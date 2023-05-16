#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# constants.py
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

QC_IMPL_DEFAULTS = {'hermitian': True,
                    'real_harmonics': True,
                    'zeta_interp': False,
                    'sph_harm_clebsch': False,
                    'g_uses_prep_mat': False,
                    'g_interpolate': False,
                    'g_smart_interpolate': False,
                    'smarter_q_rescale': False,
                    'use_cob_matrices': True}
PI = np.pi
TWOPI = 2.*PI
FOURPI2 = 4.0*PI**2
R4PI = np.sqrt(4.*PI)

RTWO = np.sqrt(2.)
RTHREE = np.sqrt(3.)
RFIVE = np.sqrt(5.)
RSIX = np.sqrt(6.)
RTEN = np.sqrt(10.)
RTWELVE = np.sqrt(12.)
RFIFTEEN = np.sqrt(15.)
RTWENTY = np.sqrt(20.)

EPSILON4 = 1.0e-4
EPSILON8 = 1.0e-8
EPSILON10 = 1.0e-10
EPSILON15 = 1.0e-15
EPSILON20 = 1.0e-20

BAD_MIN_GUESS = 100.
BAD_MAX_GUESS = 0.
POLE_CUT = 100.
DELTA_L_FOR_GRID = 0.9
DELTA_E_FOR_GRID = 0.9
L_GRID_SHIFT = 2.0
E_GRID_SHIFT = 3.0
G_TEMPLATE_DICT = {}
G_TEMPLATE_DICT[0] = np.array([[-1.]])
G_TEMPLATE_DICT[1] = np.array([[1./3., -1./RTHREE, RFIVE/3.],
                               [-1./RTHREE, 0.5, RFIFTEEN/6.],
                               [RFIVE/3., RFIFTEEN/6., 1./6.]])
G_TEMPLATE_DICT[2] = np.array([[0.5, -RTHREE/2.],
                               [-RTHREE/2., -0.5]])
G_TEMPLATE_DICT[3] = np.array([[1.]])
ISO_PROJECTOR_THREE = np.array([[1., 0., 0., 0., 0., 0., 0.]])
ISO_PROJECTOR_TWO = np.array([[0., 1., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0.]])
ISO_PROJECTOR_ONE = np.array([[0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0.]])
ISO_PROJECTOR_ZERO = np.array([[0., 0., 0., 0., 0., 0., 1.]])
ISO_PROJECTORS = [ISO_PROJECTOR_ZERO, ISO_PROJECTOR_ONE,
                  ISO_PROJECTOR_TWO, ISO_PROJECTOR_THREE]
CAL_C_ISO = np.array([[1./RTEN, 1./RTEN, 1./RTEN, RTWO/RFIVE,
                       1./RTEN, 1./RTEN, 1./RTEN],
                      [-0.5, -0.5, 0., 0., 0., 0.5, 0.5],
                      [-1./RTWELVE, 1./RTWELVE, -1./RTHREE, 0.,
                       1./RTHREE, -1./RTWELVE, 1./RTWELVE],
                      [RTHREE/RTWENTY, RTHREE/RTWENTY, -1./RFIFTEEN,
                       -2./RFIFTEEN, -1./RFIFTEEN, RTHREE/RTWENTY,
                       RTHREE/RTWENTY],
                      [0.5, -0.5, 0., 0., 0., -0.5, 0.5],
                      [0., 0., 1./RTHREE, -1./RTHREE, 1./RTHREE, 0., 0.],
                      [-1./RSIX, 1./RSIX, 1./RSIX, 0., -1./RSIX, -1./RSIX,
                       1./RSIX]])
PION_ORDERS = [[0, 1, 2], [1, 0, 2], [0, 2, 1],
               [1, 2, 0], [2, 0, 1], [2, 1, 0]]


class bcolors:
    """Class collecting text colors."""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
