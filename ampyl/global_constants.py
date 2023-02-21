#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# global_constants.py
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
                    'real harmonics': True,
                    'Zinterp': False,
                    'YYCG': False}

PI = np.pi
TWOPI = 2.*PI
FOURPI2 = 4.0*PI**2
ROOT4PI = np.sqrt(4.*PI)
ROOT_THREE = np.sqrt(3.0)
ROOT_TWO = np.sqrt(2.0)
EPSPROJ = 1.0e-8
EPSILON10 = 1.0e-10
EPSILON15 = 1.0e-15
EPSILON20 = 1.0e-20

DELTA_L_FOR_GRID = 0.9
DELTA_E_FOR_GRID = 0.9
L_GRID_SHIFT = 2.0
E_GRID_SHIFT = 3.0

G_TEMPLATE_DICT = {}
G_TEMPLATE_DICT[0] = np.array([[-1.]])
G_TEMPLATE_DICT[1] = np.array([[1./3., -1./np.sqrt(3.), np.sqrt(5.)/3.],
                               [-1./np.sqrt(3.), 0.5, np.sqrt(15.)/6.],
                               [np.sqrt(5.)/3., np.sqrt(15.)/6., 1./6.]])
G_TEMPLATE_DICT[2] = np.array([[0.5, -np.sqrt(3.)/2.],
                               [-np.sqrt(3.)/2., -0.5]])
G_TEMPLATE_DICT[3] = np.array([[1.]])

ISO_PROJECTOR_THREE = np.array([[1., 0., 0., 0., 0., 0., 0.]])
ISO_PROJECTOR_TWO = np.array([[0., 1., 0., 0., 0., 0., 0.],
                              [0., 0., 1., 0., 0., 0., 0.]])
ISO_PROJECTOR_ONE = np.array([[0., 0., 0., 1., 0., 0., 0.],
                              [0., 0., 0., 0., 1., 0., 0.],
                              [0., 0., 0., 0., 0., 1., 0.]])
ISO_PROJECTOR_ZERO = np.array([[0., 0., 0., 0., 0., 0., 1.]])
ISO_PROJECTORS = [ISO_PROJECTOR_ZERO,
                  ISO_PROJECTOR_ONE,
                  ISO_PROJECTOR_TWO,
                  ISO_PROJECTOR_THREE]

CAL_C_ISO = np.array([[1./np.sqrt(10.), 1./np.sqrt(10.), 1./np.sqrt(10.),
                       np.sqrt(2./5.), 1./np.sqrt(10.), 1./np.sqrt(10.),
                       1./np.sqrt(10.)],
                      [-0.5, -0.5, 0., 0., 0., 0.5, 0.5],
                      [-1./np.sqrt(12.), 1./np.sqrt(12.), -1./np.sqrt(3.), 0.,
                       1./np.sqrt(3.), -1./np.sqrt(12.), 1./np.sqrt(12.)],
                      [np.sqrt(3./20.), np.sqrt(3./20.), -1./np.sqrt(15.),
                       -2./np.sqrt(15.), -1./np.sqrt(15.), np.sqrt(3./20.),
                       np.sqrt(3./20.)],
                      [0.5, -0.5, 0., 0., 0., -0.5, 0.5],
                      [0., 0., 1./np.sqrt(3.), -1./np.sqrt(3.), 1./np.sqrt(3.),
                       0., 0.],
                      [-1./np.sqrt(6.), 1./np.sqrt(6.), 1./np.sqrt(6.), 0.,
                       -1./np.sqrt(6.), -1./np.sqrt(6.), 1./np.sqrt(6.)]])

PION_ORDERS = [[0, 1, 2],
               [1, 0, 2],
               [0, 2, 1],
               [1, 2, 0],
               [2, 0, 1],
               [2, 1, 0]]
