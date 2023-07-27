#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# groups.py
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
from scipy.linalg import block_diag
import quaternionic
import spherical
from .constants import RTHREE
from .constants import RTWO
from .constants import ISO_PROJECTORS
from .constants import CAL_C_ISO
from .constants import PION_ORDERS
from .constants import EPSILON8
from .constants import EPSILON10
from .constants import EPSILON15
from .constants import bcolors
import warnings
warnings.simplefilter("once")


class Groups:
    """Class for finite-volume group-theory relevant for three particles."""

    def __init__(self, ell_max, half_spin=False):
        self.wigner = spherical.Wigner(ell_max=ell_max)
        self.half_spin = half_spin

        self.OhP = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
             [[0, -1, 0], [-1, 0, 0], [0, 0, -1]],
             [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[-1, 0, 0], [0, 0, -1], [0, -1, 0]],
             [[0, 0, 1], [0, -1, 0], [1, 0, 0]],
             [[0, 0, -1], [0, -1, 0], [-1, 0, 0]],
             [[0, -1, 0], [0, 0, -1], [1, 0, 0]],
             [[0, 0, 1], [-1, 0, 0], [0, -1, 0]],
             [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
             [[0, -1, 0], [0, 0, 1], [-1, 0, 0]],
             [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
             [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
             [[0, 1, 0], [0, 0, -1], [-1, 0, 0]],
             [[0, 0, -1], [1, 0, 0], [0, -1, 0]],
             [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
             [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
             [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
             [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
             [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
             [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
             [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
             [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
             [[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
             [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
             [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, 0, -1], [0, -1, 0]],
             [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[0, 0, -1], [0, 1, 0], [-1, 0, 0]],
             [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
             [[0, 1, 0], [0, 0, 1], [-1, 0, 0]],
             [[0, 0, -1], [1, 0, 0], [0, 1, 0]],
             [[0, 0, 1], [1, 0, 0], [0, -1, 0]],
             [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
             [[0, 0, -1], [-1, 0, 0], [0, -1, 0]],
             [[0, -1, 0], [0, 0, -1], [-1, 0, 0]],
             [[0, -1, 0], [0, 0, 1], [1, 0, 0]],
             [[0, 0, 1], [-1, 0, 0], [0, 1, 0]],
             [[0, 1, 0], [-1, 0, 0], [0, 0, -1]],
             [[0, -1, 0], [1, 0, 0], [0, 0, -1]],
             [[0, 0, -1], [0, -1, 0], [1, 0, 0]],
             [[0, 0, 1], [0, -1, 0], [-1, 0, 0]],
             [[-1, 0, 0], [0, 0, -1], [0, 1, 0]],
             [[-1, 0, 0], [0, 0, 1], [0, -1, 0]],
             [[1, 0, 0], [0, 1, 0], [0, 0, -1]],
             [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.bTdict = {}

        self.chardict = {}

        self.bTdict['OhP_A1PLUS'] = np.array([[1.]*48])

        self.chardict['OhP_A1PLUS'] = self.bTdict['OhP_A1PLUS']

        self.bTdict['OhP_A2PLUS'] = np.array([[1., -1., -1., -1., -1.,
                                               -1., -1., 1., 1., 1., 1.,
                                               1., 1., 1., 1., -1., -1.,
                                               -1., -1., -1., -1., 1.,
                                               1., 1., 1., -1., -1., -1.,
                                               -1., -1., -1., 1., 1., 1.,
                                               1., 1., 1., 1., 1., -1.,
                                               -1., -1., -1., -1., -1.,
                                               1., 1., 1.]])

        self.chardict['OhP_A2PLUS'] = self.bTdict['OhP_A2PLUS']

        self.bTdict['OhP_EPLUS'] = np.array([[1., 1., 1., 1., 1., -2., -2.,
                                              1., -2., -2., 1., -2., 1., 1.,
                                              -2., 1., 1., -2., -2., 1., 1.,
                                              1., 1., 1., 1., 1., 1., 1., 1.,
                                              -2., -2., 1., -2., -2., 1., -2.,
                                              1., 1., -2., 1., 1., -2., -2.,
                                              1., 1., 1., 1., 1.],
                                             [-RTHREE, RTHREE,
                                              RTHREE, -RTHREE,
                                              -RTHREE, 0., 0., RTHREE,
                                              0., 0., RTHREE, 0.,
                                              RTHREE, RTHREE, 0.,
                                              RTHREE, RTHREE, 0., 0.,
                                              -RTHREE, -RTHREE,
                                              -RTHREE, -RTHREE,
                                              -RTHREE, -RTHREE,
                                              RTHREE, RTHREE,
                                              -RTHREE, -RTHREE, 0., 0.,
                                              RTHREE, 0., 0., RTHREE,
                                              0., RTHREE, RTHREE, 0.,
                                              RTHREE, RTHREE, 0., 0.,
                                              -RTHREE, -RTHREE,
                                              -RTHREE, -RTHREE,
                                              -RTHREE]])

        self.chardict['OhP_EPLUS'] = np.array(2*[[2., 0., 0., 0., 0., 0., 0.,
                                                  -1., -1., -1., -1., -1., -1.,
                                                  -1., -1., 0., 0., 0., 0., 0.,
                                                  0., 2., 2., 2., 2., 0., 0.,
                                                  0., 0., 0., 0., -1., -1.,
                                                  -1., -1., -1., -1., -1., -1.,
                                                  0., 0., 0., 0., 0., 0., 2.,
                                                  2., 2.]])

        self.bTdict['OhP_T1PLUS'] = np.array([[-1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.+1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               1.-1.*1j, -1.-1.*1j, 1.-1.*1j,
                                               1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                               -1.-1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               1.-1.*1j, 1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.+1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               1.-1.*1j, -1.-1.*1j, 1.-1.*1j,
                                               1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                               -1.-1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               1.-1.*1j, 1.+1.*1j, -1.-1.*1j],
                                              [RTWO, -RTWO, -RTWO,
                                               RTWO, -RTWO, RTWO,
                                               -RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               RTWO, RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               RTWO, -RTWO, RTWO,
                                               -RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               RTWO, RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO],
                                              [1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                               1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.-1.*1j, -1.+1.*1j, -1.+1.*1j,
                                               1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                               1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.-1.*1j, -1.+1.*1j, -1.+1.*1j,
                                               1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, 1.-1.*1j]]
                                             )

        self.chardict['OhP_T1PLUS'] = np.array(3*[[3., -1., -1., -1., -1., -1.,
                                                   -1., 0., 0., 0., 0., 0., 0.,
                                                   0., 0., 1., 1., 1., 1., 1.,
                                                   1., -1., -1., -1., 3., -1.,
                                                   -1., -1., -1., -1., -1., 0.,
                                                   0., 0., 0., 0., 0., 0., 0.,
                                                   1., 1., 1., 1., 1., 1., -1.,
                                                   -1., -1.]])

        self.bTdict['OhP_T2PLUS'] = np.array([[1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                               1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               1.-1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                               1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.+1.*1j, 1.-1.*1j, -1.-1.*1j,
                                               1.-1.*1j, -1.-1.*1j, -1.+1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, 1.-1.*1j],
                                              [RTWO, RTWO, RTWO,
                                               RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               -RTWO, RTWO, RTWO,
                                               -RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               RTWO, RTWO, RTWO,
                                               RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO,
                                               -RTWO, RTWO, RTWO,
                                               -RTWO, -RTWO, RTWO,
                                               -RTWO, RTWO, -RTWO,
                                               RTWO, -RTWO, -RTWO],
                                              [1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.-1.*1j, 1.+1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, 1.+1.*1j, -1.+1.*1j,
                                               1.+1.*1j, -1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                               1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                               1.-1.*1j, 1.+1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, -1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, 1.+1.*1j, 1.+1.*1j,
                                               -1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                               -1.-1.*1j, 1.+1.*1j, -1.+1.*1j,
                                               1.+1.*1j, -1.+1.*1j, -1.-1.*1j,
                                               -1.+1.*1j, -1.-1.*1j, 1.+1.*1j]]
                                             )

        self.chardict['OhP_T2PLUS'] = np.array(3*[[3., 1., 1., 1., 1., 1., 1.,
                                                   0., 0., 0., 0., 0., 0., 0.,
                                                   0., -1., -1., -1., -1., -1.,
                                                   -1., -1., -1., -1., 3., 1.,
                                                   1., 1., 1., 1., 1., 0., 0.,
                                                   0., 0., 0., 0., 0., 0., -1.,
                                                   -1., -1., -1., -1., -1.,
                                                   -1., -1., -1.]])

        self.bTdict['OhP_A1MINUS'] = np.array([[1.0]*24+[-1.0]*24])

        self.chardict['OhP_A1MINUS'] = self.bTdict['OhP_A1MINUS']

        self.bTdict['OhP_A2MINUS'] = np.array([[1., -1., -1., -1., -1., -1.,
                                                -1., 1., 1., 1., 1., 1., 1.,
                                                1., 1., -1., -1., -1., -1.,
                                                -1., -1., 1., 1., 1., -1., 1.,
                                                1., 1., 1., 1., 1., -1., -1.,
                                                -1., -1., -1., -1., -1., -1.,
                                                1., 1., 1., 1., 1., 1., -1.,
                                                -1., -1.]])

        self.chardict['OhP_A2MINUS'] = self.bTdict['OhP_A2MINUS']

        self.bTdict['OhP_EMINUS'] = np.array([[1., 1., 1., 1., 1., -2., -2.,
                                               1., -2., -2., 1., -2., 1., 1.,
                                               -2., 1., 1., -2., -2., 1., 1.,
                                               1., 1., 1., -1., -1., -1., -1.,
                                               -1., 2., 2., -1., 2., 2., -1.,
                                               2., -1., -1., 2., -1., -1., 2.,
                                               2., -1., -1., -1., -1., -1.],
                                              [-RTHREE, RTHREE,
                                               RTHREE, -RTHREE,
                                               -RTHREE, 0., 0., RTHREE,
                                               0., 0., RTHREE, 0.,
                                               RTHREE, RTHREE, 0.,
                                               RTHREE, RTHREE, 0., 0.,
                                               -RTHREE, -RTHREE,
                                               -RTHREE, -RTHREE,
                                               -RTHREE, RTHREE,
                                               -RTHREE, -RTHREE,
                                               RTHREE, RTHREE, 0., 0.,
                                               -RTHREE, 0., 0.,
                                               -RTHREE, 0., -RTHREE,
                                               -RTHREE, 0., -RTHREE,
                                               -RTHREE, 0., 0., RTHREE,
                                               RTHREE, RTHREE,
                                               RTHREE, RTHREE]])

        self.chardict['OhP_EMINUS'] = np.array(2*[[2., 0., 0., 0., 0., 0., 0.,
                                                   -1., -1., -1., -1., -1.,
                                                   -1., -1., -1., 0., 0., 0.,
                                                   0., 0., 0., 2., 2., 2., -2.,
                                                   0., 0., 0., 0., 0., 0., 1.,
                                                   1., 1., 1., 1., 1., 1., 1.,
                                                   0., 0., 0., 0., 0., 0., -2.,
                                                   -2., -2.]])

        self.bTdict['OhP_T1MINUS'] = np.array([[-1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                1.+1.*1j, 1.-1.*1j, -1.-1.*1j,
                                                1.-1.*1j, -1.-1.*1j, 1.-1.*1j,
                                                1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                                -1.-1.*1j, 1.+1.*1j, 1.+1.*1j,
                                                -1.+1.*1j, -1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                                -1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, -1.-1.*1j,
                                                -1.+1.*1j, 1.+1.*1j, -1.+1.*1j,
                                                1.+1.*1j, -1.+1.*1j, -1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                -1.-1.*1j, 1.+1.*1j, 1.+1.*1j,
                                                -1.-1.*1j, -1.-1.*1j, 1.-1.*1j,
                                                1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                -1.-1.*1j, 1.+1.*1j],
                                               [RTWO, -RTWO, -RTWO,
                                                RTWO, -RTWO, RTWO,
                                                -RTWO, -RTWO, RTWO,
                                                -RTWO, RTWO, RTWO,
                                                RTWO, -RTWO, -RTWO,
                                                RTWO, RTWO, RTWO,
                                                -RTWO, RTWO, -RTWO,
                                                RTWO, -RTWO, -RTWO,
                                                -RTWO, RTWO, RTWO,
                                                -RTWO, RTWO, -RTWO,
                                                RTWO, RTWO, -RTWO,
                                                RTWO, -RTWO, -RTWO,
                                                -RTWO, RTWO, RTWO,
                                                -RTWO, -RTWO,
                                                -RTWO, RTWO, -RTWO,
                                                RTWO, -RTWO, RTWO,
                                                RTWO],
                                               [1.+1.*1j, 1.+1.*1j, -1.-1.*1j,
                                                -1.+1.*1j, -1.-1.*1j, 1.-1.*1j,
                                                -1.-1.*1j, 1.-1.*1j, -1.-1.*1j,
                                                -1.+1.*1j, -1.-1.*1j, 1.+1.*1j,
                                                1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                1.-1.*1j, -1.+1.*1j, -1.+1.*1j,
                                                1.+1.*1j, 1.-1.*1j, 1.+1.*1j,
                                                -1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                -1.-1.*1j, -1.-1.*1j, 1.+1.*1j,
                                                1.-1.*1j, 1.+1.*1j, -1.+1.*1j,
                                                1.+1.*1j, -1.+1.*1j, 1.+1.*1j,
                                                1.-1.*1j, 1.+1.*1j, -1.-1.*1j,
                                                -1.-1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                -1.-1.*1j, -1.+1.*1j,
                                                -1.-1.*1j, 1.+1.*1j, 1.-1.*1j,
                                                -1.+1.*1j]])

        self.chardict['OhP_T1MINUS'] = np.array(3*[[3., -1., -1., -1., -1.,
                                                    -1., -1., 0., 0., 0., 0.,
                                                    0., 0., 0., 0., 1., 1., 1.,
                                                    1., 1., 1., -1., -1., -1.,
                                                    -3., 1., 1., 1., 1., 1.,
                                                    1., 0., 0., 0., 0., 0., 0.,
                                                    0., 0., -1., -1., -1., -1.,
                                                    -1., -1., 1., 1., 1.]])

        self.bTdict['OhP_T2MINUS'] = np.array([[RTWO, -RTWO, RTWO,
                                                RTWO, 0.-RTWO*1j,
                                                RTWO, 0.+RTWO*1j,
                                                -RTWO, 0.+RTWO*1j,
                                                -RTWO, 0.-RTWO*1j,
                                                0.-RTWO*1j, 0.+RTWO*1j,
                                                RTWO, RTWO,
                                                0.+RTWO*1j, 0.-RTWO*1j,
                                                -RTWO, 0.-RTWO*1j,
                                                -RTWO, 0.+RTWO*1j,
                                                -RTWO, 0.+RTWO*1j,
                                                0.-RTWO*1j, -RTWO,
                                                RTWO, -RTWO, -RTWO,
                                                0.+RTWO*1j, -RTWO,
                                                0.-RTWO*1j, RTWO,
                                                0.-RTWO*1j, RTWO,
                                                0.+RTWO*1j, 0.+RTWO*1j,
                                                0.-RTWO*1j, -RTWO,
                                                -RTWO, 0.-RTWO*1j,
                                                0.+RTWO*1j, RTWO,
                                                0.+RTWO*1j, RTWO,
                                                0.-RTWO*1j, RTWO,
                                                0.-RTWO*1j,
                                                0.+RTWO*1j],
                                               [1.-1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                1.-1.*1j, -1.+1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                1.-1.*1j, -1.+1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                1.-1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, 1.-1.*1j,
                                                -1.+1.*1j, 1.-1.*1j, -1.+1.*1j,
                                                1.-1.*1j, -1.+1.*1j, 1.-1.*1j,
                                                1.-1.*1j],
                                               [0.-RTWO*1j, 0.+RTWO*1j,
                                                0.-RTWO*1j, 0.-RTWO*1j,
                                                RTWO, 0.-RTWO*1j,
                                                -RTWO, 0.+RTWO*1j,
                                                -RTWO, 0.+RTWO*1j,
                                                RTWO, RTWO, -RTWO,
                                                0.-RTWO*1j, 0.-RTWO*1j,
                                                -RTWO, RTWO,
                                                0.+RTWO*1j, RTWO,
                                                0.+RTWO*1j, -RTWO,
                                                0.+RTWO*1j, -RTWO,
                                                RTWO, 0.+RTWO*1j,
                                                0.-RTWO*1j, 0.+RTWO*1j,
                                                0.+RTWO*1j, -RTWO,
                                                0.+RTWO*1j, RTWO,
                                                0.-RTWO*1j, RTWO,
                                                0.-RTWO*1j, -RTWO,
                                                -RTWO, RTWO,
                                                0.+RTWO*1j, 0.+RTWO*1j,
                                                RTWO, -RTWO,
                                                0.-RTWO*1j, -RTWO,
                                                0.-RTWO*1j, RTWO,
                                                0.-RTWO*1j, RTWO,
                                                -RTWO]])

        self.chardict['OhP_T2MINUS'] = np.array(3*[[3., 1., 1., 1., 1., 1., 1.,
                                                    0., 0., 0., 0., 0., 0., 0.,
                                                    0., -1., -1., -1., -1.,
                                                    -1., -1., -1., -1., -1.,
                                                    -3., -1., -1., -1., -1.,
                                                    -1., -1., 0., 0., 0., 0.,
                                                    0., 0., 0., 0., 1., 1., 1.,
                                                    1., 1., 1., 1., 1., 1.]])

        self.Dic4 = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
             [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
             [[0, -1, 0], [-1, 0, 0], [0, 0, 1]],
             [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
             [[1, 0, 0], [0, -1, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.bTdict['Dic4_A1'] = np.array([[1.]*8])

        self.chardict['Dic4_A1'] = self.bTdict['Dic4_A1']

        self.bTdict['Dic4_A2'] = np.array([[1.]*4+[-1.]*4])

        self.chardict['Dic4_A2'] = self.bTdict['Dic4_A2']

        self.bTdict['Dic4_B1'] = np.array([[1., -1., -1., 1.,
                                            -1., -1., 1., 1.]])

        self.chardict['Dic4_B1'] = self.bTdict['Dic4_B1']

        self.bTdict['Dic4_B2'] = np.array([[1., -1., -1., 1.,
                                            1., 1., -1., -1.]])

        self.chardict['Dic4_B2'] = self.bTdict['Dic4_B2']

        self.bTdict['Dic4_E2'] = np.array([[0., -1j*RTWO, 1j*RTWO, 0.,
                                            -1j*RTWO, 1j*RTWO, 0., 0.],
                                           [RTWO, 0., 0., -RTWO,
                                            0., 0., RTWO, -RTWO]])

        self.chardict['Dic4_E2'] = np.array(2*[[2., 0., 0., -2.,
                                                0., 0., 0., 0.]])

        self.Dic2 = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.bTdict['Dic2_A1'] = np.array([[1.]*4])

        self.chardict['Dic2_A1'] = self.bTdict['Dic2_A1']

        self.bTdict['Dic2_A2'] = np.array([[1.]*2+[-1.]*2])

        self.chardict['Dic2_A2'] = self.bTdict['Dic2_A2']

        self.bTdict['Dic2_B1'] = np.array([[1., -1., -1., 1.]])

        self.chardict['Dic2_B1'] = self.bTdict['Dic2_B1']

        self.bTdict['Dic2_B2'] = np.array([[1., -1., 1., -1.]])

        self.chardict['Dic2_B2'] = self.bTdict['Dic2_B2']

    def get_little_group(self, nP=np.array([0, 0, 0])):
        """Get the little group."""
        if nP@nP == 0:
            return self.OhP
        elif (nP@nP == 1) and (nP == np.array([0, 0, 1])).all():
            return self.Dic4
        elif (nP@nP == 2) and (nP == np.array([0, 1, 1])).all():
            return self.Dic2
        lg = []
        for g_elem in self.OhP:
            nP_rotated = (g_elem@(nP.reshape((3, 1)))).reshape(3)
            if (nP_rotated == nP).all():
                lg = lg+[g_elem]
        return np.array(lg)

    def generate_wigner_d(self, ell, g_elem=np.identity(3),
                          real_harmonics=True):
        """Generate the Wigner D matrix."""
        if np.linalg.det(g_elem) < 0.0:
            g_rot = -1.0*g_elem
            multiplier = (-1.0)**ell
        else:
            g_rot = 1.0*g_elem
            multiplier = 1.0
        R = quaternionic.array.from_rotation_matrix(g_rot)
        D = self.wigner.D(R)
        wig_d = [[]]
        for m in range(-ell, ell+1):
            row = []
            for mp in range(-ell, ell+1):
                entry = D[self.wigner.Dindex(ell, m, mp)]
                row = row+[entry]
            wig_d = wig_d+[row]
        wig_d = np.array(wig_d[1:])

        if real_harmonics:
            U = np.zeros((2*ell+1, 2*ell+1))*1j
            for m_real in range(-ell, ell+1):
                for m_imag in range(-ell, ell+1):
                    if m_real == m_imag == 0:
                        U[m_real+ell][m_imag+ell] = 1.+0.*1j
                    elif m_real == m_imag < 0:
                        U[m_real+ell][m_imag+ell] = 1j/np.sqrt(2.)
                    elif m_real == -m_imag < 0:
                        U[m_real+ell][m_imag+ell] = -1j*(-1.)**m_real\
                            / np.sqrt(2.)
                    elif m_real == m_imag > 0:
                        U[m_real+ell][m_imag+ell] = (-1.)**m_real/np.sqrt(2.)
                    elif m_real == -m_imag > 0:
                        U[m_real+ell][m_imag+ell] = 1./np.sqrt(2.)
            Udagger = np.conjugate(U).T
            wig_d = U@wig_d@Udagger
            if not (np.abs(wig_d.imag) < EPSILON8).all():
                raise ValueError("real Wigner-D is complex")
            wig_d = wig_d.real
        wig_d = wig_d*multiplier
        return wig_d

    def generate_induced_rep(self, nvec_arr=np.zeros((1, 3)),
                             ellm_set=[[0, 0]],
                             g_elem=np.identity(3)):
        """Generate the induced representation matrix."""
        nvec_arr_rot = (g_elem@(nvec_arr.T)).T
        nvec_arr_loc_inds = []
        for i in range(len(nvec_arr)):
            nvec_arr_rot_entry = nvec_arr_rot[i]
            loc_ind = np.where(np.all(nvec_arr == nvec_arr_rot_entry, axis=1))
            nvec_arr_loc_inds = nvec_arr_loc_inds+[loc_ind[0][0]]
        nvec_arr_rot_matrix = [[]]
        for loc_ind in nvec_arr_loc_inds:
            nvec_arr_rot_row = np.zeros(len(nvec_arr))
            nvec_arr_rot_row[loc_ind] = 1.0
            nvec_arr_rot_matrix = nvec_arr_rot_matrix+[nvec_arr_rot_row]
        nvec_arr_rot_matrix = np.array(nvec_arr_rot_matrix[1:])
        wig_d_ell_set = []

        ell_set = np.unique((np.array(ellm_set).T)[0])
        for ell in ell_set:
            wig_d_ell_set = wig_d_ell_set\
                + [self.generate_wigner_d(ell, g_elem,
                                          real_harmonics=True).T]
        wig_d_block = block_diag(*wig_d_ell_set)
        induced_rep = np.kron(nvec_arr_rot_matrix, wig_d_block)
        return induced_rep

    def generate_induced_rep_noninttwo(self,
                                       nvecset_batched=np.zeros((1, 2, 3)),
                                       first_spin=0.0, second_spin=0.0,
                                       g_elem=np.identity(3),
                                       particles_are_identical=False):
        """Generate the non-interacting induced representation matrix."""
        loc_inds = []
        nonidentical_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(nvecset_batched, 0, 2), 2, 0)
        for i in range(len(nvecset_batched)):
            nonidentical_arr_rot_entry = nonidentical_arr_rot[i]
            loc_ind = np.where(
                np.all(nvecset_batched
                       == nonidentical_arr_rot_entry, axis=(1, 2))
                )[0]
            if particles_are_identical:
                nonidentical_arr_rot_entry_swap =\
                    np.array([nonidentical_arr_rot_entry[1],
                              nonidentical_arr_rot_entry[0]])
                loc_ind = np.append(loc_ind, np.where(
                    np.all(nvecset_batched
                           == nonidentical_arr_rot_entry_swap, axis=(1, 2))
                    )[0])
            if len(loc_ind) == 2:
                assert loc_ind[0] == loc_ind[1]
                loc_ind = [loc_ind[0]]
            else:
                assert len(loc_ind) == 1
            loc_inds = loc_inds+[loc_ind[0]]

        nonint_rot_matrix = [[]]
        for loc_ind in loc_inds:
            nonint_rot_row = np.zeros(len(loc_inds))
            nonint_rot_row[loc_ind] = 1.0
            nonint_rot_matrix = nonint_rot_matrix+[nonint_rot_row]
        nonint_rot_matrix = np.array(nonint_rot_matrix[1:])

        assert np.abs(int(first_spin)-first_spin) < EPSILON15
        assert np.abs(int(second_spin)-second_spin) < EPSILON15
        first_spin = int(first_spin)
        second_spin = int(second_spin)
        wig_d_first_spin = self.generate_wigner_d(first_spin, g_elem,
                                                  real_harmonics=True).T
        wig_d_second_spin = self.generate_wigner_d(second_spin, g_elem,
                                                   real_harmonics=True).T
        induced_rep_tmp = np.kron(nonint_rot_matrix, wig_d_first_spin)
        induced_rep = np.kron(induced_rep_tmp, wig_d_second_spin)
        return induced_rep

    def generate_induced_rep_nonint(self, identical_arr=np.zeros((1, 3, 3)),
                                    nonidentical_arr=np.zeros((1, 3, 3)),
                                    g_elem=np.identity(3),
                                    definite_iso=True):
        """Generate the non-interacting induced representation matrix."""
        loc_inds = []
        identical_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(identical_arr, 0, 2), 2, 0)
        for i in range(len(identical_arr)):
            identical_arr_rot_entry = identical_arr_rot[i]
            loc_ind = []
            for pion_order in PION_ORDERS:
                loc_ind_tmp = np.where(
                    np.all(identical_arr
                           == identical_arr_rot_entry[pion_order],
                           axis=(1, 2))
                    )[0]
                loc_ind = loc_ind+list(loc_ind_tmp)
            loc_ind = np.unique(loc_ind)
            assert len(loc_ind) == 1
            loc_inds = loc_inds+[loc_ind[0]]

        if definite_iso:
            nonidentical_arr_rot = np.moveaxis(
                g_elem@np.moveaxis(nonidentical_arr, 0, 2), 2, 0)
            for i in range(len(nonidentical_arr)):
                nonidentical_arr_rot_entry = nonidentical_arr_rot[i]
                loc_ind = np.where(
                    np.all(nonidentical_arr
                           == nonidentical_arr_rot_entry, axis=(1, 2))
                    )[0]
                assert len(loc_ind) == 1
                loc_inds = loc_inds+[loc_ind[0]+len(identical_arr)]

        nonint_rot_matrix = [[]]
        for loc_ind in loc_inds:
            nonint_rot_row = np.zeros(len(loc_inds))
            nonint_rot_row[loc_ind] = 1.0
            nonint_rot_matrix = nonint_rot_matrix+[nonint_rot_row]
        nonint_rot_matrix = np.array(nonint_rot_matrix[1:])
        return nonint_rot_matrix

    def generate_induced_rep_nonint_spin(self,
                                         identical_arr=np.zeros((1, 3, 3)),
                                         nonidentical_arr=np.zeros((1, 3, 3)),
                                         first_spin=0.0, second_spin=0.0,
                                         third_spin=0.0,
                                         g_elem=np.identity(3),
                                         definite_iso=True):
        """
        Generate the non-interacting induced representation matrix,
        including spin.
        """
        loc_inds = []
        identical_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(identical_arr, 0, 2), 2, 0)
        for i in range(len(identical_arr)):
            identical_arr_rot_entry = identical_arr_rot[i]
            loc_ind = []
            for pion_order in PION_ORDERS:
                loc_ind_tmp = np.where(
                    np.all(identical_arr
                           == identical_arr_rot_entry[pion_order],
                           axis=(1, 2))
                    )[0]
                loc_ind = loc_ind+list(loc_ind_tmp)
            loc_ind = np.unique(loc_ind)
            assert len(loc_ind) == 1
            loc_inds = loc_inds+[loc_ind[0]]
        nonint_rot_matrix = [[]]
        for loc_ind in loc_inds:
            nonint_rot_row = np.zeros(len(loc_inds))
            nonint_rot_row[loc_ind] = 1.0
            nonint_rot_matrix = nonint_rot_matrix+[nonint_rot_row]
        nonint_rot_matrix = np.array(nonint_rot_matrix[1:])
        if np.abs(first_spin - int(first_spin)) > EPSILON10:
            raise ValueError("first_spin must be an integer")
        if np.abs(second_spin - int(second_spin)) > EPSILON10:
            raise ValueError("second_spin must be an integer")
        if np.abs(third_spin - int(third_spin)) > EPSILON10:
            raise ValueError("third_spin must be an integer")
        first_spin = int(first_spin)
        second_spin = int(second_spin)
        third_spin = int(third_spin)
        wig_d_first_spin = self.generate_wigner_d(first_spin, g_elem,
                                                  real_harmonics=True).T
        wig_d_second_spin = self.generate_wigner_d(second_spin, g_elem,
                                                   real_harmonics=True).T
        wig_d_third_spin = self.generate_wigner_d(third_spin, g_elem,
                                                  real_harmonics=True).T
        induced_rep_tmp = np.kron(nonint_rot_matrix, wig_d_first_spin)
        induced_rep_tmp2 = np.kron(induced_rep_tmp, wig_d_second_spin)
        induced_rep = np.kron(induced_rep_tmp2, wig_d_third_spin)
        return induced_rep

    def get_large_proj(self, nP=np.array([0, 0, 0]), irrep='A1PLUS', irow=0,
                       nvec_arr=np.zeros((1, 3)),
                       ellm_set=[[0, 0]]):
        """Get a particular large projector."""
        if (nP == np.array([0, 0, 0])).all():
            group_str = 'OhP'
            group = self.OhP
            bT = self.bTdict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 0, 1])).all():
            group_str = 'Dic4'
            group = self.Dic4
            bT = self.bTdict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 1, 1])).all():
            group_str = 'Dic2'
            group = self.Dic2
            bT = self.bTdict[group_str+'_'+irrep][irow]
        else:
            return ValueError("group not yet supported by get_large_proj")
        dim = len(nvec_arr)*len(ellm_set)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep(nvec_arr, ellm_set,
                                                    g_elem)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_large_proj_nonint(self, nP=np.array([0, 0, 0]), irrep='A1PLUS',
                              irow=0,
                              identical_arr=np.zeros((1, 3, 3)),
                              nonidentical_arr=np.zeros((1, 3, 3)),
                              definite_iso=True):
        """Get a particular large projector."""
        if (nP == np.array([0, 0, 0])).all():
            group_str = 'OhP'
            group = self.OhP
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 0, 1])).all():
            group_str = 'Dic4'
            group = self.Dic4
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 1, 1])).all():
            group_str = 'Dic2'
            group = self.Dic2
            bT = self.chardict[group_str+'_'+irrep][irow]
        else:
            return ValueError("group not yet supported by get_large_proj")
        if definite_iso:
            dim = len(identical_arr)+len(nonidentical_arr)
        else:
            dim = len(identical_arr)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep_nonint(identical_arr,
                                                           nonidentical_arr,
                                                           g_elem,
                                                           definite_iso)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_large_proj_nonint_spin(self, nP=np.array([0, 0, 0]),
                                   irrep='A1PLUS', irow=0,
                                   identical_arr=np.zeros((1, 3, 3)),
                                   nonidentical_arr=np.zeros((1, 3, 3)),
                                   first_spin=0.0, second_spin=0.0,
                                   third_spin=0.0,
                                   definite_iso=True):
        """Get a particular large projector, including spin."""
        if (nP == np.array([0, 0, 0])).all():
            group_str = 'OhP'
            group = self.OhP
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 0, 1])).all():
            group_str = 'Dic4'
            group = self.Dic4
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 1, 1])).all():
            group_str = 'Dic2'
            group = self.Dic2
            bT = self.chardict[group_str+'_'+irrep][irow]
        else:
            return ValueError("group not yet supported by get_large_proj")
        if definite_iso:
            dim = len(identical_arr)+len(nonidentical_arr)
        else:
            dim = len(identical_arr)
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0)
                                   * (2.0*third_spin+1.0))
        dim = dim*total_spin_dimension
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.\
                generate_induced_rep_nonint_spin(identical_arr,
                                                 nonidentical_arr,
                                                 first_spin,
                                                 second_spin,
                                                 third_spin,
                                                 g_elem,
                                                 definite_iso)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_large_proj_nonint_two(self, nP=np.array([0, 0, 0]), irrep='A1PLUS',
                                  irow=0,
                                  nvecset_batched=np.zeros((1, 2, 3)),
                                  first_spin=0.0, second_spin=0.0,
                                  particles_are_identical=False):
        """Get a particular large projector."""
        if (nP == np.array([0, 0, 0])).all():
            group_str = 'OhP'
            group = self.OhP
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 0, 1])).all():
            group_str = 'Dic4'
            group = self.Dic4
            bT = self.chardict[group_str+'_'+irrep][irow]
        elif (nP == np.array([0, 1, 1])).all():
            group_str = 'Dic2'
            group = self.Dic2
            bT = self.chardict[group_str+'_'+irrep][irow]
        else:
            return ValueError("group not yet supported by get_large_proj")
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0))
        dim = len(nvecset_batched)*total_spin_dimension
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self\
                .generate_induced_rep_noninttwo(nvecset_batched,
                                                first_spin,
                                                second_spin,
                                                g_elem,
                                                particles_are_identical)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def _get_final_proj(self, proj):
        eigvals, eigvecs = np.linalg.eig(proj)
        eigvecsT = eigvecs.T
        eigvals_chop = []
        for eigval in eigvals:
            if (np.abs(eigval.imag) < EPSILON8):
                eigval = eigval.real
            if isinstance(eigval, float)\
               and (np.abs(eigval) < EPSILON8):
                eigval = 0.0
            if (np.abs(eigval.real) < EPSILON8):
                eigval = eigval.imag*1j
            eigvals_chop = eigvals_chop+[eigval]
        eigvals_chop = np.array(eigvals_chop)
        eigvecsT_nonzero = eigvecsT[np.where(eigvals_chop
                                             != np.array(0.0))[0]]
        eigvecsT_chop = np.zeros(eigvecsT_nonzero.shape,
                                 dtype=complex)
        for i in range(len(eigvecsT_nonzero)):
            for j in range(len(eigvecsT_nonzero[i])):
                eigvecT_entry = eigvecsT_nonzero[i][j]
                if (np.abs(eigvecT_entry.imag) < EPSILON8):
                    eigvecT_entry = eigvecT_entry.real
                if isinstance(eigvecT_entry, float)\
                   and (np.abs(eigvecT_entry) < EPSILON8):
                    eigvecT_entry = 0.0
                if (np.abs(eigvecT_entry.real) < EPSILON8):
                    eigvecT_entry = eigvecT_entry.imag*1j
                eigvecsT_chop[i][j] = eigvecT_entry
        eigvecs_chop_orthog, r = np.linalg.qr(eigvecsT_chop.T)
        eigvecsT_chop_onorm = [[]]
        for eigvecT in eigvecs_chop_orthog.T:
            norm = np.sqrt(eigvecT@np.conjugate(eigvecT))
            eigvecT_norm = eigvecT/norm
            eigvecsT_chop_onorm = eigvecsT_chop_onorm+[eigvecT_norm]
        eigvecsT_chop_onorm = np.array(eigvecsT_chop_onorm[1:])
        eigvecsT_final = np.zeros(eigvecsT_chop_onorm.shape,
                                  dtype=complex)
        for i in range(len(eigvecsT_chop_onorm)):
            for j in range(len(eigvecsT_chop_onorm[i])):
                eigvecT_entry = eigvecsT_chop_onorm[i][j]
                if (np.abs(eigvecT_entry.imag) < EPSILON8):
                    eigvecT_entry = eigvecT_entry.real
                if isinstance(eigvecT_entry, float)\
                   and (np.abs(eigvecT_entry) < EPSILON8):
                    eigvecT_entry = 0.0
                if (np.abs(eigvecT_entry.real) < EPSILON8):
                    eigvecT_entry = eigvecT_entry.imag*1j
                eigvecsT_final[i][j] = eigvecT_entry
        finalproj = np.array(eigvecsT_final).T
        return finalproj

    def _get_summary(self, proj_dict, group_str, qcis, totalsize):
        contrib_irreps = []
        for key in proj_dict:
            if key[1] == 0:
                contrib_irreps = contrib_irreps+[key[0]]

        best_irreps = []
        for irrep_tmp in contrib_irreps:
            irrep_dim = len(self.bTdict[group_str+'_'+irrep_tmp])
            i = irrep_dim-1
            i_best = 0
            while i >= 0:
                proj_tmp = proj_dict[(irrep_tmp, i)]
                if proj_tmp.dtype == np.float64:
                    i_best = i
                i -= 1
            best_irreps = best_irreps+[(irrep_tmp, i_best)]

        summary_str = f"kellm space has size {str(totalsize)}\n\n"
        total = 0
        for key in best_irreps:
            n_times = len(proj_dict[key].T)
            n_dim = len(self.bTdict[group_str+'_'+key[0]])
            n_tot = n_times*n_dim

            pad = (7-len(key[0]))*" "
            summary_str += f"    {key[0]+pad} covers {n_times}x{n_dim}"\
                + f" = {n_tot} slots\n"
            total = total+n_tot
        summary_str += f"\ntotal is {total} \n"
        if total == totalsize:
            summary_str += "total matches size of kellm space"
        else:
            summary_str += "does not match size of kellm space, "\
                + "something went wrong"
        return best_irreps, summary_str

    def get_iso_projection(self, qcis=None, cindex=0, iso_index=0,
                           shell_index=0):
        """Get the iso-projector for non-interacting vectors."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        identical_arr = qcis.nvecset_ident_batched[cindex][shell_index]
        nonidentical_arr = qcis.nvecset_batched[cindex][shell_index]
        iso_projector = ISO_PROJECTORS[iso_index]
        iso_prepare_sets = []
        id_sub_len = len(identical_arr)
        for ident_subset_index in range(id_sub_len):
            ident_subset_entry = identical_arr[ident_subset_index]
            iso_prepare_entry = [ident_subset_index-id_sub_len]
            for pion_order in PION_ORDERS:
                loc_indices = np.where(
                    (nonidentical_arr
                     == ident_subset_entry[pion_order]).all(axis=(1, 2))
                    )
                assert len(loc_indices) == 1
                loc_index = loc_indices[0][0]
                iso_prepare_entry = iso_prepare_entry+[loc_index]
            iso_prepare_sets = iso_prepare_sets+[iso_prepare_entry]
        iso_prepare_sets = np.array(iso_prepare_sets).T+id_sub_len
        three_ident_entry = iso_prepare_sets[0]
        iso_prepare_sets = np.insert(iso_prepare_sets, 4,
                                     three_ident_entry, axis=0)
        iso_prepare_sets = np.delete(iso_prepare_sets, 0, axis=0)
        iso_prepare = []
        for iso_prepare_set in iso_prepare_sets.T:
            iso_prepare = iso_prepare+list(iso_prepare_set)
        iso_prepare_mat = (np.identity(len(iso_prepare))[iso_prepare])
        iso_prepare_matT = iso_prepare_mat.T
        mask = []
        for i in range(len(iso_prepare_matT)):
            mask = mask+[not (iso_prepare_matT[i] == 0.).all()]
        iso_prepare_matT_masked = iso_prepare_matT[mask]
        iso_prepare_mat = iso_prepare_matT_masked.T

        iso_rot = block_diag(*(id_sub_len*[CAL_C_ISO]))
        full_chbasis = iso_rot@iso_prepare_mat
        full_iso_proj = block_diag(*(id_sub_len*[iso_projector]))
        # assert (((full_chbasis.T)@full_chbasis
        #         - np.identity(len(full_chbasis))) < 1.e-10).all()
        final = full_iso_proj@full_chbasis
        return final

    def get_nonint_proj_dict_shell(self, qcis=None, cindex=0,
                                   definite_iso=False, isovalue=None,
                                   shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        identical_arr = qcis.nvecset_ident_batched[cindex][shell_index]
        nonidentical_arr = qcis.nvecset_batched[cindex][shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        non_proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'

        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                proj = self.get_large_proj_nonint(nP, irrep, irow,
                                                  identical_arr,
                                                  nonidentical_arr,
                                                  definite_iso)
                eigvals, eigvecs = np.linalg.eig(proj)
                eigvalsround = (np.round(np.abs(eigvals), 10))
                example_eigval = 0.0
                for i in range(len(eigvalsround)):
                    eigval = eigvalsround[i]
                    if np.abs(eigval) > 1.0e-10:
                        if example_eigval == 0.0:
                            example_eigval = eigval
                        else:
                            assert np.abs(
                                example_eigval-eigval
                                ) < 1.0e-10
                if np.abs(example_eigval) > 1.0e-10:
                    proj = proj/example_eigval
                if definite_iso:
                    isoproj = self.get_iso_projection(qcis, cindex, isovalue,
                                                      shell_index)
                    isorotproj = isoproj@proj@np.transpose(isoproj)
                else:
                    isorotproj = proj
                finalproj = self._get_final_proj(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_nonint_proj_dict_shell_vector(self, qcis=None, cindex=0,
                                          definite_iso=False, isovalue=None,
                                          shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        identical_arr = qcis.nvecset_ident_batched[cindex][shell_index]
        nonidentical_arr = qcis.nvecset_batched[cindex][shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        non_proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'

        first_spin = qcis.fcs.ni_list[cindex].spins[0]
        second_spin = qcis.fcs.ni_list[cindex].spins[1]
        third_spin = qcis.fcs.ni_list[cindex].spins[2]
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                proj = self.get_large_proj_nonint_spin(nP, irrep, irow,
                                                       identical_arr,
                                                       nonidentical_arr,
                                                       first_spin,
                                                       second_spin,
                                                       third_spin,
                                                       definite_iso)
                eigvals, eigvecs = np.linalg.eig(proj)
                eigvalsround = (np.round(np.abs(eigvals), 10))
                example_eigval = 0.0
                for i in range(len(eigvalsround)):
                    eigval = eigvalsround[i]
                    if np.abs(eigval) > 1.0e-10:
                        if example_eigval == 0.0:
                            example_eigval = eigval
                        else:
                            assert np.abs(
                                example_eigval-eigval
                                ) < 1.0e-10
                if np.abs(example_eigval) > 1.0e-10:
                    proj = proj/example_eigval
                if definite_iso:
                    isoproj = self.get_iso_projection(qcis, cindex, isovalue,
                                                      shell_index)
                    isorotproj = isoproj@proj@np.transpose(isoproj)
                else:
                    isorotproj = proj
                finalproj = self._get_final_proj(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_noninttwo_proj_dict_shell(self, qcis=None, cindex=0,
                                      definite_iso=False, isovalue=None,
                                      shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        flavors = qcis.fcs.ni_list[cindex].flavors
        particles_are_identical = (flavors[0] == flavors[1])
        if particles_are_identical:
            nvecset_batched = qcis.nvecset_ident_batched[cindex][shell_index]
        else:
            nvecset_batched = qcis.nvecset_batched[cindex][shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        non_proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'

        first_spin = qcis.fcs.ni_list[cindex].spins[0]
        second_spin = qcis.fcs.ni_list[cindex].spins[1]
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.chardict[group_str+'_'+irrep])):
                proj = self.get_large_proj_nonint_two(nP, irrep, irow,
                                                      nvecset_batched,
                                                      first_spin,
                                                      second_spin,
                                                      particles_are_identical)
                some_zero_vec = False
                for batch in nvecset_batched:
                    for single_vec in batch:
                        some_zero_vec = some_zero_vec\
                            or (single_vec@single_vec == 0)
                proj = np.round(proj, 10)
                some_zero_vec = True
                if some_zero_vec:
                    zero_rows = []
                    nonzero_rows = []
                    for i in range(len(proj)):
                        if np.abs(proj[i]@proj[i]) < 1.0e-10:
                            zero_rows = zero_rows+[i]
                        else:
                            nonzero_rows = nonzero_rows+[i]
                    proj = np.concatenate([proj[nonzero_rows],
                                           proj[zero_rows]])
                    projT = proj.T
                    zero_rows = []
                    nonzero_rows = []
                    for i in range(len(projT)):
                        if np.abs(projT[i]@projT[i]) < EPSILON10:
                            zero_rows = zero_rows+[i]
                        else:
                            nonzero_rows = nonzero_rows+[i]
                    projT = np.concatenate([projT[nonzero_rows],
                                            projT[zero_rows]])
                    proj = projT.T
                eigvals, eigvecs = np.linalg.eig(proj)
                eigvalsround = (np.round(np.abs(eigvals), 10))
                example_eigval = 0.0
                for i in range(len(eigvalsround)):
                    eigval = eigvalsround[i]
                    if np.abs(eigval) > 1.0e-10:
                        if example_eigval == 0.0:
                            example_eigval = eigval
                        else:
                            # assert np.abs(
                            #     example_eigval-eigval
                            #     ) < 1.0e-10
                            pass
                if np.abs(example_eigval) > 1.0e-10:
                    proj = proj/example_eigval
                if definite_iso:
                    # isoproj = self.get_iso_projection(qcis, cindex, isovalue,
                    #                                   shell_index)
                    # isorotproj = isoproj@proj@np.transpose(isoproj)
                    isorotproj = proj
                else:
                    isorotproj = proj
                finalproj = self._get_final_proj(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_nonint_proj_dict(self, qcis=None, nic_index=0):
        """Get it."""
        master_dict = {}
        if qcis is None:
            raise ValueError("qcis cannot be None")
        definite_iso = qcis.fcs.fc_list[nic_index].isospin_channel
        if not (qcis.fcs.fc_list[nic_index].flavors[0]
                == qcis.fcs.fc_list[nic_index].flavors[1]
                == qcis.fcs.fc_list[nic_index].flavors[2]):
            raise ValueError("get_nonint_proj_dict currently only supports "
                             + "identical flavors")
        row_zero_value = 0
        summary_str = ""
        nshells = len(qcis.nvecset_ident_reps[nic_index])
        for shell_index in range(nshells):
            shell_total = 0
            if definite_iso:
                nident =\
                    len(qcis.nvecset_ident_batched[nic_index][shell_index])
                nrest =\
                    len(qcis.nvecset_batched[nic_index][shell_index])
                nstates = nident+nrest
            else:
                nstates =\
                    len(qcis.nvecset_ident_batched[nic_index][shell_index])
            summary_str +=\
                f"shell_index = {shell_index} ({nstates} states):\n"
            rep_mom = str(qcis.nvecset_ident_reps[nic_index][shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            if definite_iso:
                isoset = range(4)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict =\
                    self.get_nonint_proj_dict_shell(qcis, nic_index,
                                                    definite_iso,
                                                    isovalue,
                                                    shell_index)
                master_dict[(shell_index, isovalue)] = non_proj_dict
                iso_shell_total = 0
                if definite_iso:
                    if len(non_proj_dict) == 0:
                        summary_str\
                            += f"    I3 = {isovalue} does not contain this "\
                            + "shell\n"
                    else:
                        summary_str\
                            += f"    I3 = {isovalue} contains...\n"
                else:
                    if len(non_proj_dict) == 0:
                        summary_str\
                            += "    Channel does not contain this shell\n"
                    else:
                        summary_str += "    Channel contains...\n"
                for dict_ent in non_proj_dict:
                    irrep, row = dict_ent
                    dim = 1
                    if irrep[0] == 'E':
                        dim = 2
                    if irrep[0] == 'T':
                        dim = 3
                    n_embedded = int(len(non_proj_dict[dict_ent].T)/dim)
                    if row == 0:
                        row_zero_value = n_embedded
                    else:
                        if row_zero_value != n_embedded:
                            print(f'Warning: row_zero_value = '
                                  f'{row_zero_value}, n_embedded = '
                                  f'{n_embedded}')
                    shell_total = shell_total+n_embedded
                    iso_shell_total = iso_shell_total+n_embedded
                    if row == 0:
                        if n_embedded == 1:
                            s = ''
                        else:
                            s = 's'
                        shell_covered = shell_total+n_embedded*(dim-1)
                        iso_shell_covered = iso_shell_total+n_embedded*(dim-1)
                        if definite_iso:
                            summary_str +=\
                                (f"       {irrep} "
                                 f"(appears {n_embedded} time{s}), "
                                 f"covered {shell_covered}/{nstates} "
                                 f"({iso_shell_covered} for this isospin)\n")
                        else:
                            summary_str +=\
                                (f"       {irrep} "
                                 f"(appears {n_embedded} time{s}), "
                                 f"covered {shell_covered}/{nstates}\n")
            assert shell_total == nstates
        summary_str = summary_str[:-1]
        master_dict['summary'] = summary_str
        return master_dict

    def get_nonint_proj_dict_half(self, qcis=None, nic_index=0):
        raise ValueError("get_nonint_proj_dict currently only supports "
                         "integer spins")

    def get_nonint_proj_dict_vector(self, qcis=None, nic_index=0):
        """Get the dictionary of small projectors for a given qcis."""
        master_dict = {}
        if qcis is None:
            raise ValueError("qcis cannot be None")
        definite_iso = qcis.fcs.fc_list[nic_index].isospin_channel
        if not (qcis.fcs.fc_list[nic_index].flavors[0]
                == qcis.fcs.fc_list[nic_index].flavors[1]
                == qcis.fcs.fc_list[nic_index].flavors[2]):
            raise ValueError("get_nonint_proj_dict currently only supports "
                             + "identical flavors")
        row_zero_value = 0
        summary_str = ""
        nshells = len(qcis.nvecset_ident_reps[nic_index])
        first_spin = qcis.fcs.ni_list[nic_index].spins[0]
        second_spin = qcis.fcs.ni_list[nic_index].spins[1]
        third_spin = qcis.fcs.ni_list[nic_index].spins[2]
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0)
                                   * (2.0*third_spin+1.0))
        # flavors = qcis.fcs.ni_list[nic_index].flavors
        for shell_index in range(nshells):
            shell_total = 0
            if definite_iso:
                nident =\
                    len(qcis.nvecset_ident_batched[nic_index][shell_index])\
                    * total_spin_dimension
                nrest =\
                    len(qcis.nvecset_batched[nic_index][shell_index])\
                    * total_spin_dimension
                nstates = nident+nrest
            else:
                nstates =\
                    len(qcis.nvecset_ident_batched[nic_index][shell_index])\
                    * total_spin_dimension
            summary_str +=\
                f"shell_index = {shell_index} ({nstates} states):\n"
            rep_mom = str(qcis.nvecset_ident_reps[nic_index][shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            if definite_iso:
                isoset = range(4)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict =\
                    self.get_nonint_proj_dict_shell_vector(qcis, nic_index,
                                                           definite_iso,
                                                           isovalue,
                                                           shell_index)
                iso_shell_total = 0
                if definite_iso:
                    if len(non_proj_dict) == 0:
                        summary_str\
                            += f"    I3 = {isovalue} does not contain this "\
                            + "shell\n"
                    else:
                        summary_str\
                            += f"    I3 = {isovalue} contains...\n"
                else:
                    if len(non_proj_dict) == 0:
                        summary_str\
                            += "    Channel does not contain this shell\n"
                    else:
                        summary_str += "    Channel contains...\n"
                for dict_ent in non_proj_dict:
                    irrep, row = dict_ent
                    dim = 1
                    if irrep[0] == 'E':
                        dim = 2
                    if irrep[0] == 'T':
                        dim = 3
                    n_embedded = int(len(non_proj_dict[dict_ent].T)/dim)
                    if row == 0:
                        row_zero_value = n_embedded
                    else:
                        if row_zero_value != n_embedded:
                            print(f'Warning: row_zero_value = '
                                  f'{row_zero_value}, n_embedded = '
                                  f'{n_embedded}')
                    shell_total = shell_total+n_embedded
                    iso_shell_total = iso_shell_total+n_embedded
                    if row == 0:
                        if n_embedded == 1:
                            s = ''
                        else:
                            s = 's'
                        shell_covered = shell_total+n_embedded*(dim-1)
                        iso_shell_covered = iso_shell_total+n_embedded*(dim-1)
                        if definite_iso:
                            summary_str +=\
                                (f"       {irrep} "
                                 f"(appears {n_embedded} time{s}), "
                                 f"covered {shell_covered}/{nstates} "
                                 f"({iso_shell_covered} for this isospin)\n")
                        else:
                            summary_str +=\
                                (f"       {irrep} "
                                 f"(appears {n_embedded} time{s}), "
                                 f"covered {shell_covered}/{nstates}\n")
            assert shell_total == nstates
        summary_str = summary_str[:-1]
        master_dict['summary'] = summary_str
        return master_dict

    def get_noninttwo_proj_dict(self, qcis=None, nic_index=0,
                                isospin_channel=True):
        """Get it."""
        master_dict = {}
        if qcis is None:
            raise ValueError("qcis cannot be None")
        row_zero_value = 0
        summary_str = ""
        nshells = len(qcis.nvecset_reps[nic_index])
        first_spin = qcis.fcs.ni_list[nic_index].spins[0]
        second_spin = qcis.fcs.ni_list[nic_index].spins[1]
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0))
        flavors = qcis.fcs.ni_list[nic_index].flavors
        particles_are_identical = (flavors[0] == flavors[1])
        if particles_are_identical:
            nvecset_batched_cindex = qcis.nvecset_ident_batched[nic_index]
        else:
            nvecset_batched_cindex = qcis.nvecset_batched[nic_index]
        if particles_are_identical:
            nvecset_reps_cindex = qcis.nvecset_ident_reps[nic_index]
        else:
            nvecset_reps_cindex = qcis.nvecset_reps[nic_index]
        for shell_index in range(nshells):
            shell_total = 0
            nstates = len(nvecset_batched_cindex[shell_index])\
                * total_spin_dimension
            summary_str\
                += f"shell_index = {shell_index} ({nstates} states):\n"

            rep_mom = str(nvecset_reps_cindex[shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            if isospin_channel:
                isoset = [int(qcis.fcs.ni_list[nic_index].isospin)]
                warnings.warn(f"\n{bcolors.WARNING}"
                              f"isoset is being used in ni_proj_dict, but "
                              "a set is not needed because only one value is "
                              "selected . Also casting to an int will create "
                              "problems for spin-half particles."
                              f"{bcolors.ENDC}", stacklevel=2)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict = self\
                    .get_noninttwo_proj_dict_shell(qcis, nic_index,
                                                   isospin_channel,
                                                   isovalue,
                                                   shell_index)
                master_dict[(shell_index, isovalue)] = non_proj_dict
                iso_shell_total = 0
                if len(non_proj_dict) == 0:
                    if isospin_channel:
                        summary_str\
                            += f"    I2 = {isovalue} does not contain "\
                            + "this shell\n"
                    else:
                        summary_str\
                            += "    channel does not contain this shell\n"
                else:
                    if isospin_channel:
                        summary_str\
                            += f"    I2 = {isovalue} contains...\n"
                    else:
                        summary_str\
                            += "    channel contains...\n"
                for dict_ent in non_proj_dict:
                    irrep, row = dict_ent
                    dim = 1
                    if irrep[0] == 'E':
                        dim = 2
                    if irrep[0] == 'T':
                        dim = 3
                    n_embedded = int(len(non_proj_dict[dict_ent].T)/dim)
                    if row == 0:
                        row_zero_value = n_embedded
                    else:
                        if row_zero_value != n_embedded:
                            print(f'Warning: row_zero_value = '
                                  f'{row_zero_value}, n_embedded = '
                                  f'{n_embedded}')
                    n_embedded = row_zero_value
                    shell_total = shell_total+n_embedded
                    iso_shell_total = iso_shell_total+n_embedded
                    if row == 0:
                        if n_embedded == 1:
                            s = ''
                        else:
                            s = 's'
                        shell_covered = shell_total+n_embedded*(dim-1)
                        iso_shell_covered = iso_shell_total+n_embedded*(dim-1)
                        summary_str +=\
                            (f"       {irrep} "
                             f"(appears {n_embedded} time{s}), "
                             f"covered {shell_covered}/{nstates} "
                             f"({iso_shell_covered} for this isospin)\n")
            assert shell_total == nstates
        summary_str = summary_str[:-1]
        master_dict['summary'] = summary_str
        return master_dict

    def get_channel_proj_dict(self, qcis=None, sc_index=0):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis.verbosity >= 2:
            print("getting the dict for channel =", sc_index)
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                if sc_index < qcis.n_two_channels:
                    slot_index = 0
                else:
                    cindex_shift = sc_index-qcis.n_two_channels
                    slot_index = -1
                    for k in range(len(qcis.fcs.slices_by_three_masses)):
                        three_slice = qcis.fcs.slices_by_three_masses[k]
                        if three_slice[0] <= cindex_shift < three_slice[1]:
                            slot_index = k
                    if qcis.n_two_channels > 0:
                        slot_index = slot_index+1
                slice_index = 0
                # Check issue in following slice code
                # for three_slice in qcis.fcs.slices_by_three_masses:
                #     if slot_index-qcis.n_two_channels > three_slice[1]:
                #         slice_index = slice_index+1
                nvec_arr = qcis.tbks_list[slot_index][slice_index].nvec_arr
                ellm_set = qcis.ellm_sets[sc_index]
                proj = self.get_large_proj(nP=nP, irrep=irrep,
                                           irow=irow,
                                           nvec_arr=nvec_arr,
                                           ellm_set=ellm_set)

                finalproj = self._get_final_proj(proj)
                if len(finalproj) != 0:
                    proj_dict[(irrep, irow)] = finalproj
                for keytmp in proj_dict:
                    proj_tmp = proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        proj_dict[keytmp] = proj_tmp.real

        totalsize = len(qcis.kellm_spaces[sc_index][0])
        proj_dict['best_irreps'], proj_dict['summary']\
            = self._get_summary(proj_dict, group_str, qcis, totalsize)

        return proj_dict

    def get_shell_proj_dict(self, qcis=None, cindex=0, kellm_shell=None,
                            shell_index=None):
        """Get the dictionary of small projectors for one kellm_shell."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        if kellm_shell is None:
            raise ValueError("kellm_shell cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                if shell_index is None:
                    shell_index = 0
                if cindex < qcis.n_two_channels:
                    slot_index = 0
                else:
                    cindex_shift = cindex-qcis.n_two_channels
                    slot_index = -1
                    for k in range(len(qcis.fcs.slices_by_three_masses)):
                        three_slice = qcis.fcs.slices_by_three_masses[k]
                        if three_slice[0] <= cindex_shift < three_slice[1]:
                            slot_index = k
                    if qcis.n_two_channels > 0:
                        slot_index = slot_index+1
                nvec_arr = qcis.tbks_list[slot_index][shell_index].nvec_arr
                ellm_set = qcis.ellm_sets[cindex]
                nshell = [int(kellm_shell[0]/len(ellm_set)),
                          int(kellm_shell[1]/len(ellm_set))]
                assert np.abs(int(kellm_shell[0]/len(ellm_set))
                              - kellm_shell[0]/len(ellm_set)) < EPSILON8
                assert np.abs(int(kellm_shell[1]/len(ellm_set))
                              - kellm_shell[1]/len(ellm_set)) < EPSILON8
                nvec_arr = qcis.tbks_list[slot_index][shell_index].nvec_arr[
                    nshell[0]:nshell[1]]
                ellm_set = qcis.ellm_sets[cindex]
                proj = self.get_large_proj(nP=nP, irrep=irrep,
                                           irow=irow,
                                           nvec_arr=nvec_arr,
                                           ellm_set=ellm_set)

                finalproj = self._get_final_proj(proj)
                if len(finalproj) != 0:
                    proj_dict[(irrep, irow)] = finalproj
                for keytmp in proj_dict:
                    proj_tmp = proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        proj_dict[keytmp] = proj_tmp.real
        return proj_dict

    def get_full_proj_dict(self, qcis=None):
        """Get the dictionary of small projectors for the entire qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        proj_dict = {}
        nP = qcis.nP
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(qcis.fvs.irrep_set)):
            irrep = qcis.fvs.irrep_set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                proj_list = []
                for cindex in range(qcis.n_channels):
                    if cindex < qcis.n_two_channels:
                        slot_index = 0
                    else:
                        cindex_shift = cindex-qcis.n_two_channels
                        slot_index = -1
                        for k in range(len(qcis.fcs.slices_by_three_masses)):
                            three_slice = qcis.fcs.slices_by_three_masses[k]
                            if three_slice[0] <= cindex_shift < three_slice[1]:
                                slot_index = k
                        if qcis.n_two_channels > 0:
                            slot_index = slot_index+1
                    slice_index = 0
                    nvec_arr = qcis.tbks_list[slot_index][slice_index].nvec_arr
                    ellm_set = qcis.ellm_sets[cindex]
                    proj_tmp = self.get_large_proj(nP=nP, irrep=irrep,
                                                   irow=irow,
                                                   nvec_arr=nvec_arr,
                                                   ellm_set=ellm_set)
                    proj_list = proj_list+[proj_tmp]
                proj = block_diag(*proj_list)
                finalproj = self._get_final_proj(proj)
                if len(finalproj) != 0:
                    proj_dict[(irrep, irow)] = finalproj
                for keytmp in proj_dict:
                    proj_tmp = proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        proj_dict[keytmp] = proj_tmp.real

        totalsize = 0
        for cindex in range(qcis.n_channels):
            totalsize = totalsize+len(qcis.kellm_spaces[cindex][0])
        proj_dict['best_irreps'], proj_dict['summary']\
            = self._get_summary(proj_dict, group_str, qcis, totalsize)

        return proj_dict
