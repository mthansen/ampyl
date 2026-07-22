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
from scipy.linalg import expm
from .constants import RTHREE
from .constants import RTWO
from .constants import ISO_PROJECTORS
from .constants import CAL_C_ISO
from .constants import PION_ORDERS
from .constants import EPSILON8
from .constants import EPSILON10
from .constants import EPSILON15
from .constants import bcolors
from .wigner import generate_wigner_d
import warnings
warnings.simplefilter("once")


class Groups:
    """Class for finite-volume group-theory relevant for three particles."""

    def __init__(self, ell_max, spin_half=False):
        self.ell_max = ell_max
        self.spin_half = spin_half

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

        self.OhP_double_PLUS, self.OhP_double_PLUS_intspin =\
            self._get_double_cover_group()

        self.OhP_double_MINUS, self.OhP_double_MINUS_intspin =\
            self._get_double_cover_group(parity=-1)

        self.chardict = {}

        self.chardict['OhP_G1PLUS'] = self._get_G1_char()

        self.chardict['OhP_G1MINUS'] = self._get_G1_char(parity=-1)

        self.chardict['OhP_G2PLUS'] =\
            np.array(2*[[2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
                         1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
                         -RTWO, RTWO, RTWO, -RTWO, -RTWO, RTWO, RTWO, -RTWO,
                         RTWO, -RTWO, -RTWO, RTWO, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, -1.0,
                         1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
                         1.0, -RTWO, RTWO, RTWO, -RTWO, -RTWO, RTWO, RTWO,
                         -RTWO, RTWO, -RTWO, -RTWO, RTWO, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0]])

        self.chardict['OhP_G2MINUS'] =\
            np.array(2*[[2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
                         1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
                         -RTWO, RTWO, RTWO, -RTWO, -RTWO, RTWO, RTWO, -RTWO,
                         RTWO, -RTWO, -RTWO, RTWO, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0, 1.0,
                         -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0,
                         -1.0, RTWO, -RTWO, -RTWO, RTWO, RTWO, -RTWO, -RTWO,
                         RTWO, -RTWO, RTWO, RTWO, -RTWO, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0]])

        self.chardict['OhP_HPLUS'] =\
            np.array(4*[[4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                         -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, -4.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0,
                         -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0]])

        self.chardict['OhP_HMINUS'] =\
            np.array(4*[[4.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                         -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 4.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0,
                         1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0]])

        self.bTdict = {}

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

        self.bTdict['Dic4_E2'] = np.array([[RTWO, 0., 0., -RTWO,
                                            0., 0., -RTWO, RTWO],
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

    def _get_double_cover_group(self, parity=1):
        OhP = self.OhP
        real_counts = []
        real_eigenvectors = []
        real_eigenvalues = []
        for group_element in OhP:
            if np.linalg.det(group_element) < 0.:
                group_element = -group_element
            eigenvalues, eigenvectors = np.linalg.eig(group_element)
            real_counter = 0
            for i in range(len(eigenvectors.T)):
                eigenvector = eigenvectors[:, i]
                eigenvalue = eigenvalues[i]
                assert np.sum(np.abs(group_element@eigenvector
                                     - eigenvalue*eigenvector)) < EPSILON10
                evec_imag_mag = np.abs(eigenvector.imag)
                evec_imag_magSQ = evec_imag_mag@evec_imag_mag
                evalue_imagSQ = np.abs(eigenvalue.imag)**2
                if (evec_imag_magSQ < EPSILON10 and evalue_imagSQ < EPSILON10):
                    if real_counter == 0:
                        real_eigenvectors.append(eigenvector.real)
                        real_eigenvalues.append(eigenvalue.real)
                    real_counter += 1
            real_counts.append(real_counter)
        assert len(real_eigenvalues) == len(OhP)
        assert len(real_eigenvectors) == len(OhP)

        xhat = np.array([1, 0, 0])
        zhat = np.array([0, 0, 1])
        theta_values = []
        for i in range(len(real_eigenvectors)):
            eigenvector = real_eigenvectors[i]
            along_zhat = (np.sum(np.abs(eigenvector - eigenvector[2]*zhat))
                          < EPSILON10)
            if along_zhat:
                normal_vector = xhat.reshape((1, 3))
            else:
                normal_vector = np.cross(eigenvector, zhat)
                normal_vector =\
                    (normal_vector
                     / np.linalg.norm(normal_vector)).reshape((1, 3))

            normal_vector_T = normal_vector.T
            group_element = OhP[i]
            if np.linalg.det(group_element) < 0.:
                group_element = -group_element
            costheta = (normal_vector@(group_element)@normal_vector_T)[0, 0]
            cross_product = np.cross(normal_vector,
                                     ((group_element)@normal_vector_T).T)
            sintheta = cross_product@eigenvector
            eigenvalue = real_eigenvalues[i]
            assert np.abs(eigenvalue-1.) < EPSILON10 or real_counts[i] == 3
            theta = np.nan
            if np.abs(costheta - (-1.)) < EPSILON10:
                theta = np.pi
            elif np.abs(costheta - (-.5)) < EPSILON10:
                if sintheta > 0.:
                    theta = 2*np.pi/3
                else:
                    theta = 4*np.pi/3
            elif np.abs(costheta - 0.) < EPSILON10:
                if sintheta > 0.:
                    theta = np.pi/2
                else:
                    theta = 3*np.pi/2
            elif np.abs(costheta - .5) < EPSILON10:
                if sintheta > 0.:
                    theta = np.pi/3
                else:
                    theta = 5*np.pi/3
            elif np.abs(costheta - 1.) < EPSILON10:
                theta = 0.
            theta_values.append(theta)

        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])

        # also the generators for spin 1
        spinone_sigma_x = 1j*np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]])
        spinone_sigma_y = 1j*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        spinone_sigma_z = 1j*np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])

        OhP_double = []
        OhP_double_intspin = []
        for i in range(len(real_eigenvectors)):
            eigenvector = real_eigenvectors[i]
            theta = theta_values[i]
            group_element = OhP[i]
            if np.linalg.det(group_element) < 0.:
                group_element = -group_element
            ############################
            real_counter = real_counts[i]
            if real_counter == 3:
                eigenvalues, eigenvectors = np.linalg.eig(group_element)
                location = np.where(np.abs(eigenvalues - 1.) < EPSILON10)[0][0]
                eigenvector = eigenvectors[:, location]
                theta = np.pi
                if np.sum(np.abs(group_element - np.eye(3))) < EPSILON10:
                    theta = 0.
            ############################
            sigma_dot_eigenvector = (sigma_x*eigenvector[0]
                                     + sigma_y*eigenvector[1]
                                     + sigma_z*eigenvector[2])
            sigma_spinone_dot_eigenvector = (spinone_sigma_x*eigenvector[0]
                                             + spinone_sigma_y*eigenvector[1]
                                             + spinone_sigma_z*eigenvector[2])
            parity_matrix = np.eye(2)
            spinone_parity_matrix = np.eye(3)
            element_parity = np.linalg.det(OhP[i])
            spinone_parity_matrix = element_parity*spinone_parity_matrix
            if parity == -1:
                parity_matrix = element_parity*parity_matrix
            U = np.cos(theta/2)*np.eye(2)\
                - 1j*np.sin(theta/2)*sigma_dot_eigenvector
            Uplustwopi = np.cos(theta/2+np.pi)*np.eye(2)\
                - 1j*np.sin(theta/2+np.pi)*sigma_dot_eigenvector
            U = parity_matrix@U
            Uspinone = expm(-1j*theta*sigma_spinone_dot_eigenvector)
            Uspinone = spinone_parity_matrix@Uspinone
            assert np.sum(np.abs(Uspinone - OhP[i])) < EPSILON10
            Uplustwopi = parity_matrix@Uplustwopi
            for j in range(len(U)):
                for k in range(len(U[j])):
                    if np.abs(U[j, k].imag) < EPSILON10:
                        U[j, k] = U[j, k].real+0.*1j
                    if np.abs(U[j, k]) < EPSILON10:
                        U[j, k] = 0.+0.*1j
            for j in range(len(Uplustwopi)):
                for k in range(len(Uplustwopi[j])):
                    if np.abs(Uplustwopi[j, k].imag) < EPSILON10:
                        Uplustwopi[j, k] = Uplustwopi[j, k].real+0.*1j
                    if np.abs(Uplustwopi[j, k]) < EPSILON10:
                        Uplustwopi[j, k] = 0.+0.*1j
            if parity == -1:
                group_element = OhP[i]
            OhP_double.append(U)
            OhP_double.append(Uplustwopi)
            OhP_double_intspin.append(group_element)
            OhP_double_intspin.append(group_element)
        assert len(OhP_double) == 2*len(OhP)
        assert len(OhP_double_intspin) == 2*len(OhP)
        return OhP_double, OhP_double_intspin

    def _get_G1_char(self, parity=1):
        if parity == -1:
            OhP_double = self.OhP_double_MINUS
        else:
            OhP_double = self.OhP_double_PLUS

        characters = []
        for i in range(len(OhP_double)):
            trace = np.trace(OhP_double[i])
            if np.abs(trace.imag) < EPSILON10:
                trace = trace.real
            if np.abs(trace) < EPSILON10:
                trace = 0.
            if np.abs(trace - 1.) < EPSILON10:
                trace = 1.
            if np.abs(trace + 1.) < EPSILON10:
                trace = -1.
            characters.append(trace)
        return np.array(2*[characters])

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
        return generate_wigner_d(ell, g_elem, real_harmonics)

    def generate_wigner_d_half(self, spin,
                               g_elem=[np.identity(3), np.identity(2)]):
        """Generate the Wigner D matrix."""
        return g_elem[1]

    def generate_induced_rep_kellm(self, nvec_arr=np.zeros((1, 3)),
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

    def generate_induced_rep_nonint_two_particles(
            self, nvecset_ab_batched=np.zeros((1, 2, 3)),
            first_spin=0.0, second_spin=0.0, g_elem=np.identity(3),
            particles_are_aa=False):
        """Generate the non-interacting induced representation matrix."""
        loc_inds = []
        ab_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(nvecset_ab_batched, 0, 2), 2, 0)
        for i in range(len(nvecset_ab_batched)):
            ab_arr_rot_entry = ab_arr_rot[i]
            loc_ind = np.where(
                np.all(nvecset_ab_batched
                       == ab_arr_rot_entry, axis=(1, 2))
                )[0]
            if particles_are_aa:
                ab_arr_rot_entry_swap =\
                    np.array([ab_arr_rot_entry[1],
                              ab_arr_rot_entry[0]])
                loc_ind = np.append(loc_ind, np.where(
                    np.all(nvecset_ab_batched
                           == ab_arr_rot_entry_swap, axis=(1, 2))
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

    def generate_induced_rep_nonint_three_scalars(
            self, aaa_arr=np.zeros((1, 3, 3)),
            abc_arr=np.zeros((1, 3, 3)), g_elem=np.identity(3),
            definite_iso=True):
        """Generate the non-interacting induced representation matrix."""
        loc_inds = []
        aaa_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(aaa_arr, 0, 2), 2, 0)
        for i in range(len(aaa_arr)):
            aaa_arr_rot_entry = aaa_arr_rot[i]
            loc_ind = []
            for pion_order in PION_ORDERS:
                loc_ind_tmp = np.where(
                    np.all(aaa_arr
                           == aaa_arr_rot_entry[pion_order],
                           axis=(1, 2))
                    )[0]
                loc_ind = loc_ind+list(loc_ind_tmp)
            loc_ind = np.unique(loc_ind)
            assert len(loc_ind) == 1
            loc_inds = loc_inds+[loc_ind[0]]

        if definite_iso:
            abc_arr_rot = np.moveaxis(
                g_elem@np.moveaxis(abc_arr, 0, 2), 2, 0)
            for i in range(len(abc_arr)):
                abc_arr_rot_entry = abc_arr_rot[i]
                loc_ind = np.where(
                    np.all(abc_arr
                           == abc_arr_rot_entry, axis=(1, 2))
                    )[0]
                assert len(loc_ind) == 1
                loc_inds = loc_inds+[loc_ind[0]+len(aaa_arr)]

        nonint_rot_matrix = [[]]
        for loc_ind in loc_inds:
            nonint_rot_row = np.zeros(len(loc_inds))
            nonint_rot_row[loc_ind] = 1.0
            nonint_rot_matrix = nonint_rot_matrix+[nonint_rot_row]
        nonint_rot_matrix = np.array(nonint_rot_matrix[1:])
        return nonint_rot_matrix

    def generate_induced_rep_nonint_three_scalars_labeled(
            self, nvecset_batched=np.zeros((1, 3, 3)),
            permutations=None, g_elem=np.identity(3)):
        """Generate a scalar three-particle induced representation."""
        if permutations is None:
            permutations = [[0, 1, 2]]
        loc_inds = []
        nvecset_rot = np.moveaxis(
            g_elem@np.moveaxis(nvecset_batched, 0, 2), 2, 0)
        for i in range(len(nvecset_batched)):
            nvecset_rot_entry = nvecset_rot[i]
            loc_ind = []
            for permutation in permutations:
                loc_ind_tmp = np.where(
                    np.all(
                        nvecset_batched
                        == nvecset_rot_entry[permutation],
                        axis=(1, 2),
                    )
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
        return nonint_rot_matrix

    def generate_induced_rep_nonint_three_particles_spin(
            self, aaa_arr=np.zeros((1, 3, 3)),
            abc_arr=np.zeros((1, 3, 3)),
            first_spin=0.0, second_spin=0.0, third_spin=0.0,
            g_elem=np.identity(3), definite_iso=True):
        """
        Generate the non-interacting induced representation matrix,
        including spin.
        """
        loc_inds = []
        if self.spin_half:
            [g_elem_intspin, g_elem_halfspin] = g_elem
            aaa_arr_rot = np.moveaxis(
                g_elem_intspin@np.moveaxis(aaa_arr, 0, 2), 2, 0)
        else:
            aaa_arr_rot = np.moveaxis(
                g_elem@np.moveaxis(aaa_arr, 0, 2), 2, 0)
        for i in range(len(aaa_arr)):
            aaa_arr_rot_entry = aaa_arr_rot[i]
            loc_ind = []
            for pion_order in PION_ORDERS:
                loc_ind_tmp = np.where(
                    np.all(aaa_arr
                           == aaa_arr_rot_entry[pion_order],
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
        if not self.spin_half:
            if np.abs(first_spin - int(first_spin)) > EPSILON10:
                raise ValueError("first_spin must be an integer")
            if np.abs(second_spin - int(second_spin)) > EPSILON10:
                raise ValueError("second_spin must be an integer")
            if np.abs(third_spin - int(third_spin)) > EPSILON10:
                raise ValueError("third_spin must be an integer")
            first_spin = int(first_spin)
            second_spin = int(second_spin)
            third_spin = int(third_spin)
            wig_d_first_spin = self.generate_wigner_d(
                first_spin, g_elem, real_harmonics=True).T
            wig_d_second_spin = self.generate_wigner_d(
                second_spin, g_elem, real_harmonics=True).T
            wig_d_third_spin = self.generate_wigner_d(
                third_spin, g_elem, real_harmonics=True).T
            induced_rep_tmp = np.kron(nonint_rot_matrix, wig_d_first_spin)
            induced_rep_tmp2 = np.kron(induced_rep_tmp, wig_d_second_spin)
            induced_rep = np.kron(induced_rep_tmp2, wig_d_third_spin)
        else:
            wig_d_first_spin = np.conjugate(self.generate_wigner_d_half(
                first_spin, g_elem).T)
            wig_d_second_spin = np.conjugate(self.generate_wigner_d_half(
                second_spin, g_elem).T)
            wig_d_third_spin = np.conjugate(self.generate_wigner_d_half(
                third_spin, g_elem).T)
            induced_rep_tmp = np.kron(nonint_rot_matrix, wig_d_first_spin)
            induced_rep_tmp2 = np.kron(induced_rep_tmp, wig_d_second_spin)
            induced_rep = np.kron(induced_rep_tmp2, wig_d_third_spin)
        return induced_rep

    def get_kellm_proj(self, nP=np.array([0, 0, 0]), irrep='A1PLUS',
                       irrep_row=0, nvec_arr=np.zeros((1, 3)),
                       ellm_set=[[0, 0]]):
        """Get a particular large projector."""
        if (nP == np.array([0, 0, 0])).all():
            group_str = 'OhP'
            group = self.OhP
            bT = self.bTdict[group_str+'_'+irrep][irrep_row]
        elif (nP == np.array([0, 0, 1])).all():
            group_str = 'Dic4'
            group = self.Dic4
            bT = self.bTdict[group_str+'_'+irrep][irrep_row]
        elif (nP == np.array([0, 1, 1])).all():
            group_str = 'Dic2'
            group = self.Dic2
            bT = self.bTdict[group_str+'_'+irrep][irrep_row]
        else:
            return ValueError("group not yet supported by get_large_proj")
        dim = len(nvec_arr)*len(ellm_set)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep_kellm(
                nvec_arr, ellm_set, g_elem)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_proj_nonint_three_scalars(
            self, nP=np.array([0, 0, 0]), irrep='A1PLUS', irow=0,
            aaa_arr=np.zeros((1, 3, 3)),
            abc_arr=np.zeros((1, 3, 3)),
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
            dim = len(aaa_arr)+len(abc_arr)
        else:
            dim = len(aaa_arr)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep_nonint_three_scalars(
                aaa_arr, abc_arr, g_elem, definite_iso)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_proj_nonint_three_scalars_labeled(
            self, nP=np.array([0, 0, 0]), irrep='A1PLUS', irow=0,
            nvecset_batched=np.zeros((1, 3, 3)), permutations=None):
        """Get a scalar three-particle projector for a particle label."""
        if permutations is None:
            permutations = [[0, 1, 2]]
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
        dim = len(nvecset_batched)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep =\
                self.generate_induced_rep_nonint_three_scalars_labeled(
                    nvecset_batched, permutations, g_elem)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_proj_nonint_three_particles_spin(
            self, nP=np.array([0, 0, 0]), irrep='A1PLUS', irow=0,
            aaa_arr=np.zeros((1, 3, 3)),
            abc_arr=np.zeros((1, 3, 3)),
            first_spin=0.0, second_spin=0.0, third_spin=0.0,
            definite_iso=True):
        """Get a particular large projector, including spin."""
        spin_half = self.spin_half
        if not spin_half:
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
        else:
            if (nP == np.array([0, 0, 0])).all():
                group_str = 'OhP'
                group_intspin = self.OhP_double_MINUS_intspin
                group_halfspin = self.OhP_double_PLUS
                bT = self.chardict[group_str+'_'+irrep][irow]
                if len(group_intspin) == 2*len(bT):
                    bT = np.repeat(bT, 2)
        if definite_iso:
            dim = len(aaa_arr)+len(abc_arr)
        else:
            dim = len(aaa_arr)
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0)
                                   * (2.0*third_spin+1.0))
        dim = dim*total_spin_dimension
        if not spin_half:
            proj = np.zeros((dim, dim))
        else:
            proj = np.zeros((dim, dim), dtype=complex)
        if not spin_half:
            for g_ind in range(len(group)):
                g_elem = group[g_ind]
                induced_rep = self.\
                    generate_induced_rep_nonint_three_particles_spin(
                        aaa_arr, abc_arr,
                        first_spin, second_spin, third_spin, g_elem,
                        definite_iso)
                proj = proj+induced_rep*bT[g_ind]
        else:
            for g_ind in range(len(group_intspin)):
                g_elem_intspin = group_intspin[g_ind]
                g_elem_halfspin = group_halfspin[g_ind]
                g_elem = [g_elem_intspin, g_elem_halfspin]
                induced_rep = self.\
                    generate_induced_rep_nonint_three_particles_spin(
                        aaa_arr, abc_arr,
                        first_spin, second_spin, third_spin, g_elem,
                        definite_iso)
                proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_proj_nonint_two_particles(
            self, nP=np.array([0, 0, 0]), irrep='A1PLUS', irow=0,
            nvecset_ab_batched=np.zeros((1, 2, 3)),
            first_spin=0.0, second_spin=0.0,
            particles_are_aa=False):
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
        dim = len(nvecset_ab_batched)*total_spin_dimension
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self\
                .generate_induced_rep_nonint_two_particles(
                    nvecset_ab_batched, first_spin, second_spin, g_elem,
                    particles_are_aa)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def _clean_projector(self, proj):
        eigvals, eigvecs = np.linalg.eig(proj)
        eigvecsT = eigvecs.T
        # numerical noise in the eigenvalues scales with the size of the
        # entries of proj, so the chop threshold must be relative
        eigval_chop_threshold = EPSILON8*max(1.0, np.max(np.abs(proj)))
        eigvals_chop = []
        for eigval in eigvals:
            if (np.abs(eigval.imag) < eigval_chop_threshold):
                eigval = eigval.real
            if isinstance(eigval, float)\
               and (np.abs(eigval) < eigval_chop_threshold):
                eigval = 0.0
            if (np.abs(eigval.real) < eigval_chop_threshold):
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

    def _get_summary(self, proj_dict, group_str, qcis, total_size):
        contributing_irreps = []
        irrep_index = 0
        irrep_row_index = 1
        for key in proj_dict:
            if key[irrep_row_index] == 0:
                contributing_irreps.append(key[irrep_index])

        best_irreps = []
        for irrep in contributing_irreps:
            irrep_dim = len(self.bTdict[group_str+'_'+irrep])
            irrep_row = irrep_dim-1
            best_irrep_row = 0
            while irrep_row >= 0:
                proj_tmp = proj_dict[(irrep, irrep_row)]
                if proj_tmp.dtype == np.float64:
                    best_irrep_row = irrep_row
                irrep_row -= 1
            best_irreps = best_irreps+[(irrep, best_irrep_row)]

        summary_str = f"kellm space has size {str(total_size)}\n\n"
        grand_total = 0
        for key in best_irreps:
            n_times = len(proj_dict[key].T)
            irrep_dim = len(self.bTdict[group_str+'_'+key[0]])
            total_covered = n_times*irrep_dim
            pad_offset = 7
            pad = (pad_offset-len(key[irrep_index]))*" "
            summary_str += (f"    {key[irrep_index]+pad} covers "
                            f"{n_times}x{irrep_dim}"
                            f" = {total_covered} slots\n")
            grand_total = grand_total+total_covered
        summary_str += f"\ntotal is {grand_total} \n"
        if grand_total == total_size:
            summary_str += "total matches size of kellm space"
        else:
            summary_str += ("does not match size of kellm space, "
                            "something went wrong")
        return best_irreps, summary_str

    def get_iso_projection(self, qcis=None, cindex=0, iso_index=0,
                           shell_index=0):
        """Get the iso-projector for non-interacting vectors."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        aaa_arr = qcis.nis.nvecset_aaa_batched[cindex][shell_index]
        abc_arr = qcis.nis.nvecset_abc_batched[cindex][shell_index]
        iso_projector = ISO_PROJECTORS[iso_index]
        iso_prepare_sets = []
        id_sub_len = len(aaa_arr)
        for ident_subset_index in range(id_sub_len):
            ident_subset_entry = aaa_arr[ident_subset_index]
            iso_prepare_entry = [ident_subset_index-id_sub_len]
            for pion_order in PION_ORDERS:
                loc_indices = np.where(
                    (abc_arr
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

    def get_proj_nonint_three_pions_shell(
            self, qcis=None, cindex=0, definite_iso=False, isovalue=None,
            shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        aaa_arr = qcis.nis.nvecset_aaa_batched[cindex][shell_index]
        abc_arr = qcis.nis.nvecset_abc_batched[cindex][shell_index]

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
                proj = self.get_proj_nonint_three_scalars(
                    nP, irrep, irow, aaa_arr, abc_arr,
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
                finalproj = self._clean_projector(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_proj_nonint_three_spinning_shell(
            self, qcis=None, cindex=0, definite_iso=False, isovalue=None,
            shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        spin_half = qcis.fvs.spin_half
        aaa_arr = qcis.nis.nvecset_aaa_batched[cindex][shell_index]
        abc_arr = qcis.nis.nvecset_abc_batched[cindex][shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError("momentum = ", nP, " is not yet supported")
        non_proj_dict = {}
        if not spin_half:
            if (nP@nP == 0):
                group_str = 'OhP'
            if (nP@nP == 1):
                group_str = 'Dic4'
            if (nP@nP == 2):
                group_str = 'Dic2'
        else:
            if (nP@nP == 0):
                group_str = 'OhP'

        first_spin = qcis.fcs.ni_list[cindex].spins[0]
        second_spin = qcis.fcs.ni_list[cindex].spins[1]
        third_spin = qcis.fcs.ni_list[cindex].spins[2]
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irow in range(len(self.chardict[group_str+'_'+irrep])):
                proj = self.get_proj_nonint_three_particles_spin(
                    nP, irrep, irow, aaa_arr, abc_arr,
                    first_spin, second_spin, third_spin, definite_iso)
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
                finalproj = self._clean_projector(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_proj_nonint_three_scalars_labeled_shell(
            self, qcis=None, cindex=0, shell_index=None):
        """Get scalar projectors for a three-particle label shell."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        particle_label = qcis.nis._nonint_channel_particle_label(cindex)
        nvecset_batched = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_batched'
        )[cindex][shell_index]
        permutations = self._three_particle_label_permutations(particle_label)

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
            for irow in range(len(self.chardict[group_str+'_'+irrep])):
                proj = self.get_proj_nonint_three_scalars_labeled(
                    nP, irrep, irow, nvecset_batched, permutations)
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
                finalproj = self._clean_projector(proj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    @staticmethod
    def _three_particle_label_permutations(particle_label):
        if particle_label == 'aaa':
            return PION_ORDERS
        if particle_label == 'aab':
            return [[0, 1, 2], [1, 0, 2]]
        if particle_label == 'abc':
            return [[0, 1, 2]]
        raise ValueError(f"unsupported three-particle label {particle_label}")

    def get_proj_nonint_two_particles_shell(
            self, qcis=None, cindex=0, definite_iso=False,
            isovalue=None, shell_index=None):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.nP
        irrep_set = qcis.fvs.irrep_set
        particle_label = qcis.nis._nonint_channel_particle_label(cindex)
        particles_are_aa = (particle_label == 'aa')
        nvecset_ab_batched = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_batched'
        )[cindex][shell_index]

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
                proj = self.get_proj_nonint_two_particles(
                    nP, irrep, irow, nvecset_ab_batched,
                    first_spin, second_spin, particles_are_aa)
                some_zero_vec = False
                for batch in nvecset_ab_batched:
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
                finalproj = self._clean_projector(isorotproj)
                if len(finalproj) != 0:
                    non_proj_dict[(irrep, irow)] = finalproj
                for keytmp in non_proj_dict:
                    proj_tmp = non_proj_dict[keytmp]
                    if (proj_tmp.imag == np.zeros(proj_tmp.shape)).all():
                        non_proj_dict[keytmp] = proj_tmp.real
        return non_proj_dict

    def get_proj_nonint_three_pions_dict(self, qcis=None, nic_index=0):
        """Get it."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        if qcis.nis._nonint_channel_particle_label(nic_index) != 'aaa':
            return self.get_proj_nonint_three_scalars_labeled_dict(
                qcis=qcis, nic_index=nic_index)
        master_dict = {}
        definite_iso = qcis.fcs.ni_list[nic_index].isospin_channel
        row_zero_value = 0
        summary_str = ""
        nshells = len(qcis.nis.nvecset_aaa_reps[nic_index])
        for shell_index in range(nshells):
            shell_total = 0
            if definite_iso:
                nident =\
                    len(qcis.nis.nvecset_aaa_batched[nic_index][shell_index])
                nrest =\
                    len(qcis.nis.nvecset_abc_batched[nic_index][shell_index])
                nstates = nident+nrest
            else:
                nstates =\
                    len(qcis.nis.nvecset_aaa_batched[nic_index][shell_index])
            summary_str +=\
                f"shell_index = {shell_index} ({nstates} states):\n"
            rep_mom = str(qcis.nis.nvecset_aaa_reps[nic_index][shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            if definite_iso:
                isoset = range(4)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict =\
                    self.get_proj_nonint_three_pions_shell(
                        qcis, nic_index, definite_iso, isovalue, shell_index)
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

    def get_proj_nonint_three_scalars_labeled_dict(
            self, qcis=None, nic_index=0):
        """Get scalar three-particle projectors for non-aaa labels."""
        master_dict = {}
        if qcis is None:
            raise ValueError("qcis cannot be None")
        particle_label = qcis.nis._nonint_channel_particle_label(nic_index)
        nvecset_reps_cindex = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_reps'
        )[nic_index]
        nvecset_batched_cindex = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_batched'
        )[nic_index]
        row_zero_value = 0
        summary_str = ""
        nshells = len(nvecset_reps_cindex)
        if qcis.fcs.ni_list[nic_index].isospin_channel:
            isoset = [int(qcis.fcs.ni_list[nic_index].isospin)]
        else:
            isoset = range(1)
        for shell_index in range(nshells):
            shell_total = 0
            nstates = len(nvecset_batched_cindex[shell_index])
            summary_str +=\
                f"shell_index = {shell_index} ({nstates} states):\n"
            rep_mom = str(nvecset_reps_cindex[shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            for isovalue in isoset:
                non_proj_dict =\
                    self.get_proj_nonint_three_scalars_labeled_shell(
                        qcis, nic_index, shell_index)
                master_dict[(shell_index, isovalue)] = non_proj_dict
                if len(non_proj_dict) == 0:
                    summary_str += "    Channel does not contain this shell\n"
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
                    if row == 0:
                        if n_embedded == 1:
                            s = ''
                        else:
                            s = 's'
                        shell_covered = shell_total+n_embedded*(dim-1)
                        summary_str +=\
                            (f"       {irrep} "
                             f"(appears {n_embedded} time{s}), "
                             f"covered {shell_covered}/{nstates}\n")
            assert shell_total == nstates
        summary_str = summary_str[:-1]
        master_dict['summary'] = summary_str
        return master_dict

    def get_proj_nonint_three_spinning_dict(self, qcis=None, nic_index=0):
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
        nshells = len(qcis.nis.nvecset_aaa_reps[nic_index])
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
                    len(qcis.nis.nvecset_aaa_batched[nic_index][shell_index])\
                    * total_spin_dimension
                nrest =\
                    len(qcis.nis.nvecset_abc_batched[nic_index][shell_index])\
                    * total_spin_dimension
                nstates = nident+nrest
            else:
                nstates =\
                    len(qcis.nis.nvecset_aaa_batched[nic_index][shell_index])\
                    * total_spin_dimension
            summary_str +=\
                f"shell_index = {shell_index} ({nstates} states):\n"
            rep_mom = str(qcis.nis.nvecset_aaa_reps[nic_index][shell_index])
            rep_mom = rep_mom.replace(' [', (' '*30)+'[')
            summary_str += "    representative momenta = "+rep_mom+"\n"
            if definite_iso:
                isoset = range(4)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict =\
                    self.get_proj_nonint_three_spinning_shell(
                        qcis, nic_index, definite_iso, isovalue, shell_index)
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
                    if irrep[0] == 'G':
                        dim = 2
                    if irrep[0] == 'H':
                        dim = 4
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

    def get_proj_nonint_two_particles_dict(self, qcis=None, nic_index=0,
                                           isospin_channel=True):
        """Get it."""
        master_dict = {}
        if qcis is None:
            raise ValueError("qcis cannot be None")
        row_zero_value = 0
        summary_str = ""
        particle_label = qcis.nis._nonint_channel_particle_label(nic_index)
        nvecset_batched_cindex = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_batched'
        )[nic_index]
        nvecset_reps_cindex = getattr(
            qcis.nis,
            f'nvecset_{particle_label}_reps'
        )[nic_index]
        nshells = len(nvecset_reps_cindex)
        first_spin = qcis.fcs.ni_list[nic_index].spins[0]
        second_spin = qcis.fcs.ni_list[nic_index].spins[1]
        total_spin_dimension = int((2.0*first_spin+1.0)*(2.0*second_spin+1.0))
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
                    .get_proj_nonint_two_particles_shell(
                        qcis, nic_index, isospin_channel, isovalue,
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

    def get_fixed_sc_proj_dict(self, qcis=None, sc_index=0):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis.verbosity >= 2:
            print("getting the dict for channel =", sc_index)
        if qcis is None:
            raise ValueError("qcis cannot be None")
        nP = qcis.fvs.nP
        irrep_set = qcis.fvs.irrep_set
        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2) and (nP@nP != 4):
            raise ValueError("momentum = ", nP, " is not yet supported")
        proj_dict = {}
        if (nP@nP == 0):
            group_str = 'OhP'
        elif (nP@nP == 1):
            group_str = 'Dic4'
        elif (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(irrep_set)):
            irrep = irrep_set[i]
            for irrep_row in range(len(self.bTdict[group_str+'_'+irrep])):
                three_slice_index = qcis.sc_to_three_slice[sc_index]
                shell_index = 0
                nvec_arr = qcis.tbks_list[three_slice_index][
                    shell_index].nvec_arr
                ellm_set = qcis.ellm_sets[sc_index]
                kellm_proj = self.get_kellm_proj(nP=nP, irrep=irrep,
                                                 irrep_row=irrep_row,
                                                 nvec_arr=nvec_arr,
                                                 ellm_set=ellm_set)

                clean_kellm_proj = self._clean_projector(kellm_proj)
                if len(clean_kellm_proj) != 0:
                    proj_dict[(irrep, irrep_row)] = clean_kellm_proj
        for key in proj_dict:
            kellm_proj = proj_dict[key]
            if (kellm_proj.imag == np.zeros(kellm_proj.shape)).all():
                proj_dict[key] = kellm_proj.real

        kellm_shell_index = 0
        total_size = len(qcis.kellm_spaces[sc_index][kellm_shell_index])
        proj_dict['best_irreps'], proj_dict['summary']\
            = self._get_summary(proj_dict, group_str, qcis, total_size)

        return proj_dict

    def get_fixed_sc_and_shell_proj_dict(self, qcis=None, sc_index=0,
                                         kellm_shell=None,
                                         kellm_shell_index=None):
        """Get the dictionary of small projectors for one kellm_shell."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        if kellm_shell is None:
            raise ValueError("kellm_shell cannot be None")
        nP = qcis.fvs.nP
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
            for irrep_row in range(len(self.bTdict[group_str+'_'+irrep])):
                if kellm_shell_index is None:
                    shell_index = 0
                else:
                    shell_index = kellm_shell_index
                three_slice_index = qcis.sc_to_three_slice[sc_index]
                nvec_arr = qcis.tbks_list[three_slice_index][
                    shell_index].nvec_arr
                ellm_set = qcis.ellm_sets[sc_index]
                nshell = [kellm_shell[0]//len(ellm_set),
                          kellm_shell[1]//len(ellm_set)]
                assert np.abs(kellm_shell[0]//len(ellm_set)
                              - kellm_shell[0]/len(ellm_set)) < EPSILON8
                assert np.abs(kellm_shell[1]//len(ellm_set)
                              - kellm_shell[1]/len(ellm_set)) < EPSILON8
                nvec_arr = qcis.tbks_list[three_slice_index][shell_index]\
                    .nvec_arr[nshell[0]:nshell[1]]
                ellm_set = qcis.ellm_sets[sc_index]
                kellm_proj = self.get_kellm_proj(nP=nP, irrep=irrep,
                                                 irrep_row=irrep_row,
                                                 nvec_arr=nvec_arr,
                                                 ellm_set=ellm_set)

                clean_kellm_proj = self._clean_projector(kellm_proj)
                if len(clean_kellm_proj) != 0:
                    proj_dict[(irrep, irrep_row)] = clean_kellm_proj
        for key in proj_dict:
            kellm_proj = proj_dict[key]
            if (kellm_proj.imag == np.zeros(kellm_proj.shape)).all():
                proj_dict[key] = kellm_proj.real
        return proj_dict

    def get_full_proj_dict(self, qcis=None):
        """Get the dictionary of small projectors for the entire qcis."""
        if qcis is None:
            raise ValueError("qcis cannot be None")
        proj_dict = {}
        nP = qcis.fvs.nP
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(qcis.fvs.irrep_set)):
            irrep = qcis.fvs.irrep_set[i]
            for irrep_row in range(len(self.bTdict[group_str+'_'+irrep])):
                proj_list = []
                for sc_index in range(qcis.n_channels):
                    three_slice_index = qcis.sc_to_three_slice[sc_index]
                    shell_index = 0
                    nvec_arr = qcis.tbks_list[three_slice_index][
                        shell_index].nvec_arr
                    ellm_set = qcis.ellm_sets[sc_index]
                    kellm_proj = self.get_kellm_proj(nP=nP, irrep=irrep,
                                                     irrep_row=irrep_row,
                                                     nvec_arr=nvec_arr,
                                                     ellm_set=ellm_set)
                    proj_list = proj_list+[kellm_proj]
                proj = block_diag(*proj_list)
                clean_proj = self._clean_projector(proj)
                if len(clean_proj) != 0:
                    proj_dict[(irrep, irrep_row)] = clean_proj
        for key in proj_dict:
            proj = proj_dict[key]
            if (proj.imag == np.zeros(proj.shape)).all():
                proj_dict[key] = proj.real

        total_size = 0
        kellm_shell_index = 0
        for sc_index in range(qcis.n_channels):
            total_size += len(qcis.kellm_spaces[sc_index][
                kellm_shell_index])
        proj_dict['best_irreps'], proj_dict['summary']\
            = self._get_summary(proj_dict, group_str, qcis, total_size)

        return proj_dict
