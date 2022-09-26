#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# group_theory.py
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

ROOT_THREE = np.sqrt(3.0)
ROOT_TWO = np.sqrt(2.0)
EPSPROJ = 1.0e-9


class Irreps:
    """Class collecting data for finite-volume irreducible representations."""

    def __init__(self, nP=np.array([0, 0, 0])):
        self._nP = nP
        if (self._nP == np.array([0, 0, 0])).all():
            self.A1PLUS = 'A1PLUS'
            self.A2PLUS = 'A2PLUS'
            self.T1PLUS = 'T1PLUS'
            self.T2PLUS = 'T2PLUS'
            self.EPLUS = 'EPLUS'
            self.A1MINUS = 'A1MINUS'
            self.A2MINUS = 'A2MINUS'
            self.T1MINUS = 'T1MINUS'
            self.T2MINUS = 'T2MINUS'
            self.EMINUS = 'EMINUS'
            self.set = [self.A1PLUS, self.A2PLUS, self.EPLUS, self.T1PLUS,
                        self.T2PLUS, self.A1MINUS, self.A2MINUS, self.EMINUS,
                        self.T1MINUS, self.T2MINUS]
        elif (self._nP == np.array([0, 0, 1])).all():
            self.A1 = 'A1'
            self.A2 = 'A2'
            self.B1 = 'B1'
            self.B2 = 'B2'
            self.E = 'E2'
            self.set = [self.A1, self.A2, self.B1, self.B2, self.E]
        elif (self._nP == np.array([0, 1, 1])).all():
            self.A1 = 'A1'
            self.A2 = 'A2'
            self.B1 = 'B1'
            self.B2 = 'B2'
            self.set = [self.A1, self.A2, self.B1, self.B2]
        else:
            raise ValueError('unsupported value of nP in irreps: '
                             + str(self._nP))

    def __str__(self):
        """Summary of the irrep set."""
        strtmp = ''
        for enttmp in self.set:
            strtmp = strtmp+enttmp+', '
        return strtmp[:-2]


class Groups:
    """Class for finite-volume group-theory relevant for three particles."""

    def __init__(self, ell_max):
        self.wigner = spherical.Wigner(ell_max)

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

        self.bTdict['OhP_A1PLUS'] = np.array([[1.]*48])

        self.bTdict['OhP_A2PLUS'] = np.array([[1., -1., -1., -1., -1.,
                                               -1., -1., 1., 1., 1., 1.,
                                               1., 1., 1., 1., -1., -1.,
                                               -1., -1., -1., -1., 1.,
                                               1., 1., 1., -1., -1., -1.,
                                               -1., -1., -1., 1., 1., 1.,
                                               1., 1., 1., 1., 1., -1.,
                                               -1., -1., -1., -1., -1.,
                                               1., 1., 1.]])

        self.bTdict['OhP_EPLUS'] = np.array([[1., 1., 1., 1., 1., -2., -2.,
                                              1., -2., -2., 1., -2., 1., 1.,
                                              -2., 1., 1., -2., -2., 1., 1.,
                                              1., 1., 1., 1., 1., 1., 1., 1.,
                                              -2., -2., 1., -2., -2., 1., -2.,
                                              1., 1., -2., 1., 1., -2., -2.,
                                              1., 1., 1., 1., 1.],
                                             [-ROOT_THREE, ROOT_THREE,
                                              ROOT_THREE, -ROOT_THREE,
                                              -ROOT_THREE, 0., 0., ROOT_THREE,
                                              0., 0., ROOT_THREE, 0.,
                                              ROOT_THREE, ROOT_THREE, 0.,
                                              ROOT_THREE, ROOT_THREE, 0., 0.,
                                              -ROOT_THREE, -ROOT_THREE,
                                              -ROOT_THREE, -ROOT_THREE,
                                              -ROOT_THREE, -ROOT_THREE,
                                              ROOT_THREE, ROOT_THREE,
                                              -ROOT_THREE, -ROOT_THREE, 0., 0.,
                                              ROOT_THREE, 0., 0., ROOT_THREE,
                                              0., ROOT_THREE, ROOT_THREE, 0.,
                                              ROOT_THREE, ROOT_THREE, 0., 0.,
                                              -ROOT_THREE, -ROOT_THREE,
                                              -ROOT_THREE, -ROOT_THREE,
                                              -ROOT_THREE]])

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
                                              [ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO],
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
                                              [ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                               -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                               ROOT_TWO, -ROOT_TWO, -ROOT_TWO],
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

        self.bTdict['OhP_A1MINUS'] = np.array([[1.0]*24+[-1.0]*24])

        self.bTdict['OhP_A2MINUS'] = np.array([[1., -1., -1., -1., -1., -1.,
                                                -1., 1., 1., 1., 1., 1., 1.,
                                                1., 1., -1., -1., -1., -1.,
                                                -1., -1., 1., 1., 1., -1., 1.,
                                                1., 1., 1., 1., 1., -1., -1.,
                                                -1., -1., -1., -1., -1., -1.,
                                                1., 1., 1., 1., 1., 1., -1.,
                                                -1., -1.]])

        self.bTdict['OhP_EMINUS'] = np.array([[1., 1., 1., 1., 1., -2., -2.,
                                               1., -2., -2., 1., -2., 1., 1.,
                                               -2., 1., 1., -2., -2., 1., 1.,
                                               1., 1., 1., -1., -1., -1., -1.,
                                               -1., 2., 2., -1., 2., 2., -1.,
                                               2., -1., -1., 2., -1., -1., 2.,
                                               2., -1., -1., -1., -1., -1.],
                                              [-ROOT_THREE, ROOT_THREE,
                                               ROOT_THREE, -ROOT_THREE,
                                               -ROOT_THREE, 0., 0., ROOT_THREE,
                                               0., 0., ROOT_THREE, 0.,
                                               ROOT_THREE, ROOT_THREE, 0.,
                                               ROOT_THREE, ROOT_THREE, 0., 0.,
                                               -ROOT_THREE, -ROOT_THREE,
                                               -ROOT_THREE, -ROOT_THREE,
                                               -ROOT_THREE, ROOT_THREE,
                                               -ROOT_THREE, -ROOT_THREE,
                                               ROOT_THREE, ROOT_THREE, 0., 0.,
                                               -ROOT_THREE, 0., 0.,
                                               -ROOT_THREE, 0., -ROOT_THREE,
                                               -ROOT_THREE, 0., -ROOT_THREE,
                                               -ROOT_THREE, 0., 0., ROOT_THREE,
                                               ROOT_THREE, ROOT_THREE,
                                               ROOT_THREE, ROOT_THREE]])

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
                                               [ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                                -ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, ROOT_TWO,
                                                -ROOT_TWO, -ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                                ROOT_TWO],
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

        self.bTdict['OhP_T2MINUS'] = np.array([[ROOT_TWO, -ROOT_TWO, ROOT_TWO,
                                                ROOT_TWO, 0.-ROOT_TWO*1j,
                                                ROOT_TWO, 0.+ROOT_TWO*1j,
                                                -ROOT_TWO, 0.+ROOT_TWO*1j,
                                                -ROOT_TWO, 0.-ROOT_TWO*1j,
                                                0.-ROOT_TWO*1j, 0.+ROOT_TWO*1j,
                                                ROOT_TWO, ROOT_TWO,
                                                0.+ROOT_TWO*1j, 0.-ROOT_TWO*1j,
                                                -ROOT_TWO, 0.-ROOT_TWO*1j,
                                                -ROOT_TWO, 0.+ROOT_TWO*1j,
                                                -ROOT_TWO, 0.+ROOT_TWO*1j,
                                                0.-ROOT_TWO*1j, -ROOT_TWO,
                                                ROOT_TWO, -ROOT_TWO, -ROOT_TWO,
                                                0.+ROOT_TWO*1j, -ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                0.+ROOT_TWO*1j, 0.+ROOT_TWO*1j,
                                                0.-ROOT_TWO*1j, -ROOT_TWO,
                                                -ROOT_TWO, 0.-ROOT_TWO*1j,
                                                0.+ROOT_TWO*1j, ROOT_TWO,
                                                0.+ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j,
                                                0.+ROOT_TWO*1j],
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
                                               [0.-ROOT_TWO*1j, 0.+ROOT_TWO*1j,
                                                0.-ROOT_TWO*1j, 0.-ROOT_TWO*1j,
                                                ROOT_TWO, 0.-ROOT_TWO*1j,
                                                -ROOT_TWO, 0.+ROOT_TWO*1j,
                                                -ROOT_TWO, 0.+ROOT_TWO*1j,
                                                ROOT_TWO, ROOT_TWO, -ROOT_TWO,
                                                0.-ROOT_TWO*1j, 0.-ROOT_TWO*1j,
                                                -ROOT_TWO, ROOT_TWO,
                                                0.+ROOT_TWO*1j, ROOT_TWO,
                                                0.+ROOT_TWO*1j, -ROOT_TWO,
                                                0.+ROOT_TWO*1j, -ROOT_TWO,
                                                ROOT_TWO, 0.+ROOT_TWO*1j,
                                                0.-ROOT_TWO*1j, 0.+ROOT_TWO*1j,
                                                0.+ROOT_TWO*1j, -ROOT_TWO,
                                                0.+ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j, -ROOT_TWO,
                                                -ROOT_TWO, ROOT_TWO,
                                                0.+ROOT_TWO*1j, 0.+ROOT_TWO*1j,
                                                ROOT_TWO, -ROOT_TWO,
                                                0.-ROOT_TWO*1j, -ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                0.-ROOT_TWO*1j, ROOT_TWO,
                                                -ROOT_TWO]])

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

        self.bTdict['Dic4_A2'] = np.array([[1.]*4+[-1.]*4])

        self.bTdict['Dic4_B1'] = np.array([[1., -1., -1., 1.,
                                            -1., -1., 1., 1.]])

        self.bTdict['Dic4_B2'] = np.array([[1., -1., -1., 1.,
                                            1., 1., -1., -1.]])

        self.bTdict['Dic4_E2'] = np.array([[0., -1j*ROOT_TWO, 1j*ROOT_TWO, 0.,
                                            -1j*ROOT_TWO, 1j*ROOT_TWO, 0., 0.],
                                           [ROOT_TWO, 0., 0., -ROOT_TWO,
                                            0., 0., ROOT_TWO, -ROOT_TWO]])

        self.Dic2 = np.array(
            [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
             [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
             [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]])

        self.bTdict['Dic2_A1'] = np.array([[1.]*4])

        self.bTdict['Dic2_A2'] = np.array([[1.]*2+[-1.]*2])

        self.bTdict['Dic2_B1'] = np.array([[1., -1., -1., 1.]])

        self.bTdict['Dic2_B2'] = np.array([[1., -1., 1., -1.]])

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
            if not (np.abs(wig_d.imag) < EPSPROJ).all():
                raise ValueError('real Wigner-D is complex')
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
            return ValueError('group not yet supported by get_large_proj')
        dim = len(nvec_arr)*len(ellm_set)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep(nvec_arr, ellm_set,
                                                    g_elem)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def _get_final_proj(self, proj):
        eigvals, eigvecs = np.linalg.eig(proj)
        eigvecsT = eigvecs.T
        eigvals_chop = []
        for eigval in eigvals:
            if (np.abs(eigval.imag) < EPSPROJ):
                eigval = eigval.real
            if isinstance(eigval, float)\
               and (np.abs(eigval) < EPSPROJ):
                eigval = 0.0
            if (np.abs(eigval.real) < EPSPROJ):
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
                if (np.abs(eigvecT_entry.imag) < EPSPROJ):
                    eigvecT_entry = eigvecT_entry.real
                if isinstance(eigvecT_entry, float)\
                   and (np.abs(eigvecT_entry) < EPSPROJ):
                    eigvecT_entry = 0.0
                if (np.abs(eigvecT_entry.real) < EPSPROJ):
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
                if (np.abs(eigvecT_entry.imag) < EPSPROJ):
                    eigvecT_entry = eigvecT_entry.real
                if isinstance(eigvecT_entry, float)\
                   and (np.abs(eigvecT_entry) < EPSPROJ):
                    eigvecT_entry = 0.0
                if (np.abs(eigvecT_entry.real) < EPSPROJ):
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

        summarystr = ""
        summarystr = summarystr+"kellm space has size "\
            + str(totalsize)+"\n\n"
        total = 0
        for key in best_irreps:
            n_times = len(proj_dict[key].T)
            n_dim = len(self.bTdict[group_str+'_'+key[0]])
            n_tot = n_times*n_dim

            summarystr = summarystr+"    "+key[0]+((7-len(key[0]))*" ")\
                + " covers " + str(n_times)+'x'+str(n_dim)+' = '\
                + str(n_tot)+" slots\n"
            total = total+n_tot
        summarystr = summarystr+"\ntotal is "+str(total)+" \n"
        if total == totalsize:
            summarystr = summarystr+"total matches size of kellm space"
        else:
            summarystr = summarystr+"does not match size of kellm space, "\
                + "something went wrong"
        return best_irreps, summarystr

    def get_channel_proj_dict(self, qcis=None, cindex=0):
        """Get the dictionary of small projectors for a given qcis."""
        if qcis.verbosity >= 2:
            print("getting the dict for channel =", cindex)
        if qcis is None:
            raise ValueError('qcis cannot be None')
        nP = qcis.nP
        irrep_set = Irreps(nP=nP).set
        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError('momentum = ', nP, ' is not yet supported')
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
                slice_index = 0
                for three_slice in qcis.fcs.three_slices:
                    if cindex > three_slice[1]:
                        slice_index = slice_index+1
                nvec_arr = qcis.tbks_list[slice_index][0].nvec_arr
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

        totalsize = len(qcis.kellm_spaces[cindex][0])
        proj_dict['best_irreps'], proj_dict['summary']\
            = self._get_summary(proj_dict, group_str, qcis, totalsize)

        return proj_dict

    def get_slice_proj_dict(self, qcis=None, cindex=0, kellm_slice=None):
        """Get the dictionary of small projectors for one kellm_slice."""
        if qcis is None:
            raise ValueError('qcis cannot be None')
        if kellm_slice is None:
            raise ValueError('kellm_slice cannot be None')
        nP = qcis.nP
        irrep_set = Irreps(nP=nP).set
        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError('momentum = ', nP, ' is not yet supported')
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
                slice_index = 0
                for three_slice in qcis.fcs.three_slices:
                    if cindex > three_slice[1]:
                        slice_index = slice_index+1
                nslice = [int(kellm_slice[0]/len(qcis.ellm_sets[cindex])),
                          int(kellm_slice[1]/len(qcis.ellm_sets[cindex]))]
                nvec_arr = qcis.tbks_list[slice_index][0].nvec_arr[
                    nslice[0]:nslice[1]]
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
            raise ValueError('qcis cannot be None')
        proj_dict = {}
        nP = qcis.nP
        if (nP@nP == 0):
            group_str = 'OhP'
        if (nP@nP == 1):
            group_str = 'Dic4'
        if (nP@nP == 2):
            group_str = 'Dic2'
        for i in range(len(qcis.fvs.irreps.set)):
            irrep = qcis.fvs.irreps.set[i]
            for irow in range(len(self.bTdict[group_str+'_'+irrep])):
                proj_list = []
                for cindex in range(qcis.n_channels):
                    slice_index = 0
                    for three_slice in qcis.fcs.three_slices:
                        if cindex > three_slice[1]:
                            slice_index = slice_index+1
                    nvec_arr = qcis.tbks_list[slice_index][0].nvec_arr
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
