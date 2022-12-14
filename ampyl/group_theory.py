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
EPSPROJ = 1.0e-8
PION_ORDERS = [[0, 1, 2],
               [1, 0, 2],
               [0, 2, 1],
               [1, 2, 0],
               [2, 0, 1],
               [2, 1, 0]]

ISO_PROJECTOR_ZERO = np.diag([0.]*6+[1.])
ISO_PROJECTOR_ONE = np.diag([0.]*3+[1.]*3+[0.])
ISO_PROJECTOR_TWO = np.diag([0.]+[1.]*2+[0.]*4)
ISO_PROJECTOR_THREE = np.diag([1.]+[0.]*6)

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

        self.chardict['OhP_T1MINUS'] = np.array(3*[[3., -1., -1., -1., -1.,
                                                    -1., -1., 0., 0., 0., 0.,
                                                    0., 0., 0., 0., 1., 1., 1.,
                                                    1., 1., 1., -1., -1., -1.,
                                                    -3., 1., 1., 1., 1., 1.,
                                                    1., 0., 0., 0., 0., 0., 0.,
                                                    0., 0., -1., -1., -1., -1.,
                                                    -1., -1., 1., 1., 1.]])

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

    def generate_induced_rep_noninttwo(self,
                                       nonidentical_arr=np.zeros((1, 2, 3)),
                                       g_elem=np.identity(3)):
        """Generate the non-interacting induced representation matrix."""
        loc_inds = []
        nonidentical_arr_rot = np.moveaxis(
            g_elem@np.moveaxis(nonidentical_arr, 0, 2), 2, 0)
        for i in range(len(nonidentical_arr)):
            nonidentical_arr_rot_entry = nonidentical_arr_rot[i]
            loc_ind = np.where(
                np.all(nonidentical_arr
                       == nonidentical_arr_rot_entry, axis=(1, 2))
                )[0]
            assert len(loc_ind) == 1
            loc_inds = loc_inds+[loc_ind[0]]

        nonint_rot_matrix = [[]]
        for loc_ind in loc_inds:
            nonint_rot_row = np.zeros(len(loc_inds))
            nonint_rot_row[loc_ind] = 1.0
            nonint_rot_matrix = nonint_rot_matrix+[nonint_rot_row]
        nonint_rot_matrix = np.array(nonint_rot_matrix[1:])

        wig_d_ell_set = []
        ell_set = [1]
        for ell in ell_set:
            wig_d_ell_set = wig_d_ell_set\
                + [self.generate_wigner_d(ell, g_elem,
                                          real_harmonics=True).T]
        wig_d_block = block_diag(*wig_d_ell_set)
        induced_rep = np.kron(nonint_rot_matrix, wig_d_block)
        return induced_rep

    def generate_induced_rep_nonint(self, identical_arr=np.zeros((1, 3, 3)),
                                    nonidentical_arr=np.zeros((1, 3, 3)),
                                    g_elem=np.identity(3)):
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

    def get_large_proj_nonint(self, nP=np.array([0, 0, 0]), irrep='A1PLUS',
                              irow=0,
                              identical_arr=np.zeros((1, 3, 3)),
                              nonidentical_arr=np.zeros((1, 3, 3))):
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
            return ValueError('group not yet supported by get_large_proj')
        dim = len(identical_arr)+len(nonidentical_arr)
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep_nonint(identical_arr,
                                                           nonidentical_arr,
                                                           g_elem)
            proj = proj+induced_rep*bT[g_ind]
        return proj

    def get_large_proj_nonint_two(self, nP=np.array([0, 0, 0]), irrep='A1PLUS',
                                  irow=0,
                                  nonidentical_arr=np.zeros((1, 2, 3))):
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
            return ValueError('group not yet supported by get_large_proj')
        dim = len(nonidentical_arr)*3
        proj = np.zeros((dim, dim))
        for g_ind in range(len(group)):
            g_elem = group[g_ind]
            induced_rep = self.generate_induced_rep_noninttwo(nonidentical_arr,
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

    def get_iso_projection(self, qcis=None, iso_index=0, shell_index=0):
        """Get the iso-projector for non-interacting vectors."""
        if qcis is None:
            raise ValueError('qcis cannot be None')
        identical_arr = qcis.n1n2n3_ident_batched[shell_index]
        nonidentical_arr = qcis.n1n2n3_batched[shell_index]
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
            raise ValueError('qcis cannot be None')
        nP = qcis.nP
        irrep_set = Irreps(nP=nP).set
        identical_arr = qcis.n1n2n3_ident_batched[shell_index]
        nonidentical_arr = qcis.n1n2n3_batched[shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError('momentum = ', nP, ' is not yet supported')
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
                slice_index = 0
                for three_slice in qcis.fcs.three_slices:
                    if cindex > three_slice[1]:
                        slice_index = slice_index+1
                proj = self.get_large_proj_nonint(nP, irrep, irow,
                                                  identical_arr,
                                                  nonidentical_arr)
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
                    isoproj = self.get_iso_projection(qcis, isovalue,
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
            raise ValueError('qcis cannot be None')
        nP = qcis.nP
        irrep_set = Irreps(nP=nP).set
        nonidentical_arr = qcis.n1n2_batched[shell_index]

        if (nP@nP != 0) and (nP@nP != 1) and (nP@nP != 2):
            raise ValueError('momentum = ', nP, ' is not yet supported')
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
                proj = self.get_large_proj_nonint_two(nP, irrep, irow,
                                                      nonidentical_arr)
                some_zero_vec = False
                for nonidentical_entry in nonidentical_arr:
                    for single_vec in nonidentical_entry:
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
                        if np.abs(projT[i]@projT[i]) < 1.0e-10:
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
                    # isoproj = self.get_iso_projection(qcis, isovalue,
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

    def get_nonint_proj_dict(self, qcis=None, cindex=0, definite_iso=True):
        """Get it."""
        master_dict = {}
        if qcis is None:
            raise ValueError('qcis cannot be None')
        row_zero_value = 0
        summary_string = ''
        nshells = len(qcis.n1n2n3_ident_reps)
        for shell_index in range(nshells):
            shell_total = 0
            nstates = len(qcis.n1n2n3_ident_batched[shell_index])\
                + len(qcis.n1n2n3_batched[shell_index])
            summary_string = summary_string\
                + f'shell_index = {shell_index} ({nstates} states):\n'
            summary_string = summary_string\
                + f'representative momenta = \n{qcis.n1n2n3_ident_reps[shell_index]}\n'
            if definite_iso:
                isoset = range(4)
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict = self.get_nonint_proj_dict_shell(qcis, 0,
                                                                definite_iso,
                                                                isovalue,
                                                                shell_index)
                master_dict[(shell_index, isovalue)] = non_proj_dict
                iso_shell_total = 0
                if len(non_proj_dict) == 0:
                    summary_string = summary_string\
                        + f'    I3 = {isovalue} does not contain this shell\n'
                else:
                    summary_string = summary_string\
                        + f'    I3 = {isovalue} contains...\n'
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
                        summary_string = summary_string\
                            + (f'       {irrep} '
                               f'(appears {n_embedded} time{s}), '
                               f'covered {shell_covered}/{nstates} '
                               f'({iso_shell_covered} for this isospin)\n')
                    if shell_total == nstates:
                        summary_string = summary_string\
                            + 'The shell is covered!\n\n'
        summary_string = summary_string[:-1]
        master_dict['summary'] = summary_string
        return master_dict

    def get_noninttwo_proj_dict(self, qcis=None, cindex=0, definite_iso=True):
        """Get it."""
        master_dict = {}
        if qcis is None:
            raise ValueError('qcis cannot be None')
        row_zero_value = 0
        summary_string = ''
        nshells = len(qcis.n1n2_reps)
        for shell_index in range(nshells):
            shell_total = 0
            nstates = len(qcis.n1n2_batched[shell_index])*3
            summary_string = summary_string\
                + f'shell_index = {shell_index} ({nstates} states):\n'
            summary_string = summary_string\
                + f'representative momenta = \n{qcis.n1n2_reps[shell_index]}\n'
            if definite_iso:
                isoset = [2]
            else:
                isoset = range(1)
            for isovalue in isoset:
                non_proj_dict = self.get_noninttwo_proj_dict_shell(qcis, 0,
                                                                   definite_iso,
                                                                   isovalue,
                                                                   shell_index)
                master_dict[(shell_index, isovalue)] = non_proj_dict
                iso_shell_total = 0
                if len(non_proj_dict) == 0:
                    summary_string = summary_string\
                        + f'    I2 = {isovalue} does not contain this shell\n'
                else:
                    summary_string = summary_string\
                        + f'    I2 = {isovalue} contains...\n'
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
                        summary_string = summary_string\
                            + (f'       {irrep} '
                               f'(appears {n_embedded} time{s}), '
                               f'covered {shell_covered}/{nstates} '
                               f'({iso_shell_covered} for this isospin)\n')
                    if shell_total == nstates:
                        summary_string = summary_string\
                            + 'The shell is covered!\n\n'
        summary_string = summary_string[:-1]
        master_dict['summary'] = summary_string
        return master_dict

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
