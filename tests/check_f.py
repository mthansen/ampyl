#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# ampyl.py
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
import unittest
import ampyl


class TestFlavorChannel(unittest.TestCase):
    """Class to test the FlavorChannel class."""

    def get_f_abbreviated(self, E, nP, L, k_entry, C1cut, alphaKSS,
                          ell_row, mazi_row, ell_col, mazi_col):
        """Get F, abbreviated form."""
        masses = [1.]*3
        alphabeta = [-1., 0.]
        F = ampyl.QCFunctions.getF_single_entry(E, nP, L, k_entry,
                                                *masses, C1cut, alphaKSS,
                                                *alphabeta, ell_row, mazi_row,
                                                ell_col, mazi_col,
                                                'relativistic pole',
                                                {'hermitian': True,
                                                 'real harmonics': True})
        return F

    def get_value_direct(self, E, nP, L, kellm_space, C1cut, alphaKSS):
        """Get F matrix in a direct way."""
        F_direct = [[]]
        for kellm_entry_row in kellm_space:
            F_row = []
            for kellm_entry_col in kellm_space:
                kentry_row = kellm_entry_row[:3]
                kentry_col = kellm_entry_col[:3]
                ell_row = kellm_entry_row[3]
                mazi_row = kellm_entry_row[4]
                ell_col = kellm_entry_col[3]
                mazi_col = kellm_entry_col[4]
                if (kentry_row == kentry_col).all():
                    k_entry = kentry_row
                    Ftmp = self.get_f_abbreviated(E, nP, L, k_entry, C1cut,
                                                  alphaKSS, ell_row, mazi_row,
                                                  ell_col, mazi_col)
                else:
                    Ftmp = 0.
                F_row = F_row+[Ftmp]
            F_direct = F_direct+[F_row]
        F_direct = np.array(F_direct[1:])
        return F_direct

    def setUp(self):
        """Exectue set-up."""
        fc = ampyl.FlavorChannel(3)
        fcs = ampyl.FlavorChannelSpace(fc_list=[fc])
        fvs = ampyl.FiniteVolumeSetup()
        tbis = ampyl.ThreeBodyInteractionScheme()
        qcis = ampyl.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                  Emax=5., Lmax=7.)

        self.epsilon = 1.0e-15

        self.qcis = qcis
        self.fcs = fcs
        self.f = ampyl.F(qcis=qcis)

        fvs = ampyl.FiniteVolumeSetup(nP=np.array([0, 0, 1]))
        tbis = ampyl.ThreeBodyInteractionScheme()
        qcis_001 = ampyl.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                      Emax=5., Lmax=7.)
        self.qcis_001 = qcis_001
        self.f_001 = ampyl.F(qcis=qcis_001)

    def tearDown(self):
        """Execute tear-down."""
        pass

    def test_f_one(self):
        """Test G, first test."""
        C1cut = 3
        alphaKSS = 1.
        F = self.f.get_value(5., 7.)
        F_direct = self.get_value_direct(5., np.array([0, 0, 0]), 7.,
                                         self.qcis.kellm_spaces[0][0],
                                         C1cut, alphaKSS)
        self.assertTrue((F-F_direct < self.epsilon).all())

    def test_f_two(self):
        """Test G, second test."""
        C1cut = 3
        alphaKSS = 1.
        F = self.f_001.get_value(5., 7.)
        F_direct = self.get_value_direct(5., np.array([0, 0, 1]), 7.,
                                         self.qcis_001.kellm_spaces[0][0],
                                         C1cut, alphaKSS)
        self.assertTrue((F-F_direct < self.epsilon).all())


class Template(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Exectue set-up."""
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def __example(self, x):
        return x

    def test(self):
        """Example test."""
        self.assertEqual(10., self.__example(10.))


unittest.main()
