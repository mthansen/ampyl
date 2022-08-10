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

    def get_g_abbreviated(self, E, nP, L, kentry_row, kentry_col,
                          ell_row, mazi_row, ell_col, mazi_col):
        """Get G abbreviated."""
        masses = [1.0]*3
        alphabeta = [-1.0, 0.0]
        gtmp = ampyl.QCFunctions.getG_single_entry(E, nP, L,
                                                   kentry_row, kentry_col,
                                                   ell_row, mazi_row,
                                                   ell_col, mazi_col,
                                                   *masses, *alphabeta,
                                                   False,
                                                   'relativistic pole',
                                                   'hermitian, real harmonics')
        return gtmp

    def get_value_direct(self, E, nP, L,
                         kellm_space_row, kellm_space_col, fcs):
        """Get G matrix in a direct way."""
        G_direct = [[]]
        for kellm_entry_row in kellm_space_row:
            G_row = []
            for kellm_entry_col in kellm_space_col:
                kentry_row = kellm_entry_row[:3]
                kentry_col = kellm_entry_col[:3]
                ell_row = kellm_entry_row[3]
                mazi_row = kellm_entry_row[4]
                ell_col = kellm_entry_col[3]
                mazi_col = kellm_entry_col[4]
                Gtmp = self.get_g_abbreviated(E, nP, L,
                                              kentry_row, kentry_col,
                                              ell_row, mazi_row,
                                              ell_col, mazi_col)
                G_row = G_row+[Gtmp*fcs.g_templates[0][0][0][0]]
            G_direct = G_direct+[G_row]
        G_direct = np.array(G_direct[1:])
        return G_direct

    def setUp(self):
        """Exectue set-up."""
        fc = ampyl.FlavorChannel(3)
        fcs = ampyl.FlavorChannelSpace(fc_list=[fc])
        fvs = ampyl.FiniteVolumeSetup()
        tbis = ampyl.ThreeBodyInteractionScheme()
        qcis = ampyl.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                  Emax=5.0, Lmax=7.0)
        self.qcis = qcis
        self.fcs = fcs
        self.g = ampyl.G(qcis=qcis)

    def tearDown(self):
        """Execute tear-down."""
        pass

    def test_g_one(self):
        """Test G, first test."""
        G = self.g.get_value(5.0, 7.0)
        G_direct = self.get_value_direct(5.0, np.array([0, 0, 0]), 7.0,
                                         self.qcis.kellm_spaces[0][0],
                                         self.qcis.kellm_spaces[0][0],
                                         self.fcs)
        self.assertTrue((G-G_direct < 1.0e-15).all())


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
        self.assertEqual(10.0, self.__example(10.0))


unittest.main()
