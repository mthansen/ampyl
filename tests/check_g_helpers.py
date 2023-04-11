#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# check_g_helpers.py
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
from ampyl import QCIndexSpace
from ampyl import G
from ampyl import FiniteVolumeSetup


class TestGHelpers(unittest.TestCase):

    def setUp(self):
        self.qcis = QCIndexSpace()
        self.qcis.populate()
        self.g = G(qcis=self.qcis)
        self.tbks_set = self.qcis.tbks_list[0]
        self.fvs_011 = FiniteVolumeSetup(nP=np.array([0, 1, 1]))
        self.qcis_011 = QCIndexSpace(fvs=self.fvs_011)
        self.qcis_011.populate()
        self.g_011 = G(qcis=self.qcis_011)
        self.tbks_set_011 = self.qcis_011.tbks_list[0]

    def test_get_masks_and_shells_nP_zero(self):
        E = self.qcis.Emax
        nP = np.zeros((3,))
        L = self.qcis.Lmax
        tbks_entry = self.tbks_set[0]
        cindex_row = 0
        cindex_col = 0
        expected_shells = [[0, 1], [1, 7], [7, 19], [19, 27]]
        for row_shell_index in range(len(tbks_entry.shells)):
            for col_shell_index in range(len(tbks_entry.shells)):
                mask_row_shells, mask_col_shells, row_shell, col_shell\
                      = self.g._get_masks_and_shells(E, nP, L, tbks_entry,
                                                     cindex_row, cindex_col,
                                                     row_shell_index,
                                                     col_shell_index)
                self.assertIsNone(mask_row_shells)
                self.assertIsNone(mask_col_shells)
                expected_row_shell = expected_shells[row_shell_index]
                expected_col_shell = expected_shells[col_shell_index]
                self.assertEqual(row_shell, expected_row_shell)
                self.assertEqual(col_shell, expected_col_shell)

    def test_get_masks_and_shells_nP_nonzero(self):
        E = self.qcis_011.Emax
        nP = np.array([0, 1, 1])
        L = self.qcis_011.Lmax
        tbks_entry = self.tbks_set_011[0]
        cindex_row = 0
        cindex_col = 0
        expected_shells = [[0, 1], [1, 3], [3, 5], [5, 7], [7, 8], [8, 12],
                           [12, 14], [14, 16], [16, 18], [18, 20]]
        for row_shell_index in range(len(tbks_entry.shells)):
            for col_shell_index in range(len(tbks_entry.shells)):
                mask_row_shells, mask_col_shells, row_shell, col_shell\
                    = self.g._get_masks_and_shells(E, nP, L, tbks_entry,
                                                   cindex_row, cindex_col,
                                                   row_shell_index,
                                                   col_shell_index)
                expected_row_shell = expected_shells[row_shell_index]
                expected_col_shell = expected_shells[col_shell_index]
                row_truth_list = [True]*len(expected_shells)
                col_truth_list = [True]*len(expected_shells)
                self.assertEqual(mask_row_shells, row_truth_list)
                self.assertEqual(mask_col_shells, col_truth_list)
                self.assertEqual(row_shell, expected_row_shell)
                self.assertEqual(col_shell, expected_col_shell)

    def test_get_masks_and_shells_nP_nonzero_lowerE(self):
        E = 4.11
        nP = np.array([0, 1, 1])
        L = self.qcis_011.Lmax
        tbks_entry = self.tbks_set_011[0]
        cindex_row = 0
        cindex_col = 0
        expected_shells_unmasked = [[0, 1], [1, 3], [3, 5], [5, 7], [7, 8],
                                    [8, 12], [12, 14], [14, 16], [16, 18],
                                    [18, 20]]
        expected_mask = [True, True, True, False, True,
                         True, False, True, False, False]
        expected_shells = list(np.array(
            expected_shells_unmasked)[expected_mask])
        for row_shell_index in range(len(expected_shells)):
            for col_shell_index in range(len(expected_shells)):
                mask_row_shells, mask_col_shells, row_shell, col_shell\
                    = self.g._get_masks_and_shells(E, nP, L, tbks_entry,
                                                   cindex_row, cindex_col,
                                                   row_shell_index,
                                                   col_shell_index)
                self.assertEqual(mask_row_shells, expected_mask)
                self.assertEqual(mask_col_shells, expected_mask)
                expected_row_shell = list(expected_shells[row_shell_index])
                expected_col_shell = list(expected_shells[col_shell_index])
                self.assertEqual(row_shell, expected_row_shell)
                self.assertEqual(col_shell, expected_col_shell)

    def test_get_masks_and_shells_invalid_slice_index(self):
        E = self.qcis.Emax
        nP = np.zeros((3,))
        L = self.qcis.Lmax
        tbks_entry = self.tbks_set_011[0]
        cindex_row = 1
        cindex_col = 1
        row_shell_index = 0
        col_shell_index = 0
        with self.assertRaises(ValueError):
            self.g._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
                                         row_shell_index, col_shell_index)

    def test_nPzero_projectors(self):
        sc_index_col = 0
        sc_index_row = 0
        row_shell_index = 0
        col_shell_index = 0
        irrep = ('A1PLUS', 0)
        [proj_left, proj_right]\
            = self.g._nPzero_projectors(sc_index_row, sc_index_col,
                                        row_shell_index, col_shell_index,
                                        irrep)
        for row in proj_left:
            for entry in row:
                self.assertEqual(entry, 1.)
        for row in proj_right:
            for entry in row:
                self.assertEqual(entry, 1.)


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
