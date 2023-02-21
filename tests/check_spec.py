#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2023.

@author: M.T. Hansen
"""

###############################################################################
#
# check_spec.py
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

import unittest
from scipy.optimize import root_scalar
import numpy as np
from ampyl import QCIndexSpace
from ampyl import QC


class TestEnergyPrediction(unittest.TestCase):
    """A class for unit testing the output energies."""

    def test_weakly_interacting_groundstate(self):
        """A method that tests the simplest prediction."""
        qcis = QCIndexSpace()
        qcis.fvs.qc_impl['g_uses_prep_mat'] = True
        qc = QC(qcis=qcis)
        L = 5.0
        a0 = 0.0001
        delta1 = 1.e-6
        delta2 = 1.e-1
        root_tmp = root_scalar(qc.get_value, args=(L,
                                                   [[[a0]], [0.0]],
                                                   True,
                                                   ('A1PLUS', 0)),
                               bracket=[3.+delta1, 3.+delta2]).root
        result = (root_tmp-3.0)*L**3/12./a0
        expected = np.pi
        self.assertAlmostEqual(result, expected, delta=0.001)


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


if __name__ == '__main__':
    unittest.main()
