#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_qcis.py
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
import numpy as np
from ampyl import FlavorChannel
from ampyl import FlavorChannelSpace
from ampyl import FiniteVolumeSetup
from ampyl import ThreeBodyInteractionScheme
from ampyl import QCIndexSpace


class TestQCIndexSpace(unittest.TestCase):

    def setUp(self):
        # Initialize objects for testing
        self.fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        self.fvs = FiniteVolumeSetup(nP=np.array([0, 0, 1]))
        self.tbis = ThreeBodyInteractionScheme()
        self.Emax = 5.0
        self.Lmax = 5.0

    def test_init(self):
        # Test initialization
        qcis = QCIndexSpace(fcs=self.fcs, fvs=self.fvs, tbis=self.tbis,
                            Emax=self.Emax, Lmax=self.Lmax, verbosity=2)
        self.assertEqual(qcis.verbosity, 2)
        self.assertEqual(qcis.Emax, 5.0)
        self.assertEqual(qcis.Lmax, 5.0)
        self.assertEqual(qcis.fvs.nP.tolist(), [0, 0, 1])
        self.assertIsInstance(qcis.fcs, FlavorChannelSpace)
        self.assertIsInstance(qcis.tbis, ThreeBodyInteractionScheme)

    def test_populate(self):
        # Test population of index space
        qcis = QCIndexSpace(fcs=self.fcs, fvs=self.fvs, tbis=self.tbis,
                            Emax=self.Emax, Lmax=self.Lmax)
        qcis.populate()
        self.assertIsInstance(qcis.kellm_spaces, list)
        kellm_spaces_as_numpy = np.array(qcis.kellm_spaces, dtype=object)
        self.assertEqual(kellm_spaces_as_numpy.shape, (1, 16))
        self.assertIsInstance(qcis.kellm_spaces[0][0], np.ndarray)
        self.assertEqual(qcis.kellm_spaces[0][0].shape, (28, 5))


if __name__ == '__main__':
    unittest.main()
