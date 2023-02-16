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

import unittest
import numpy as np
from ampyl import FlavorChannel
from ampyl import FlavorChannelSpace
from ampyl import QCIndexSpace


class TestNonInteracting(unittest.TestCase):
    """Unit tests for the FlavorChannel class."""

    def test_load_ni_data(self):
        fc2 = FlavorChannel(2)
        fc3 = FlavorChannel(3)
        fcs = FlavorChannelSpace(fc_list=[fc2, fc3])
        qcis = QCIndexSpace(fcs=fcs)
        ni_data_two = qcis._load_ni_data_two(fc2)
        expected_set = [1., 1., 5., np.array([0, 0, 0]), 5., 1]
        for j in range(len(expected_set)):
            expectation = expected_set[j]
            reality = ni_data_two[j]
            if type(expectation) is np.ndarray:
                self.assertTrue((expectation == reality).all())
            else:
                self.assertEqual(expectation, reality)
        self.assertEqual(ni_data_two[6].shape, (27, 3))

        ni_data_three = qcis._load_ni_data_three(fc3)
        expected_set = [1., 1., 1., 5., np.array([0, 0, 0]), 5., 2]
        for j in range(len(expected_set)):
            expectation = expected_set[j]
            reality = ni_data_three[j]
            if type(expectation) is np.ndarray:
                self.assertTrue((expectation == reality).all())
            else:
                self.assertEqual(expectation, reality)
        self.assertEqual(ni_data_three[7].shape, (125, 3))


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
