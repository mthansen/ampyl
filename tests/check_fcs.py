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
import ampyl


class TestFlavorChannel(unittest.TestCase):
    """Class to test the FlavorChannel class."""

    def setUp(self):
        """Exectue set-up."""
        self.fc = ampyl.FlavorChannel(3)

    def tearDown(self):
        """Execute tear-down."""
        pass

    def test_default_fc(self):
        """Test default three-particle flavor channel."""
        self.assertEqual(self.fc.n_particles, 3)
        self.assertEqual(self.fc.masses, 3*[1.0])
        self.assertEqual(self.fc.twospins, 3*[0])
        self.assertTrue(self.fc.explicit_flavor_channel)
        self.assertEqual(self.fc.explicit_flavors, 3*[1])
        self.assertFalse(self.fc.isospin_channel)
        self.assertIsNone(self.fc.isospin_value)
        self.assertIsNone(self.fc.isospin_flavor)


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
