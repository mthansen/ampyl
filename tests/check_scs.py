#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# check_scs.py
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
from ampyl import QCFunctions
from ampyl import FlavorChannel
from ampyl import SpectatorChannel


class TestSpectatorChannel(unittest.TestCase):
    """A class for unit testing the SpectatorChannel class."""

    def test_init(self):
        """A method that tests the init for the class SpectatorChannel."""
        sc = SpectatorChannel()
        self.assertIsInstance(sc, SpectatorChannel)
        self.assertEqual(sc.fc.flavors, ['pi', 'pi', 'pi'])
        self.assertEqual(sc.indexing, [0, 1, 2])
        self.assertIsNone(sc.sub_twoisospin)
        self.assertEqual(sc.ell_set, [0])
        self.assertEqual(sc.p_cot_deltas,
                         [QCFunctions.pcotdelta_scattering_length])
        self.assertEqual(sc.n_params_set, [1])

    def test_allowed_sub_twoisospins(self):
        """A method that tests the allowed_sub_twoisospins property."""
        sc = SpectatorChannel(fc=FlavorChannel(3))
        self.assertEqual(sc.allowed_sub_twoisospins, None)
        sc = SpectatorChannel(fc=FlavorChannel(2))
        self.assertIsNone(sc.allowed_sub_twoisospins)
        sc = SpectatorChannel(fc=FlavorChannel(3, isospin_channel=True,
                                               twoisospins=[2, 2, 2],
                                               twoisospin_value=2),
                              sub_twoisospin=4)
        self.assertEqual(sc.allowed_sub_twoisospins, [0, 2, 4])

    def test_fc_property(self):
        """A method that tests the `fc` property."""
        sc = SpectatorChannel()
        self.assertIsInstance(sc.fc, FlavorChannel)
        self.assertEqual(sc.fc.flavors, ['pi', 'pi', 'pi'])

    def test_indexing_property(self):
        """A method that tests the indexing property."""
        sc = SpectatorChannel()
        self.assertEqual(sc.indexing, [0, 1, 2])

    def test_sub_twoisospin_property(self):
        """A method that tests the sub_twoisospin property."""
        sc = SpectatorChannel()
        self.assertIsNone(sc.sub_twoisospin)

    def test_ell_set_property(self):
        """A method that tests the ell_set property."""
        sc = SpectatorChannel()
        self.assertEqual(sc.ell_set, [0])

    def test_p_cot_deltas_property(self):
        """A method that tests the p_cot_deltas property."""
        sc = SpectatorChannel()
        self.assertEqual(sc.p_cot_deltas,
                         [QCFunctions.pcotdelta_scattering_length])

    def test_n_params_set_property(self):
        """A method that tests the n_params_set property."""
        sc = SpectatorChannel()
        self.assertEqual(sc.n_params_set, [1])


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
