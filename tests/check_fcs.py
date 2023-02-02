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


class TestFlavorChannel(unittest.TestCase):
    """Unit tests for the FlavorChannel class."""

    def test_n_particles(self):
        # Test with n_particles as an int
        channel = FlavorChannel(2)
        self.assertEqual(channel.n_particles, 2)

        # Test with n_particles less than 2
        with self.assertRaises(ValueError):
            channel = FlavorChannel(1)

        # Test with n_particles not being an int
        with self.assertRaises(ValueError):
            channel = FlavorChannel("2")

    def test_flavors(self):
        # Test with flavors provided
        channel = FlavorChannel(2, flavors=['pi', 'pi'])
        self.assertEqual(channel.flavors, ['pi', 'pi'])

        # Test with no flavors provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.flavors, ['pi', 'pi'])

    def test_isospin_channel(self):
        # Test with isospin_channel and twoisospins provided
        channel = FlavorChannel(2, isospin_channel=True, twoisospins=[2, 1])
        self.assertTrue(channel.isospin_channel)
        # set back to equal as flavors are matched
        self.assertEqual(channel.twoisospins, [2, 2])

        channel = FlavorChannel(2, flavors=['K', 'pi'],
                                isospin_channel=True, twoisospins=[1, 2])
        self.assertEqual(channel.twoisospins, [1, 2])

        # Test with no isospin_channel and twoisospin_value provided
        channel = FlavorChannel(2, twoisospin_value=3)
        self.assertTrue(channel.isospin_channel)
        # 3 is not allowed, defaults to maximum
        self.assertEqual(channel.twoisospin_value, 4)

        # Test with no isospin_channel
        channel = FlavorChannel(2)
        self.assertFalse(channel.isospin_channel)

    def test_masses(self):
        # Test with masses provided
        channel = FlavorChannel(2, masses=[1.0, 2.0])
        # set back to equal as flavors are matched
        self.assertEqual(channel.masses, [1.0, 1.0])
        channel = FlavorChannel(2, masses=[1.0, 2.0], flavors=['pi', 'K'])
        self.assertEqual(channel.masses, [1.0, 2.0])

        # Test with no masses provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.masses, [1.0, 1.0])

    def test_twospins(self):
        # Test with twospins provided
        channel = FlavorChannel(2, twospins=[2, 2])
        self.assertEqual(channel.twospins, [2, 2])

        # Test with no twospins provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.twospins, [0, 0])

    def test_with_four_particles(self):
        # Create a FlavorChannel object with 4 particles
        channel = FlavorChannel(4)

        # Check that the number of particles is correct
        self.assertEqual(channel.n_particles, 4)

        # Check that the masses are set to default values
        self.assertEqual(channel.masses, [1.0, 1.0, 1.0, 1.0])

        # Check that the twospins are set to default values
        self.assertEqual(channel.twospins, [0, 0, 0, 0])

        # Check that the flavors are set to default values
        self.assertEqual(channel.flavors, ['pi', 'pi', 'pi', 'pi'])

        # Check that the isospin channel is set to default value
        self.assertFalse(channel.isospin_channel)

    def test_isospin_channel_setter(self):
        channel = FlavorChannel(3)

        # test setting isospin_channel to True
        channel.isospin_channel = True
        self.assertTrue(channel.isospin_channel)
        self.assertIsNotNone(channel.twoisospins)
        self.assertIsNotNone(channel.allowed_total_twoisospins)
        self.assertIsNotNone(channel.twoisospin_value)

        # test setting isospin_channel to False
        channel.isospin_channel = False
        self.assertFalse(channel.isospin_channel)
        self.assertIsNone(channel.twoisospins)
        self.assertIsNone(channel.allowed_total_twoisospins)
        self.assertIsNone(channel.twoisospin_value)

        # test setting isospin_channel to non-bool
        with self.assertRaises(ValueError):
            channel.isospin_channel = 'True'

    def test_get_allowed_three_particles(self):
        twoisospins = [1, 2, 3]
        channel = FlavorChannel(3, flavors=['K', 'pi', 'Omega'],
                                twoisospins=twoisospins)
        expected_result = [0, 2, 4, 6]
        result = channel._get_allowed()
        self.assertEqual(expected_result, result)

    def test_get_allowed_three_particles_summary(self):
        channel = FlavorChannel(3, flavors=['a', 'b', 'c'],
                                twoisospins=[1, 2, 3])
        channel._get_allowed()
        expected_summary = np.array([
            [0, 1, 'a', 1, ['b', 'c'], [2, 3]],
            [2, 1, 'a', 1, ['b', 'c'], [2, 3]],
            [2, 3, 'a', 1, ['b', 'c'], [2, 3]],
            [4, 3, 'a', 1, ['b', 'c'], [2, 3]],
            [4, 5, 'a', 1, ['b', 'c'], [2, 3]],
            [6, 5, 'a', 1, ['b', 'c'], [2, 3]],
            [0, 2, 'b', 2, ['a', 'c'], [1, 3]],
            [2, 2, 'b', 2, ['a', 'c'], [1, 3]],
            [4, 2, 'b', 2, ['a', 'c'], [1, 3]],
            [2, 4, 'b', 2, ['a', 'c'], [1, 3]],
            [4, 4, 'b', 2, ['a', 'c'], [1, 3]],
            [6, 4, 'b', 2, ['a', 'c'], [1, 3]],
            [2, 1, 'c', 3, ['a', 'b'], [1, 2]],
            [4, 1, 'c', 3, ['a', 'b'], [1, 2]],
            [0, 3, 'c', 3, ['a', 'b'], [1, 2]],
            [2, 3, 'c', 3, ['a', 'b'], [1, 2]],
            [4, 3, 'c', 3, ['a', 'b'], [1, 2]],
            [6, 3, 'c', 3, ['a', 'b'], [1, 2]],
            ], dtype=object)

        self.assertTrue(np.array_equal(channel.summary, expected_summary))



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
