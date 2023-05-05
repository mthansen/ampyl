#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_flavor_channel_two.py
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
from ampyl import Particle
from ampyl import FlavorChannel


class TestFlavorChannel(unittest.TestCase):
    """Test the FlavorChannel class."""

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
        pion = Particle(flavor='pi')
        channel = FlavorChannel(2, particles=[pion, pion])
        self.assertEqual(channel.flavors, ['pi', 'pi'])

        # Test with no flavors provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.flavors, ['pi', 'pi'])

    def test_isospin_channel(self):
        # Test with isospin_channel and isospins provided
        pion = Particle(flavor='pi', isospin=1.0)
        kaon = Particle(flavor='K', isospin=0.5)
        channel = FlavorChannel(2, particles=[pion, kaon],
                                isospin_channel=True, isospin=1.5)
        self.assertTrue(channel.isospin_channel)
        self.assertEqual(channel.isospins, [1.0, 0.5])

        # Test with no isospin_channel and isospin provided
        channel = FlavorChannel(2, particles=[pion, pion], isospin=2.)
        self.assertTrue(channel.isospin_channel)
        self.assertEqual(channel.isospin, 2.)

        # Test with no isospin_channel
        channel = FlavorChannel(2)
        self.assertFalse(channel.isospin_channel)

    def test_masses(self):
        # Test with masses provided
        pion = Particle(flavor='pi', mass=1.0)
        kaon = Particle(flavor='K', mass=2.0)
        channel = FlavorChannel(2, particles=[pion, kaon])
        self.assertEqual(channel.masses, [1.0, 2.0])

        # Test with no masses provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.masses, [1.0, 1.0])

    def test_spins(self):
        # Test with spins provided
        rho = Particle(flavor='rho', spin=1.0)
        channel = FlavorChannel(2, particles=[rho, rho])
        self.assertEqual(channel.spins, [1., 1.])

        # Test with no spins provided
        channel = FlavorChannel(2)
        self.assertEqual(channel.spins, [0., 0.])

    def test_with_four_particles(self):
        # Create a FlavorChannel object with 4 particles
        channel = FlavorChannel(4)

        # Check that the number of particles is correct
        self.assertEqual(channel.n_particles, 4)

        # Check that the masses are set to default values
        self.assertEqual(channel.masses, [1.0, 1.0, 1.0, 1.0])

        # Check that the spins are set to default values
        self.assertEqual(channel.spins, [0, 0, 0, 0])

        # Check that the flavors are set to default values
        self.assertEqual(channel.flavors, ['pi', 'pi', 'pi', 'pi'])

        # Check that the isospin channel is set to default value
        self.assertFalse(channel.isospin_channel)

    # def test_isospin_channel_setter(self):
    #     channel = FlavorChannel(3)

    #     # test setting isospin_channel to True
    #     channel.isospin_channel = True
    #     self.assertTrue(channel.isospin_channel)
    #     self.assertIsNotNone(channel.isospins)
    #     self.assertIsNotNone(channel.allowed_total_isospins)
    #     self.assertIsNotNone(channel.isospin)

    #     # test setting isospin_channel to False
    #     channel.isospin_channel = False
    #     self.assertFalse(channel.isospin_channel)
    #     self.assertIsNone(channel.isospins)
    #     self.assertIsNone(channel.allowed_total_isospins)
    #     self.assertIsNone(channel.isospin)

    #     # test setting isospin_channel to non-bool
    #     with self.assertRaises(ValueError):
    #         channel.isospin_channel = 'True'

    def test_get_allowed_three_particles(self):
        kaon = Particle(flavor='K', isospin=0.5)
        pion = Particle(flavor='pi', isospin=1.0)
        Omega = Particle(flavor='Omega', isospin=1.5)
        channel = FlavorChannel(3, particles=[kaon, pion, Omega],
                                isospin_channel=True, isospin=1.0)
        expected_result = [0.0, 1., 2., 3.]
        result = channel._get_allowed_total_isospins()
        self.assertEqual(expected_result, result)

    def test_get_allowed_three_particles_summary(self):

        kaon = Particle(flavor='K', isospin=0.5)
        pion = Particle(flavor='pi', isospin=1.0)
        Omega = Particle(flavor='Omega', isospin=1.5)
        channel = FlavorChannel(3, particles=[kaon, pion, Omega],
                                isospin_channel=True, isospin=1.0)

        expected_summary = np.array([
            [0.0, 0.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [1.0, 0.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [1.0, 1.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [2.0, 1.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [2.0, 2.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [3.0, 2.5, 'K', 0.5, 'pi', 'Omega', 1.0, 1.5],
            [1.0, 0.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [2.0, 0.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [0.0, 1.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [1.0, 1.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [2.0, 1.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [3.0, 1.5, 'Omega', 1.5, 'K', 'pi', 0.5, 1.0],
            [0.0, 1.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5],
            [1.0, 1.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5],
            [2.0, 1.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5],
            [1.0, 2.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5],
            [2.0, 2.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5],
            [3.0, 2.0, 'pi', 1.0, 'K', 'Omega', 0.5, 1.5]
            ], dtype=object)

        self.assertTrue(np.array_equal(channel.summary, expected_summary))


if __name__ == '__main__':
    unittest.main()
