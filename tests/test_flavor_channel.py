#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_flavor_channel.py
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
from ampyl import Particle
from ampyl import FlavorChannel


class TestFlavorChannel(unittest.TestCase):
    """Test the FlavorChannel class."""

    def test_init(self):
        """Test the initialization of the FlavorChannel class."""
        # test that the initialization works
        particle_pi = Particle(1., flavor="pi")
        particle_K = Particle(2., flavor="K")
        channel = FlavorChannel(2, particles=[
            particle_pi, particle_K])
        self.assertEqual(channel.particles, [
            particle_pi, particle_K])
        self.assertEqual(channel.isospin_channel, False)
        self.assertEqual(channel.isospin, None)
        self.assertEqual(channel.n_particles, 2)
        self.assertEqual(channel.masses, [particle_pi.mass,
                                          particle_K.mass])
        self.assertEqual(channel.spins, [particle_pi.spin,
                                         particle_K.spin])
        self.assertEqual(channel.flavors, [particle_pi.flavor,
                                           particle_K.flavor])
        self.assertEqual(channel.isospins, [particle_pi.isospin,
                                            particle_K.isospin])
        self.assertEqual(channel.allowed_total_isospins, None)

        # test that the initialization fails for invalid inputs
        with self.assertRaises(ValueError):
            channel = FlavorChannel(1, particles=[Particle()])
        with self.assertRaises(ValueError):
            channel = FlavorChannel(2, particles=[Particle(), Particle()],
                                    isospin=1.)
        with self.assertRaises(ValueError):
            channel = FlavorChannel(3, particles=[Particle(), Particle()])

    def test_get_masses(self):
        """Test the _get_masses method of the FlavorChannel class."""
        # test that the method works
        particle_pi = Particle(1., flavor="pi")
        particle_K = Particle(2., flavor="K")
        channel = FlavorChannel(2, particles=[particle_pi, particle_K])
        self.assertEqual(channel._get_masses(), [particle_pi.mass,
                                                 particle_K.mass])

    def test_get_spins(self):
        """Test the _get_spins method of the FlavorChannel class."""
        # test that the method works
        particle_pi = Particle(1., flavor="pi")
        particle_K = Particle(2., flavor="K")
        channel = FlavorChannel(2, particles=[particle_pi, particle_K])
        self.assertEqual(channel._get_spins(), [particle_pi.spin,
                                                particle_K.spin])

    def test_get_flavors(self):
        """Test the _get_flavors method of the FlavorChannel class."""
        # test that the method works
        particle_pi = Particle(1., flavor="pi")
        particle_K = Particle(2., flavor="K")
        channel = FlavorChannel(2, particles=[particle_pi, particle_K])
        self.assertEqual(channel._get_flavors(), [particle_pi.flavor,
                                                  particle_K.flavor])

    def test_get_isospins(self):
        pass

    def test_get_allowed_total_isospins(self):
        pass

    def test_flavor_channel_str(self):
        """Test the __str__ method."""
        particles = [Particle(mass=1.0, spin=0.5, flavor="1"),
                     Particle(mass=2.0, spin=0.5, flavor="2")]
        flavor_channel = FlavorChannel(2, particles=particles)
        self.assertEqual(str(flavor_channel),
                         "FlavorChannel with the following details:\n"
                         "    2 particles,\n"
                         "    masses: [1.0, 2.0],\n"
                         "    spins: [0.5, 0.5],\n"
                         "    flavors: ['1', '2'],\n"
                         "    isospin_channel: False.")
        particles = [Particle(mass=1.0, spin=0.5, flavor="1"),
                     Particle(mass=2.0, spin=0.5, flavor="2"),
                     Particle(mass=3.0, spin=0.5, flavor="3")]
        flavor_channel = FlavorChannel(3, particles=particles)
        self.assertEqual(str(flavor_channel),
                         "FlavorChannel with the following details:\n"
                         "    3 particles,\n"
                         "    masses: [1.0, 2.0, 3.0],\n"
                         "    spins: [0.5, 0.5, 0.5],\n"
                         "    flavors: ['1', '2', '3'],\n"
                         "    isospin_channel: False.")
        particles = [Particle(mass=1.0, spin=0.5, flavor="1"),
                     Particle(mass=2.0, spin=0.5, flavor="2"),
                     Particle(mass=3.0, spin=0.5, flavor="3"),
                     Particle(mass=4.0, spin=0.5, flavor="4")]
        flavor_channel = FlavorChannel(4, particles=particles)
        self.assertEqual(str(flavor_channel),
                         "FlavorChannel with the following details:\n"
                         "    4 particles,\n"
                         "    masses: [1.0, 2.0, 3.0, 4.0],\n"
                         "    spins: [0.5, 0.5, 0.5, 0.5],\n"
                         "    flavors: ['1', '2', '3', '4'],\n"
                         "    isospin_channel: False.")


if __name__ == "__main__":
    unittest.main()
