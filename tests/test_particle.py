#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_particle.py
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


class TestParticle(unittest.TestCase):
    """Test the Particle class."""

    def test_init(self):
        """Test the initialization of a Particle."""
        particle = Particle()
        self.assertEqual(particle.mass, 1.)
        self.assertEqual(particle.spin, 0.)
        self.assertEqual(particle.flavor, 'pi')
        self.assertEqual(particle.isospin_multiplet, False)
        self.assertEqual(particle.isospin, None)

        particle = Particle(mass=2., spin=1., flavor='K',
                            isospin_multiplet=True, isospin=0.)
        self.assertEqual(particle.mass, 2.)
        self.assertEqual(particle.spin, 1.)
        self.assertEqual(particle.flavor, 'K')
        self.assertEqual(particle.isospin_multiplet, True)
        self.assertEqual(particle.isospin, 0.)

    # def test_check_type(self):
    #     """Test the check_type method."""
    #     particle = Particle()
    #     with self.assertRaises(TypeError):
    #         particle._check_type(1, 'mass', float, 'float')
    #     with self.assertRaises(TypeError):
    #         particle._check_type(1., 'spin', float, 'float')
    #     with self.assertRaises(TypeError):
    #         particle._check_type(1., 'flavor', str, 'str')
    #     with self.assertRaises(TypeError):
    #         particle._check_type(1., 'isospin_multiplet', bool, 'bool')
    #     with self.assertRaises(TypeError):
    #         particle._check_type(1., 'isospin', float, 'float')

    def test_mass(self):
        """Test the mass property."""
        particle = Particle()
        particle.mass = 2.
        self.assertEqual(particle.mass, 2.)

    def test_spin(self):
        """Test the spin property."""
        particle = Particle()
        particle.spin = 1.
        self.assertEqual(particle.spin, 1.)

    def test_flavor(self):
        """Test the flavor property."""
        particle = Particle()
        particle.flavor = 'K'
        self.assertEqual(particle.flavor, 'K')

    def test_isospin_multiplet(self):
        """Test the isospin_multiplet property."""
        particle = Particle()
        particle.isospin_multiplet = True
        self.assertEqual(particle.isospin_multiplet, True)

    def test_isospin(self):
        """Test the isospin property."""
        particle = Particle()
        particle.isospin = 0.
        self.assertEqual(particle.isospin, 0.)

    def test_eq(self):
        """Test the equality operator."""
        particle1 = Particle()
        particle2 = Particle()
        self.assertTrue(particle1 == particle2)
        particle2.mass = 2.
        self.assertFalse(particle1 == particle2)
        particle2.mass = 1.
        self.assertTrue(particle1 == particle2)
        particle2.spin = 1.
        self.assertFalse(particle1 == particle2)
        particle2.spin = 0.
        self.assertTrue(particle1 == particle2)
        particle2.flavor = 'K'
        self.assertFalse(particle1 == particle2)
        particle2.flavor = 'pi'
        self.assertTrue(particle1 == particle2)
        particle2.isospin_multiplet = True
        self.assertFalse(particle1 == particle2)
        particle2.isospin_multiplet = False
        self.assertTrue(particle1 == particle2)
        particle2.isospin = 0.
        self.assertFalse(particle1 == particle2)
        particle2.isospin = None
        self.assertTrue(particle1 == particle2)

    def test_particle_str(self):
        particle = Particle()
        self.assertEqual(particle.__str__(),
                         "Particle with the following properties:\n"
                         "    mass: 1.0,\n"
                         "    spin: 0.0,\n"
                         "    flavor: pi,\n"
                         "    isospin_multiplet: False.")


if __name__ == '__main__':
    unittest.main()
