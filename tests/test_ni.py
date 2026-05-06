#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_ni.py
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
from ampyl.flavor import Particle
from ampyl.flavor import FlavorChannel
from ampyl.flavor import FlavorChannelSpace
from ampyl.groups import Groups
from ampyl.spaces import QCIndexSpace


class TestNonInteracting(unittest.TestCase):
    """Unit tests for the FlavorChannel class."""

    def test_load_ni_data(self):
        fc2 = FlavorChannel(2)
        fc3 = FlavorChannel(3)
        fcs = FlavorChannelSpace(fc_list=[fc2, fc3])
        qcis = QCIndexSpace(fcs=fcs)
        ni_data_two = qcis._load_ni_data_two(fc2)
        # expected_set = [1., 1., 5., np.array([0, 0, 0]), 5., 1]
        expected_set = [1., 1., 5., np.array([0, 0, 0]), 5., 2]
        for j in range(len(expected_set)):
            expectation = expected_set[j]
            reality = ni_data_two[j]
            if type(expectation) is np.ndarray:
                self.assertTrue((expectation == reality).all())
            else:
                self.assertEqual(expectation, reality)
        # self.assertEqual(ni_data_two[6].shape, (27, 3))
        self.assertEqual(ni_data_two[6].shape, (125, 3))

        ni_data_three = qcis._load_ni_data_three(fc3)
        expected_set = [1., 1., 1., 5., np.array([0, 0, 0]), 5., 2]
        for j in range(len(expected_set)):
            expectation = expected_set[j]
            reality = ni_data_three[j]
            if type(expectation) is np.ndarray:
                self.assertTrue((expectation == reality).all())
            else:
                self.assertEqual(expectation, reality)
        self.assertEqual(ni_data_three[7].shape, (27, 3))

    def test_populate_nonint_data_particle_labels(self):
        pion = Particle(mass=1.0, flavor="pi")
        kaon = Particle(mass=1.2, flavor="K")
        eta = Particle(mass=1.4, flavor="eta")
        fc_aaa = FlavorChannel(3, particles=[pion, pion, pion])
        fc_aab = FlavorChannel(3, particles=[pion, pion, kaon])
        fc_abc = FlavorChannel(3, particles=[pion, kaon, eta])
        fc_aa = FlavorChannel(2, particles=[pion, pion])
        fc_ab = FlavorChannel(2, particles=[pion, kaon])
        fcs = FlavorChannelSpace(
            fc_list=[], ni_list=[fc_aaa, fc_aab, fc_abc, fc_aa, fc_ab])
        qcis = QCIndexSpace(fcs=fcs, Emax=4.5, Lmax=4.0)
        qcis.group = Groups(ell_max=4, spin_half=False)

        qcis.populate_all_nonint_data()

        self.assertEqual(qcis._nonint_channel_particle_label(0), 'aaa')
        self.assertEqual(qcis._nonint_channel_particle_label(1), 'aab')
        self.assertEqual(qcis._nonint_channel_particle_label(2), 'abc')
        self.assertEqual(qcis._nonint_channel_particle_label(3), 'aa')
        self.assertEqual(qcis._nonint_channel_particle_label(4), 'ab')
        for label in ['abc', 'aab', 'aaa']:
            self.assertEqual(len(getattr(qcis, f'nvecset_{label}')), 5)
            self.assertEqual(len(getattr(qcis, f'nvecset_{label}_batched')),
                             5)
            self.assertIsNone(getattr(qcis, f'nvecset_{label}')[3])
        for label in ['ab', 'aa']:
            self.assertEqual(len(getattr(qcis, f'nvecset_{label}')), 5)
            self.assertEqual(len(getattr(qcis, f'nvecset_{label}_batched')),
                             5)
            self.assertIsNone(getattr(qcis, f'nvecset_{label}')[0])
        self.assertGreaterEqual(len(qcis.nvecset_abc[1]),
                                len(qcis.nvecset_aab[1]))
        self.assertGreaterEqual(len(qcis.nvecset_aab[1]),
                                len(qcis.nvecset_aaa[1]))
        self.assertGreaterEqual(len(qcis.nvecset_ab[3]),
                                len(qcis.nvecset_aa[3]))


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
