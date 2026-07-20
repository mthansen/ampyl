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
from ampyl import nonint_utils
from ampyl.constants import FOURPI2
from ampyl.spaces import QCIndexSpace


class TestNonInteracting(unittest.TestCase):
    """Unit tests for the FlavorChannel class."""

    def test_load_ni_data(self):
        fc2 = FlavorChannel(2)
        fc3 = FlavorChannel(3)
        fcs = FlavorChannelSpace(fc_list=[fc2, fc3])
        qcis = QCIndexSpace(fcs=fcs)
        ni_data_two = nonint_utils._load_ni_data_two(qcis.nis, fc2)
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

        ni_data_three = nonint_utils._load_ni_data_three(qcis.nis, fc3)
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

        qcis.nis.populate_all_nonint_data()

        self.assertEqual(qcis.nis._nonint_channel_particle_label(0), 'aaa')
        self.assertEqual(qcis.nis._nonint_channel_particle_label(1), 'aab')
        self.assertEqual(qcis.nis._nonint_channel_particle_label(2), 'abc')
        self.assertEqual(qcis.nis._nonint_channel_particle_label(3), 'aa')
        self.assertEqual(qcis.nis._nonint_channel_particle_label(4), 'ab')
        for label in ['abc', 'aab', 'aaa']:
            self.assertEqual(len(getattr(qcis.nis, f'nvecset_{label}')), 5)
            self.assertEqual(len(getattr(qcis.nis,
                                         f'nvecset_{label}_batched')),
                             5)
            self.assertIsNone(getattr(qcis.nis, f'nvecset_{label}')[3])
        for label in ['ab', 'aa']:
            self.assertEqual(len(getattr(qcis.nis, f'nvecset_{label}')), 5)
            self.assertEqual(len(getattr(qcis.nis,
                                         f'nvecset_{label}_batched')),
                             5)
            self.assertIsNone(getattr(qcis.nis, f'nvecset_{label}')[0])
        self.assertGreaterEqual(len(qcis.nis.nvecset_abc[1]),
                                len(qcis.nis.nvecset_aab[1]))
        self.assertGreaterEqual(len(qcis.nis.nvecset_aab[1]),
                                len(qcis.nis.nvecset_aaa[1]))
        self.assertGreaterEqual(len(qcis.nis.nvecset_ab[3]),
                                len(qcis.nis.nvecset_aa[3]))


class TestNonIntTwoParticleEnergies(unittest.TestCase):
    """Two-particle non-interacting energies must use the channel masses."""

    def _build(self, mass1, mass2, Emax=4.5, Lmax=4.0):
        """Build a populated qcis whose only ni channel is (mass1, mass2)."""
        first = Particle(mass=mass1, flavor="pi")
        if mass1 == mass2:
            second = first
        else:
            second = Particle(mass=mass2, flavor="K")
        fc_three = FlavorChannel(3, particles=[first, first, first])
        fc_two = FlavorChannel(2, particles=[first, second])
        fcs = FlavorChannelSpace(fc_list=[fc_three], ni_list=[fc_two])
        qcis = QCIndexSpace(fcs=fcs, Emax=Emax, Lmax=Lmax)
        qcis.group = Groups(ell_max=4, spin_half=False)
        qcis.populate()
        qcis.nis.populate_nonint_functions()
        return qcis

    def _levels(self, qcis):
        """Yield (nSQ1, nSQ2, function) for every two-particle level."""
        for key, functions in qcis.nis.nonint_functions[0].items():
            multiplicities = qcis.nis.nonint_multiplicities[0][key]
            for multiplicity, function in zip(multiplicities, functions):
                yield multiplicity[0], multiplicity[1], function

    def _expected(self, mass1, mass2, nSQ1, nSQ2, L):
        """Evaluate the analytic sum of two finite-volume energies."""
        return (np.sqrt(mass1**2+FOURPI2*nSQ1/L**2)
                + np.sqrt(mass2**2+FOURPI2*nSQ2/L**2))

    def test_degenerate_channel_uses_channel_masses(self):
        """Equal-mass levels must match the analytic expression."""
        mass = 1.0
        L = 4.0
        qcis = self._build(mass, mass)
        self.assertEqual(qcis.nis._nonint_channel_particle_label(0), 'aa')
        checked = 0
        for nSQ1, nSQ2, function in self._levels(qcis):
            self.assertAlmostEqual(
                function(L), self._expected(mass, mass, nSQ1, nSQ2, L),
                places=12)
            checked += 1
        self.assertGreater(checked, 0)

    def test_nondegenerate_channel_pairs_masses_with_momenta(self):
        """Unequal-mass levels pin mass 1 to slot 1 and mass 2 to slot 2."""
        mass1, mass2 = 1.0, 1.7
        L = 4.0
        qcis = self._build(mass1, mass2)
        self.assertEqual(qcis.nis._nonint_channel_particle_label(0), 'ab')
        checked = 0
        for nSQ1, nSQ2, function in self._levels(qcis):
            self.assertAlmostEqual(
                function(L), self._expected(mass1, mass2, nSQ1, nSQ2, L),
                places=12)
            checked += 1
        self.assertGreater(checked, 0)

    def test_nonint_two_levels_respect_Emax(self):
        """Enumeration and evaluation must agree on the mass used.

        The momentum sets are cut on ``E <= Emax`` using the channel masses
        in ``nonint_utils._get_nvecset_ab_two``. Any level exceeding Emax
        means the evaluation is using a different mass than the cut did.
        """
        for mass1, mass2 in [(1.0, 1.0), (1.0, 1.7)]:
            qcis = self._build(mass1, mass2)
            for nSQ1, nSQ2, function in self._levels(qcis):
                self.assertLessEqual(function(qcis.Lmax),
                                     qcis.Emax+1.0e-12)


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
