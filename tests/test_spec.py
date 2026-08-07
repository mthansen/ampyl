#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 2023.

@author: M.T. Hansen
"""

###############################################################################
#
# test_spec.py
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
from scipy.optimize import root_scalar
import numpy as np
from ampyl.spaces import FiniteVolumeSetup
from ampyl.spaces import QCIndexSpace
from ampyl import QC


class TestEnergyPrediction(unittest.TestCase):
    """A class for unit testing the output energies."""

    def test_weakly_interacting_groundstate(self):
        """A method that tests the simplest prediction."""
        qcis = QCIndexSpace()
        qcis.populate()
        qcis.fvs.qc_impl['g_uses_prep_mat'] = True
        qcis.fvs.qc_impl['smarter_q_rescale'] = True
        qc = QC(qcis=qcis)
        L = 5.0
        a0 = 0.0001
        delta1 = 1.e-6
        delta2 = 1.e-1
        qc_dict = {'k_params': [[[a0]], [0.0]],
                   'project': True,
                   'irrep': ('A1PLUS', 0)}
        root_tmp = root_scalar(qc.get_value, args=(L, qc_dict),
                               bracket=[3.+delta1, 3.+delta2]).root
        result = (root_tmp-3.0)*L**3/12./a0
        expected = np.pi
        self.assertAlmostEqual(result, expected, delta=0.001)

    def test_weakly_interacting_first_excited_state(self):
        """Test the leading-order shift of the first excited state.

        The first excited non-interacting level in the rest-frame A1PLUS
        irrep is built from the momentum set {q, -q, 0} with q = 2 pi/L,
        giving the free energy E_1 = m + 2 omega_q, omega_q
        = sqrt(m^2 + q^2). The expected leading-order (in a0) shift is
        taken from:

        [1] D. M. Grabowska and M. T. Hansen, "Analytic expansions of
            multi-hadron finite-volume energies: I. Two-particle states",
            arXiv:2110.06878.
        [2] D. M. Grabowska and M. T. Hansen, "Analytic Expansions of
            Two- and Three-Particle Excited-State Energies",
            arXiv:2112.11996 (proceedings of LATTICE2021).

        Two equations are being matched. First, Eq. (16) of Ref. [2]
        (derived in detail in Ref. [1]) gives the leading-order shift of a
        two-particle level with total momentum d (2 pi/L)*[d] and
        constituent momenta nu, d - nu:

            E_n = E_n^(0)
                  + g_n [E_n^(0)/(4 omega_nu omega_(d-nu))]
                    [8 pi a0/(gamma_n^(0) L^3)] + O(a0^2),

        where g_n is the degeneracy of the free level and gamma^(0)
        = E^(0)/E* is the boost factor to the pair CM frame. Second,
        Eq. (26) of Ref. [2] states that the leading-order shift of a
        non-degenerate three-particle level is the sum of the Eq.-(16)
        shifts of its three two-particle subsystems. For {q, -q, 0} these
        are: the rest-frame pair (q, -q) with E^(0) = 2 omega_q, gamma
        = 1, g_n = 6, contributing 24 pi a0/(omega_q L^3); and the two
        moving-frame pairs (+-q, 0) with E^(0) = m + omega_q, g_n = 2,
        gamma = (m + omega_q)/sqrt(s'), each contributing
        4 pi sqrt(s') a0/(omega_q L^3), where sqrt(s')
        = sqrt(2 m^2 + 2 m omega_q) is that pair's CM energy. In total,

            Delta E_1 = (8 pi a0/(omega_q L^3)) (3 + sqrt(s')),

        which reduces to 40 pi a0/L^3 in the nonrelativistic (large-L)
        limit, the analogue of the ground-state result 12 pi a0/L^3. The
        same coefficient follows from relativistic degenerate perturbation
        theory over the three Fock states (q along x, y, or z), providing
        an independent check of the pair-sum result for this level.
        """
        qcis = QCIndexSpace()
        qcis.populate()
        qcis.fvs.qc_impl['g_uses_prep_mat'] = True
        qcis.fvs.qc_impl['smarter_q_rescale'] = True
        qc = QC(qcis=qcis)
        L = 5.0
        a0 = 0.0001
        delta1 = 1.e-6
        delta2 = 1.e-1
        q = 2.*np.pi/L
        omega_q = np.sqrt(1.+q**2)
        E1_free = 1.+2.*omega_q
        sqrt_s_prime = np.sqrt(2.+2.*omega_q)
        qc_dict = {'k_params': [[[a0]], [0.0]],
                   'project': True,
                   'irrep': ('A1PLUS', 0)}
        root_tmp = root_scalar(qc.get_value, args=(L, qc_dict),
                               bracket=[E1_free+delta1, E1_free+delta2]).root
        result = (root_tmp-E1_free)*omega_q*L**3/8./a0/(3.+sqrt_s_prime)
        expected = np.pi
        self.assertAlmostEqual(result, expected, delta=0.001)

    def test_weakly_interacting_moving_frame_groundstate(self):
        """Test the leading-order shift for nonzero total momentum.

        The lowest non-interacting level with total momentum
        P = (2 pi/L) zhat is built from the momentum set {q, 0, 0} with
        q = (2 pi/L) zhat, giving the free energy E_0 = 2 m + omega_q,
        omega_q = sqrt(m^2 + q^2). It lies in the A1 irrep of the
        little group of P and, being a single Fock configuration, is
        non-degenerate. The expected leading-order (in a0) shift is
        taken from:

        [1] D. M. Grabowska and M. T. Hansen, "Analytic expansions of
            multi-hadron finite-volume energies: I. Two-particle states",
            arXiv:2110.06878.
        [2] D. M. Grabowska and M. T. Hansen, "Analytic Expansions of
            Two- and Three-Particle Excited-State Energies",
            arXiv:2112.11996 (proceedings of LATTICE2021).

        As in test_weakly_interacting_first_excited_state, Eq. (26) of
        Ref. [2] expresses the shift as the sum of the two-particle
        subsystem shifts of Eq. (16). Here the three pairs are: the
        rest-frame threshold pair (0, 0) with E^(0) = 2 m, gamma = 1,
        g_n = 1, contributing 4 pi a0/(m L^3); and the two moving-frame
        ground-state pairs (q, 0) with E^(0) = m + omega_q, g_n = 2,
        gamma = (m + omega_q)/sqrt(s'), each contributing
        4 pi sqrt(s') a0/(omega_q L^3), where sqrt(s')
        = sqrt(2 m^2 + 2 m omega_q) is that pair's CM energy. In total,

            Delta E_0 = (4 pi a0/L^3) (1 + 2 sqrt(s')/omega_q),

        which reduces to 20 pi a0/L^3 in the nonrelativistic (large-L)
        limit, consistent with counting one same-mode pair (4 pi a0/L^3)
        and two distinct-mode pairs (8 pi a0/L^3 each) in first-order
        perturbation theory.
        """
        fvs = FiniteVolumeSetup(nP=np.array([0, 0, 1]))
        qcis = QCIndexSpace(fvs=fvs)
        qcis.populate()
        qcis.fvs.qc_impl['g_uses_prep_mat'] = True
        qcis.fvs.qc_impl['smarter_q_rescale'] = True
        qc = QC(qcis=qcis)
        L = 5.0
        a0 = 0.0001
        delta1 = 1.e-6
        delta2 = 1.e-1
        q = 2.*np.pi/L
        omega_q = np.sqrt(1.+q**2)
        E0_free = 2.+omega_q
        sqrt_s_prime = np.sqrt(2.+2.*omega_q)
        qc_dict = {'k_params': [[[a0]], [0.0]],
                   'project': True,
                   'irrep': ('A1', 0)}
        root_tmp = root_scalar(qc.get_value, args=(L, qc_dict),
                               bracket=[E0_free+delta1, E0_free+delta2]).root
        result = (root_tmp-E0_free)*L**3/4./a0/(1.+2.*sqrt_s_prime/omega_q)
        expected = np.pi
        self.assertAlmostEqual(result, expected, delta=0.001)


if __name__ == '__main__':
    unittest.main()
