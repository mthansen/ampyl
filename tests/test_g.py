#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from itertools import permutations, product
from types import SimpleNamespace
import numpy as np
from ampyl.spaces import ThreeBodyKinematicSpace
from ampyl import qc_functions


class TestGArrayVariantsAgree(unittest.TestCase):
    """The two G-block builders must produce identical output.

    getG_array slices shell blocks out of the full TBKS arrays on the
    fly; getG_array_prep_mat reads the shell blocks precomputed by
    ThreeBodyKinematicSpace. They are alternate data paths into the same
    computation, selected by qc_impl['g_uses_prep_mat'], and historically
    diverged when the below-cutoff zero-guard was added to only one of
    them."""

    @classmethod
    def setUpClass(cls):
        nvecs = [[nx, ny, nz]
                 for nx in range(-1, 2)
                 for ny in range(-1, 2)
                 for nz in range(-1, 2)
                 if nx**2+ny**2+nz**2 <= 1]
        cls.tbks = ThreeBodyKinematicSpace(nvec_arr=np.array(nvecs))
        cls.nP = np.array([0, 0, 0])
        cls.L = 5.0

    def _compare_all_shell_pairs(self, E, masses, ell, three_scheme):
        m1, m2, m3 = masses
        blocks = []
        for row_index, row_shell in enumerate(self.tbks.shells):
            for col_index, col_shell in enumerate(self.tbks.shells):
                g_direct = qc_functions.getG_array(
                    E, self.nP, self.L, m1, m2, m3, self.tbks,
                    row_shell, col_shell, ell, ell, -1.0, 0.0,
                    {}, three_scheme, 1.0)
                g_prep = qc_functions.getG_array_prep_mat(
                    E, self.nP, self.L, m1, m2, m3, self.tbks,
                    row_index, col_index, ell, ell, -1.0, 0.0,
                    {}, three_scheme, 1.0)
                self.assertTrue(
                    np.array_equal(g_direct, g_prep),
                    msg=(f"shell pair ({row_index}, {col_index}) differs "
                         f"at E={E}, masses={masses}, ell={ell}, "
                         f"scheme={three_scheme}"))
                blocks.append(g_direct)
        return blocks

    def test_agreement_above_cutoff(self):
        blocks = self._compare_all_shell_pairs(
            4.2, (1.0, 1.0, 1.0), 0, 'original pole')
        self.assertTrue(any(np.any(block != 0.) for block in blocks))

    def test_agreement_nondegenerate_masses_and_ell_one(self):
        blocks = self._compare_all_shell_pairs(
            4.5, (1.0, 1.0, 1.3), 1, 'original pole')
        self.assertTrue(any(np.any(block != 0.) for block in blocks))

    def test_agreement_relativistic_pole_scheme(self):
        blocks = self._compare_all_shell_pairs(
            4.2, (1.0, 1.0, 1.0), 0, 'relativistic pole')
        self.assertTrue(any(np.any(block != 0.) for block in blocks))

    def test_agreement_in_cutoff_transition_region(self):
        # E = 2.05 puts the zero-momentum shell inside the smooth-cutoff
        # transition (H ~ 0.4), so blocks are nonzero but suppressed.
        blocks = self._compare_all_shell_pairs(
            2.05, (1.0, 1.0, 1.0), 0, 'original pole')
        self.assertTrue(any(np.any(block != 0.) for block in blocks))

    def test_below_cutoff_blocks_are_exactly_zero_in_both_paths(self):
        # At E = 1.2 every shell has H exactly zero or underflowing
        # (~1e-17), so the zero-guard must fire in both paths.
        # Regression: getG_array_prep_mat previously lacked the guard and
        # returned cutoff-suppressed junk instead of exact zeros.
        blocks = self._compare_all_shell_pairs(
            1.2, (1.0, 1.0, 1.0), 0, 'original pole')
        for block in blocks:
            self.assertTrue(np.all(block == 0.))


class TestGSingleEntryMatchesArray(unittest.TestCase):
    """getG_single_entry must reproduce getG_array entrywise.

    Regression: it indexed calY's (calY, calYconj) pair before
    unpacking, which raised on every call. Covers nonzero total
    momentum, unequal masses in every role, ell = 1 harmonics, both
    pole schemes and distinct row- and column-side cutoffs (alpha2,
    beta2)."""

    def test_entrywise_agreement(self):
        nvecs = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, -1, 0],
                          [0, 0, -1], [1, 1, 0], [1, 0, 1], [0, -1, 1]])
        tbks = SimpleNamespace(nvec_arr=nvecs,
                               nvecSQ_arr=(nvecs**2).sum(1))
        m_K = 1.4
        L = 4.6
        E = 2.0*m_K+1.9
        alpha, beta, alpha2, beta2 = -0.6, 0.02, -0.35, -0.01
        full = [0, len(nvecs)]
        for nP in [np.array([0, 0, 1]), np.array([0, 1, 1])]:
            for m1, m2, m3 in set(permutations([m_K, m_K, 1.0])):
                for ell, scheme in product(
                        [0, 1], ['relativistic pole', 'original pole']):
                    g_array = qc_functions.getG_array(
                        E, nP, L, m1, m2, m3, tbks, full, full, ell, ell,
                        alpha, beta, {}, scheme, 1.0,
                        alpha2=alpha2, beta2=beta2)
                    scale = np.max(np.abs(g_array))
                    self.assertGreater(scale, 0.0)
                    d = 2*ell+1
                    for i, j, mi, mj in product(range(len(nvecs)),
                                                range(len(nvecs)),
                                                range(d), range(d)):
                        g_entry = qc_functions.getG_single_entry(
                            E, nP, L, nvecs[i], nvecs[j],
                            ell, mi-ell, ell, mj-ell, m1, m2, m3,
                            alpha, beta, three_scheme=scheme,
                            alpha2=alpha2, beta2=beta2)
                        self.assertLess(
                            abs(g_entry-g_array[i*d+mi, j*d+mj]),
                            1.0e-13*scale)


class TestGOriginalPoleNormalization(unittest.TestCase):
    """The original pole puts the exchanged particle on shell.

    The relativistic pole carries 1/(2 w1 L^3 (E-w1-w3+w2)); the
    original pole replaces E-w1-w3+w2 by its on-shell value 2 w2, so
    G_orig/G_rel = (E-w1-w3+w2)/(2 w2) entrywise, which tends to 1 at
    the pole. Regression: getG_array used 1/(2 w1 w2 L^3), twice the
    correct 1/(4 w1 w2 L^3)."""

    def test_ratio_to_relativistic_pole(self):
        nvecs = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, -1, 0],
                          [1, 1, 0], [1, 0, 1]])
        tbks = SimpleNamespace(nvec_arr=nvecs,
                               nvecSQ_arr=(nvecs**2).sum(1))
        m1, m2, m3 = 1.4, 1.0, 1.2
        nP = np.array([0, 0, 1])
        L = 4.6
        E = 5.4
        full = [0, len(nvecs)]
        g = {}
        for scheme in ['relativistic pole', 'original pole']:
            g[scheme] = qc_functions.getG_array(
                E, nP, L, m1, m2, m3, tbks, full, full, 0, 0,
                -0.6, 0.0, {}, scheme, 1.0)
        P = 2.0*np.pi*nP/L
        for i, j in product(range(len(nvecs)), repeat=2):
            p3 = 2.0*np.pi*nvecs[i]/L
            p1 = 2.0*np.pi*nvecs[j]/L
            p2 = P-p1-p3
            omega1 = np.sqrt(m1**2+p1@p1)
            omega2 = np.sqrt(m2**2+p2@p2)
            omega3 = np.sqrt(m3**2+p3@p3)
            g_rel = g['relativistic pole'][i, j]
            if g_rel == 0.0:
                continue
            self.assertAlmostEqual(
                g['original pole'][i, j]/g_rel,
                (E-omega1-omega3+omega2)/(2.0*omega2), places=13)


if __name__ == '__main__':
    unittest.main()
