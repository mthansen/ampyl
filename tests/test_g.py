#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
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


if __name__ == '__main__':
    unittest.main()
