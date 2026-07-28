#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import warnings
import numpy as np
from ampyl.kinematic_functions import standard_boost
from ampyl.kinematic_functions import standard_boost_array


class TestStandardBoostArray(unittest.TestCase):
    """standard_boost_array must match standard_boost entry by entry.

    Regression: the array version previously computed beta**2 from entry
    (0, 0) only and broadcast it, which was correct only when every
    entry shared the same boost magnitude (as shell-blocked G-matrix
    calls happen to guarantee), and it dropped the scalar version's
    superluminal guard, producing NaN instead of zeros."""

    # A 2x3 grid of boost velocities with deliberately mixed magnitudes,
    # a vanishing entry, and a superluminal entry.
    BETA_GRID = np.array([
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.3, 0.4, 0.0]],
        [[0.0, 0.0, 0.9], [-0.2, 0.5, 0.1], [1.2, 0.0, 0.0]]])

    FOURMOM_GRID = np.array([
        [[1.0, 0.0, 0.0, 0.0], [2.0, 0.3, -0.1, 0.7], [1.5, 1.0, 1.0, 1.0]],
        [[3.0, -1.0, 0.5, 0.2], [1.1, 0.0, 0.4, -0.3], [2.5, 0.6, 0.6, 0.6]]])

    def test_matches_scalar_boost_elementwise(self):
        boosted = standard_boost_array(self.BETA_GRID, self.FOURMOM_GRID)
        for i in range(self.BETA_GRID.shape[0]):
            for j in range(self.BETA_GRID.shape[1]):
                expected = standard_boost(self.BETA_GRID[i][j],
                                          self.FOURMOM_GRID[i][j])
                self.assertTrue(
                    np.allclose(boosted[i][j], expected, atol=1.0e-14),
                    msg=(f"entry ({i}, {j}): got {boosted[i][j]}, "
                         f"expected {expected}"))

    def test_zero_beta_entries_pass_through_unchanged(self):
        boosted = standard_boost_array(self.BETA_GRID, self.FOURMOM_GRID)
        self.assertTrue(np.array_equal(boosted[0][0],
                                       self.FOURMOM_GRID[0][0]))

    def test_superluminal_entries_are_zeroed_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            boosted = standard_boost_array(self.BETA_GRID,
                                           self.FOURMOM_GRID)
        self.assertTrue(np.all(boosted[1][2] == 0.0))
        self.assertTrue(np.all(np.isfinite(boosted)))

    def test_uniform_magnitude_grid_matches_scalar(self):
        # The regime the old implementation was correct in: every entry
        # shares one magnitude but points along different axes.
        beta_grid = np.array([
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]],
            [[0.0, 0.0, -0.5], [0.3, 0.4, 0.0]]])
        fourmom_grid = self.FOURMOM_GRID[:, :2]
        boosted = standard_boost_array(beta_grid, fourmom_grid)
        for i in range(2):
            for j in range(2):
                expected = standard_boost(beta_grid[i][j],
                                          fourmom_grid[i][j])
                self.assertTrue(
                    np.allclose(boosted[i][j], expected, atol=1.0e-14))


if __name__ == '__main__':
    unittest.main()
