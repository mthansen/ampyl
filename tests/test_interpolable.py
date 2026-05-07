#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np

from ampyl.interpolable import _build_matrix_interpolator


class TestInterpolable(unittest.TestCase):
    def test_singleton_energy_axis_uses_cubic_in_volume(self):
        E_grid = np.array([2.0])
        L_grid = np.array([0.0, 1.0, 2.0, 3.0])
        interp_tensor = (L_grid**3).reshape(1, len(L_grid), 1, 1)

        interp = _build_matrix_interpolator(E_grid, L_grid, interp_tensor)

        self.assertAlmostEqual(interp((2.0, 1.5))[0][0], 1.5**3)
        with self.assertRaises(ValueError):
            interp((2.1, 1.5))

    def test_singleton_volume_axis_uses_cubic_in_energy(self):
        E_grid = np.array([0.0, 1.0, 2.0, 3.0])
        L_grid = np.array([5.0])
        interp_tensor = (E_grid**3).reshape(len(E_grid), 1, 1, 1)

        interp = _build_matrix_interpolator(E_grid, L_grid, interp_tensor)

        self.assertAlmostEqual(interp((1.5, 5.0))[0][0], 1.5**3)
        with self.assertRaises(ValueError):
            interp((1.5, 5.1))


if __name__ == '__main__':
    unittest.main()
