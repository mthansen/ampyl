#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from types import SimpleNamespace

from ampyl.interpolable import Interpolable
from ampyl.interpolable import _build_matrix_interpolator
from ampyl import interpolable_utils


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

    def test_strict_pole_residue_matrices_restore_other_poles(self):
        irrep = ('A1', 0)

        def smooth_interp(point):
            E, L = point
            return np.array([
                [E+L, 2.*E],
                [3.*L, E-L],
            ])

        interpolable = SimpleNamespace(
            interps={irrep: smooth_interp},
            pole_lists={irrep: [np.array([[0, 0, 0], [0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.],
                                               [2., 2., 2.]])]},
            pole_textures_lists={irrep: [np.array([
                [[1., 1.],
                 [0., 0.]],
                [[0., 1.],
                 [0., 1.]],
            ])]},
        )

        residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep)

        self.assertEqual(len(residues), 1)
        np.testing.assert_allclose(
            residues[0][0],
            [[5., -2.],
             [0., 0.]])
        np.testing.assert_allclose(
            residues[0][1],
            [[0., 4.],
             [0., 4.]])

    def test_strict_pole_residue_matrices_rotate_to_standard_basis(self):
        irrep = ('A1', 0)
        cob_matrix = np.array([
            [0., 1.],
            [1., 0.],
            [0., 0.],
        ])
        interpolable = SimpleNamespace(
            interps={irrep: lambda point: np.array([
                [1., 2.],
                [3., 4.],
            ])},
            cob_matrix_lists={irrep: [cob_matrix]},
            pole_lists={irrep: [np.array([[0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.]])]},
            pole_textures_lists={irrep: [np.array([[
                [1., 1.],
                [1., 1.],
            ]])]},
        )

        residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep)

        expected_smooth_residue = np.array([
            [1., 2.],
            [3., 4.],
        ])
        expected_standard_residue = (
            cob_matrix@expected_smooth_residue@cob_matrix.T
        )
        np.testing.assert_allclose(residues[0][0],
                                   expected_standard_residue)

    def test_strict_pole_residue_matrices_can_stay_in_smooth_basis(self):
        irrep = ('A1', 0)
        cob_matrix = np.array([
            [0., 1.],
            [1., 0.],
            [0., 0.],
        ])
        interpolable = SimpleNamespace(
            interps={irrep: lambda point: np.array([
                [1., 2.],
                [3., 4.],
            ])},
            cob_matrix_lists={irrep: [cob_matrix]},
            pole_lists={irrep: [np.array([[0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.]])]},
            pole_textures_lists={irrep: [np.array([[
                [1., 1.],
                [1., 1.],
            ]])]},
        )

        residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep, basis='smooth')

        np.testing.assert_allclose(
            residues[0][0],
            [[1., 2.],
             [3., 4.]])

    def test_strict_pole_residue_warns_on_overlapping_coincident_poles(self):
        irrep = ('A1', 0)
        interpolable = SimpleNamespace(
            interps={irrep: lambda point: np.ones((1, 1))},
            pole_lists={irrep: [np.array([[0, 0, 0], [0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.],
                                               [1., 1., 1.]])]},
            pole_textures_lists={irrep: [np.array([[[1.]], [[1.]]])]},
        )

        with self.assertWarnsRegex(
                UserWarning,
                "coincident poles share at least one matrix entry"):
            residues = interpolable_utils._get_pole_residue_matrix_list(
                interpolable, 2., irrep)
        np.testing.assert_allclose(residues[0], [[[1.]], [[1.]]])

    def test_public_pole_residue_method_caches_by_volume(self):
        irrep = ('A1', 0)
        interpolable = Interpolable()
        interpolable.interps[irrep] = lambda point: np.ones((1, 1))
        interpolable.pole_lists[irrep] = [np.array([[0, 0, 0]])]
        interpolable.pole_mass_lists[irrep] = [np.array([[1., 1., 1.]])]
        interpolable.pole_textures_lists[irrep] = [np.array([[[1.]]])]
        interpolable._store_interpolator(name='toy')

        residues = interpolable.get_pole_residue_matrices(
            L=2., irrep=irrep, interpolator_name='toy')

        np.testing.assert_allclose(residues[0][0], [[1.]])
        self.assertIn(2., interpolable.pole_residue_matrix_lists[irrep])
        self.assertIn(
            2.,
            interpolable.interpolators[0]['pole_residue_matrix_lists'][irrep])

    def test_public_pole_residue_method_rejects_unknown_basis(self):
        interpolable = Interpolable()

        with self.assertRaises(ValueError):
            interpolable.get_pole_residue_matrices(
                L=2., irrep=('A1', 0), basis='diagonal')

    def test_pole_residue_cache_separates_standard_and_smooth_basis(self):
        irrep = ('A1', 0)
        cob_matrix = np.array([
            [0., 1.],
            [1., 0.],
            [0., 0.],
        ])
        interpolable = Interpolable()
        interpolable.interps[irrep] = lambda point: np.array([
            [1., 2.],
            [3., 4.],
        ])
        interpolable.cob_matrix_lists[irrep] = [cob_matrix]
        interpolable.pole_lists[irrep] = [np.array([[0, 0, 0]])]
        interpolable.pole_mass_lists[irrep] = [np.array([[1., 1., 1.]])]
        interpolable.pole_textures_lists[irrep] = [np.array([[
            [1., 1.],
            [1., 1.],
        ]])]
        interpolable._store_interpolator(name='toy')

        standard_residues = interpolable.get_pole_residue_matrices(
            L=2., irrep=irrep, interpolator_name='toy')
        smooth_residues = interpolable.get_pole_residue_matrices(
            L=2., irrep=irrep, interpolator_name='toy', basis='smooth')

        self.assertIn(2., interpolable.pole_residue_matrix_lists[irrep])
        self.assertIn(('smooth', 2.),
                      interpolable.pole_residue_matrix_lists[irrep])
        self.assertEqual(standard_residues[0][0].shape, (3, 3))
        self.assertEqual(smooth_residues[0][0].shape, (2, 2))

    def test_pole_residue_energy_range_skips_out_of_range_poles(self):
        irrep = ('A1', 0)

        class FakeInterp:
            grid = (np.array([2.5, 4.0]), np.array([2.0]))

            def __call__(self, point):
                E, L = point
                if E > 4.0:
                    raise ValueError("out-of-range pole was evaluated")
                return np.array([[E+L]])

        interpolable = Interpolable()
        interpolable.interps[irrep] = FakeInterp()
        interpolable.pole_lists[irrep] = [
            np.array([[0, 0, 0], [0, 0, 0]])
        ]
        interpolable.pole_mass_lists[irrep] = [
            np.array([[1., 1., 1.], [2., 2., 2.]])
        ]
        interpolable.pole_textures_lists[irrep] = [
            np.array([[[1.]], [[1.]]])
        ]
        interpolable._store_interpolator(name='toy')

        residues = interpolable.get_pole_residue_matrices(
            L=2., irrep=irrep, interpolator_name='toy',
            E_range=[2.5, 6.5])

        np.testing.assert_allclose(residues[0][0], [[-5./3.]])
        np.testing.assert_allclose(residues[0][1], [[0.]])
        cache_key = (2., 2.5, 6.5)
        self.assertIn(cache_key,
                      interpolable.pole_residue_matrix_lists[irrep])

    def test_pole_residue_energy_range_keeps_endpoint_poles(self):
        irrep = ('A1', 0)
        calls = []

        def fake_interp(point):
            E, L = point
            calls.append(float(E))
            return np.array([[E+L]])

        interpolable = SimpleNamespace(
            interps={irrep: fake_interp},
            pole_lists={irrep: [np.array([[0, 0, 0], [0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.],
                                               [2., 2., 2.]])]},
            pole_textures_lists={irrep: [np.array([[[1.]], [[1.]]])]},
            cob_matrix_lists={},
        )

        residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep, E_range=[3., 6.])

        np.testing.assert_allclose(calls, [3., 6.])
        np.testing.assert_allclose(residues[0][0], [[-5./3.]])
        np.testing.assert_allclose(residues[0][1], [[8./3.]])

    def test_pole_residue_energy_range_zero_shape_matches_basis(self):
        irrep = ('A1', 0)
        cob_matrix = np.array([
            [0., 1.],
            [1., 0.],
            [0., 0.],
        ])

        def fail_if_called(point):
            raise AssertionError("out-of-range pole was evaluated")

        interpolable = SimpleNamespace(
            interps={irrep: fail_if_called},
            cob_matrix_lists={irrep: [cob_matrix]},
            pole_lists={irrep: [np.array([[0, 0, 0]])]},
            pole_mass_lists={irrep: [np.array([[1., 1., 1.]])]},
            pole_textures_lists={irrep: [np.array([[
                [1., 1.],
                [1., 1.],
            ]])]},
        )

        standard_residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep, E_range=[4., 5.])
        smooth_residues = interpolable_utils._get_pole_residue_matrix_list(
            interpolable, 2., irrep, E_range=[4., 5.], basis='smooth')

        self.assertEqual(standard_residues[0][0].shape, (3, 3))
        self.assertEqual(smooth_residues[0][0].shape, (2, 2))
        np.testing.assert_allclose(standard_residues[0][0], np.zeros((3, 3)))
        np.testing.assert_allclose(smooth_residues[0][0], np.zeros((2, 2)))


if __name__ == '__main__':
    unittest.main()
