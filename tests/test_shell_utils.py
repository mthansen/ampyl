#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_shell_utils.py
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
import warnings
from types import SimpleNamespace
import numpy as np
from ampyl import shell_utils
from ampyl.constants import TWOPI


def _make_qcmatrix(nP, qc_impl=None, alpha=-1.0, beta=0.0,
                   sc_to_three_slice=None):
    """Build a minimal stand-in for a QC matrix element (K, F or G).

    All masses are 1 so the two-particle threshold is 2, and with the
    default alpha = -1, beta = 0 the zero support point vanishes and the
    shell mask reduces to (E-omega_k)^2 - (P-k)^2 > 0.
    """
    sc = SimpleNamespace(
        spectator=SimpleNamespace(mass=1.0),
        first_dimer=SimpleNamespace(mass=1.0),
        second_dimer=SimpleNamespace(mass=1.0))
    qcis = SimpleNamespace(
        fvs=SimpleNamespace(nP=np.array(nP),
                            qc_impl={} if qc_impl is None else qc_impl),
        sc_to_three_slice=([0, 0] if sc_to_three_slice is None
                           else sc_to_three_slice),
        fcs=SimpleNamespace(sc_list_sorted=[sc],
                            slices_by_three_masses=[[0]]),
        tbis=SimpleNamespace(scheme_data=[(alpha, beta)]))
    return SimpleNamespace(qcis=qcis)


def _make_tbks_entry():
    """Three unit-mass momenta at L = 2 pi, grouped into two shells.

    With E = 2.5, nP = (1, 0, 0) and L = 2 pi (so k = n exactly), the
    mask (E-omega_k)^2 - (P-k)^2 > 0 evaluates to
    [True, True, False]: the shell [0, 1] survives and [1, 3] is cut.
    """
    return SimpleNamespace(
        nvec_arr=np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]]),
        nvecSQ_arr=np.array([0, 1, 1]),
        shells=[[0, 1], [1, 3]])


_E = 2.5
_L = TWOPI
_NP_ZERO = [0, 0, 0]
_NP_UNIT = [1, 0, 0]


class TestVerifyIrrepIsKnown(unittest.TestCase):
    """Tests for the irrep sanity check."""

    def _qcis(self):
        return SimpleNamespace(proj_dicts_by_sc_and_shellset=[
            [[{('A1PLUS', 0): 'projector'}],
             [{('EPLUS', 0): 'projector'}]]])

    def test_known_irrep_passes(self):
        shell_utils._verify_irrep_is_known(self._qcis(), ('EPLUS', 0))

    def test_unknown_irrep_raises(self):
        with self.assertRaises(ValueError) as caught:
            shell_utils._verify_irrep_is_known(self._qcis(), ('T1PLUS', 0))
        self.assertIn('T1PLUS', str(caught.exception))
        self.assertIn('A1PLUS', str(caught.exception))


class TestGetZeroSupportPoint(unittest.TestCase):
    """Tests for the zero-support-point evaluation."""

    def test_uses_scheme_data_when_no_attributes(self):
        qcmatrix = _make_qcmatrix(_NP_ZERO, alpha=-1.0, beta=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = shell_utils._get_zero_support_point(qcmatrix, 2.0)
        self.assertAlmostEqual(value, 0.0, places=12)

    def test_uses_alpha_beta_attributes_when_present(self):
        qcmatrix = _make_qcmatrix(_NP_ZERO)
        qcmatrix.alpha = 0.5
        qcmatrix.beta = 0.25
        threshold = 2.0
        expected = (1.5*threshold**2/4.
                    - 0.25*(2.5*threshold**2/4.))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            value = shell_utils._get_zero_support_point(qcmatrix, threshold)
        self.assertAlmostEqual(value, expected, places=12)

    def test_warns_about_hardcoded_scheme(self):
        qcmatrix = _make_qcmatrix(_NP_ZERO)
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            shell_utils._get_zero_support_point(qcmatrix, 2.0)


class TestGetMasksAndShellsForK(unittest.TestCase):
    """Tests for the K-matrix shell selection."""

    def test_rest_frame_returns_shell_without_mask(self):
        k = _make_qcmatrix(_NP_ZERO)
        tbks_entry = _make_tbks_entry()
        mask_slices, slice_entry = shell_utils._get_masks_and_shells_for_k(
            k, _E, _L, tbks_entry, 0, 1)
        self.assertIsNone(mask_slices)
        self.assertEqual(slice_entry, [1, 3])

    def test_moving_frame_masks_shells_above_threshold(self):
        k = _make_qcmatrix(_NP_UNIT)
        tbks_entry = _make_tbks_entry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_slices, slice_entry = \
                shell_utils._get_masks_and_shells_for_k(
                    k, _E, _L, tbks_entry, 0, 0)
        self.assertEqual(mask_slices, [True, False])
        self.assertEqual(list(slice_entry), [0, 1])

    def test_moving_frame_keeps_all_shells_at_high_energy(self):
        k = _make_qcmatrix(_NP_UNIT)
        tbks_entry = _make_tbks_entry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_slices, slice_entry = \
                shell_utils._get_masks_and_shells_for_k(
                    k, 6.0, _L, tbks_entry, 0, 1)
        self.assertEqual(mask_slices, [True, True])
        self.assertEqual(list(slice_entry), [1, 3])


class TestGetMasksAndShellsForF(unittest.TestCase):
    """Tests for the F-matrix shell selection."""

    def test_rest_frame_returns_shell_without_mask(self):
        f = _make_qcmatrix(_NP_ZERO)
        tbks_entry = _make_tbks_entry()
        mask_slices, slice_entry = shell_utils._get_masks_and_shells_for_f(
            f, _E, _L, tbks_entry, 0, 0)
        self.assertIsNone(mask_slices)
        self.assertEqual(slice_entry, [0, 1])

    def test_moving_frame_with_reduce_size_masks_shells(self):
        f = _make_qcmatrix(_NP_UNIT, qc_impl={'reduce_size': True})
        tbks_entry = _make_tbks_entry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_slices, slice_entry = \
                shell_utils._get_masks_and_shells_for_f(
                    f, _E, _L, tbks_entry, 0, 0)
        self.assertEqual(mask_slices, [True, False])
        self.assertEqual(list(slice_entry), [0, 1])

    def test_moving_frame_without_reduce_size_keeps_all_shells(self):
        f = _make_qcmatrix(_NP_UNIT, qc_impl={'reduce_size': False})
        tbks_entry = _make_tbks_entry()
        mask_slices, slice_entry = shell_utils._get_masks_and_shells_for_f(
            f, _E, _L, tbks_entry, 0, 1)
        self.assertEqual(mask_slices, [True, True])
        self.assertEqual(slice_entry, [1, 3])

    def test_moving_frame_defaults_to_reduce_size(self):
        f = _make_qcmatrix(_NP_UNIT)
        tbks_entry = _make_tbks_entry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_slices, _ = shell_utils._get_masks_and_shells_for_f(
                f, _E, _L, tbks_entry, 0, 0)
        self.assertEqual(mask_slices, [True, False])


class TestGetMasksAndShellsForNondiagonal(unittest.TestCase):
    """Tests for the G-matrix (nondiagonal) shell selection."""

    def test_rest_frame_uses_row_and_column_entries(self):
        nondiagonal = _make_qcmatrix(_NP_ZERO)
        row_entry = _make_tbks_entry()
        col_entry = SimpleNamespace(
            nvec_arr=row_entry.nvec_arr,
            nvecSQ_arr=row_entry.nvecSQ_arr,
            shells=[[0, 2], [2, 3]])
        mask_row, mask_col, row_shell, col_shell = \
            shell_utils._get_masks_and_shells_for_nondiagonal(
                nondiagonal, _E, _L, row_entry, 0, 1, 0, 1,
                col_tbks_entry=col_entry)
        self.assertIsNone(mask_row)
        self.assertIsNone(mask_col)
        self.assertEqual(row_shell, [0, 1])
        self.assertEqual(col_shell, [2, 3])

    def test_rest_frame_defaults_column_entry_to_row_entry(self):
        nondiagonal = _make_qcmatrix(_NP_ZERO)
        tbks_entry = _make_tbks_entry()
        _, _, row_shell, col_shell = \
            shell_utils._get_masks_and_shells_for_nondiagonal(
                nondiagonal, _E, _L, tbks_entry, 0, 1, 0, 1)
        self.assertEqual(row_shell, [0, 1])
        self.assertEqual(col_shell, [1, 3])

    def test_helper_nPzero_defaults_column_entry(self):
        tbks_entry = _make_tbks_entry()
        mask_row, mask_col, row_shell, col_shell = \
            shell_utils._mask_and_shell_helper_nPzero(
                None, tbks_entry, 1, 0)
        self.assertIsNone(mask_row)
        self.assertIsNone(mask_col)
        self.assertEqual(row_shell, [1, 3])
        self.assertEqual(col_shell, [0, 1])

    def test_moving_frame_rejects_mismatched_mass_slices(self):
        nondiagonal = _make_qcmatrix(_NP_UNIT, sc_to_three_slice=[0, 1])
        tbks_entry = _make_tbks_entry()
        with self.assertRaises(NotImplementedError):
            shell_utils._get_masks_and_shells_for_nondiagonal(
                nondiagonal, _E, _L, tbks_entry, 0, 1, 0, 0)

    def test_moving_frame_with_reduce_size_masks_rows_and_columns(self):
        nondiagonal = _make_qcmatrix(
            _NP_UNIT, qc_impl={'reduce_size': True})
        tbks_entry = _make_tbks_entry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_row, mask_col, row_shell, col_shell = \
                shell_utils._get_masks_and_shells_for_nondiagonal(
                    nondiagonal, _E, _L, tbks_entry, 0, 1, 0, 0)
        self.assertEqual(mask_row, [True, False])
        self.assertEqual(mask_col, [True, False])
        self.assertEqual(row_shell, [0, 1])
        self.assertEqual(col_shell, [0, 1])

    def test_moving_frame_without_reduce_size_keeps_all_shells(self):
        nondiagonal = _make_qcmatrix(
            _NP_UNIT, qc_impl={'reduce_size': False})
        tbks_entry = _make_tbks_entry()
        mask_row, mask_col, row_shell, col_shell = \
            shell_utils._get_masks_and_shells_for_nondiagonal(
                nondiagonal, _E, _L, tbks_entry, 0, 1, 0, 1)
        self.assertEqual(mask_row, [True, True])
        self.assertEqual(mask_col, [True, True])
        self.assertEqual(row_shell, [0, 1])
        self.assertEqual(col_shell, [1, 3])


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
