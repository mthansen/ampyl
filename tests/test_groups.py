#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_groups.py
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
from types import SimpleNamespace
import numpy as np
from ampyl.groups import Groups

OHP_IRREPS = ['A1PLUS', 'A2PLUS', 'EPLUS', 'T1PLUS', 'T2PLUS',
              'A1MINUS', 'A2MINUS', 'EMINUS', 'T1MINUS', 'T2MINUS']
DIC4_IRREPS = ['A1', 'A2', 'B1', 'B2', 'E2']
DIC2_IRREPS = ['A1', 'A2', 'B1', 'B2']
SPIN_IRREPS = ['G1PLUS', 'G1MINUS', 'G2PLUS', 'G2MINUS', 'HPLUS', 'HMINUS']


def contains_matrix(group, candidate, atol=1.0e-10):
    """Return True if candidate matches an element of group."""
    return bool(np.any(np.all(np.abs(group-candidate) < atol, axis=(1, 2))))


class TestGroupsFixture(unittest.TestCase):
    """Shared Groups instance and a kellm space closed under OhP."""

    @classmethod
    def setUpClass(cls):
        cls.groups = Groups(ell_max=1)
        nvecs = [[nx, ny, nz]
                 for nx in range(-2, 3)
                 for ny in range(-2, 3)
                 for nz in range(-2, 3)
                 if nx**2+ny**2+nz**2 <= 2]
        cls.nvec_arr = np.array(nvecs)
        cls.ellm_set = [[0, 0], [1, -1], [1, 0], [1, 1]]
        cls.space_dim = len(cls.nvec_arr)*len(cls.ellm_set)

    def get_induced_reps(self, group_elems):
        """Induced representation matrices for the fixture kellm space."""
        return [self.groups.generate_induced_rep_kellm(
            self.nvec_arr, self.ellm_set, g_elem) for g_elem in group_elems]


class TestOhPGroupAxioms(TestGroupsFixture):
    """OhP is the full octahedral group with parity, order 48."""

    def test_ohp_elements_are_orthogonal_with_unit_determinant(self):
        for g_elem in self.groups.OhP:
            self.assertTrue(np.allclose(g_elem@g_elem.T, np.identity(3)))
            self.assertAlmostEqual(np.abs(np.linalg.det(g_elem)), 1.0)

    def test_ohp_has_24_proper_and_24_improper_elements(self):
        dets = [round(np.linalg.det(g_elem)) for g_elem in self.groups.OhP]
        self.assertEqual(len(self.groups.OhP), 48)
        self.assertEqual(dets.count(1), 24)
        self.assertEqual(dets.count(-1), 24)

    def test_ohp_elements_are_unique(self):
        for i in range(len(self.groups.OhP)):
            for j in range(i+1, len(self.groups.OhP)):
                self.assertFalse(
                    np.allclose(self.groups.OhP[i], self.groups.OhP[j]))

    def test_ohp_contains_identity_and_inverses(self):
        self.assertTrue(contains_matrix(self.groups.OhP, np.identity(3)))
        for g_elem in self.groups.OhP:
            self.assertTrue(contains_matrix(self.groups.OhP, g_elem.T))

    def test_ohp_is_closed_under_multiplication(self):
        for g_one in self.groups.OhP:
            products = g_one@self.groups.OhP
            for product in products:
                self.assertTrue(contains_matrix(self.groups.OhP, product))


class TestLittleGroups(TestGroupsFixture):
    """Little groups are stabilizer subgroups of OhP."""

    def _check_little_group(self, little_group, nP, expected_order):
        self.assertEqual(len(little_group), expected_order)
        for g_elem in little_group:
            self.assertTrue(contains_matrix(self.groups.OhP, g_elem))
            self.assertTrue(np.allclose(g_elem@nP, nP))
        for g_one in little_group:
            for product in g_one@little_group:
                self.assertTrue(contains_matrix(little_group, product))

    def test_dic4_is_stabilizer_of_001(self):
        nP = np.array([0, 0, 1])
        self._check_little_group(self.groups.Dic4, nP, 8)
        self.assertTrue(np.allclose(self.groups.get_little_group(nP),
                                    self.groups.Dic4))

    def test_dic2_is_stabilizer_of_011(self):
        nP = np.array([0, 1, 1])
        self._check_little_group(self.groups.Dic2, nP, 4)
        self.assertTrue(np.allclose(self.groups.get_little_group(nP),
                                    self.groups.Dic2))

    def test_generic_little_group_stabilizes_111(self):
        nP = np.array([1, 1, 1])
        little_group = self.groups.get_little_group(nP)
        self._check_little_group(little_group, nP, 6)


class TestCharacterTables(TestGroupsFixture):
    """Character rows satisfy the standard orthogonality relations."""

    def _check_character_table(self, group_str, irreps, order):
        dims = {}
        for irrep in irreps:
            chars = self.groups.chardict[f'{group_str}_{irrep}']
            for row in chars[1:]:
                self.assertTrue(np.allclose(row, chars[0]))
            dim = len(chars)
            dims[irrep] = dim
            self.assertAlmostEqual(chars[0][0], dim)
        self.assertEqual(sum(dim**2 for dim in dims.values()), order)
        for i, irrep_one in enumerate(irreps):
            char_one = self.groups.chardict[f'{group_str}_{irrep_one}'][0]
            for irrep_two in irreps[i:]:
                char_two =\
                    self.groups.chardict[f'{group_str}_{irrep_two}'][0]
                inner = np.sum(char_one*np.conjugate(char_two))
                expected = order if irrep_one == irrep_two else 0.
                self.assertAlmostEqual(np.abs(inner-expected), 0.)

    def test_ohp_character_table(self):
        self._check_character_table('OhP', OHP_IRREPS, 48)

    def test_dic4_character_table(self):
        self._check_character_table('Dic4', DIC4_IRREPS, 8)

    def test_dic2_character_table(self):
        self._check_character_table('Dic2', DIC2_IRREPS, 4)


class TestDoubleCover(TestGroupsFixture):
    """The double cover of OhP and its spin-half characters."""

    def test_double_cover_has_96_unitary_elements_48_unique(self):
        # Parity acts trivially on spinors in the PLUS convention, so an
        # improper element shares the 2x2 matrix of its proper part: the
        # 96 elements consist of 48 distinct matrices, each appearing
        # exactly twice.
        double_group = np.array(self.groups.OhP_double_PLUS)
        self.assertEqual(len(double_group), 96)
        for g_elem in double_group:
            self.assertTrue(np.allclose(g_elem@np.conjugate(g_elem.T),
                                        np.identity(2)))
        counts = [int(np.sum(np.all(np.abs(double_group-g_elem) < 1.0e-10,
                                    axis=(1, 2))))
                  for g_elem in double_group]
        self.assertEqual(counts, [2]*96)

    def test_double_cover_pairs_are_negatives(self):
        # Elements are appended in pairs (U, U rotated by an extra 2*pi),
        # which differ by a sign in the spinor representation.
        for name in ['OhP_double_PLUS', 'OhP_double_MINUS']:
            double_group = getattr(self.groups, name)
            for i in range(0, len(double_group), 2):
                self.assertTrue(np.allclose(double_group[i],
                                            -double_group[i+1]),
                                msg=f"{name} pair {i}")

    def test_double_cover_spinor_rotation_homomorphism(self):
        # For each element, the proper rotation part R and the spinor
        # matrix U must satisfy R_ab = Re tr(sigma_a U sigma_b U^dag)/2.
        sigma = [np.array([[0., 1.], [1., 0.]]),
                 np.array([[0., -1j], [1j, 0.]]),
                 np.array([[1., 0.], [0., -1.]])]
        double_group = self.groups.OhP_double_PLUS
        for i, ohp_elem in enumerate(self.groups.OhP):
            rotation = np.linalg.det(ohp_elem)*ohp_elem
            spinor = double_group[2*i]
            recovered = np.zeros((3, 3))
            for a in range(3):
                for b in range(3):
                    recovered[a, b] = np.real(np.trace(
                        sigma[a]@spinor@sigma[b]@np.conjugate(
                            spinor.T)))/2.
            self.assertTrue(np.allclose(recovered, rotation, atol=1.0e-10),
                            msg=f"element {i} spinor does not reproduce "
                            "its rotation")

    def test_double_cover_is_closed(self):
        double_group = np.array(self.groups.OhP_double_PLUS)
        for g_one in double_group:
            products = g_one@double_group
            for product in products:
                self.assertTrue(
                    contains_matrix(double_group, product, atol=1.0e-8))

    def test_double_cover_intspin_alignment(self):
        # PLUS intspin stores the proper part of each OhP element; MINUS
        # intspin stores the OhP element itself. Both are aligned so that
        # entries 2*i and 2*i+1 correspond to OhP element i.
        intspin_plus = np.array(self.groups.OhP_double_PLUS_intspin)
        intspin_minus = np.array(self.groups.OhP_double_MINUS_intspin)
        for i, ohp_elem in enumerate(self.groups.OhP):
            proper_part = np.linalg.det(ohp_elem)*ohp_elem
            for offset in range(2):
                self.assertTrue(np.allclose(intspin_plus[2*i+offset],
                                            proper_part))
                self.assertTrue(np.allclose(intspin_minus[2*i+offset],
                                            ohp_elem))

    def test_g1_characters_match_double_cover_traces(self):
        double_group = np.array(self.groups.OhP_double_PLUS)
        traces = np.array([np.trace(g_elem) for g_elem in double_group])
        char = self.groups.chardict['OhP_G1PLUS'][0]
        self.assertTrue(np.allclose(char, traces))

    def test_spin_half_character_orthogonality(self):
        for i, irrep_one in enumerate(SPIN_IRREPS):
            char_one = self.groups.chardict[f'OhP_{irrep_one}'][0]
            self.assertEqual(len(char_one), 96)
            dim = len(self.groups.chardict[f'OhP_{irrep_one}'])
            self.assertAlmostEqual(char_one[0], dim)
            for irrep_two in SPIN_IRREPS[i:]:
                char_two = self.groups.chardict[f'OhP_{irrep_two}'][0]
                inner = np.sum(char_one*np.conjugate(char_two))
                expected = 96. if irrep_one == irrep_two else 0.
                self.assertAlmostEqual(np.abs(inner-expected), 0.)


class TestKellmProjectors(TestGroupsFixture):
    """Character projectors on the kellm space are a complete orthogonal
    set of hermitian idempotents; bTdict row operators live inside the
    matching isotypic subspace."""

    EXPECTED_OHP_MULTIPLICITIES = {
        'A1PLUS': 5, 'A2PLUS': 1, 'EPLUS': 5, 'T1PLUS': 3, 'T2PLUS': 4,
        'A1MINUS': 0, 'A2MINUS': 1, 'EMINUS': 1, 'T1MINUS': 8, 'T2MINUS': 4}

    def _character_projectors(self, group_str, nP, irreps):
        little_group = self.groups.get_little_group(nP)
        order = len(little_group)
        induced_reps = self.get_induced_reps(little_group)
        projectors = {}
        for irrep in irreps:
            char = self.groups.chardict[f'{group_str}_{irrep}'][0]
            dim = len(self.groups.chardict[f'{group_str}_{irrep}'])
            projector = sum(induced_reps[i]*np.conjugate(char[i])
                            for i in range(order))*(dim/order)
            projectors[irrep] = (projector, dim)
        return projectors

    def _check_group(self, group_str, nP, irreps,
                     expected_multiplicities=None):
        projectors = self._character_projectors(group_str, nP, irreps)
        total = np.zeros((self.space_dim, self.space_dim), dtype=complex)
        covered = 0
        for irrep, (projector, dim) in projectors.items():
            self.assertLess(
                np.max(np.abs(projector-np.conjugate(projector.T))),
                1.0e-10, msg=f"{group_str}_{irrep} not hermitian")
            self.assertLess(
                np.max(np.abs(projector@projector-projector)),
                1.0e-10, msg=f"{group_str}_{irrep} not idempotent")
            multiplicity = np.trace(projector).real/dim
            self.assertAlmostEqual(
                multiplicity, round(multiplicity), places=8,
                msg=f"{group_str}_{irrep} multiplicity not an integer")
            covered += dim*round(multiplicity)
            total = total+projector
            if expected_multiplicities is not None:
                self.assertEqual(round(multiplicity),
                                 expected_multiplicities[irrep])
        self.assertLess(
            np.max(np.abs(total-np.identity(self.space_dim))), 1.0e-10,
            msg=f"{group_str} projectors do not sum to the identity")
        self.assertEqual(covered, self.space_dim)
        self._check_bT_rows(group_str, nP, projectors)

    def _check_bT_rows(self, group_str, nP, projectors):
        for irrep, (char_projector, dim) in projectors.items():
            multiplicity = round(np.trace(char_projector).real/dim)
            bT_rows = self.groups.bTdict[f'{group_str}_{irrep}']
            for irrep_row in range(len(bT_rows)):
                row_op = self.groups.get_kellm_proj(
                    nP=nP, irrep=irrep, irrep_row=irrep_row,
                    nvec_arr=self.nvec_arr, ellm_set=self.ellm_set)
                scale = np.max(np.abs(row_op))
                if scale < 1.0e-10:
                    self.assertEqual(
                        multiplicity, 0,
                        msg=(f"{group_str}_{irrep} row {irrep_row} vanishes "
                             "although the irrep appears in the space"))
                    continue
                self.assertLess(
                    np.max(np.abs(char_projector@row_op-row_op))/scale,
                    1.0e-8,
                    msg=(f"{group_str}_{irrep} row {irrep_row} not in "
                         "its isotypic subspace"))
                row_sq = row_op@row_op
                overlap = np.sum(np.conjugate(row_op)*row_sq)
                norm_sq = np.sum(np.abs(row_op)**2)
                scalar = overlap/norm_sq
                self.assertLess(
                    np.max(np.abs(row_sq-scalar*row_op))/scale**2,
                    1.0e-8,
                    msg=(f"{group_str}_{irrep} row {irrep_row} squared "
                         "is not proportional to itself"))
                self.assertGreater(
                    np.abs(scalar), 1.0e-6*scale,
                    msg=(f"{group_str}_{irrep} row {irrep_row} is "
                         "nilpotent: not a valid row projector"))
                eigvals = np.linalg.eigvals(row_op)
                rank = int(np.sum(np.abs(eigvals) > 1.0e-6*np.abs(scalar)))
                self.assertEqual(
                    rank, multiplicity,
                    msg=(f"{group_str}_{irrep} row {irrep_row} has rank "
                         f"{rank} but the irrep multiplicity is "
                         f"{multiplicity}"))

    def test_ohp_kellm_projectors(self):
        self._check_group('OhP', np.array([0, 0, 0]), OHP_IRREPS,
                          self.EXPECTED_OHP_MULTIPLICITIES)

    def test_dic4_kellm_projectors(self):
        self._check_group('Dic4', np.array([0, 0, 1]), DIC4_IRREPS)

    def test_dic2_kellm_projectors(self):
        self._check_group('Dic2', np.array([0, 1, 1]), DIC2_IRREPS)


class TestErrorHandlingRegressions(TestGroupsFixture):
    """Regression tests for error-path bugs in projector entry points."""

    def test_get_fixed_sc_proj_dict_raises_on_none_qcis(self):
        with self.assertRaises(ValueError):
            self.groups.get_fixed_sc_proj_dict(qcis=None)

    def test_get_fixed_sc_proj_dict_raises_on_unsupported_momentum(self):
        fake_qcis = SimpleNamespace(
            verbosity=0,
            fvs=SimpleNamespace(nP=np.array([0, 0, 2]), irrep_set=[]))
        with self.assertRaises(ValueError):
            self.groups.get_fixed_sc_proj_dict(qcis=fake_qcis)

    def test_get_kellm_proj_raises_on_unsupported_momentum(self):
        with self.assertRaises(ValueError):
            self.groups.get_kellm_proj(nP=np.array([0, 0, 2]),
                                       irrep='A1PLUS', irrep_row=0,
                                       nvec_arr=self.nvec_arr,
                                       ellm_set=self.ellm_set)


if __name__ == '__main__':
    unittest.main()
