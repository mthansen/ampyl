#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_groups_nonint.py
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
from ampyl import groups as groups_module
from ampyl.groups import Groups

OHP_IRREPS = ['A1PLUS', 'A2PLUS', 'EPLUS', 'T1PLUS', 'T2PLUS',
              'A1MINUS', 'A2MINUS', 'EMINUS', 'T1MINUS', 'T2MINUS']
DIC4_IRREPS = ['A1', 'A2', 'B1', 'B2', 'E2']
DIC2_IRREPS = ['A1', 'A2', 'B1', 'B2']
SPIN_IRREPS = ['G1PLUS', 'G1MINUS', 'G2PLUS', 'G2MINUS', 'HPLUS', 'HMINUS']

ZERO_TRIPLE = np.zeros((1, 3, 3), dtype=int)
EMPTY_TRIPLE = np.zeros((0, 3, 3), dtype=int)
DIC4_TRIPLE = np.array([[[0, 0, 1], [0, 0, 0], [0, 0, 0]]])
DIC2_TRIPLE = np.array([[[0, 1, 1], [0, 0, 0], [0, 0, 0]]])
ZERO_PAIR = np.zeros((1, 2, 3), dtype=int)
DIC4_PAIR = np.array([[[0, 0, 1], [0, 0, 0]]])
DIC2_PAIR = np.array([[[0, 1, 1], [0, 0, 0]]])

NP_REST = np.array([0, 0, 0])
NP_DIC4 = np.array([0, 0, 1])
NP_DIC2 = np.array([0, 1, 1])
NP_BAD = np.array([1, 1, 1])

KELLM_ELLM_SET = [[0, 0], [1, -1], [1, 0], [1, 1]]

_GROUPS_CACHE = {}


def get_groups(spin_half=False):
    """Cache the two Groups instances across the whole module."""
    if spin_half not in _GROUPS_CACHE:
        _GROUPS_CACHE[spin_half] = Groups(ell_max=1, spin_half=spin_half)
    return _GROUPS_CACHE[spin_half]


def closed_nvec_arr():
    """All integer vectors with squared norm at most 2 (closed under OhP).

    The origin comes first so that the single-vector slice [0:1] used by
    the kellm_shell tests is itself closed under every little group.
    """
    nvecs = [[nx, ny, nz]
             for nx in range(-2, 3)
             for ny in range(-2, 3)
             for nz in range(-2, 3)
             if 0 < nx**2+ny**2+nz**2 <= 2]
    return np.array([[0, 0, 0]]+nvecs)


def make_kellm_qcis(nP, irrep_set):
    """Fake qcis carrying just what the kellm dict entry points read."""
    nvec_arr = closed_nvec_arr()
    space = np.zeros(len(nvec_arr)*len(KELLM_ELLM_SET))
    return SimpleNamespace(
        verbosity=2,
        fvs=SimpleNamespace(nP=nP, irrep_set=irrep_set),
        sc_to_three_slice=[0],
        tbks_list=[[SimpleNamespace(nvec_arr=nvec_arr)]],
        ellm_sets=[KELLM_ELLM_SET],
        kellm_spaces=[[space]],
        n_channels=1)


class ProjectorAssertions(unittest.TestCase):
    """Shared checks for character-projector families."""

    def check_family(self, groups, group_str, irreps, order, space_dim,
                     raw_proj_getter, expected_multiplicities):
        """Verify hermiticity, idempotency, multiplicities, completeness.

        raw_proj_getter(irrep) must return the raw character sum
        sum_g chi(g)* R(g); the normalized projector is dim/order times
        that. The projectors must resolve the identity on the space.
        """
        total = np.zeros((space_dim, space_dim), dtype=complex)
        for irrep in irreps:
            raw = raw_proj_getter(irrep)
            dim = len(groups.chardict[f'{group_str}_{irrep}'])
            proj = np.array(raw, dtype=complex)*dim/order
            self.assertLess(
                np.max(np.abs(proj-np.conjugate(proj.T))), 1.0e-10,
                msg=f"{group_str}_{irrep} projector not hermitian")
            self.assertLess(
                np.max(np.abs(proj@proj-proj)), 1.0e-10,
                msg=f"{group_str}_{irrep} projector not idempotent")
            multiplicity = np.trace(proj).real/dim
            self.assertAlmostEqual(
                multiplicity, expected_multiplicities.get(irrep, 0),
                places=8,
                msg=f"{group_str}_{irrep} has wrong multiplicity")
            total = total+proj
        self.assertLess(
            np.max(np.abs(total-np.identity(space_dim))), 1.0e-10,
            msg=f"{group_str} projectors do not sum to the identity")


class TestThreeParticlesIntegerSpin(ProjectorAssertions):
    """Integer-spin three-particle projectors against exact answers."""

    def test_rest_frame_spin_one_decomposes_as_T1MINUS(self):
        # Three particles at rest, spins (0, 0, 1): the 3-dim space is a
        # single ell=1, parity-odd multiplet, so exactly one T1MINUS.
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_three_particles_spin(
                NP_REST, irrep, 0, ZERO_TRIPLE, EMPTY_TRIPLE,
                0.0, 0.0, 1.0, definite_iso=False)

        self.check_family(groups, 'OhP', OHP_IRREPS, 48, 3, raw,
                          {'T1MINUS': 1})

    def test_dic4_frame_scalar_state_is_A1(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_three_particles_spin(
                NP_DIC4, irrep, 0, DIC4_TRIPLE, EMPTY_TRIPLE,
                0.0, 0.0, 0.0, definite_iso=False)

        self.check_family(groups, 'Dic4', DIC4_IRREPS, 8, 1, raw, {'A1': 1})

    def test_dic2_frame_scalar_state_is_A1(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_three_particles_spin(
                NP_DIC2, irrep, 0, DIC2_TRIPLE, EMPTY_TRIPLE,
                0.0, 0.0, 0.0, definite_iso=False)

        self.check_family(groups, 'Dic2', DIC2_IRREPS, 4, 1, raw, {'A1': 1})

    def test_unsupported_momentum_raises(self):
        groups = get_groups()
        with self.assertRaises(ValueError):
            groups.get_proj_nonint_three_particles_spin(
                NP_BAD, 'A1PLUS', 0, ZERO_TRIPLE, EMPTY_TRIPLE,
                0.0, 0.0, 0.0, definite_iso=False)

    def test_half_integer_spins_rejected_without_spin_half(self):
        groups = get_groups()
        for spins in [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]:
            with self.assertRaises(ValueError):
                groups.generate_induced_rep_nonint_three_particles_spin(
                    ZERO_TRIPLE, EMPTY_TRIPLE, *spins,
                    g_elem=np.identity(3), definite_iso=False)


class TestThreeParticlesHalfSpin(ProjectorAssertions):
    """Spin-half three-particle projectors: 1/2 x 1/2 x 1/2 = 2 G1 + H."""

    def _raw_half_spin(self, irrep):
        groups = get_groups(spin_half=True)
        return groups.get_proj_nonint_three_particles_spin(
            NP_REST, irrep, 0, ZERO_TRIPLE, EMPTY_TRIPLE,
            0.5, 0.5, 0.5, definite_iso=False)

    def test_three_spin_half_at_rest_decomposition(self):
        # Coupling three spin-1/2 particles gives two spin-1/2 multiplets
        # and one spin-3/2 multiplet; in the double cover of OhP with the
        # PLUS parity convention these are 2 x G1PLUS and 1 x HPLUS,
        # filling all 2x2 + 2x2 + 4 = 8 spinor states.
        groups = get_groups(spin_half=True)
        self.check_family(
            groups, 'OhP', SPIN_IRREPS, 96, 8, self._raw_half_spin,
            {'G1PLUS': 2, 'HPLUS': 1})

    def test_single_valued_irrep_vanishes_on_spinor_space(self):
        # A1PLUS has 48 characters, so the projector loop repeats them
        # across the 96 double-cover elements; U and -U then contribute
        # with opposite sign and the projector must cancel exactly.
        raw = self._raw_half_spin('A1PLUS')
        self.assertLess(np.max(np.abs(raw)), 1.0e-10)

    def test_half_spin_induced_rep_is_unitary(self):
        groups = get_groups(spin_half=True)
        intspin = groups.OhP_double_MINUS_intspin
        halfspin = groups.OhP_double_PLUS
        for g_ind in [0, 1, 30, 95]:
            g_elem = [intspin[g_ind], halfspin[g_ind]]
            rep = groups.generate_induced_rep_nonint_three_particles_spin(
                ZERO_TRIPLE, EMPTY_TRIPLE, 0.5, 0.5, 0.5, g_elem,
                definite_iso=False)
            self.assertEqual(rep.shape, (8, 8))
            self.assertTrue(np.allclose(
                rep@np.conjugate(rep.T), np.identity(8)))

    def test_generate_wigner_d_half_returns_spinor_part(self):
        groups = get_groups(spin_half=True)
        spinor = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = groups.generate_wigner_d_half(
            0.5, [np.identity(3), spinor])
        self.assertTrue(np.array_equal(result, spinor))


class TestThreeScalarsMovingFrames(ProjectorAssertions):
    """Moving-frame branches of the scalar three-particle projectors."""

    def test_three_scalars_dic2(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_three_scalars(
                NP_DIC2, irrep, 0, DIC2_TRIPLE, EMPTY_TRIPLE,
                definite_iso=False)

        self.check_family(groups, 'Dic2', DIC2_IRREPS, 4, 1, raw, {'A1': 1})

    def test_three_scalars_unsupported_momentum_raises(self):
        groups = get_groups()
        with self.assertRaises(ValueError):
            groups.get_proj_nonint_three_scalars(
                NP_BAD, 'A1PLUS', 0, ZERO_TRIPLE, EMPTY_TRIPLE,
                definite_iso=False)

    def test_three_scalars_labeled_dic4_and_dic2(self):
        groups = get_groups()
        for nP, group_str, irreps, order, nvecset in [
                (NP_DIC4, 'Dic4', DIC4_IRREPS, 8, DIC4_TRIPLE),
                (NP_DIC2, 'Dic2', DIC2_IRREPS, 4, DIC2_TRIPLE)]:
            def raw(irrep):
                return groups.get_proj_nonint_three_scalars_labeled(
                    nP, irrep, 0, nvecset,
                    permutations=[[0, 1, 2], [1, 0, 2]])

            self.check_family(groups, group_str, irreps, order, 1, raw,
                              {'A1': 1})

    def test_three_scalars_labeled_unsupported_momentum_raises(self):
        groups = get_groups()
        with self.assertRaises(ValueError):
            groups.get_proj_nonint_three_scalars_labeled(
                NP_BAD, 'A1PLUS', 0, ZERO_TRIPLE)

    def test_labeled_induced_rep_defaults_to_identity_permutation(self):
        groups = get_groups()
        rep = groups.generate_induced_rep_nonint_three_scalars_labeled(
            DIC4_TRIPLE)
        self.assertTrue(np.array_equal(rep, np.identity(1)))

    def test_three_particle_label_permutations(self):
        perms = Groups._three_particle_label_permutations('aaa')
        self.assertEqual(len(perms), 6)
        self.assertIn([0, 1, 2], [list(p) for p in perms])
        self.assertEqual(Groups._three_particle_label_permutations('aab'),
                         [[0, 1, 2], [1, 0, 2]])
        self.assertEqual(Groups._three_particle_label_permutations('abc'),
                         [[0, 1, 2]])
        with self.assertRaises(ValueError):
            Groups._three_particle_label_permutations('ab')


class TestTwoParticlesProjectors(ProjectorAssertions):
    """Two-particle projectors in moving frames and with spin."""

    def test_two_particles_dic4(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_two_particles(
                NP_DIC4, irrep, 0, DIC4_PAIR, 0.0, 0.0)

        self.check_family(groups, 'Dic4', DIC4_IRREPS, 8, 1, raw, {'A1': 1})

    def test_two_particles_dic2(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_two_particles(
                NP_DIC2, irrep, 0, DIC2_PAIR, 0.0, 0.0)

        self.check_family(groups, 'Dic2', DIC2_IRREPS, 4, 1, raw, {'A1': 1})

    def test_two_particles_rest_frame_spin_one_is_T1MINUS(self):
        groups = get_groups()

        def raw(irrep):
            return groups.get_proj_nonint_two_particles(
                NP_REST, irrep, 0, ZERO_PAIR, 1.0, 0.0)

        self.check_family(groups, 'OhP', OHP_IRREPS, 48, 3, raw,
                          {'T1MINUS': 1})

    def test_two_particles_unsupported_momentum_raises(self):
        groups = get_groups()
        with self.assertRaises(ValueError):
            groups.get_proj_nonint_two_particles(
                NP_BAD, 'A1PLUS', 0, ZERO_PAIR, 0.0, 0.0)


class TestSpinningShellAndDict(unittest.TestCase):
    """Shell- and dict-level entry points for spin-half channels."""

    @staticmethod
    def _make_spin_half_qcis():
        ni_channel = SimpleNamespace(spins=[0.5, 0.5, 0.5],
                                     isospin_channel=False)
        fc_channel = SimpleNamespace(isospin_channel=False,
                                     flavors=['N', 'N', 'N'])
        return SimpleNamespace(
            nP=NP_REST,
            fvs=SimpleNamespace(nP=NP_REST, spin_half=True,
                                irrep_set=list(SPIN_IRREPS)),
            nis=SimpleNamespace(
                nvecset_aaa_batched=[[ZERO_TRIPLE]],
                nvecset_abc_batched=[[EMPTY_TRIPLE]],
                nvecset_aaa_reps=[[np.zeros((3, 3), dtype=int)]]),
            fcs=SimpleNamespace(ni_list=[ni_channel], fc_list=[fc_channel]))

    def test_spinning_shell_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups(spin_half=True).get_proj_nonint_three_spinning_shell(
                qcis=None)

    def test_spinning_shell_decomposition(self):
        groups = get_groups(spin_half=True)
        qcis = self._make_spin_half_qcis()
        non_proj_dict = groups.get_proj_nonint_three_spinning_shell(
            qcis=qcis, cindex=0, definite_iso=False, shell_index=0)
        # 2 x G1PLUS (2 rows, 4 columns each) and 1 x HPLUS (4 rows,
        # 4 columns each); G2 and all MINUS irreps must be absent.
        expected_keys = {('G1PLUS', 0), ('G1PLUS', 1),
                         ('HPLUS', 0), ('HPLUS', 1),
                         ('HPLUS', 2), ('HPLUS', 3)}
        self.assertEqual(set(non_proj_dict.keys()), expected_keys)
        for key, proj in non_proj_dict.items():
            self.assertEqual(proj.shape, (8, 4),
                             msg=f"unexpected shape for {key}")
            overlaps = np.conjugate(proj.T)@proj
            self.assertTrue(np.allclose(overlaps, np.identity(4)),
                            msg=f"columns of {key} not orthonormal")

    def test_spinning_dict_summary(self):
        groups = get_groups(spin_half=True)
        qcis = self._make_spin_half_qcis()
        master_dict = groups.get_proj_nonint_three_spinning_dict(
            qcis=qcis, nic_index=0)
        self.assertIn('summary', master_dict)
        summary = master_dict['summary']
        self.assertIn('shell_index = 0 (8 states)', summary)
        self.assertIn('G1PLUS (appears 2 times)', summary)
        self.assertIn('HPLUS (appears 1 time)', summary)

    def test_spinning_dict_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups(spin_half=True).get_proj_nonint_three_spinning_dict(
                qcis=None)

    def test_spinning_dict_rejects_mixed_flavors(self):
        groups = get_groups(spin_half=True)
        qcis = self._make_spin_half_qcis()
        qcis.fcs.fc_list[0].flavors = ['N', 'N', 'X']
        with self.assertRaises(ValueError):
            groups.get_proj_nonint_three_spinning_dict(qcis=qcis,
                                                       nic_index=0)

    def test_definite_iso_with_empty_abc_matches_no_iso(self):
        # With an empty abc array the definite_iso dimension bookkeeping
        # must reduce to the aaa-only result.
        groups = get_groups(spin_half=True)
        with_iso = groups.get_proj_nonint_three_particles_spin(
            NP_REST, 'G1PLUS', 0, ZERO_TRIPLE, EMPTY_TRIPLE,
            0.5, 0.5, 0.5, definite_iso=True)
        without_iso = groups.get_proj_nonint_three_particles_spin(
            NP_REST, 'G1PLUS', 0, ZERO_TRIPLE, EMPTY_TRIPLE,
            0.5, 0.5, 0.5, definite_iso=False)
        self.assertTrue(np.allclose(with_iso, without_iso))


class TestSpinningShellIntegerSpin(unittest.TestCase):
    """The spinning shell and dict also serve integer-spin channels."""

    @staticmethod
    def _make_integer_spin_qcis(nP, irrep_set, aaa_batched, spins):
        ni_channel = SimpleNamespace(spins=list(spins),
                                     isospin_channel=False)
        fc_channel = SimpleNamespace(isospin_channel=False,
                                     flavors=['rho', 'rho', 'rho'])
        return SimpleNamespace(
            nP=nP,
            fvs=SimpleNamespace(nP=nP, spin_half=False,
                                irrep_set=list(irrep_set)),
            nis=SimpleNamespace(
                nvecset_aaa_batched=[[aaa_batched]],
                nvecset_abc_batched=[[EMPTY_TRIPLE]],
                nvecset_aaa_reps=[[aaa_batched[0]]]),
            fcs=SimpleNamespace(ni_list=[ni_channel], fc_list=[fc_channel]))

    def test_two_vector_particles_at_rest(self):
        # Spins (1, 1, 0) at rest: 1 x 1 = 0 + 1 + 2 with overall even
        # parity, so the 9 states split as A1PLUS + T1PLUS + (EPLUS +
        # T2PLUS), each appearing exactly once.
        qcis = self._make_integer_spin_qcis(
            NP_REST, OHP_IRREPS, ZERO_TRIPLE, (1.0, 1.0, 0.0))
        master_dict = get_groups().get_proj_nonint_three_spinning_dict(
            qcis=qcis, nic_index=0)
        summary = master_dict['summary']
        self.assertIn('shell_index = 0 (9 states)', summary)
        for irrep in ['A1PLUS', 'T1PLUS', 'EPLUS', 'T2PLUS']:
            self.assertIn(f'{irrep} (appears 1 time)', summary)

    def test_moving_frame_scalar_channels(self):
        for nP, irreps, triple in [(NP_DIC4, DIC4_IRREPS, DIC4_TRIPLE),
                                   (NP_DIC2, DIC2_IRREPS, DIC2_TRIPLE)]:
            qcis = self._make_integer_spin_qcis(
                nP, irreps, triple, (0.0, 0.0, 0.0))
            non_proj_dict =\
                get_groups().get_proj_nonint_three_spinning_shell(
                    qcis=qcis, cindex=0, definite_iso=False, shell_index=0)
            self.assertEqual(set(non_proj_dict.keys()), {('A1', 0)})
            self.assertEqual(non_proj_dict[('A1', 0)].dtype, np.float64)

    def test_unsupported_momentum_raises(self):
        qcis = self._make_integer_spin_qcis(
            NP_BAD, DIC4_IRREPS, DIC4_TRIPLE, (0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_spinning_shell(
                qcis=qcis, cindex=0, definite_iso=False, shell_index=0)


class TestScalarShellsMovingFrames(unittest.TestCase):
    """Moving-frame and error branches of the scalar shell functions."""

    @staticmethod
    def _make_pions_qcis(nP, irrep_set, aaa_batched):
        return SimpleNamespace(
            nP=nP,
            fvs=SimpleNamespace(nP=nP, irrep_set=list(irrep_set)),
            nis=SimpleNamespace(
                _nonint_channel_particle_label=lambda cindex: 'aaa',
                nvecset_aaa_batched=[[aaa_batched]],
                nvecset_abc_batched=[[EMPTY_TRIPLE]],
                nvecset_aaa_reps=[[aaa_batched[0]]]),
            fcs=SimpleNamespace(
                ni_list=[SimpleNamespace(isospin_channel=False)]))

    def test_pions_shell_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_pions_shell(qcis=None)

    def test_pions_shell_unsupported_momentum_raises(self):
        qcis = self._make_pions_qcis(NP_BAD, OHP_IRREPS, ZERO_TRIPLE)
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_pions_shell(
                qcis=qcis, cindex=0, definite_iso=False, shell_index=0)

    def test_pions_shell_dic2(self):
        qcis = self._make_pions_qcis(NP_DIC2, DIC2_IRREPS, DIC2_TRIPLE)
        non_proj_dict = get_groups().get_proj_nonint_three_pions_shell(
            qcis=qcis, cindex=0, definite_iso=False, shell_index=0)
        self.assertEqual(set(non_proj_dict.keys()), {('A1', 0)})
        self.assertTrue(np.allclose(np.abs(non_proj_dict[('A1', 0)]),
                                    np.identity(1)))

    def test_pions_dict_without_isospin(self):
        qcis = self._make_pions_qcis(NP_REST, OHP_IRREPS, ZERO_TRIPLE)
        master_dict = get_groups().get_proj_nonint_three_pions_dict(
            qcis=qcis, nic_index=0)
        self.assertIn((0, 0), master_dict)
        self.assertEqual(set(master_dict[(0, 0)].keys()), {('A1PLUS', 0)})
        self.assertIn('Channel contains', master_dict['summary'])
        self.assertIn('A1PLUS (appears 1 time)', master_dict['summary'])

    def test_pions_dict_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_pions_dict(qcis=None)

    @staticmethod
    def _make_labeled_qcis(nP, irrep_set, nvecset_batched, label='aab'):
        nis = SimpleNamespace(
            _nonint_channel_particle_label=lambda cindex: label)
        setattr(nis, f'nvecset_{label}_batched', [[nvecset_batched]])
        setattr(nis, f'nvecset_{label}_reps', [[nvecset_batched[:1]]])
        return SimpleNamespace(
            nP=nP,
            fvs=SimpleNamespace(nP=nP, irrep_set=list(irrep_set)),
            nis=nis,
            fcs=SimpleNamespace(
                ni_list=[SimpleNamespace(isospin_channel=False)]))

    def test_labeled_shell_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_scalars_labeled_shell(
                qcis=None)

    def test_labeled_shell_unsupported_momentum_raises(self):
        qcis = self._make_labeled_qcis(NP_BAD, DIC4_IRREPS, DIC4_TRIPLE)
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_scalars_labeled_shell(
                qcis=qcis, cindex=0, shell_index=0)

    def test_labeled_shell_moving_frames(self):
        for nP, irreps, nvecset in [(NP_DIC4, DIC4_IRREPS, DIC4_TRIPLE),
                                    (NP_DIC2, DIC2_IRREPS, DIC2_TRIPLE)]:
            qcis = self._make_labeled_qcis(nP, irreps, nvecset)
            non_proj_dict =\
                get_groups().get_proj_nonint_three_scalars_labeled_shell(
                    qcis=qcis, cindex=0, shell_index=0)
            self.assertEqual(set(non_proj_dict.keys()), {('A1', 0)})

    def test_labeled_dict_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_three_scalars_labeled_dict(
                qcis=None)

    def test_labeled_dict_single_state(self):
        qcis = self._make_labeled_qcis(NP_REST, OHP_IRREPS, ZERO_TRIPLE,
                                       label='abc')
        master_dict =\
            get_groups().get_proj_nonint_three_scalars_labeled_dict(
                qcis=qcis, nic_index=0)
        self.assertIn('A1PLUS (appears 1 time)', master_dict['summary'])
        self.assertEqual(set(master_dict[(0, 0)].keys()), {('A1PLUS', 0)})


class TestTwoParticleShellAndDict(unittest.TestCase):
    """Shell- and dict-level entry points for two-particle channels."""

    @staticmethod
    def _make_two_particle_qcis(nP, irrep_set, nvecset_batched,
                                isospin_channel=False, isospin=1.0):
        nis = SimpleNamespace(
            _nonint_channel_particle_label=lambda cindex: 'ab',
            nvecset_ab_batched=[[nvecset_batched]],
            nvecset_ab_reps=[[nvecset_batched[:1]]])
        ni_channel = SimpleNamespace(spins=[0.0, 0.0],
                                     isospin_channel=isospin_channel,
                                     isospin=isospin)
        return SimpleNamespace(
            nP=nP,
            fvs=SimpleNamespace(nP=nP, irrep_set=list(irrep_set)),
            nis=nis,
            fcs=SimpleNamespace(ni_list=[ni_channel]))

    def test_two_particle_shell_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_two_particles_shell(qcis=None)

    def test_two_particle_shell_unsupported_momentum_raises(self):
        qcis = self._make_two_particle_qcis(NP_BAD, DIC4_IRREPS, DIC4_PAIR)
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_two_particles_shell(
                qcis=qcis, cindex=0, definite_iso=False, shell_index=0)

    def test_two_particle_shell_moving_frames(self):
        for nP, irreps, pair in [(NP_DIC4, DIC4_IRREPS, DIC4_PAIR),
                                 (NP_DIC2, DIC2_IRREPS, DIC2_PAIR)]:
            qcis = self._make_two_particle_qcis(nP, irreps, pair)
            non_proj_dict =\
                get_groups().get_proj_nonint_two_particles_shell(
                    qcis=qcis, cindex=0, definite_iso=False, shell_index=0)
            self.assertEqual(set(non_proj_dict.keys()), {('A1', 0)})

    def test_two_particle_dict_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_proj_nonint_two_particles_dict(qcis=None)

    def test_two_particle_dict_without_isospin(self):
        qcis = self._make_two_particle_qcis(NP_REST, OHP_IRREPS, ZERO_PAIR)
        master_dict = get_groups().get_proj_nonint_two_particles_dict(
            qcis=qcis, nic_index=0, isospin_channel=False)
        self.assertIn('A1PLUS (appears 1 time)', master_dict['summary'])
        self.assertEqual(set(master_dict[(0, 0)].keys()), {('A1PLUS', 0)})


class TestKellmDictEntryPoints(unittest.TestCase):
    """Fixed-channel kellm projector dictionaries in moving frames."""

    def _column_total(self, proj_dict):
        return sum(proj.shape[1] for key, proj in proj_dict.items()
                   if isinstance(key, tuple))

    def test_fixed_sc_proj_dict_dic4(self):
        qcis = make_kellm_qcis(NP_DIC4, DIC4_IRREPS)
        proj_dict = get_groups().get_fixed_sc_proj_dict(qcis=qcis,
                                                        sc_index=0)
        self.assertIn('summary', proj_dict)
        self.assertIn('best_irreps', proj_dict)
        self.assertIn(('A1', 0), proj_dict)
        total_size = len(qcis.kellm_spaces[0][0])
        self.assertEqual(self._column_total(
            {k: v for k, v in proj_dict.items()
             if k not in ('summary', 'best_irreps')}), total_size)

    def test_fixed_sc_proj_dict_dic2(self):
        qcis = make_kellm_qcis(NP_DIC2, DIC2_IRREPS)
        proj_dict = get_groups().get_fixed_sc_proj_dict(qcis=qcis,
                                                        sc_index=0)
        self.assertIn(('A1', 0), proj_dict)
        self.assertIn('total matches size of kellm space',
                      proj_dict['summary'])

    def test_fixed_sc_and_shell_proj_dict_defaults_shell_index(self):
        qcis = make_kellm_qcis(NP_DIC4, DIC4_IRREPS)
        n_ellm = len(KELLM_ELLM_SET)
        proj_dict = get_groups().get_fixed_sc_and_shell_proj_dict(
            qcis=qcis, sc_index=0, kellm_shell=[0, n_ellm])
        self.assertEqual(self._column_total(proj_dict), n_ellm)

    def test_fixed_sc_and_shell_proj_dict_dic2(self):
        qcis = make_kellm_qcis(NP_DIC2, DIC2_IRREPS)
        n_ellm = len(KELLM_ELLM_SET)
        proj_dict = get_groups().get_fixed_sc_and_shell_proj_dict(
            qcis=qcis, sc_index=0, kellm_shell=[0, n_ellm],
            kellm_shell_index=0)
        self.assertEqual(self._column_total(proj_dict), n_ellm)

    def test_fixed_sc_and_shell_error_paths(self):
        groups = get_groups()
        with self.assertRaises(ValueError):
            groups.get_fixed_sc_and_shell_proj_dict(qcis=None)
        qcis = make_kellm_qcis(NP_DIC4, DIC4_IRREPS)
        with self.assertRaises(ValueError):
            groups.get_fixed_sc_and_shell_proj_dict(qcis=qcis,
                                                    kellm_shell=None)
        qcis_bad = make_kellm_qcis(NP_BAD, DIC4_IRREPS)
        with self.assertRaises(ValueError):
            groups.get_fixed_sc_and_shell_proj_dict(
                qcis=qcis_bad, kellm_shell=[0, len(KELLM_ELLM_SET)])

    def test_full_proj_dict_dic2(self):
        qcis = make_kellm_qcis(NP_DIC2, DIC2_IRREPS)
        proj_dict = get_groups().get_full_proj_dict(qcis=qcis)
        self.assertIn(('A1', 0), proj_dict)
        total_size = len(qcis.kellm_spaces[0][0])
        self.assertEqual(self._column_total(
            {k: v for k, v in proj_dict.items()
             if k not in ('summary', 'best_irreps')}), total_size)

    def test_full_proj_dict_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_full_proj_dict(qcis=None)


class TestSummaryAndIsoProjection(unittest.TestCase):
    """Direct tests for the summary helper and iso-projection guard."""

    def test_get_summary_raises_when_space_not_spanned(self):
        groups = get_groups()
        proj_dict = {('A1PLUS', 0): np.ones((5, 2))}
        with self.assertRaises(RuntimeError):
            groups._get_summary(proj_dict, 'OhP', None, 3)

    def test_get_summary_success_reports_coverage(self):
        groups = get_groups()
        proj_dict = {('A1PLUS', 0): np.ones((5, 2))}
        best_irreps, summary = groups._get_summary(proj_dict, 'OhP', None, 2)
        self.assertEqual(best_irreps, [('A1PLUS', 0)])
        self.assertIn('total matches size of kellm space', summary)

    def test_get_iso_projection_requires_qcis(self):
        with self.assertRaises(ValueError):
            get_groups().get_iso_projection(qcis=None)

    def test_pion_orders_are_all_three_particle_permutations(self):
        perms = {tuple(p) for p in groups_module.PION_ORDERS}
        self.assertEqual(len(perms), 6)
        self.assertIn((0, 1, 2), perms)


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
