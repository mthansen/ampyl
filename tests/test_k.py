#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
import numpy as np
import ampyl
from ampyl.constants import EPSILON20
from ampyl.constants import EPSILON30
from ampyl.k_matrices import Kdf


class TestK(unittest.TestCase):
    """Class to test the two-particle K matrix."""

    def test_k_dimer_symmetry_factor(self):
        """Test K is doubled only for non-identical dimer particles."""
        pion = ampyl.flavor.Particle(mass=1.0, flavor='pi')
        kaon = ampyl.flavor.Particle(mass=2.5, flavor='K')
        fc = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion])
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc])
        fvs = ampyl.spaces.FiniteVolumeSetup(
            qc_impl={'smarter_q_rescale': True})
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, fvs=fvs, tbis=tbis, Emax=7., Lmax=4.)
        qcis.populate()
        k = ampyl.K(qcis=qcis)
        E = 6.5
        L = 4.0
        epsilon = 1.0e-15

        for sc_ind, expected_factor in [(0, 1.0), (1, 2.0)]:
            sc = qcis.fcs.sc_list_sorted[sc_ind]
            three_slice_index = qcis.sc_to_three_slice[sc_ind]
            tbks_entry = qcis.tbks_list[three_slice_index][0]
            slice_entry = tbks_entry.shells[0]
            m1 = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            ell = sc.ell_set[0]
            pcotdelta_function = sc.p_cot_deltas[0]
            pcotdelta_parameter_list = [1.0]
            alpha, beta = qcis.tbis.scheme_data[sc_ind]
            raw_k = ampyl.qc_functions.getK_array(
                E, k.qcis.fvs.nP, L, m1, m2, m3, tbks_entry, slice_entry,
                ell, pcotdelta_function, pcotdelta_parameter_list,
                alpha, beta, k.qcis.fvs.qc_impl, k.qcis.tbis.three_scheme)
            shell_k = k.get_shell(
                E, L, m1, m2, m3, sc_ind, sc_ind, ell,
                pcotdelta_function, pcotdelta_parameter_list, tbks_entry, 0,
                False, None)
            self.assertTrue(
                np.allclose(shell_k, expected_factor*raw_k, rtol=0.0,
                            atol=epsilon))


class TestKdfGetShell(unittest.TestCase):
    """Unit tests for Kdf.get_shell block sizing and projection.

    The block dimensions come from the shell slices and the angular
    momenta. Regression: they were previously read off the projector
    shapes, so the unprojected path (project=False, the signature
    default) raised UnboundLocalError."""

    SHELLS = [[0, 1], [1, 3]]

    @staticmethod
    def _make_kdf(proj_dicts=None):
        qcis = SimpleNamespace(
            fvs=SimpleNamespace(nP=np.array([0, 0, 0])),
            sc_to_three_slice=[0, 0],
            proj_dicts_by_sc_and_shellset=proj_dicts)
        return Kdf(qcis=qcis)

    def _tbks_entry(self):
        return SimpleNamespace(shells=self.SHELLS)

    def test_unprojected_ell_zero_block_is_constant(self):
        # The isotropic term lives on the S wave. Regression: the gate
        # read ell1 == ell2 == 1, so an ell_set of [0] -- which is what
        # the KKpi spaces use -- silently produced a Kdf of zeros.
        kdf = self._make_kdf()
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=0,
            ell1=0, ell2=0, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=False, irrep=None)
        self.assertEqual(shell.shape, (1, 2))
        self.assertTrue(np.all(shell == 2.5))

    def test_unprojected_ell_off_diagonal_block_is_zero(self):
        kdf = self._make_kdf()
        for ell1, ell2 in [(0, 1), (1, 0), (1, 2)]:
            shell = kdf.get_shell(
                E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
                cindex_row=0, cindex_col=0, sc_index_row=0,
                sc_index_col=0, ell1=ell1, ell2=ell2,
                tbks_entry=self._tbks_entry(),
                row_shell_index=0, col_shell_index=1,
                project=False, irrep=None)
            self.assertEqual(shell.shape,
                             (1*(2*ell1+1), 2*(2*ell2+1)))
            self.assertTrue(np.all(shell == 0.0))

    def test_unprojected_ell_one_block_is_still_constant(self):
        # The ell = 1 behaviour predates the ell = 0 case and is fixed
        # by the three-pion benchmark in tests/test_qc.py; widening the
        # gate must not change it.
        kdf = self._make_kdf()
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=0,
            ell1=1, ell2=1, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=False, irrep=None)
        self.assertEqual(shell.shape, (3, 6))
        self.assertTrue(np.all(shell == 2.5))

    def test_column_shells_come_from_the_column_entry(self):
        # Two spectator channels with different spectator masses sit in
        # different three-particle slices and carry different shell
        # sets; the column block must be sized by the column channel's
        # own entry.
        kdf = self._make_kdf()
        col_tbks_entry = SimpleNamespace(shells=[[0, 2], [2, 5], [5, 9]])
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=1,
            ell1=0, ell2=0, tbks_entry=self._tbks_entry(),
            row_shell_index=1, col_shell_index=2,
            project=False, irrep=None,
            col_tbks_entry=col_tbks_entry)
        self.assertEqual(shell.shape, (2, 4))
        self.assertTrue(np.all(shell == 2.5))

    def test_unknown_irrep_raises_instead_of_empty_block(self):
        # Regression: an irrep absent from every projector dictionary
        # (e.g. a typo) previously returned an empty array, which was
        # zero-padded downstream into a plausible all-zero block.
        irrep = ('A1PLUS', 0)
        proj_dicts = [
            [{irrep: np.identity(3)}, {irrep: np.identity(6)}],
            [{irrep: np.identity(3)}, {irrep: np.identity(6)}]]
        kdf = self._make_kdf(proj_dicts)
        with self.assertRaises(ValueError):
            kdf.get_shell(
                E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
                cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=1,
                ell1=1, ell2=1, tbks_entry=self._tbks_entry(),
                row_shell_index=0, col_shell_index=1,
                project=True, irrep=('BOGUS', 0))

    def test_known_irrep_missing_for_shell_returns_empty(self):
        # A known irrep that this particular shell does not contribute
        # to is legitimate and must still yield the empty block.
        irrep = ('A1PLUS', 0)
        proj_dicts = [
            [{irrep: np.identity(3)}, {}],
            [{irrep: np.identity(3)}, {}]]
        kdf = self._make_kdf(proj_dicts)
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=1,
            ell1=1, ell2=1, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=True, irrep=irrep)
        self.assertEqual(len(shell), 0)

    def test_projected_block_matches_manual_projection(self):
        irrep = 'A1PLUS'
        # shells [[0, 1], [1, 3]] at ell = 0 give a one-dimensional row
        # block and a two-dimensional column block
        proj_row = np.arange(2.).reshape((1, 2))
        proj_col = np.arange(8.).reshape((2, 4))
        proj_dicts = [
            [{irrep: proj_row}, {irrep: proj_col}],
            [{irrep: proj_row}, {irrep: proj_col}]]
        kdf = self._make_kdf(proj_dicts)
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=1,
            ell1=0, ell2=0, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=True, irrep=irrep)
        expected = np.conjugate(proj_row.T)@(
            np.ones((1, 2))*2.5)@proj_col
        self.assertTrue(np.allclose(shell, expected))


class TestKdfNondegenerate(unittest.TestCase):
    """Kdf on a space whose spectator channels have different masses.

    Two kaons and a pion give two spectator channels with different
    spectator masses, hence two three-particle slices, hence two TBKS
    slots carrying different numbers of shells. Regressions covered:
    ``get_value`` raised because it assumed a single slot, and the
    ell = 0 gate returned zeros.
    """

    IRREP = ('A1PLUS', 0)
    K3 = 1.5
    E = 6.5
    L = 4.0

    @classmethod
    def setUpClass(cls):
        pion = ampyl.flavor.Particle(mass=1.0, flavor='pi')
        kaon = ampyl.flavor.Particle(mass=2.5, flavor='K')
        fc = ampyl.flavor.FlavorChannel(3, particles=[kaon, kaon, pion])
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc])
        fvs = ampyl.spaces.FiniteVolumeSetup(
            qc_impl={'smarter_q_rescale': True})
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, fvs=fvs, tbis=tbis, Emax=7., Lmax=4.)
        qcis.populate()
        cls.qcis = qcis
        cls.kdf = Kdf(qcis=qcis)
        cls.k = ampyl.K(qcis=qcis)

    def _shell_counts(self):
        _, slices_by_three_slice = self.kdf._get_entry_and_slices(
            self.E, self.L, self.qcis.fvs.nP)
        return [len(slices) for slices in slices_by_three_slice]

    def test_space_really_has_two_slices_of_different_size(self):
        self.assertEqual(self.qcis.fcs.n_three_slices, 2)
        self.assertEqual(len(self.qcis.tbks_list), 2)
        counts = self._shell_counts()
        self.assertEqual(len(counts), 2)
        self.assertNotEqual(counts[0], counts[1])

    def test_unprojected_value_is_constant_and_matches_k(self):
        kdf_value = self.kdf.get_value(self.E, self.L, [self.K3],
                                       False, None)
        k_value = self.k.get_value(
            self.E, self.L, pcotdelta_parameter_lists=[[0.3], [0.3]],
            project=False, irrep=None)
        self.assertEqual(kdf_value.shape, np.asarray(k_value).shape)
        self.assertTrue(np.all(kdf_value == self.K3))

    def test_projected_value_matches_k_and_is_symmetric(self):
        kdf_value = self.kdf.get_value(self.E, self.L, [self.K3],
                                       True, self.IRREP)
        k_value = self.k.get_value(
            self.E, self.L, pcotdelta_parameter_lists=[[0.3], [0.3]],
            project=True, irrep=self.IRREP)
        self.assertEqual(kdf_value.shape, np.asarray(k_value).shape)
        self.assertTrue(np.allclose(kdf_value, kdf_value.T))

    def test_projected_value_is_the_projected_constant_matrix(self):
        # Independent assembly: a constant matrix is an outer product of
        # vectors of ones, so its projection is the outer product of the
        # projected vectors of ones, built here straight from the
        # projector dictionaries.
        counts = self._shell_counts()
        left = []
        right = []
        for sc_index in range(len(self.qcis.fcs.sc_list_sorted)):
            three_slice = self.qcis.sc_to_three_slice[sc_index]
            for shell_index in range(counts[three_slice]):
                projector = self.qcis.proj_dicts_by_sc_and_shellset[
                    sc_index][shell_index][self.IRREP]
                ones = np.ones(projector.shape[0])
                left.append(np.conjugate(projector.T)@ones)
                right.append(projector.T@ones)
        left = np.concatenate(left)
        right = np.concatenate(right)
        expected = self.K3*np.outer(left, right)
        kdf_value = self.kdf.get_value(self.E, self.L, [self.K3],
                                       True, self.IRREP)
        self.assertEqual(kdf_value.shape, expected.shape)
        self.assertTrue(np.allclose(kdf_value, expected))

    def test_asymmetric_qc_reduces_to_the_kdf_zero_version(self):
        # det(F3inv + Kdf) at Kdf = 0 must be 1/det(F3), which is what
        # the kdf_zero version returns. This is the end-to-end check
        # that Kdf enters the asymmetric quantization condition in the
        # same basis as F+G and K.
        qc = ampyl.QC(qcis=self.qcis)

        def value(version, k3):
            return complex(qc.get_value(
                self.E, self.L,
                {'k_params': [[[0.3], [0.3]], [k3]],
                 'project': True, 'irrep': self.IRREP,
                 'version': version}))

        reference = value('kdf_zero_detf3inv_asym_fgcombo', 0.0)
        at_zero = value('kdf+f3inv_asym_fgcombo', 0.0)
        self.assertTrue(np.isclose(at_zero, reference, rtol=1.0e-10))
        for k3 in (-7.0, 7.0):
            self.assertFalse(np.isclose(value('kdf+f3inv_asym_fgcombo',
                                              k3),
                                        reference, rtol=1.0e-12))


class TestPcotdeltaScatteringLength(unittest.TestCase):
    """Unit tests for the scattering-length parametrization."""

    def test_matches_minus_inverse_scattering_length(self):
        pcotdelta = ampyl.qc_functions.pcotdelta_scattering_length
        self.assertAlmostEqual(pcotdelta(1.5, 0.25), -4.0, places=12)
        self.assertAlmostEqual(pcotdelta(1.5, -0.25), 4.0, places=12)

    def test_vanishing_scattering_length_uses_epsilon(self):
        pcotdelta = ampyl.qc_functions.pcotdelta_scattering_length
        self.assertAlmostEqual(pcotdelta(1.5, 0.0), -1.0/EPSILON30, places=12)

    def test_vanishing_scattering_length_gives_vanishing_k(self):
        qcis = ampyl.spaces.QCIndexSpace()
        qcis.populate()
        k = ampyl.K(qcis=qcis)
        k_matrix = k.get_value(E=4.0, L=5.0,
                               pcotdelta_parameter_lists=[[0.0]],
                               project=False)
        self.assertTrue(np.all(np.abs(k_matrix) < EPSILON20))


if __name__ == '__main__':
    unittest.main()
