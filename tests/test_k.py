#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
from types import SimpleNamespace
import numpy as np
import ampyl
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

    def test_unprojected_ell_one_block_is_constant(self):
        kdf = self._make_kdf()
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=0,
            ell1=1, ell2=1, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=False, irrep=None)
        self.assertEqual(shell.shape, (3, 6))
        self.assertTrue(np.all(shell == 2.5))

    def test_unprojected_other_ell_block_is_zero(self):
        kdf = self._make_kdf()
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=0,
            ell1=0, ell2=0, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=False, irrep=None)
        self.assertEqual(shell.shape, (1, 2))
        self.assertTrue(np.all(shell == 0.0))

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
        proj_row = np.arange(6.).reshape((3, 2))
        proj_col = np.arange(24.).reshape((6, 4))
        proj_dicts = [
            [{irrep: proj_row}, {irrep: proj_col}],
            [{irrep: proj_row}, {irrep: proj_col}]]
        kdf = self._make_kdf(proj_dicts)
        shell = kdf.get_shell(
            E=5.0, L=5.0, k3_params=[2.5], m1=1.0, m2=1.0, m3=1.0,
            cindex_row=0, cindex_col=0, sc_index_row=0, sc_index_col=1,
            ell1=1, ell2=1, tbks_entry=self._tbks_entry(),
            row_shell_index=0, col_shell_index=1,
            project=True, irrep=irrep)
        expected = np.conjugate(proj_row.T)@(
            np.ones((3, 6))*2.5)@proj_col
        self.assertTrue(np.allclose(shell, expected))


if __name__ == '__main__':
    unittest.main()
