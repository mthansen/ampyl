#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for shell-set selection in projection dictionaries."""

import unittest
import warnings
import numpy as np
import ampyl


class TestShellsetIndex(unittest.TestCase):
    """Test shell-set handling at zero total momentum.

    At zero total momentum ``proj_dicts_by_sc_and_shellset`` has no
    shell-set rank: it stores one flat per-shell table per spectator
    channel, built for shell set 0. This is exact because the
    precomputed shell sets are nested (each smaller set is a prefix of
    shell set 0) and each per-shell projector depends only on the
    shell's own momentum orbit. These tests pin both properties, and
    that ``get_shellset_index`` refuses zero-momentum calls.
    """

    @classmethod
    def setUpClass(cls):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = ampyl.flavor.FlavorChannel(3)
            fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc])
            fvs = ampyl.spaces.FiniteVolumeSetup()
            tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
            cls.qcis = ampyl.spaces.QCIndexSpace(
                fcs=fcs, fvs=fvs, tbis=tbis, Emax=5.0, Lmax=6.0)
            cls.qcis.populate()

    def test_zero_momentum_shellset_index_raises(self):
        """The shell-set selector refuses zero-momentum calls."""
        with self.assertRaises(ValueError) as caught:
            self.qcis.get_shellset_index(4.2, 5.0)
        self.assertIn("nonzero total momentum", str(caught.exception))

    def test_flat_per_shell_table_is_stored(self):
        """At nP = 0 each channel stores a flat list of per-shell dicts."""
        for shell_dicts in self.qcis.proj_dicts_by_sc_and_shellset:
            self.assertGreater(len(shell_dicts), 0)
            for shell_dict in shell_dicts:
                self.assertIsInstance(shell_dict, dict)

    def test_shell_sets_are_nested_prefixes(self):
        """Each smaller kinematic space keeps a prefix of the shells."""
        for tbks_entries in self.qcis.tbks_list:
            shells_zero = tbks_entries[0].shells
            for tbks_entry in tbks_entries:
                self.assertEqual(
                    tbks_entry.shells,
                    shells_zero[:len(tbks_entry.shells)])

    def test_truncated_set_projectors_equal_stored_set(self):
        """Rebuilt truncated-set projectors match the stored table.

        The stored table is built for shell set 0 only; here the
        per-shell projectors of two smaller shell sets are rebuilt from
        scratch and must coincide with the stored entries, justifying
        the single flat table.
        """
        sc_index = 0
        stored = self.qcis.proj_dicts_by_sc_and_shellset[sc_index]
        compared = 0
        for shellset_index in (2, 4):
            kellm_shell_set = self.qcis.kellm_shells[sc_index][
                shellset_index]
            for shell_index, kellm_shell in enumerate(kellm_shell_set):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rebuilt = self.qcis.group\
                        .get_fixed_sc_and_shell_proj_dict(
                            qcis=self.qcis, sc_index=sc_index,
                            kellm_shell=kellm_shell,
                            kellm_shell_index=shellset_index)
                self.assertEqual(set(rebuilt.keys()),
                                 set(stored[shell_index].keys()))
                for irrep, projector in rebuilt.items():
                    reference = stored[shell_index][irrep]
                    self.assertEqual(projector.shape, reference.shape)
                    self.assertTrue(np.allclose(projector, reference))
                    compared += 1
        self.assertGreater(compared, 0)

    def test_mixed_space_stores_flat_tables(self):
        """A mixed two-plus-three-particle space also stores flat tables."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi',
                                         isospin_multiplet=True,
                                         isospin=1.0)
            fc_two = ampyl.flavor.FlavorChannel(
                2, particles=[pion, pion], isospin=2.0)
            fc_three = ampyl.flavor.FlavorChannel(
                3, particles=[pion, pion, pion], isospin=3.0)
            fcs = ampyl.flavor.FlavorChannelSpace(
                fc_list=[fc_two, fc_three])
            fvs = ampyl.spaces.FiniteVolumeSetup()
            tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
            qcis = ampyl.spaces.QCIndexSpace(
                fcs=fcs, fvs=fvs, tbis=tbis, Emax=4.6, Lmax=5.0)
            qcis.populate()
        self.assertEqual(qcis.n_two_channels, 1)
        for shell_dicts in qcis.proj_dicts_by_sc_and_shellset:
            self.assertGreater(len(shell_dicts), 0)
            for shell_dict in shell_dicts:
                self.assertIsInstance(shell_dict, dict)


if __name__ == '__main__':
    unittest.main()
