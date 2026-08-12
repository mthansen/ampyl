#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests for shell-set selection in projection dictionaries."""

import unittest
import warnings
import numpy as np
import ampyl


class TestShellsetIndex(unittest.TestCase):
    """Test shell-set selection at zero total momentum.

    ``QCIndexSpace.get_shellset_index`` returns 0 at zero total
    momentum. This is exact because the precomputed shell sets are
    nested (each smaller set is a prefix of shell set 0) and the
    per-shell projectors agree across shell sets. These tests pin both
    properties.
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

    def test_zero_momentum_returns_zero_silently(self):
        """Index 0 is selected at nP = 0 without emitting a warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            shellset_index = self.qcis.get_shellset_index(4.2, 5.0)
        self.assertEqual(shellset_index, 0)
        self.assertEqual(len(caught), 0)

    def test_shell_sets_are_nested_prefixes(self):
        """Each smaller kinematic space keeps a prefix of the shells."""
        for tbks_entries in self.qcis.tbks_list:
            shells_zero = tbks_entries[0].shells
            for tbks_entry in tbks_entries:
                self.assertEqual(
                    tbks_entry.shells,
                    shells_zero[:len(tbks_entry.shells)])

    def test_projectors_match_shell_set_zero(self):
        """Per-shell projectors equal their shell-set-0 counterparts."""
        compared = 0
        for pd_by_shellset in self.qcis.proj_dicts_by_sc_and_shellset:
            shellset_zero = pd_by_shellset[0]
            for shellset in pd_by_shellset[1:]:
                for shell_index, shell_dict in enumerate(shellset):
                    for irrep, projector in shell_dict.items():
                        self.assertIn(irrep, shellset_zero[shell_index])
                        reference = shellset_zero[shell_index][irrep]
                        self.assertEqual(projector.shape, reference.shape)
                        self.assertTrue(np.allclose(projector, reference))
                        compared += 1
        self.assertGreater(compared, 0)


if __name__ == '__main__':
    unittest.main()
