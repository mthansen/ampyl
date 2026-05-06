#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import ampyl


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


if __name__ == '__main__':
    unittest.main()
