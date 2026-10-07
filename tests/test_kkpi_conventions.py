#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conventions fixed by the K K pi fast-QC program.

Every expected value here is hand-coded from the physics, so the tests
stand alone.
"""

import unittest
import numpy as np
import ampyl
from ampyl.constants import PI


MK = 0.09698/0.06906
IRREP = ('A1PLUS', 0)


def _build_kkpi_qcis(m_K=MK, Emax=None, Lmax=5.0):
    pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi')
    kaon = ampyl.flavor.Particle(mass=m_K, spin=0.0, flavor='K')
    fc = ampyl.flavor.FlavorChannel(3, particles=[kaon, kaon, pion])
    fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc], ni_list=[fc])
    fvs = ampyl.spaces.FiniteVolumeSetup()
    tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
    if Emax is None:
        Emax = 2.0*m_K+1.5
    qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                     Emax=Emax, Lmax=Lmax)
    qcis.populate()
    return qcis


class TestK2DimerSymmetryFactor(unittest.TestCase):
    """K2 on a non-identical dimer carries 1/2, not 2."""

    def test_zero_shell_threshold_values(self):
        """At threshold, K2 = 2 omega (16 pi / S) sqrt(sigma) (-a).

        S = 1 for the identical K K pair and S = 2 for the
        non-identical K pi pair. With m_K = 1.5 and E = 2 m_K + m_pi
        both pairs sit exactly at threshold on the zero shell, so
        q (1 - H) vanishes in floating point.
        """
        m_K = 1.5
        qcis = _build_kkpi_qcis(m_K=m_K, Emax=4.5, Lmax=4.0)
        k = ampyl.K(qcis=qcis)
        E = 2.0*m_K+1.0
        L = 4.0
        a_val = 0.3
        for sc_ind, sc in enumerate(qcis.fcs.sc_list_sorted):
            mspec = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            identical = sc.first_dimer == sc.second_dimer
            symmetry = 1.0 if identical else 2.0
            tbks_entry = qcis.tbks_list[qcis.sc_to_three_slice[sc_ind]][0]
            kshell = k.get_shell(
                E, L, mspec, m2, m3, sc_ind, sc_ind, 0,
                sc.p_cot_deltas[0], [a_val], tbks_entry, 0, False, None)
            expected = 2.0*mspec*16.0*PI/symmetry*(m2+m3)*(-a_val)
            self.assertEqual(kshell.shape, (1, 1))
            self.assertAlmostEqual(kshell[0, 0]/expected, 1.0, places=13)

    def test_threshold_shift_matches_pairwise_sum(self):
        """The a -> 0 ground shift of det[1 + (F+G) K2] at rest.

        It must approach the sum of the leading two-particle shifts,
        (2 pi / L^3)(a_KK / mu_KK + 2 a_Kpi / mu_Kpi). A spurious
        factor 4 on the K pi channel (2 on F times the former 2 on
        K2) drives the K-pi-only ratio to ~4 and the full one to ~3.1.
        """
        qcis = _build_kkpi_qcis()
        qc = ampyl.QC(qcis=qcis)
        L = 4.756853
        a_val = 0.001/MK
        thr = 2.0*MK+1.0
        mu_kk = MK/2.0
        mu_kpi = MK/(MK+1.0)
        shift_kk = 2.0*PI*a_val/(mu_kk*L**3)
        shift_kpi2 = 2.0*2.0*PI*a_val/(mu_kpi*L**3)

        def det(E, a_kk, a_kpi):
            f = qc.f.get_value(E=E, L=L, project=True, irrep=IRREP)
            g = qc.g.get_value(E=E, L=L, project=True, irrep=IRREP)
            k = qc.k.get_value(
                E=E, L=L, pcotdelta_parameter_lists=[[a_kk], [a_kpi]],
                project=True, irrep=IRREP)
            return np.linalg.det(np.eye(len(k))+(f+g)@k).real

        for a_kk, a_kpi, pred in [(a_val, a_val, shift_kk+shift_kpi2),
                                  (a_val, 0.0, shift_kk),
                                  (0.0, a_val, shift_kpi2)]:
            lo, hi = thr+1.0e-8, thr+8.0*pred
            f_lo = det(lo, a_kk, a_kpi)
            self.assertNotEqual(np.sign(f_lo), np.sign(det(hi, a_kk, a_kpi)))
            for _ in range(30):
                mid = 0.5*(lo+hi)
                if np.sign(det(mid, a_kk, a_kpi)) == np.sign(f_lo):
                    lo = mid
                else:
                    hi = mid
            ratio = (0.5*(lo+hi)-thr)/pred
            self.assertLess(abs(ratio-1.0), 1.0e-2)


if __name__ == '__main__':
    unittest.main()
