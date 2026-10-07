#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conventions fixed by the K K pi fast-QC program.

Every expected value here is hand-coded from the physics, so the tests
stand alone.
"""

import unittest
from types import SimpleNamespace
import numpy as np
import ampyl
from ampyl import qc_functions
from ampyl import shell_utils
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


class TestSpectatorBoxAtNonzeroMomentum(unittest.TestCase):
    """The spectator box admits nonzero ESQmin at nonzero nP."""

    @staticmethod
    def _npspecmax(m_spec, Emax, Lmax, nPSQ, ESQmin):
        qcis = SimpleNamespace(
            fcs=SimpleNamespace(
                sc_list_sorted=[SimpleNamespace(
                    spectator=SimpleNamespace(mass=m_spec))],
                slices_by_three_masses=[[0]]),
            Emax=Emax, Lmax=Lmax, nPSQ=nPSQ,
            _get_ESQmin=lambda three_slice_index: ESQmin)
        return ampyl.spaces.QCIndexSpace._get_nPspecmax(qcis, 0)

    def test_rest_frame_value(self):
        # sigma = ESQmin = 0 with the spectator recoiling against the
        # dimer at rest: |k| = (E^2 - m^2)/(2 E)
        nmax = self._npspecmax(1.0, 5.0, 6.0, 0, 0.0)
        self.assertAlmostEqual(nmax, 6.0*(25.0-1.0)/(10.0*2.0*PI),
                               places=12)

    def test_moving_frame_bound_is_reached_parallel_to_P(self):
        m_spec, Emax, Lmax, ESQmin = MK, 5.8, 6.0, 0.972
        nP = np.array([0, 1, 1])
        nmax = self._npspecmax(m_spec, Emax, Lmax, nP@nP, ESQmin)
        nvec = nmax*nP/np.sqrt(nP@nP)
        kSQ = (2.0*PI/Lmax)**2*(nvec@nvec)
        PmkSQ = (2.0*PI/Lmax)**2*((nP-nvec)@(nP-nvec))
        sigma = (Emax-np.sqrt(m_spec**2+kSQ))**2-PmkSQ
        self.assertAlmostEqual(sigma, ESQmin, places=10)

    def test_kkpi_populates_with_default_ESQmins(self):
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = ampyl.flavor.Particle(mass=MK, spin=0.0, flavor='K')
        fc = ampyl.flavor.FlavorChannel(3, particles=[kaon, kaon, pion])
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc], ni_list=[fc])
        fvs = ampyl.spaces.FiniteVolumeSetup(nP=np.array([0, 0, 1]))
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        self.assertTrue(all(ESQmin > 0.0 for ESQmin in tbis.ESQmins))
        qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                         Emax=5.0, Lmax=5.0)
        qcis.populate()
        for slot in range(2):
            self.assertGreater(len(qcis.tbks_list[slot][0].shells), 0)


class TestMultiSliceMovingFrameFK(unittest.TestCase):
    """Multi-slice F and K at nonzero total momentum.

    For the trivial irrep at ell = 0 both are diagonal and constant on
    a little-group orbit, so each projected entry is the single-entry
    kernel at the orbit's first momentum, times the dimer factor (2 on
    F and 1/2 on K for the non-identical K pi dimer). Channels are
    ordered pion spectator then kaon spectator, orbits in shell order
    and masked with each channel's own cutoff."""

    def test_projected_entries(self):
        E, L = 5.2, 5.0
        a_vals = [0.4, -0.7]
        for nP in [np.array([0, 0, 1]), np.array([0, 1, 1])]:
            pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi')
            kaon = ampyl.flavor.Particle(mass=MK, spin=0.0, flavor='K')
            fc = ampyl.flavor.FlavorChannel(
                3, particles=[kaon, kaon, pion])
            fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc],
                                                  ni_list=[fc])
            fvs = ampyl.spaces.FiniteVolumeSetup(nP=nP)
            tbis = ampyl.spaces.ThreeBodyInteractionScheme(
                fcs=fcs, scheme_data=[[-0.4, 0.0], [-0.7, 0.0]])
            qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                             Emax=5.4, Lmax=5.2)
            qcis.populate()
            qc = ampyl.QC(qcis=qcis)
            irrep = ('A1', 0)
            f = qc.f.get_value(E=E, L=L, project=True, irrep=irrep)
            k = qc.k.get_value(E=E, L=L,
                               pcotdelta_parameter_lists=[[a] for a
                                                          in a_vals],
                               project=True, irrep=irrep)
            f_expected = []
            k_expected = []
            for sc_index, sc in enumerate(qcis.fcs.sc_list_sorted):
                identical = sc.first_dimer == sc.second_dimer
                mspec = sc.spectator.mass
                m2 = sc.first_dimer.mass
                m3 = sc.second_dimer.mass
                alpha, beta = qcis.tbis.scheme_data[sc_index]
                tbks_entry = qcis.tbks_list[
                    qcis.sc_to_three_slice[sc_index]][0]
                _, shells = shell_utils._get_active_shells(
                    qcis, sc_index, E, L, tbks_entry)
                for shell in shells:
                    nvec = tbks_entry.nvec_arr[shell[0]]
                    f_expected.append(
                        (1.0 if identical else 2.0)
                        * qc_functions.getF_single_entry(
                            E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3,
                            mspec=mspec, C1cut=qc.f.C1cut,
                            alphaKSS=qc.f.alphaKSS, alpha=alpha,
                            beta=beta))
                    k_expected.append(
                        (1.0 if identical else 0.5)
                        * qc_functions.getK_single_entry(
                            pcotdelta_parameter_list=[a_vals[sc_index]],
                            E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3,
                            mspec=mspec, alpha=alpha, beta=beta))
            self.assertGreater(len(k_expected), 4)
            for matrix, expected in [(f, f_expected), (k, k_expected)]:
                expected = np.diag(np.real(expected))
                self.assertEqual(matrix.shape, expected.shape)
                scale = np.max(np.abs(expected))
                self.assertLess(np.max(np.abs(matrix-expected)),
                                1.0e-13*scale)


if __name__ == '__main__':
    unittest.main()
