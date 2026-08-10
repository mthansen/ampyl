#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_two_particle_channels.py
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

import contextlib
import io
import unittest
import numpy as np
from scipy.optimize import root_scalar

import ampyl
from ampyl import qc_functions as qcf


class TestTwoParticleChannelsInFcList(unittest.TestCase):
    """Two-particle channels in fc_list must give the Luscher condition.

    The space couples one two-particle channel (an identical pair with
    mass 1.0) to a heavy three-particle channel whose threshold (4.5)
    sits above every energy probed, with its interaction switched off,
    so the QC determinant factorizes and the two-particle roots must
    match the standalone condition 1 + F2*K2 = 0 built directly from
    ``qc_functions``.
    """

    @classmethod
    def setUpClass(cls):
        cls.mass = 1.0
        cls.a_scatter = 0.1
        light = ampyl.flavor.Particle(mass=cls.mass, flavor='a')
        heavy = ampyl.flavor.Particle(mass=1.5, flavor='c')
        fc_two = ampyl.flavor.FlavorChannel(2, particles=[light, light])
        fc_three = ampyl.flavor.FlavorChannel(3, particles=[heavy]*3)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_two, fc_three],
                                              ni_list=[fc_two])
        fvs = ampyl.spaces.FiniteVolumeSetup()
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        cls.qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                             Emax=4.6, Lmax=5.5)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.qcis.populate()
        cls.qc = ampyl.QC(qcis=cls.qcis)
        cls.qc_dict = {
            'k_params': [[[cls.a_scatter], [1.0e-10]], [0.0]],
            'project': True,
            'irrep': ('A1PLUS', 0),
        }

    def _qc_full(self, E, L):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.qc.get_value(E=E, L=L, qc_dict=self.qc_dict)

    def _qc_direct(self, E, L):
        Ftwo = qcf.getFtwo_single_entry(
            E2=E, nP2=np.array([0, 0, 0]), L=L,
            m1=self.mass, m2=self.mass, C1cut=5, alphaKSS=1.0)
        Ktwo = qcf.getK_single_entry(
            pcotdelta_function=qcf.pcotdelta_scattering_length,
            pcotdelta_parameter_list=[self.a_scatter],
            E=E, npspec=np.array([0, 0, 0]), L=L,
            m1=self.mass, m2=self.mass, mspec=0.0, ell=0,
            qc_impl={'hermitian': False})
        return 1.0 + Ftwo*Ktwo

    def test_slot_mappings(self):
        """Two-particle channels map to slot 0, three-slices shift by one."""
        self.assertEqual(self.qcis.n_two_channels, 1)
        self.assertEqual(self.qcis.sc_to_three_slice, [0, 1])
        sub_indices = self.qcis.get_tbks_sub_indices(E=3.0, L=5.0)
        self.assertEqual(len(sub_indices), len(self.qcis.tbks_list))

    def test_matrix_shapes_agree(self):
        """F, G and K must share one block layout."""
        E, L = 3.0, 5.0
        project, irrep = True, ('A1PLUS', 0)
        with contextlib.redirect_stdout(io.StringIO()):
            f_mat = self.qc.f.get_value(E=E, L=L, project=project,
                                        irrep=irrep)
            g_mat = self.qc.g.get_value(E=E, L=L, project=project,
                                        irrep=irrep)
            k_mat = self.qc.k.get_value(
                E=E, L=L,
                pcotdelta_parameter_lists=self.qc_dict['k_params'][0],
                project=project, irrep=irrep)
        self.assertEqual(f_mat.shape, g_mat.shape)
        self.assertEqual(f_mat.shape, k_mat.shape)
        # pair block is first and G has no two-particle coupling
        self.assertEqual(g_mat[0, 0], 0.0)
        self.assertTrue((g_mat[0, 1:] == 0.0).all())
        self.assertTrue((g_mat[1:, 0] == 0.0).all())

    def test_ground_state_matches_direct_luscher(self):
        """Full-QC ground state equals the standalone two-particle root."""
        L = 5.0
        root_direct = root_scalar(self._qc_direct, args=(L,),
                                  bracket=[2.0+1.0e-9, 2.5]).root
        root_full = root_scalar(self._qc_full, args=(L,),
                                bracket=[root_direct-1.0e-4,
                                         root_direct+1.0e-4]).root
        self.assertAlmostEqual(root_full, root_direct, places=10)

    def test_ground_state_matches_threshold_expansion(self):
        """Ground-state shift follows the Luscher threshold expansion."""
        L = 5.0
        root_full = root_scalar(self._qc_full, args=(L,),
                                bracket=[2.0+1.0e-9, 2.5]).root
        shift = root_full-2.0*self.mass
        a_over_L = self.a_scatter/L
        expansion = (4.0*np.pi*self.a_scatter/(self.mass*L**3)
                     * (1.0+2.837297*a_over_L+6.375183*a_over_L**2))
        self.assertAlmostEqual(shift, expansion, places=4)


class TestPureTwoParticleSpace(unittest.TestCase):
    """A purely two-particle fc_list must work without a dummy channel."""

    @classmethod
    def setUpClass(cls):
        cls.masses = [1.0, 1.3]
        cls.a_values = [0.14, -0.06]
        particles = [ampyl.flavor.Particle(mass=m, flavor=f)
                     for m, f in zip(cls.masses, 'ab')]
        fc_two = [ampyl.flavor.FlavorChannel(2, particles=[p, p])
                  for p in particles]
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=fc_two,
                                              ni_list=fc_two)
        fvs = ampyl.spaces.FiniteVolumeSetup()
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        cls.qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                             Emax=4.2, Lmax=5.5)
        with contextlib.redirect_stdout(io.StringIO()):
            cls.qcis.populate()
        cls.qc = ampyl.QC(qcis=cls.qcis)
        cls.qc_dict = {
            'k_params': [[[a] for a in cls.a_values], [0.0]],
            'project': True,
            'irrep': ('A1PLUS', 0),
        }

    def _qc_full(self, E, L):
        with contextlib.redirect_stdout(io.StringIO()):
            return self.qc.get_value(E=E, L=L, qc_dict=self.qc_dict)

    def _qc_direct(self, E, L, m, a):
        Ftwo = qcf.getFtwo_single_entry(
            E2=E, nP2=np.array([0, 0, 0]), L=L,
            m1=m, m2=m, C1cut=5, alphaKSS=1.0)
        Ktwo = qcf.getK_single_entry(
            pcotdelta_function=qcf.pcotdelta_scattering_length,
            pcotdelta_parameter_list=[a],
            E=E, npspec=np.array([0, 0, 0]), L=L,
            m1=m, m2=m, mspec=0.0, ell=0,
            qc_impl={'hermitian': False})
        return 1.0 + Ftwo*Ktwo

    def test_populate_without_three_particle_channel(self):
        """The index space populates with zero three-particle slices."""
        self.assertEqual(self.qcis.fcs.n_three_slices, 0)
        self.assertEqual(self.qcis.n_two_channels, 2)
        self.assertEqual(self.qcis.sc_to_three_slice, [0, 0])
        self.assertIn(('A1PLUS', 0), self.qcis.proj_dict.keys())

    def test_matrix_layout_is_two_by_two(self):
        """F, G and K reduce to the pair blocks alone."""
        E, L = 3.0, 5.0
        with contextlib.redirect_stdout(io.StringIO()):
            f_mat = self.qc.f.get_value(E=E, L=L, project=True,
                                        irrep=('A1PLUS', 0))
            g_mat = self.qc.g.get_value(E=E, L=L, project=True,
                                        irrep=('A1PLUS', 0))
        self.assertEqual(f_mat.shape, (2, 2))
        self.assertEqual(g_mat.shape, (2, 2))
        self.assertTrue((g_mat == 0.0).all())
        self.assertEqual(f_mat[0, 1], 0.0)
        self.assertEqual(f_mat[1, 0], 0.0)

    def test_roots_match_direct_luscher(self):
        """Both channels' ground states match the standalone condition."""
        L = 5.0
        for m, a in zip(self.masses, self.a_values):
            if a > 0:
                bracket = [2.0*m+1.0e-9, 2.0*m+0.4]
            else:
                bracket = [2.0*m-0.2, 2.0*m-1.0e-9]
            root_direct = root_scalar(self._qc_direct, args=(L, m, a),
                                      bracket=bracket).root
            root_full = root_scalar(self._qc_full, args=(L,),
                                    bracket=[root_direct-1.0e-4,
                                             root_direct+1.0e-4]).root
            self.assertAlmostEqual(root_full, root_direct, places=10)


if __name__ == '__main__':
    unittest.main()
