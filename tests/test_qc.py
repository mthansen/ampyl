#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_qc.py
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
import numpy as np
from scipy.optimize import root_scalar
import ampyl


class TestQC(unittest.TestCase):
    def build_qc(self):
        mrho = 2.197791
        pion = ampyl.flavor.Particle(mass=1., spin=0., flavor='pi',
                                     isospin_multiplet=True, isospin=1.)
        rho = ampyl.flavor.Particle(mass=mrho, spin=1., flavor='rho',
                                    isospin_multiplet=True, isospin=1.)
        fc_three_pi = ampyl.flavor.FlavorChannel(3,
                                                 particles=[pion, pion, pion],
                                                 isospin=2.)
        fc_rho_pi = ampyl.flavor.FlavorChannel(2, particles=[rho, pion],
                                               isospin=2.)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_three_pi],
                                              ni_list=[fc_three_pi, fc_rho_pi])
        fcs.sc_list[0].p_cot_deltas[0]\
            = ampyl.functions.QCFunctions.pcotdelta_breit_wigner
        fvs = ampyl.spaces.FiniteVolumeSetup()
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(
            fcs=fcs, ESQmin=0.3, scheme_data=[-0.7, 0.0],
            use_pv_shift_prescription=[True, False],
            pv_shift_parameters=[[-20.], [0.]])
        qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                         Emax=5.5, Lmax=4.0)
        qcis.populate()
        return ampyl.QC(qcis=qcis)

    def build_qc_case(self):
        L = 16*0.06906*3.444
        k_params = [[[5.80, 2.184], [0.296]], [-9.0]]
        project = True
        irrep = ('T1MINUS', 1)
        qc_dict = {'k_params': k_params, 'project': project, 'irrep': irrep,
                   'version': 'kdf+f3inv_asym_fgcombo'}
        return L, qc_dict

    def test_qc(self):
        qc = self.build_qc()
        L, qc_dict = self.build_qc_case()
        brackets = [[4.6, 4.7], [4.7, 4.9]]
        roots = []
        for bracket in brackets:
            root = root_scalar(qc.get_value, args=(L, qc_dict),
                               bracket=bracket).root
            roots.append(root)
        roots = np.array(roots)

        roots_expected = np.array([4.63304377, 4.84871987])
        diffSQ = np.sum((roots - roots_expected)**2)
        self.assertTrue(diffSQ < 1.e-15)

    def test_qc_energy_solver_is_explicit(self):
        qc = self.build_qc()
        L, qc_dict = self.build_qc_case()
        solver = ampyl.QCEnergySolver(qc)

        self.assertFalse(hasattr(qc, 'energy_solver'))
        self.assertFalse(hasattr(qc, 'get_all_energies'))

        roots = np.array([
            solver.simple_try_at_fixed_L([4.6, 4.7], L, qc_dict),
            solver.simple_try_at_fixed_L([4.7, 4.9], L, qc_dict)
        ])

        roots_expected = np.array([4.63304377, 4.84871987])
        diffSQ = np.sum((roots - roots_expected)**2)
        self.assertTrue(diffSQ < 1.e-15)


if __name__ == '__main__':
    unittest.main()
