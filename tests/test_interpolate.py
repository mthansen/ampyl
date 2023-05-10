#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_interpolate.py
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
import ampyl
import warnings
warnings.simplefilter("once")


class TestInterpolate(unittest.TestCase):
    """Class to test the G class."""

    def setUp(self):
        """Set up the test."""
        pion = ampyl.Particle(mass=1., spin=0., isospin=1.)
        fc = ampyl.FlavorChannel(3, particles=[pion, pion, pion], isospin=2.)
        fcs = ampyl.FlavorChannelSpace(fc_list=[fc])
        fvs = ampyl.FiniteVolumeSetup()
        tbis = ampyl.ThreeBodyInteractionScheme()
        qcis = ampyl.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis, Emax=5.2,
                                  Lmax=5.2)
        qcis.populate()
        g = ampyl.G(qcis=qcis)
        fplusg = ampyl.FplusG(qcis=qcis)

        Emin = 2.500000001
        Emax = 5.0
        Estep = 0.5
        Lmin = 2.
        Lmax = 5.0
        Lstep = 1.0
        g.build_interpolator(Emin, Emax, Estep,
                             Lmin, Lmax, Lstep,
                             project=True, irrep=('T1MINUS', 1))
        fplusg.build_interpolator(Emin, Emax, Estep,
                                  Lmin, Lmax, Lstep,
                                  project=True, irrep=('T1MINUS', 1))
        self.qcis = qcis
        self.g = g
        self.fplusg = fplusg

    def test_g_interpolate(self):
        E = 4.0
        L = 4.0
        self.g.qcis.fvs.qc_impl['g_interpolate'] = True
        gval1 = self.g.get_value(E=E, L=L, project=True, irrep=('T1MINUS', 1))
        self.g.qcis.fvs.qc_impl['g_interpolate'] = False
        gval2 = self.g.get_value(E=E, L=L, project=True, irrep=('T1MINUS', 1))
        diff = np.sum(np.abs(gval1-gval2))
        avg = np.sum(np.abs(gval1)+np.abs(gval2))/0.5
        self.assertLess(diff/avg, 1e-8)

    def test_fplusg_interpolate(self):
        E = 4.0
        L = 4.0
        self.fplusg.qcis.fvs.qc_impl['g_interpolate'] = True
        fplusgval1 = self.fplusg.get_value(E=E, L=L, project=True,
                                           irrep=('T1MINUS', 1))
        self.fplusg.qcis.fvs.qc_impl['g_interpolate'] = False
        fplusgval2 = self.fplusg.get_value(E=E, L=L, project=True,
                                           irrep=('T1MINUS', 1))
        diff = np.sum(np.abs(fplusgval1-fplusgval2))
        avg = np.sum(np.abs(fplusgval1)+np.abs(fplusgval2))/0.5
        print(diff, avg)


if __name__ == '__main__':
    unittest.main()
