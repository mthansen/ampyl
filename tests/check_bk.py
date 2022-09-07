#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# check_bk.py
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

import numpy as np
import unittest
from ampyl import BKFunctions


class TestBKFunctions(unittest.TestCase):
    """Class to test the FlavorChannel class."""

    def check_YY_recon(self, nvec_arr, ell1, mazi1, ell2, mazi2):
        """Check YY recon."""
        nvec_mag = np.sqrt(nvec_arr@nvec_arr)
        nvec_arr = np.array([nvec_arr])
        YY = BKFunctions.cart_sph_harm(ell1, mazi1, nvec_arr)[0]\
            / (np.sqrt(4.*np.pi)*nvec_mag**ell1)\
            * np.conjugate(
                BKFunctions.cart_sph_harm(ell2, mazi2, nvec_arr)[0])\
            / (np.sqrt(4.*np.pi)*nvec_mag**ell2)

        recombine_YY_set = BKFunctions.recombine_YY(ell1, mazi1,
                                                    ell2, mazi2)

        YY_recombined = 0.
        for entry in recombine_YY_set:
            [ell, mazi, coeff] = entry
            YY_recombined = YY_recombined\
                + coeff*BKFunctions.cart_sph_harm(ell, mazi, nvec_arr)[0]\
                / (np.sqrt(4.*np.pi)*nvec_mag**ell)
        return np.abs(YY-YY_recombined)

    def check_YY_recon_real(self, nvec_arr, ell1, mazi1, ell2, mazi2):
        """Check YY recon real."""
        nvec_mag = np.sqrt(nvec_arr@nvec_arr)
        nvec_arr = np.array([nvec_arr])
        YY = BKFunctions.cart_sph_harm_real(ell1, mazi1, nvec_arr)[0]\
            / (np.sqrt(4.*np.pi)*nvec_mag**ell1)\
            * BKFunctions.cart_sph_harm_real(ell2, mazi2, nvec_arr)[0]\
            / (np.sqrt(4.*np.pi)*nvec_mag**ell2)

        recombine_YY_real_set = BKFunctions.recombine_YY_real(ell1, mazi1,
                                                              ell2, mazi2)

        YY_recombined = 0.
        for entry in recombine_YY_real_set:
            [ell, mazi, coeff] = entry
            YY_recombined = YY_recombined\
                + coeff*BKFunctions.cart_sph_harm_real(ell, mazi, nvec_arr)[0]\
                / (np.sqrt(4.*np.pi)*nvec_mag**ell)
        return np.abs(YY-YY_recombined)

    def setUp(self):
        """Exectue set-up."""
        self.epsilon = 1.e-15
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def test_bk_one(self):
        """Test BK, first test."""
        ell1 = 2
        mazi1 = 1
        ell2 = 4
        mazi2 = 2

        nvec_arr = np.array([1., 2., 3.])
        diff = self.check_YY_recon(nvec_arr, ell1, mazi1, ell2, mazi2)
        self.assertTrue(diff < self.epsilon)

        nvec_arr = np.array([0., 1., 2.])
        diff = self.check_YY_recon(nvec_arr, ell1, mazi1, ell2, mazi2)
        self.assertTrue(diff < self.epsilon)

    def test_bk_two(self):
        """Test BK, second test."""
        ell1 = 2
        mazi1 = 1
        ell2 = 4
        mazi2 = 2

        nvec_arr = np.array([1., 2., 3.])
        diff = self.check_YY_recon_real(nvec_arr, ell1, mazi1, ell2, mazi2)
        self.assertTrue(diff < self.epsilon)

        nvec_arr = np.array([0., 1., 2.])
        diff = self.check_YY_recon_real(nvec_arr, ell1, mazi1, ell2, mazi2)
        self.assertTrue(diff < self.epsilon)


class Template(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Exectue set-up."""
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def __example(self, x):
        return x

    def test(self):
        """Example test."""
        self.assertEqual(10., self.__example(10.))


unittest.main()
