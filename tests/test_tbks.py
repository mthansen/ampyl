#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_tbks.py
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
from ampyl import ThreeBodyKinematicSpace


class TestThreeBodyKinematicSpace(unittest.TestCase):
    def test_nP_must_be_numpy_array(self):
        with self.assertRaises(ValueError):
            ThreeBodyKinematicSpace(nP=[0, 0, 0])

    def test_nP_must_have_shape_3(self):
        with self.assertRaises(ValueError):
            ThreeBodyKinematicSpace(nP=np.array([0, 0]))

    def test_nP_must_be_populated_with_ints(self):
        with self.assertRaises(ValueError):
            ThreeBodyKinematicSpace(nP=np.array([1.5, 2, 3]))

    def test_nvec_arr_sorting(self):
        nP = np.array([1, 1, 1])
        nvec_arr = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        t = ThreeBodyKinematicSpace(nP=nP, nvec_arr=nvec_arr)
        expected_nvec_arr = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        np.testing.assert_array_equal(t.nvec_arr, expected_nvec_arr)

    def test_shell_sorting(self):
        nP = np.array([1, 1, 1])
        nvec_arr = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        t = ThreeBodyKinematicSpace(nP=nP, nvec_arr=nvec_arr)
        expected_nvec_arr = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                                      [1, 0, 0]])
        np.testing.assert_array_equal(t.nvec_arr, expected_nvec_arr)
        expected_shells = [[0, 1], [1, 4]]
        self.assertEqual(t.shells, expected_shells)

    def test_get_first_sort(self):
        cut = 2
        rng = range(-cut, cut+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        t = ThreeBodyKinematicSpace()
        nvec_first_sort = t._get_first_sort(nvec_arr)
        expected = np.array([[0, 0, 0],
                            [-1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0],
                            [-1, -1, 0],
                            [-1, 0, -1],
                            [-1, 0, 1],
                            [-1, 1, 0],
                            [0, -1, -1],
                            [0, -1, 1],
                            [0, 1, -1],
                            [0, 1, 1],
                            [1, -1, 0],
                            [1, 0, -1],
                            [1, 0, 1],
                            [1, 1, 0],
                            [-1, -1, -1],
                            [-1, -1, 1],
                            [-1, 1, -1],
                            [-1, 1, 1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [1, 1, -1],
                            [1, 1, 1],
                            [-2, 0, 0],
                            [0, -2, 0],
                            [0, 0, -2],
                            [0, 0, 2],
                            [0, 2, 0],
                            [2, 0, 0],
                            [-2, -1, 0],
                            [-2, 0, -1],
                            [-2, 0, 1],
                            [-2, 1, 0],
                            [-1, -2, 0],
                            [-1, 0, -2],
                            [-1, 0, 2],
                            [-1, 2, 0],
                            [0, -2, -1],
                            [0, -2, 1],
                            [0, -1, -2],
                            [0, -1, 2],
                            [0, 1, -2],
                            [0, 1, 2],
                            [0, 2, -1],
                            [0, 2, 1],
                            [1, -2, 0],
                            [1, 0, -2],
                            [1, 0, 2],
                            [1, 2, 0],
                            [2, -1, 0],
                            [2, 0, -1],
                            [2, 0, 1],
                            [2, 1, 0],
                            [-2, -1, -1],
                            [-2, -1, 1],
                            [-2, 1, -1],
                            [-2, 1, 1],
                            [-1, -2, -1],
                            [-1, -2, 1],
                            [-1, -1, -2],
                            [-1, -1, 2],
                            [-1, 1, -2],
                            [-1, 1, 2],
                            [-1, 2, -1],
                            [-1, 2, 1],
                            [1, -2, -1],
                            [1, -2, 1],
                            [1, -1, -2],
                            [1, -1, 2],
                            [1, 1, -2],
                            [1, 1, 2],
                            [1, 2, -1],
                            [1, 2, 1],
                            [2, -1, -1],
                            [2, -1, 1],
                            [2, 1, -1],
                            [2, 1, 1],
                            [-2, -2, 0],
                            [-2, 0, -2],
                            [-2, 0, 2],
                            [-2, 2, 0],
                            [0, -2, -2],
                            [0, -2, 2],
                            [0, 2, -2],
                            [0, 2, 2],
                            [2, -2, 0],
                            [2, 0, -2],
                            [2, 0, 2],
                            [2, 2, 0],
                            [-2, -2, -1],
                            [-2, -2, 1],
                            [-2, -1, -2],
                            [-2, -1, 2],
                            [-2, 1, -2],
                            [-2, 1, 2],
                            [-2, 2, -1],
                            [-2, 2, 1],
                            [-1, -2, -2],
                            [-1, -2, 2],
                            [-1, 2, -2],
                            [-1, 2, 2],
                            [1, -2, -2],
                            [1, -2, 2],
                            [1, 2, -2],
                            [1, 2, 2],
                            [2, -2, -1],
                            [2, -2, 1],
                            [2, -1, -2],
                            [2, -1, 2],
                            [2, 1, -2],
                            [2, 1, 2],
                            [2, 2, -1],
                            [2, 2, 1],
                            [-2, -2, -2],
                            [-2, -2, 2],
                            [-2, 2, -2],
                            [-2, 2, 2],
                            [2, -2, -2],
                            [2, -2, 2],
                            [2, 2, -2],
                            [2, 2, 2]])
        self.assertTrue((nvec_first_sort == expected).all())

    def test_get_shell_sort(self):
        pass

    def test_build_shell_dict_nvec_arr(self):
        pass

    def test_build_shell_sort_with_counter(self):
        pass

    def test_populate_nvec_simple_derivatives(self):
        pass

    def test_populate_nvec_matrices(self):
        pass

    def test_populate_nvec_stacks(self):
        pass

    def test_initialize_dicts(self):
        pass

    def test_populate_dicts(self):
        pass

    def test_convert_dicts_to_lists(self):
        pass

    def test_populate_nvecSQ_stacks(self):
        pass

    def test_populate_nvec_shells(self):
        pass

    def test_build_row_shells(self):
        pass

    def test_slice_and_swap(self):
        pass

    def test_tbks_str(self):
        pass


if __name__ == '__main__':
    unittest.main()
