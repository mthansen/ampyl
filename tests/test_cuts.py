#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created April 2026.

@author: M.T. Hansen and OpenAI Codex
"""

###############################################################################
#
# test_cuts.py
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


class TestFplusG(unittest.TestCase):
    """Tests for shell nvecSQ collection in FplusG."""

    def setUp(self):
        """Create a lightweight FplusG object for helper-method tests."""
        self.fplusg = ampyl.FplusG(qcis=ampyl.spaces.QCIndexSpace())

    def _wrap_entry(self, n1vecSQs, n2vecSQs, n3vecSQs):
        """Build the nested shell structure consumed by _get_all_nvecSQs."""
        entry = [[n1vecSQs, n2vecSQs, n3vecSQs], None, None]
        return [[[[entry, []]]]]

    def _entry(self, n1vecSQs, n2vecSQs, n3vecSQs):
        """Build a non-empty shell entry for custom nested fixtures."""
        return [[n1vecSQs, n2vecSQs, n3vecSQs], None, None]

    def _expected_diagonal_nvecSQs(self, shell_nvecSQ):
        """Build the diagonal nvecSQ triples for a given shell label."""
        n3vec = self.fplusg._DIAGONAL_N3VECS[shell_nvecSQ]
        all_nvecSQs = set()
        for n1_entry in np.ndindex((5, 5, 5)):
            n1vec = np.array(n1_entry)-2
            n2vec = -n1vec-n3vec
            all_nvecSQs.add(tuple(sorted([
                int(n1vec@n1vec),
                int(n2vec@n2vec),
                int(n3vec@n3vec),
            ])))
        return all_nvecSQs

    def test_get_all_nvecSQs_for_pole_detection_collects_unique_entries(self):
        """The collector should sort entries and ignore duplicate triples."""
        nvecSQs_by_shell = self._wrap_entry(
            [[5, 2], [5, 7]],
            [[1, 4], [3, 2]],
            [[4, 3], [1, 2]],
        )

        all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )

        self.assertEqual(
            all_nvecSQs,
            [[1, 4, 5], [2, 3, 4], [1, 3, 5], [2, 2, 7]],
        )

    def test_get_all_nvecSQs_for_pole_detection_ignores_empty_blocks(self):
        """Empty nested entries should not contribute pole candidates."""
        nvecSQs_by_shell = [
            [
                [
                    [
                        [],
                        self._entry(
                            [[5, 2], [5, 7]],
                            [[1, 4], [3, 2]],
                            [[4, 3], [1, 2]],
                        ),
                    ],
                    [
                        [],
                        [],
                    ],
                ],
                [
                    [
                        self._entry(
                            [[8]],
                            [[6]],
                            [[2]],
                        ),
                    ],
                ],
            ],
        ]

        all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )

        self.assertEqual(
            {tuple(entry) for entry in all_nvecSQs},
            {
                (1, 4, 5),
                (2, 3, 4),
                (1, 3, 5),
                (2, 2, 7),
                (2, 6, 8),
            },
        )
        self.assertEqual(len(all_nvecSQs), 5)

    def test_get_all_nvecSQs_for_pole_detection_adds_diagonal_shells(self):
        """Supported diagonal shells should contribute their implied triples."""
        for shell_nvecSQ in sorted(self.fplusg._DIAGONAL_N3VECS):
            with self.subTest(shell_nvecSQ=shell_nvecSQ):
                nvecSQs_by_shell = self._wrap_entry(
                    [[shell_nvecSQ]],
                    [[shell_nvecSQ]],
                    [[shell_nvecSQ]],
                )

                all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
                    nvecSQs_by_shell
                )
                all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
                expected_nvecSQs = self._expected_diagonal_nvecSQs(
                    shell_nvecSQ
                )
                expected_nvecSQs.add((shell_nvecSQ,)*3)

                self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
                self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_get_all_nvecSQs_for_pole_detection_diagonal_shell_explicit_case(
            self):
        """A diagonal shell should include concrete and implied triples."""
        nvecSQs_by_shell = self._wrap_entry(
            [[2, 7], [9, 5]],
            [[4, 1], [2, 6]],
            [[6, 3], [1, 8]],
        )

        all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
        expected_nvecSQs = self._expected_diagonal_nvecSQs(2)
        expected_nvecSQs.update({
            (2, 4, 6),
            (1, 3, 7),
            (1, 2, 9),
            (5, 6, 8),
        })

        self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
        self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_get_all_nvecSQs_for_pole_detection_deduplicates_nested_blocks(
            self):
        """Repeated concrete and diagonal triples should appear only once."""
        nvecSQs_by_shell = [
            [
                [
                    [
                        self._entry(
                            [[1, 4], [1, 6]],
                            [[2, 5], [3, 6]],
                            [[3, 6], [4, 6]],
                        ),
                        [],
                        self._entry(
                            [[1, 4], [1, 6]],
                            [[2, 5], [3, 6]],
                            [[3, 6], [4, 6]],
                        ),
                    ],
                    [
                        [],
                        self._entry(
                            [[1]],
                            [[1]],
                            [[1]],
                        ),
                    ],
                ],
                [
                    [
                        self._entry(
                            [[9, 4], [6, 4]],
                            [[0, 5], [7, 4]],
                            [[3, 6], [8, 4]],
                        ),
                    ],
                ],
            ],
        ]

        all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
        expected_nvecSQs = self._expected_diagonal_nvecSQs(1)
        expected_nvecSQs.update({
            (1, 2, 3),
            (4, 5, 6),
            (1, 3, 4),
            (6, 6, 6),
            (1, 1, 1),
            (0, 3, 9),
            (4, 5, 6),
            (6, 7, 8),
            (4, 4, 4),
        })

        self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
        self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_get_all_nvecSQs_for_pole_detection_unsupported_diagonal_shell(
            self):
        """Unsupported diagonal labels should not add implied triples."""
        nvecSQs_by_shell = self._wrap_entry(
            [[6, 2], [7, 3]],
            [[1, 5], [8, 4]],
            [[9, 0], [2, 6]],
        )

        all_nvecSQs = self.fplusg._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )

        self.assertEqual(
            {tuple(entry) for entry in all_nvecSQs},
            {
                (1, 6, 9),
                (0, 2, 5),
                (2, 7, 8),
                (3, 4, 6),
            },
        )
        self.assertEqual(len(all_nvecSQs), 4)


if __name__ == '__main__':
    unittest.main()
