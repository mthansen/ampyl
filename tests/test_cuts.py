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
from ampyl import interpolable_utils


class TestPoleCandidates(unittest.TestCase):
    """Tests for shell nvecSQ collection in cut interpolators."""

    def setUp(self):
        """Create lightweight cut objects for helper-method tests."""
        qcis = ampyl.spaces.QCIndexSpace()
        self.f = ampyl.F(qcis=qcis)
        self.g = ampyl.G(qcis=qcis)
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
        n3vec = self.f._DIAGONAL_N3VECS[shell_nvecSQ]
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

    def test_g_pole_candidates_collect_unique_shell_pair_entries(self):
        """G should sort entries and ignore duplicate triples."""
        nvecSQs_by_shell = self._wrap_entry(
            [[5, 2], [5, 7]],
            [[1, 4], [3, 2]],
            [[4, 3], [1, 2]],
        )

        all_nvecSQs = self.g._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )

        self.assertEqual(
            all_nvecSQs,
            [[1, 4, 5], [2, 3, 4], [1, 3, 5], [2, 2, 7]],
        )

    def test_g_pole_candidates_ignore_empty_blocks(self):
        """Empty nested entries should not contribute G pole candidates."""
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

        all_nvecSQs = self.g._get_all_nvecSQs_for_pole_detection(
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

    def test_f_pole_candidates_add_diagonal_shells(self):
        """Supported diagonal shells should contribute implied triples."""
        for shell_nvecSQ in sorted(self.f._DIAGONAL_N3VECS):
            with self.subTest(shell_nvecSQ=shell_nvecSQ):
                nvecSQs_by_shell = self._wrap_entry(
                    [[shell_nvecSQ]],
                    [[shell_nvecSQ]],
                    [[shell_nvecSQ]],
                )

                all_nvecSQs = self.f._get_all_nvecSQs_for_pole_detection(
                    nvecSQs_by_shell
                )
                all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
                expected_nvecSQs = self._expected_diagonal_nvecSQs(
                    shell_nvecSQ
                )
                expected_nvecSQs.add((shell_nvecSQ,)*3)

                self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
                self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_f_pole_candidates_keep_only_diagonal_entries(self):
        """F should include concrete and implied triples only on diagonals."""
        nvecSQs_by_shell = self._wrap_entry(
            [[2, 7], [9, 5]],
            [[4, 1], [2, 6]],
            [[6, 3], [1, 8]],
        )

        all_nvecSQs = self.f._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
        expected_nvecSQs = self._expected_diagonal_nvecSQs(2)
        expected_nvecSQs.update({
            (2, 4, 6),
            (5, 6, 8),
        })

        self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
        self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_fplusg_pole_candidates_merge_f_and_g(self):
        """FplusG should merge F and G candidates without duplicates."""
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
        f_nvecSQs = self.f._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        g_nvecSQs = self.g._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        all_nvecSQs_set = {tuple(entry) for entry in all_nvecSQs}
        expected_nvecSQs = (
            {tuple(entry) for entry in f_nvecSQs}
            | {tuple(entry) for entry in g_nvecSQs}
        )

        self.assertEqual(all_nvecSQs_set, expected_nvecSQs)
        self.assertEqual(len(all_nvecSQs), len(all_nvecSQs_set))

    def test_canonicalize_pole_candidate_keeps_mass_alignment(self):
        """Sorting nvecSQs should apply the same permutation to masses."""
        pole_nvecSQs, pole_masses = interpolable_utils\
            ._canonicalize_pole_candidate(
                [5, 1, 3],
                [10.0, 20.0, 30.0],
            )

        self.assertEqual(pole_nvecSQs, (1, 3, 5))
        self.assertEqual(pole_masses, (20.0, 30.0, 10.0))

    def test_pole_textures_keep_distinct_mass_assignments(self):
        """Equal nvecSQs with different masses should remain distinct poles."""
        polefree_interp_data_list = [[[
            [],
            [],
            [
                [0, 0, [1, 2, 3], [1.0, 2.0, 3.0]],
                [0, 0, [1, 2, 3], [3.0, 2.0, 1.0]],
            ],
        ]]]

        pole_list, pole_mass_list, pole_textures_list, _ = \
            interpolable_utils._get_pole_textures(
                self.f,
                [1],
                polefree_interp_data_list,
            )

        self.assertEqual(pole_list[0].tolist(), [[1, 2, 3], [1, 2, 3]])
        self.assertEqual(
            pole_mass_list[0].tolist(),
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        )
        self.assertEqual(
            pole_textures_list[0].tolist(),
            [[[1.0]], [[1.0]]],
        )

    def test_f_pole_candidates_include_each_three_mass_slice(self):
        """F candidates should retain separate mass assignments by slice."""
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = ampyl.flavor.Particle(mass=2.0, spin=0.0, flavor='K')
        eta = ampyl.flavor.Particle(mass=3.0, spin=0.0, flavor='eta')
        fc = ampyl.flavor.FlavorChannel(3, particles=[kaon, pion, eta])
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=ampyl.flavor.FlavorChannelSpace(fc_list=[fc]),
            Emax=10.0,
        )
        f = ampyl.F(qcis=qcis)
        nvecSQs_by_shell = self._wrap_entry(
            [[2]],
            [[4]],
            [[6]],
        )

        pole_candidates = f._get_pole_candidates_for_detection(
            nvecSQs_by_shell
        )
        pole_candidate_keys = {
            (tuple(candidate[0]), tuple(candidate[1]))
            for candidate in pole_candidates
            if tuple(candidate[0]) == (2, 4, 6)
        }

        self.assertEqual(
            pole_candidate_keys,
            {
                ((2, 4, 6), (2.0, 1.0, 3.0)),
                ((2, 4, 6), (1.0, 2.0, 3.0)),
                ((2, 4, 6), (2.0, 3.0, 1.0)),
                ((2, 4, 6), (3.0, 1.0, 2.0)),
                ((2, 4, 6), (1.0, 3.0, 2.0)),
                ((2, 4, 6), (3.0, 2.0, 1.0)),
            },
        )

    def test_f_pole_candidates_unsupported_diagonal_shell(self):
        """Unsupported diagonal labels should not add implied F triples."""
        nvecSQs_by_shell = self._wrap_entry(
            [[6, 2], [7, 3]],
            [[1, 5], [8, 4]],
            [[9, 0], [2, 6]],
        )

        all_nvecSQs = self.f._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )

        self.assertEqual(
            {tuple(entry) for entry in all_nvecSQs},
            {
                (1, 6, 9),
                (3, 4, 6),
            },
        )
        self.assertEqual(len(all_nvecSQs), 2)


if __name__ == '__main__':
    unittest.main()
