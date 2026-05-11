#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-interacting momentum-set utilities used by :mod:`ampyl.spaces`.
"""

###############################################################################
#
# nonint_utils.py
#
# MIT License
# Copyright (c) 2026 Maxwell T. Hansen
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

import warnings
from copy import deepcopy

import numpy as np

from .constants import TWOPI
from .constants import FOURPI2
from .constants import bcolors

warnings.simplefilter("once")


class NonIntSpaceUtils:
    """Helper methods for non-interacting momentum-set construction."""

    def _load_ni_data_three(self, fc):
        [m1, m2, m3] = fc.masses
        Emax = self.Emax
        nP = self.nP
        Lmax = self.Lmax
        PSQ = (nP@nP)*FOURPI2/Lmax**2
        ECMSQ_max = Emax**2-PSQ

        pSQ = 0.
        m_pairs = [[m1, m2], [m1, m3], [m2, m3]]
        for m_pair in m_pairs:
            ma, mb = m_pair
            if ma == mb:
                pSQ_tmp = ECMSQ_max/4.-ma**2
                m_max = ma
            else:
                pSQ_tmp = (ECMSQ_max**2
                           + (ma**2 - mb**2)**2
                           - 2*ECMSQ_max*(ma**2 + mb**2))/(4.*ECMSQ_max)
                m_max = max(ma, mb)
            if pSQ_tmp > pSQ:
                pSQ = pSQ_tmp
                omp = np.sqrt(pSQ+m_max**2)

        beta = np.sqrt(nP@nP)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = p_cutoff*Lmax/TWOPI
        nvec_int_cutoff = int(nvec_cutoff)+1
        rng = range(-nvec_int_cutoff, nvec_int_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        carr = (nvecs*nvecs).sum(1) > nvec_cutoff**2
        nvecs = np.delete(nvecs, np.where(carr), axis=0)
        return [m1, m2, m3, Emax, nP, Lmax, nvec_int_cutoff, nvecs]

    def _get_nvecset_abc_three(self, nvecset_abc, nmin, nmax,
                               m1, m2, m3, Emax, nP, Lmax, n1, n2):
        n3 = nP-n1-n2
        n1SQ = n1@n1
        n2SQ = n2@n2
        n3SQ = n3@n3
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2)), *(list(n3))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_abc = nvecset_abc+[[n1, n2, n3]]
        return [nvecset_abc, nmin, nmax]

    def _square_and_sort_three(self, nvecset_abc, nmin, nmax,
                               m1, m2, m3, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_abc_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            n1 = nvecset_abc[i][0]
            n2 = nvecset_abc[i][1]
            n3 = nvecset_abc[i][2]
            n1SQ = n1@n1
            n2SQ = n2@n2
            n3SQ = n3@n3
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            n3_as_num = (n3[2]-nmin)\
                + (n3[1]-nmin)*numsys+(n3[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num,
                                                    n3_as_num]]
            nvecset_abc_SQs = nvecset_abc_SQs+[[n1SQ, n2SQ, n3SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_abc_SQs = np.array(nvecset_abc_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(4):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
        nvecset_abc = nvecset_abc[re_indexing]
        nvecset_abc_SQs = nvecset_abc_SQs[re_indexing]
        return [nvecset_abc, nvecset_abc_SQs]

    def _get_nvecset_aaa_three(self, nvecset_abc, nvecset_abc_SQs):
        nvecset_aaa = []
        nvecset_aaa_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            [n1, n2, n3] = nvecset_abc[i]
            candidates = self._permuted_three_candidates(
                n1, n2, n3, self._aaa_three_permutations())
            include_entry = self._include_symmetrized_entry(
                candidates, nvecset_aaa)
            if include_entry:
                nvecset_aaa = nvecset_aaa+[[n1, n2, n3]]
                nvecset_aaa_SQs = nvecset_aaa_SQs+[nvecset_abc_SQs[i]]
        nvecset_aaa = np.array(nvecset_aaa)
        nvecset_aaa_SQs = np.array(nvecset_aaa_SQs)
        return [nvecset_aaa, nvecset_aaa_SQs]

    def _get_nvecset_aab_three(self, nvecset_abc, nvecset_abc_SQs):
        nvecset_aab = []
        nvecset_aab_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            [n1, n2, n3] = nvecset_abc[i]
            candidates = self._permuted_three_candidates(
                n1, n2, n3, self._aab_three_permutations())
            include_entry = self._include_symmetrized_entry(
                candidates, nvecset_aab)
            if include_entry:
                nvecset_aab = nvecset_aab+[[n1, n2, n3]]
                nvecset_aab_SQs = nvecset_aab_SQs+[nvecset_abc_SQs[i]]
        nvecset_aab = np.array(nvecset_aab)
        nvecset_aab_SQs = np.array(nvecset_aab_SQs)
        return [nvecset_aab, nvecset_aab_SQs]

    @staticmethod
    def _aaa_three_permutations():
        return [(0, 1, 2), (1, 2, 0), (2, 0, 1),
                (2, 1, 0), (1, 0, 2), (0, 2, 1)]

    @staticmethod
    def _aab_three_permutations():
        return [(0, 1, 2), (1, 0, 2)]

    @staticmethod
    def _aaa_two_permutations():
        return [(0, 1), (1, 0)]

    @staticmethod
    def _permuted_three_candidates(n1, n2, n3, permutations):
        nvecs = [n1, n2, n3]
        return [np.array([nvecs[index] for index in permutation])
                for permutation in permutations]

    @staticmethod
    def _permuted_two_candidates(n1, n2, permutations):
        nvecs = [n1, n2]
        return [np.array([nvecs[index] for index in permutation])
                for permutation in permutations]

    @staticmethod
    def _include_symmetrized_entry(candidates, nvecset):
        include_entry = True
        for candidate in candidates:
            for nvecset_tmp_entry in nvecset:
                nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                include_entry = include_entry\
                    and (not ((candidate == nvecset_tmp_entry).all()))
        return include_entry

    def _reps_and_batches_permutations(self, nvecset, nvecset_SQs, nP,
                                       permutations):
        nvecset_reps = [nvecset[0]]
        nvecset_SQreps = [nvecset_SQs[0]]
        nvecset_inds = [0]
        nvecset_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_reps)):
                        n_included = np.array(nvecset_reps[k])
                        candidates = self._permuted_candidates(
                            nvecset[j]@g_elem, permutations)
                        if not self._include_symmetrized_entry(
                                candidates, [n_included]):
                            already_included = True
                            nvecset_counts[k] = nvecset_counts[k]+1
            if not already_included:
                nvecset_reps = nvecset_reps+[nvecset[j]]
                nvecset_SQreps = nvecset_SQreps+[nvecset_SQs[j]]
                nvecset_inds = nvecset_inds+[j]
                nvecset_counts = nvecset_counts+[1]

        nvecset_batched = list(np.arange(len(nvecset_reps)))
        for j in range(len(nvecset)):
            for k in range(len(nvecset_reps)):
                include_entry = False
                n_rep = np.array(nvecset_reps[k])
                for g_elem in G:
                    candidates = self._permuted_candidates(
                        nvecset[j]@g_elem, permutations)
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_batched[k], np.int64):
                        nvecset_batched[k] = [nvecset[j]]
                    else:
                        nvecset_batched[k] = nvecset_batched[k]\
                            + [nvecset[j]]

        for j in range(len(nvecset_batched)):
            nvecset_batched[j] = np.array(nvecset_batched[j])
        return [nvecset_reps, nvecset_SQreps, nvecset_inds,
                nvecset_counts, nvecset_batched]

    @staticmethod
    def _permuted_candidates(nvecset_entry, permutations):
        return [np.array([nvecset_entry[index] for index in permutation])
                for permutation in permutations]

    def _reps_and_batches_three(self, nvecset_abc, nvecset_abc_SQs,
                                nvecset_aaa, nvecset_aaa_SQs,
                                nP):
        nvecset_abc_reps = [nvecset_abc[0]]
        nvecset_aaa_reps = deepcopy([nvecset_aaa[0]])
        nvecset_abc_SQreps = [nvecset_abc_SQs[0]]
        nvecset_aaa_SQreps = deepcopy([nvecset_aaa_SQs[0]])
        nvecset_abc_inds = [0]
        nvecset_aaa_inds = deepcopy([0])
        nvecset_abc_counts = deepcopy([0])
        nvecset_aaa_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_abc)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_abc_reps)):
                        n_included = nvecset_abc_reps[k]
                        if (nvecset_abc[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_abc_counts[k] = nvecset_abc_counts[k]+1
            if not already_included:
                nvecset_abc_reps = nvecset_abc_reps+[nvecset_abc[j]]
                nvecset_abc_SQreps = nvecset_abc_SQreps+[nvecset_abc_SQs[j]]
                nvecset_abc_inds = nvecset_abc_inds+[j]
                nvecset_abc_counts = nvecset_abc_counts+[1]

        for j in range(len(nvecset_aaa)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_aaa_reps)):
                        n_included = nvecset_aaa_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2, n3] = nvecset_aaa[j]@g_elem
                        candidates = [np.array([n1, n2, n3]),
                                      np.array([n2, n3, n1]),
                                      np.array([n3, n1, n2]),
                                      np.array([n3, n2, n1]),
                                      np.array([n2, n1, n3]),
                                      np.array([n1, n3, n2])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_aaa_counts[k]\
                                = nvecset_aaa_counts[k]+1
            if not already_included:
                nvecset_aaa_reps = nvecset_aaa_reps\
                    + [nvecset_aaa[j]]
                nvecset_aaa_SQreps = nvecset_aaa_SQreps\
                    + [nvecset_aaa_SQs[j]]
                nvecset_aaa_inds = nvecset_aaa_inds+[j]
                nvecset_aaa_counts = nvecset_aaa_counts+[1]

        nvecset_abc_batched = list(np.arange(len(nvecset_aaa_reps)))
        for j in range(len(nvecset_abc)):
            for k in range(len(nvecset_aaa_reps)):
                include_entry = False
                n_rep = nvecset_aaa_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_abc[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_abc_batched[k], np.int64):
                        nvecset_abc_batched[k] = [nvecset_abc[j]]
                    else:
                        nvecset_abc_batched[k] = nvecset_abc_batched[k]\
                            + [nvecset_abc[j]]

        nvecset_aaa_batched\
            = list(np.arange(len(nvecset_aaa_reps)))
        for j in range(len(nvecset_aaa)):
            for k in range(len(nvecset_aaa_reps)):
                include_entry = False
                n_rep = nvecset_aaa_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_aaa[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_aaa_batched[k], np.int64):
                        nvecset_aaa_batched[k] = [nvecset_aaa[j]]
                    else:
                        nvecset_aaa_batched[k]\
                            = nvecset_aaa_batched[k]\
                            + [nvecset_aaa[j]]

        for j in range(len(nvecset_abc_batched)):
            nvecset_abc_batched[j] = np.array(nvecset_abc_batched[j])

        for j in range(len(nvecset_aaa_batched)):
            nvecset_aaa_batched[j]\
                = np.array(nvecset_aaa_batched[j])
        return [nvecset_abc_reps, nvecset_aaa_reps,
                nvecset_abc_SQreps, nvecset_aaa_SQreps,
                nvecset_abc_inds, nvecset_aaa_inds,
                nvecset_abc_counts, nvecset_aaa_counts,
                nvecset_abc_batched, nvecset_aaa_batched]

    def _load_ni_data_two(self, fc):
        Emax = self.Emax
        nP = self.nP
        nPSQ = nP@nP
        Lmax = self.Lmax

        [m1, m2] = fc.masses
        ECMSQ = Emax**2-FOURPI2*nPSQ/Lmax**2
        pSQ = (ECMSQ**2-2.0*ECMSQ*m1**2
               + m1**4-2.0*ECMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*ECMSQ)
        mmax = np.max([m1, m2])
        omp = np.sqrt(pSQ+mmax**2)
        beta = np.sqrt(nPSQ)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = int(p_cutoff*Lmax/TWOPI)+1
        warnings.warn(f"\n{bcolors.WARNING}"
                      "nvec_cutoff was increased by one. "
                      "This needs to be checked."
                      f"{bcolors.ENDC}", stacklevel=2)
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]

    def _get_nvecset_ab_two(self, nvecset_ab, nmin, nmax, m1, m2,
                            Emax, nP, Lmax, n1):
        n2 = nP-n1
        n1SQ = n1@n1
        n2SQ = n2@n2
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_ab = nvecset_ab+[[n1, n2]]
        return [nvecset_ab, nmin, nmax]

    def _square_and_sort_two(self, nvecset_ab, nmin, nmax,
                             m1, m2, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_ab_SQs = deepcopy([])
        for i in range(len(nvecset_ab)):
            n1 = nvecset_ab[i][0]
            n2 = nvecset_ab[i][1]
            n1SQ = n1@n1
            n2SQ = n2@n2
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num]]
            nvecset_ab_SQs = nvecset_ab_SQs+[[n1SQ, n2SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_ab_SQs = np.array(nvecset_ab_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(3):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
        nvecset_ab = nvecset_ab[re_indexing]
        nvecset_ab_SQs = nvecset_ab_SQs[re_indexing]
        return [nvecset_ab, nvecset_ab_SQs]

    def _get_nvecset_aa_two(self, nvecset_ab, nvecset_ab_SQs):
        nvecset_aa = []
        nvecset_aa_SQs = deepcopy([])
        for i in range(len(nvecset_ab)):
            [n1, n2] = nvecset_ab[i]
            candidates = [np.array([n1, n2]),
                          np.array([n2, n1])]
            include_entry = True
            for candidate in candidates:
                for nvecset_tmp_entry in nvecset_aa:
                    nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                    include_entry = include_entry\
                        and (not ((candidate == nvecset_tmp_entry)
                                  .all()))
            if include_entry:
                nvecset_aa = nvecset_aa+[[n1, n2]]
                nvecset_aa_SQs = nvecset_aa_SQs+[nvecset_ab_SQs[i]]
        nvecset_aa = np.array(nvecset_aa)
        nvecset_aa_SQs = np.array(nvecset_aa_SQs)
        return [nvecset_aa, nvecset_aa_SQs]

    def _reps_and_batches_two(self, nvecset_ab, nvecset_ab_SQs, nvecset_aa,
                              nvecset_aa_SQs, nP):
        nvecset_ab_reps = [nvecset_ab[0]]
        nvecset_aa_reps = deepcopy([nvecset_aa[0]])
        nvecset_ab_SQreps = [nvecset_ab_SQs[0]]
        nvecset_aa_SQreps = deepcopy([nvecset_aa_SQs[0]])
        nvecset_ab_inds = [0]
        nvecset_aa_inds = deepcopy([0])
        nvecset_ab_counts = deepcopy([0])
        nvecset_aa_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_ab)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_ab_reps)):
                        n_included = nvecset_ab_reps[k]
                        if (nvecset_ab[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_ab_counts[k] = nvecset_ab_counts[k]+1
            if not already_included:
                nvecset_ab_reps = nvecset_ab_reps+[nvecset_ab[j]]
                nvecset_ab_SQreps = nvecset_ab_SQreps+[nvecset_ab_SQs[j]]
                nvecset_ab_inds = nvecset_ab_inds+[j]
                nvecset_ab_counts = nvecset_ab_counts+[1]

        for j in range(len(nvecset_aa)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_aa_reps)):
                        n_included = nvecset_aa_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2] = nvecset_aa[j]@g_elem
                        candidates = [np.array([n1, n2]),
                                      np.array([n2, n1])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_aa_counts[k]\
                                = nvecset_aa_counts[k]+1
            if not already_included:
                nvecset_aa_reps = nvecset_aa_reps\
                    + [nvecset_aa[j]]
                nvecset_aa_SQreps = nvecset_aa_SQreps\
                    + [nvecset_aa_SQs[j]]
                nvecset_aa_inds = nvecset_aa_inds+[j]
                nvecset_aa_counts = nvecset_aa_counts+[1]

        nvecset_ab_batched = list(np.arange(len(nvecset_ab_reps)))
        for j in range(len(nvecset_ab)):
            for k in range(len(nvecset_ab_reps)):
                include_entry = False
                n_rep = nvecset_ab_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_ab[j]@g_elem
                    candidates = [np.array([n1, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_ab_batched[k], np.int64):
                        nvecset_ab_batched[k] = [nvecset_ab[j]]
                    else:
                        nvecset_ab_batched[k] = nvecset_ab_batched[k]\
                            + [nvecset_ab[j]]

        nvecset_aa_batched\
            = list(np.arange(len(nvecset_aa_reps)))
        for j in range(len(nvecset_aa)):
            for k in range(len(nvecset_aa_reps)):
                include_entry = False
                n_rep = nvecset_aa_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_aa[j]@g_elem
                    candidates = [np.array([n1, n2]),
                                  np.array([n2, n1])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_aa_batched[k], np.int64):
                        nvecset_aa_batched[k] = [nvecset_aa[j]]
                    else:
                        nvecset_aa_batched[k]\
                            = nvecset_aa_batched[k]\
                            + [nvecset_aa[j]]

        for j in range(len(nvecset_ab_batched)):
            nvecset_ab_batched[j] = np.array(nvecset_ab_batched[j])

        for j in range(len(nvecset_aa_batched)):
            nvecset_aa_batched[j]\
                = np.array(nvecset_aa_batched[j])
        return [nvecset_ab_reps, nvecset_aa_reps,
                nvecset_ab_SQreps, nvecset_aa_SQreps,
                nvecset_ab_inds, nvecset_aa_inds,
                nvecset_ab_counts, nvecset_aa_counts,
                nvecset_ab_batched, nvecset_aa_batched]
