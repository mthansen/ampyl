#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# cuts.py
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
from scipy.linalg import block_diag
from . import shell_utils
from . import check_utils
from . import interpolable_utils
from .constants import QC_IMPL_DEFAULTS
from .constants import TWOPI
from .constants import FOURPI2
from .constants import bcolors
from . import qc_functions
from .spaces import QCIndexSpace
from .interpolable import Interpolable
import warnings
warnings.simplefilter("once")


class G(Interpolable):
    """Represent the finite-volume G matrix."""

    def _get_value_not_interpolated(self, E, L, project, irrep):
        """Build the un-interpolated G matrix."""
        nP = self.qcis.fvs.nP
        if self.qcis.verbosity >= 2:
            self._g_verbose_a(E, L, nP)
        not_projecting = (irrep is None) and (project is False)
        projecting = not not_projecting
        irrep_not_in_keys = irrep not in self.qcis.proj_dict.keys()
        if projecting and irrep_not_in_keys:
            raise ValueError("irrep "+str(irrep)+" not in "
                             "qcis.proj_dict.keys()")
        tbks_entries, slices_by_three_slice = self._get_entries_and_slices(
            E, L, nP)
        g_final = self._get_value_from_tbks(
            E, L, project, irrep, tbks_entries, slices_by_three_slice)
        return g_final

    def _get_all_nvecSQs_for_pole_detection(self, nvecSQs_by_shell):
        """Collect shell-pair nvecSQ triples used to identify G poles."""
        all_nvecSQs = []
        for outer_nvecSQ_row in nvecSQs_by_shell:
            for outer_nvecSQ_entry in outer_nvecSQ_row:
                for inner_nvecSQ_row in outer_nvecSQ_entry:
                    for inner_nvecSQ_entry in inner_nvecSQ_row:
                        if len(inner_nvecSQ_entry) != 0:
                            n1vecSQs = inner_nvecSQ_entry[0][0]
                            n2vecSQs = inner_nvecSQ_entry[0][1]
                            n3vecSQs = inner_nvecSQ_entry[0][2]
                            for i in range(len(n1vecSQs)):
                                for j in range(len(n1vecSQs[i])):
                                    nvecSQ_sets = [
                                        n1vecSQs[i][j],
                                        n2vecSQs[i][j],
                                        n3vecSQs[i][j],
                                    ]
                                    nvecSQ_sets = list(np.sort(nvecSQ_sets))
                                    if nvecSQ_sets not in all_nvecSQs:
                                        all_nvecSQs = all_nvecSQs+[nvecSQ_sets]
        return all_nvecSQs

    def _get_pole_candidates_for_detection(self, nvecSQs_by_shell):
        """Collect canonicalized G pole candidates with aligned masses."""
        all_pole_candidates = []
        seen_pole_candidates = set()
        for sc_row_ind, outer_nvecSQ_row in enumerate(nvecSQs_by_shell):
            for sc_col_ind, outer_nvecSQ_entry in enumerate(outer_nvecSQ_row):
                row_three_slice = self.qcis.sc_to_three_slice[sc_row_ind]
                col_three_slice = self.qcis.sc_to_three_slice[sc_col_ind]
                row_inslice = sc_row_ind - self.qcis.fcs\
                    .slices_by_three_masses[row_three_slice][0]
                col_inslice = sc_col_ind - self.qcis.fcs\
                    .slices_by_three_masses[col_three_slice][0]
                g_rescale = self.qcis.fcs.g_templates[
                    row_three_slice][col_three_slice][
                    row_inslice][col_inslice]
                if g_rescale == 0.0:
                    continue
                masses = self._extract_g_masses(sc_row_ind, sc_col_ind)
                for inner_nvecSQ_row in outer_nvecSQ_entry:
                    for inner_nvecSQ_entry in inner_nvecSQ_row:
                        if len(inner_nvecSQ_entry) == 0:
                            continue
                        n1vecSQs = inner_nvecSQ_entry[0][0]
                        n2vecSQs = inner_nvecSQ_entry[0][1]
                        n3vecSQs = inner_nvecSQ_entry[0][2]
                        for i in range(len(n1vecSQs)):
                            for j in range(len(n1vecSQs[i])):
                                pole_candidate = (
                                    interpolable_utils
                                    ._canonicalize_pole_candidate(
                                        [n1vecSQs[i][j],
                                         n2vecSQs[i][j],
                                         n3vecSQs[i][j]],
                                        masses)
                                )
                                if pole_candidate not in seen_pole_candidates:
                                    seen_pole_candidates.add(pole_candidate)
                                    all_pole_candidates.append([
                                        list(pole_candidate[0]),
                                        list(pole_candidate[1]),
                                    ])
        return all_pole_candidates

    def _g_verbose_a(self, E, L, nP):
        """Print detailed diagnostic information for G evaluation."""
        print('evaluating G using numpy accelerated version')
        print('E = ', E, ', nP = ', nP, ', L = ', L)

        print(self.qcis.tbis.three_scheme, ',', self.qcis.fvs.qc_impl)
        print('cutoff params:', self.alpha, ',', self.beta)

        if self.qcis.tbis.three_scheme == 'original pole':
            sf = '1./(2.*w1*w2*L**3)'
        elif self.qcis.tbis.three_scheme == 'relativistic pole':
            sf = '1./(2.*w1*L**3)\n    * 1./(E-w1-w3+w2)'
        else:
            raise ValueError("three_scheme not recognized")
        hermitian = QC_IMPL_DEFAULTS['hermitian']
        if 'hermitian' in self.qcis.fvs.qc_impl:
            hermitian = self.qcis.fvs.qc_impl['hermitian']
        if hermitian:
            sf = sf+'\n    * 1./(2.0*w3)'
        print('G = YY*H1*H2\n    * '+sf+'\n    * 1./(E-w1-w2-w3)\n')

    def _get_entries_and_slices(self, E, L, nP):
        """Return the relevant TBKS entries and shell slices for G."""
        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            tbks_entries = []
            slices_by_three_slice = []
            for three_slice_index in range(self.qcis.fcs.n_three_slices):
                tbks_entry = self.qcis.tbks_list[three_slice_index][
                    tbks_sub_indices[three_slice_index]]
                tbks_entries.append(tbks_entry)
                slices_by_three_slice.append(tbks_entry.shells)
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
        else:
            if self.qcis.fcs.n_three_slices != 1:
                raise NotImplementedError(
                    "multi-slice G is implemented only for zero total momentum"
                )
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            sc_index = self.qcis.fcs.slices_by_three_masses[0][0]
            sc = self.qcis.fcs.sc_list_sorted[sc_index]
            mspec = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            ibest = self.qcis._get_ibest(E, L)
            ibest = 0
            warnings.warn(f"\n{bcolors.WARNING}"
                          "ibest is set to 0. This is a temporary fix."
                          f"{bcolors.ENDC}")
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            threshold = m2+m3
            zero_support_point = shell_utils._get_zero_support_point(
                self, threshold)
            mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)
            mask_slices = []
            slices = tbks_entry.shells
            reduce_size = QC_IMPL_DEFAULTS['reduce_size']
            if 'reduce_size' in self.qcis.fvs.qc_impl:
                reduce_size = self.qcis.fvs.qc_impl['reduce_size']
            if reduce_size:
                for slice_entry in slices:
                    mask_slices = mask_slices\
                        + [mask[slice_entry[0]:slice_entry[1]].all()]
                slices = list((np.array(slices))[mask_slices])
            tbks_entries = [tbks_entry]
            slices_by_three_slice = [slices]
        return tbks_entries, slices_by_three_slice

    def _get_value_from_tbks(self, E, L, project, irrep, tbks_entries,
                             slices_by_three_slice):
        """Assemble the full G matrix from a TBKS entry."""
        g_final = []
        if self.qcis.verbosity >= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            g_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                if self.qcis.verbosity >= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     + "supported in G")
                ell2 = col_ell_set[0]
                row_three_slice = self.qcis.sc_to_three_slice[sc_row_ind]
                col_three_slice = self.qcis.sc_to_three_slice[sc_col_ind]
                row_inslice = sc_row_ind - self.qcis.fcs\
                    .slices_by_three_masses[row_three_slice][0]
                col_inslice = sc_col_ind - self.qcis.fcs\
                    .slices_by_three_masses[col_three_slice][0]
                g_rescale = self.qcis.fcs.g_templates[
                    row_three_slice][col_three_slice][
                    row_inslice][col_inslice]
                if g_rescale == 0.0:
                    g_outer_row.append(np.array([]))
                    continue
                m1, m2, m3 = self._extract_g_masses(
                    sc_row_ind, sc_col_ind)
                row_tbks_entry = tbks_entries[row_three_slice]
                col_tbks_entry = tbks_entries[col_three_slice]
                row_slices = slices_by_three_slice[row_three_slice]
                col_slices = slices_by_three_slice[col_three_slice]
                g_inner = []
                for row_shell_index in range(len(row_slices)):
                    g_inner_row = []
                    for col_shell_index in range(len(col_slices)):
                        g_tmp = self.get_shell(E, L,
                                               m1, m2, m3,
                                               # only for non-zero nP
                                               sc_row_ind, sc_col_ind,
                                               ell1, ell2,
                                               g_rescale,
                                               row_tbks_entry,
                                               row_shell_index,
                                               col_shell_index,
                                               project, irrep,
                                               col_tbks_entry=col_tbks_entry)
                        g_inner_row.append(g_tmp)
                    g_inner.append(g_inner_row)
                g_inner = self._clean_shape(g_inner)
                g_block_tmp = np.block(g_inner)
                g_outer_row = g_outer_row+[g_block_tmp]
            g_final.append(g_outer_row)
        g_final = self._clean_shape(g_final)
        g_final = np.block(g_final)
        return g_final

    def _extract_g_masses(self, sc_row_ind, sc_col_ind):
        row_sc = self.qcis.fcs.sc_list_sorted[sc_row_ind]
        col_sc = self.qcis.fcs.sc_list_sorted[sc_col_ind]
        row_spec_flavor = row_sc.flavors_indexed[0]
        col_spec_flavor = col_sc.flavors_indexed[0]
        row_pair = [(row_sc.first_dimer.flavor, row_sc.first_dimer.mass),
                    (row_sc.second_dimer.flavor, row_sc.second_dimer.mass)]
        exchange_mass = None
        for pair_index, (flavor, mass) in enumerate(row_pair):
            if flavor == col_spec_flavor:
                row_pair.pop(pair_index)
                break
        if row_pair:
            exchange_mass = row_pair[0][1]
        else:
            exchange_mass = row_sc.first_dimer.mass
        m1 = col_sc.spectator.mass
        m2 = exchange_mass
        m3 = row_sc.spectator.mass
        if row_spec_flavor not in col_sc.flavors_indexed[1:]:
            raise ValueError("row spectator is not in the column dimer")
        return m1, m2, m3

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  sc_index_row=None, sc_index_col=None,
                  ell1=0, ell2=0,
                  g_rescale=1.0, tbks_entry=None,
                  row_shell_index=None,
                  col_shell_index=None,
                  project=False, irrep=None,
                  col_tbks_entry=None):
        """Build the G matrix block for a single pair of shells.

        Parameters
        ----------
        E : float, optional
            Energy value.
        L : float, optional
            Box length.
        m1, m2, m3 : float, optional
            Particle masses.
        cindex_row, cindex_col : int, optional
            Three-slice indices for nonzero total momentum.
        sc_index_row, sc_index_col : int, optional
            Spectator-channel indices.
        ell1, ell2 : int, optional
            Partial-wave indices.
        g_rescale : float, optional
            Overall rescaling factor for the shell block.
        tbks_entry : object, optional
            Precomputed TBKS entry used to define the shell structure.
        row_shell_index, col_shell_index : int, optional
            Shell indices for the block row and column.
        project : bool, optional
            Whether to project onto an irrep.
        irrep : tuple, optional
            Target irrep when projecting.

        Returns
        -------
        numpy.ndarray
            G-matrix block for the requested shells.
        """
        three_scheme = self.qcis.tbis.three_scheme
        nP = self.qcis.fvs.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha, beta = self.qcis.tbis.scheme_data[sc_index_col]
        alpha2, beta2 = self.qcis.tbis.scheme_data[sc_index_row]

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = shell_utils._get_masks_and_shells_for_nondiagonal(
                self, E, L, tbks_entry, sc_index_row, sc_index_col,
                row_shell_index, col_shell_index, col_tbks_entry)
        if col_tbks_entry is None:
            col_tbks_entry = tbks_entry
        if project:
            try:
                if nP@nP != 0:
                    proj_tmp_right, proj_tmp_left = self.\
                        _nP_nonzero_projectors(E, L,
                                               sc_index_row, sc_index_col,
                                               row_shell_index,
                                               col_shell_index,
                                               irrep,
                                               mask_row_shells,
                                               mask_col_shells)
                else:
                    proj_tmp_right, proj_tmp_left = self.\
                        _nPzero_projectors(sc_index_row, sc_index_col,
                                           row_shell_index, col_shell_index,
                                           irrep)
            except KeyError:
                return np.array([])

        g_uses_prep_mat = QC_IMPL_DEFAULTS['g_uses_prep_mat']
        if 'g_uses_prep_mat' in self.qcis.fvs.qc_impl:
            g_uses_prep_mat = self.qcis.fvs.qc_impl['g_uses_prep_mat']
        if g_uses_prep_mat and (nP@nP == 0):
            if col_tbks_entry is not tbks_entry:
                raise ValueError("g_uses_prep_mat does not support "
                                 "different row and column TBKS entries")
            Gshell = qc_functions.getG_array_prep_mat(E, nP, L, m1, m2, m3,
                                                      tbks_entry,
                                                      row_shell_index,
                                                      col_shell_index,
                                                      ell1, ell2,
                                                      alpha, beta,
                                                      qc_impl, three_scheme,
                                                      g_rescale,
                                                      alpha2=alpha2,
                                                      beta2=beta2)
        else:
            if col_tbks_entry is tbks_entry:
                Gshell = qc_functions.getG_array(E, nP, L, m1, m2, m3,
                                                 tbks_entry,
                                                 row_shell, col_shell,
                                                 ell1, ell2,
                                                 alpha, beta,
                                                 qc_impl, three_scheme,
                                                 g_rescale,
                                                 alpha2=alpha2,
                                                 beta2=beta2)
            else:
                Gshell = qc_functions.getG_array_two_tbks(
                    E, nP, L, m1, m2, m3, tbks_entry, col_tbks_entry,
                    row_shell, col_shell, ell1, ell2, alpha, beta,
                    qc_impl, three_scheme, g_rescale, alpha2=alpha2,
                    beta2=beta2)
        if project:
            Gshell = proj_tmp_left@Gshell@proj_tmp_right
        return Gshell

    def _nP_nonzero_projectors(self, E, L, sc_index_row, sc_index_col,
                               row_shell_index, col_shell_index, irrep,
                               mask_row_shells, mask_col_shells):
        """Return shell projectors for nonzero total momentum."""
        ibest = self.qcis._get_ibest(E, L)
        ibest = 0
        warnings.warn(f"\n{bcolors.WARNING}"
                      "ibest is set to 0. This is a temporary fix."
                      f"{bcolors.ENDC}")
        proj_tmp_right = np.array(self.qcis.proj_dicts_by_sc_and_shellset[
            sc_index_col][ibest])[mask_col_shells][
                col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
            np.array(self.qcis.
                     proj_dicts_by_sc_and_shellset[sc_index_row][ibest]
                     )[mask_row_shells][row_shell_index][irrep]).T)
        return proj_tmp_right, proj_tmp_left

    def _nPzero_projectors(self, sc_index_row, sc_index_col,
                           row_shell_index, col_shell_index, irrep):
        """Return shell projectors for zero total momentum."""
        proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_index_col][0][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
        return proj_tmp_right, proj_tmp_left

    def _clean_shape(self, g_collection):
        """Pad empty blocks so a nested G collection can be assembled."""
        nrows = len(g_collection)
        ncols = max((len(row) for row in g_collection), default=0)
        rowsizes = [0]*nrows
        colsizes = [0]*ncols
        for i in range(nrows):
            for j in range(len(g_collection[i])):
                shtmp = g_collection[i][j].shape
                if shtmp != (0,):
                    if shtmp[0] > rowsizes[i]:
                        rowsizes[i] = shtmp[0]
                    if shtmp[1] > colsizes[j]:
                        colsizes[j] = shtmp[1]
        for i in range(nrows):
            for j in range(len(g_collection[i])):
                shtmp = g_collection[i][j].shape
                if shtmp == (0,) or shtmp == (0, 0):
                    g_collection[i][j] = np.zeros((rowsizes[i], colsizes[j]))
        return g_collection


class F(Interpolable):
    """Represent the finite-volume F matrix."""

    _DIAGONAL_N3VECS = {
        0: np.array([0, 0, 0]),
        1: np.array([0, 0, 1]),
        2: np.array([0, 1, 1]),
        3: np.array([1, 1, 1]),
        4: np.array([0, 0, 2]),
    }

    def __init__(self, qcis=None, alphaKSS=1.0, C1cut=3):
        """Initialize the F matrix with zeta-function cutoff parameters."""
        super().__init__(qcis)
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS

    def _iter_nvecSQ_mats(self, nvecSQs_by_shell):
        """Yield the shell nvecSQ matrices stored in nested shell data."""
        for outer_nvecSQ_row in nvecSQs_by_shell:
            for outer_nvecSQ_entry in outer_nvecSQ_row:
                for inner_nvecSQ_row in outer_nvecSQ_entry:
                    for inner_nvecSQ_entry in inner_nvecSQ_row:
                        if len(inner_nvecSQ_entry) != 0:
                            yield inner_nvecSQ_entry[0]

    def _append_nvecSQs(self, all_nvecSQs, seen_nvecSQs, nvecSQs):
        """Append a sorted nvecSQ triple if it has not been seen yet."""
        nvecSQs_sorted = tuple(np.sort(nvecSQs))
        if nvecSQs_sorted not in seen_nvecSQs:
            seen_nvecSQs.add(nvecSQs_sorted)
            all_nvecSQs.append(list(nvecSQs_sorted))

    def _get_diagonal_nvecSQs(self, shell_nvecSQ):
        """Return extra diagonal nvecSQ triples implied by a shell label."""
        n3vec = self._DIAGONAL_N3VECS.get(int(shell_nvecSQ))
        if n3vec is None:
            return []

        diagonal_nvecSQs = []
        for n1_entry in np.ndindex((5, 5, 5)):
            n1vec = np.array(n1_entry)-2
            n2vec = -n1vec-n3vec
            diagonal_nvecSQs.append([
                n1vec@n1vec,
                n2vec@n2vec,
                n3vec@n3vec,
            ])
        return diagonal_nvecSQs

    def _get_all_nvecSQs_for_pole_detection(self, nvecSQs_by_shell):
        """Collect shell-diagonal nvecSQ triples used to identify F poles."""
        all_nvecSQs = []
        seen_nvecSQs = set()
        diagonal_nvecSQs_by_shell = {}

        for n1vecSQs, n2vecSQs, n3vecSQs in self._iter_nvecSQ_mats(
                nvecSQs_by_shell):
            for i, n1vecSQ_row in enumerate(n1vecSQs):
                for j, n1vecSQ_entry in enumerate(n1vecSQ_row):
                    if i != j:
                        continue

                    self._append_nvecSQs(
                        all_nvecSQs,
                        seen_nvecSQs,
                        [n1vecSQ_entry, n2vecSQs[i][j], n3vecSQs[i][j]],
                    )
                    shell_nvecSQ = int(n1vecSQs[i][0])
                    if shell_nvecSQ not in diagonal_nvecSQs_by_shell:
                        diagonal_nvecSQs_by_shell[shell_nvecSQ] = (
                            self._get_diagonal_nvecSQs(shell_nvecSQ)
                        )
                    for diagonal_nvecSQs in diagonal_nvecSQs_by_shell[
                            shell_nvecSQ]:
                        self._append_nvecSQs(
                            all_nvecSQs,
                            seen_nvecSQs,
                            diagonal_nvecSQs,
                        )
        return all_nvecSQs

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell1=0, ell2=0, tbks_entry=None,
                  slice_index=None, project=False, irrep=None,
                  mask=None):
        """Build the F matrix block for a single shell.

        Parameters
        ----------
        E : float, optional
            Energy value.
        L : float, optional
            Box length.
        m1, m2, m3 : float, optional
            Particle masses.
        cindex : int, optional
            Three-slice index for nonzero total momentum.
        sc_ind : int, optional
            Spectator-channel index.
        ell1, ell2 : int, optional
            Partial-wave indices.
        tbks_entry : object, optional
            Precomputed TBKS entry used to define the shell structure.
        slice_index : int, optional
            Shell index to evaluate.
        project : bool, optional
            Whether to project onto an irrep.
        irrep : tuple, optional
            Target irrep when projecting.
        mask : array-like, optional
            Shell mask used when reducing nonzero-momentum data.

        Returns
        -------
        numpy.ndarray
            F-matrix block for the requested shell.
        """
        three_scheme = self.qcis.tbis.three_scheme
        nP = self.qcis.fvs.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha, beta = self.qcis.tbis.scheme_data[sc_ind]
        C1cut = self.C1cut
        alphaKSS = self.alphaKSS
        use_pv_shift_prescription\
            = self.qcis.tbis.use_pv_shift_prescription[sc_ind]
        if use_pv_shift_prescription:
            pv_shift_parameters = self.qcis.tbis.pv_shift_parameters[sc_ind]
        else:
            pv_shift_parameters = None
        sc = self.qcis.fcs.sc_list_sorted[sc_ind]
        dimer_symmetry_factor = 1.0
        if sc.first_dimer != sc.second_dimer:
            dimer_symmetry_factor = 2.0

        mask_slices, slice_entry\
            = shell_utils._get_masks_and_shells_for_f(
                self, E, L, tbks_entry, cindex, slice_index)
        Fshell = qc_functions.getF_array(
            E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell1, ell2,
            alpha, beta, C1cut, alphaKSS, qc_impl, three_scheme,
            use_pv_shift_prescription=use_pv_shift_prescription,
            pv_shift_parameters=pv_shift_parameters,
            dimer_symmetry_factor=dimer_symmetry_factor)

        if project:
            try:
                if nP@nP != 0:
                    # ibest = self.qcis._get_ibest(E, L)
                    ibest = 0
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "ibest is set to 0. This is a temporary fix."
                                  f"{bcolors.ENDC}")
                    proj_tmp_right = np.array(
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_ind][ibest])[mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_ind][0][slice_index][irrep]
                    proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                return np.array([])
        if project:
            Fshell = proj_tmp_left@Fshell@proj_tmp_right
        return Fshell

    def _get_value_not_interpolated(self, E, L, project, irrep):
        """Build the un-interpolated F matrix shell by shell."""
        check_utils.check_value_within_qcis_bounds(self, E, L)
        nP = self.qcis.fvs.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            mask = None
        else:
            if self.qcis.fcs.n_three_slices != 1:
                raise NotImplementedError(
                    "multi-slice F is implemented only for zero total "
                    "momentum")
            three_slice_index = 0
            cindex = 0
            sc_index = self.qcis.fcs.slices_by_three_masses[0][0]
            sc = self.qcis.fcs.sc_list_sorted[sc_index]
            m1 = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            # ibest = self.qcis._get_ibest(E, L)
            ibest = 0
            warnings.warn(f"\n{bcolors.WARNING}"
                          "ibest is set to 0. This is a temporary fix."
                          f"{bcolors.ENDC}")
            reduce_size = QC_IMPL_DEFAULTS['reduce_size']
            if 'reduce_size' in self.qcis.fvs.qc_impl:
                reduce_size = self.qcis.fvs.qc_impl['reduce_size']
            if reduce_size:
                mspec = m1
                if len(self.qcis.tbks_list) > 1:
                    raise ValueError("get_value within F assumes tbks_list is "
                                     "length one.")
                three_slice_index = 0
                tbks_entry = self.qcis.tbks_list[three_slice_index][ibest]
                kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
                kvec_arr = TWOPI*tbks_entry.nvec_arr/L
                omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
                Pvec = TWOPI*nP/L
                PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
                threshold = m2+m3
                zero_support_point = shell_utils._get_zero_support_point(
                    self, threshold)
                mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
                if self.qcis.verbosity >= 2:
                    print('mask =')
                    print(mask)
                mask_slices = []
                slices = tbks_entry.shells
                if self.qcis.verbosity >= 2:
                    print('slices =')
                    print(slices)
                for slice_entry in slices:
                    mask_slices = mask_slices\
                        + [mask[slice_entry[0]:slice_entry[1]].all()]
                slices = list((np.array(slices))[mask_slices])
                if self.qcis.verbosity >= 2:
                    print('mask_slices =')
                    print(mask_slices)
                    print('range for sc_ind =')
                    print(range(len(self.qcis.fcs.sc_list_sorted)))
            else:
                tbks_entry = self.qcis.tbks_list[three_slice_index][ibest]
                slices = tbks_entry.shells
                mask = [True]*len(tbks_entry.nvecSQ_arr)
        f_final_list = []
        for sc_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            sc = self.qcis.fcs.sc_list_sorted[sc_ind]
            ell_set = sc.ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 "supported in F")
            ell1 = ell_set[0]
            ell2 = ell1
            if nP@nP == 0:
                three_slice_index = self.qcis.sc_to_three_slice[sc_ind]
                tbks_entry = self.qcis.tbks_list[three_slice_index][
                    tbks_sub_indices[three_slice_index]]
                slices = tbks_entry.shells
                m1 = sc.spectator.mass
                m2 = sc.first_dimer.mass
                m3 = sc.second_dimer.mass
                cindex = sc_ind
            for slice_index in range(len(slices)):
                if self.qcis.verbosity >= 2:
                    print('get_shell is receiving:')
                    print(E, L, m1, m2, m3)
                    print(f'cindex = {cindex}\n'
                          f'sc_ind = {sc_ind}\n'
                          f'ell1 = {ell1}\n'
                          f'ell2 = {ell2}\n')
                    print(tbks_entry)
                    print('slice_index = '+str(slice_index))
                    print(project, irrep)
                f_tmp = self.get_shell(
                    E, L, m1, m2, m3, cindex,  # only for non-zero nP
                    sc_ind, ell1, ell2, tbks_entry, slice_index,
                    project, irrep, mask)
                if len(f_tmp) != 0:
                    f_final_list = f_final_list+[f_tmp]
        return block_diag(*f_final_list)


class FplusG(Interpolable):
    """Represent the combined F+G matrix."""

    def __init__(self, qcis=QCIndexSpace(), alphaKSS=1.0, C1cut=3):
        """Initialize the combined F+G object from F and G components."""
        super().__init__(qcis)
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS
        self.f = F(qcis=qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=qcis)

    def _get_value_not_interpolated(self, E, L, project, irrep):
        """Build the un-interpolated F+G matrix."""
        return self.g.get_value(E=E, L=L, project=project, irrep=irrep)\
            + self.f.get_value(E=E, L=L, project=project, irrep=irrep)

    def _append_nvecSQs(self, all_nvecSQs, seen_nvecSQs, nvecSQs):
        """Append a sorted nvecSQ triple if it has not been seen yet."""
        nvecSQs_sorted = tuple(np.sort(nvecSQs))
        if nvecSQs_sorted not in seen_nvecSQs:
            seen_nvecSQs.add(nvecSQs_sorted)
            all_nvecSQs.append(list(nvecSQs_sorted))

    def _get_all_nvecSQs_for_pole_detection(self, nvecSQs_by_shell):
        """Merge F and G nvecSQ triples used to identify F+G poles."""
        all_nvecSQs = []
        seen_nvecSQs = set()
        for candidate_source in (self.g, self.f):
            for nvecSQs in candidate_source\
               ._get_all_nvecSQs_for_pole_detection(nvecSQs_by_shell):
                self._append_nvecSQs(all_nvecSQs, seen_nvecSQs, nvecSQs)
        return all_nvecSQs
