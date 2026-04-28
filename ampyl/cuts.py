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
from .constants import QC_IMPL_DEFAULTS
from .constants import TWOPI
from .constants import FOURPI2
from .constants import bcolors
from .functions import QCFunctions
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
        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if self.qcis.verbosity >= 2:
            print('representatives of three_slice:')
            print('    cindex_row =', cindex_row,
                  ', cindex_col =', cindex_col)
        not_projecting = (irrep is None) and (project is False)
        projecting = not not_projecting
        irrep_not_in_keys = irrep not in self.qcis.proj_dict.keys()
        if projecting and irrep_not_in_keys:
            raise ValueError("irrep "+str(irrep)+" not in "
                             "qcis.proj_dict.keys()")
        tbks_entry, slices = self._get_entry_and_slices(E, L, nP)
        g_final = self._get_value_from_tbks(E, L, project, irrep,
                                            cindex_col, cindex_row,
                                            tbks_entry, slices)
        return g_final

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

    def _get_entry_and_slices(self, E, L, nP):
        """Return the relevant TBKS entry and shell slices for G."""
        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            mspec, m2, m3 = self.extract_masses()
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
        return tbks_entry, slices

    def _get_value_from_tbks(self, E, L, project, irrep, cindex_col,
                             cindex_row, tbks_entry, slices):
        """Assemble the full G matrix from a TBKS entry."""
        m1, m2, m3 = self.extract_masses()
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
                g_rescale = self.qcis.fcs.g_templates[0][0][
                    sc_row_ind][sc_col_ind]
                g_inner = []
                for row_shell_index in range(len(slices)):
                    g_inner_row = []
                    for col_shell_index in range(len(slices)):
                        g_tmp = self.get_shell(E, L,
                                               m1, m2, m3,
                                               cindex_row, cindex_col,
                                               # only for non-zero nP
                                               sc_row_ind, sc_col_ind,
                                               ell1, ell2,
                                               g_rescale,
                                               tbks_entry,
                                               row_shell_index,
                                               col_shell_index,
                                               project, irrep)
                        g_inner_row.append(g_tmp)
                    g_inner.append(g_inner_row)
                g_inner = self._clean_shape(g_inner)
                g_block_tmp = np.block(g_inner)
                g_outer_row = g_outer_row+[g_block_tmp]
            g_final.append(g_outer_row)
        g_final = self._clean_shape(g_final)
        g_final = np.block(g_final)
        return g_final

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex_row=None, cindex_col=None,  # only for non-zero nP
                  sc_index_row=None, sc_index_col=None,
                  ell1=0, ell2=0,
                  g_rescale=1.0, tbks_entry=None,
                  row_shell_index=None,
                  col_shell_index=None,
                  project=False, irrep=None):
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
        alpha = self.alpha
        beta = self.beta

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = shell_utils._get_masks_and_shells_for_nondiagonal(
                self, E, L, tbks_entry, cindex_row, cindex_col,
                row_shell_index, col_shell_index)
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
            Gshell = QCFunctions.getG_array_prep_mat(E, nP, L, m1, m2, m3,
                                                     tbks_entry,
                                                     row_shell_index,
                                                     col_shell_index,
                                                     ell1, ell2,
                                                     alpha, beta,
                                                     qc_impl, three_scheme,
                                                     g_rescale)
        else:
            Gshell = QCFunctions.getG_array(E, nP, L, m1, m2, m3,
                                            tbks_entry,
                                            row_shell, col_shell,
                                            ell1, ell2,
                                            alpha, beta,
                                            qc_impl, three_scheme,
                                            g_rescale)
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
        rowsizes = [0]*len(g_collection)
        colsizes = [0]*len(g_collection)
        for i in range(len(g_collection)):
            for j in range(len(g_collection)):
                shtmp = g_collection[i][j].shape
                if shtmp != (0,):
                    if shtmp[0] > rowsizes[i]:
                        rowsizes[i] = shtmp[0]
                    if shtmp[1] > colsizes[j]:
                        colsizes[j] = shtmp[1]
        for i in range(len(g_collection)):
            for j in range(len(g_collection)):
                shtmp = g_collection[i][j].shape
                if shtmp == (0,) or shtmp == (0, 0):
                    g_collection[i][j].shape = (rowsizes[i], colsizes[j])
        return g_collection


class F(Interpolable):
    """Represent the finite-volume F matrix."""

    def __init__(self, qcis=None, alphaKSS=1.0, C1cut=3):
        """Initialize the F matrix with zeta-function cutoff parameters."""
        self.qcis = qcis
        three_scheme = self.qcis.tbis.three_scheme
        alpha_beta_scheme = (three_scheme == 'original pole')\
            or (three_scheme == 'relativistic pole')
        if alpha_beta_scheme:
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS

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
        alpha = self.alpha
        beta = self.beta
        C1cut = self.C1cut
        alphaKSS = self.alphaKSS
        use_pv_shift_prescription\
            = self.qcis.tbis.use_pv_shift_prescription[sc_ind]
        if use_pv_shift_prescription:
            pv_shift_parameters = self.qcis.tbis.pv_shift_parameters[sc_ind]
        else:
            pv_shift_parameters = None

        mask_slices, slice_entry\
            = shell_utils._get_masks_and_shells_for_f(
                self, E, L, tbks_entry, cindex, slice_index)
        Fshell = QCFunctions.getF_array(
            E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell1, ell2,
            alpha, beta, C1cut, alphaKSS, qc_impl, three_scheme,
            use_pv_shift_prescription=use_pv_shift_prescription,
            pv_shift_parameters=pv_shift_parameters)

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
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        three_slice_index = 0
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.fvs.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")

        cindex = 0
        m1, m2, m3 = self.extract_masses()
        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within F assumes tbks_list is "
                                 "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            mask = None
        else:
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
            ell_set = self.qcis.fcs.sc_list_sorted[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 "supported in F")
            ell1 = ell_set[0]
            ell2 = ell1
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

    def _get_all_nvecSQs(self, nvecSQs_by_shell):
        """Collect all squared momentum triples appearing in shell data."""
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
                                    nvecSQ_sets = [n1vecSQs[i][j],
                                                   n2vecSQs[i][j],
                                                   n3vecSQs[i][j]]
                                    nvecSQ_sets = list(np.sort(nvecSQ_sets))
                                    if nvecSQ_sets not in all_nvecSQs:
                                        all_nvecSQs = all_nvecSQs+[nvecSQ_sets]
                                    if i == j:
                                        rng = range(-2, 2+1)
                                        mesh = np.meshgrid(*([rng]*3))
                                        nvec_arr = np.vstack([y.flat
                                                              for y in mesh]).T
                                        if n1vecSQs[i][0] == 0:
                                            n3vec = np.array([0, 0, 0])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets =\
                                                    list(
                                                        np.sort([n1SQtmp,
                                                                 n2SQtmp,
                                                                 n3SQtmp]))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 1:
                                            n3vec = np.array([0, 0, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets =\
                                                    list(
                                                        np.sort([n1SQtmp,
                                                                 n2SQtmp,
                                                                 n3SQtmp]))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 2:
                                            n3vec = np.array([0, 1, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 3:
                                            n3vec = np.array([1, 1, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 4:
                                            n3vec = np.array([0, 0, 2])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
        return all_nvecSQs
