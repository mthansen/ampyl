#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# k_matrices.py
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
from .constants import TWOPI
from .constants import FOURPI2
from .constants import QC_IMPL_DEFAULTS
from .constants import bcolors
from .functions import QCFunctions
import warnings
warnings.simplefilter("once")


class K:
    """
    Class for the two-to-two K matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    """

    def __init__(self, qcis=None):
        self.qcis = qcis
        three_scheme = self.qcis.tbis.three_scheme
        alpha_beta_scheme = (three_scheme == 'original pole')\
            or (three_scheme == 'relativistic pole')
        if alpha_beta_scheme:
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell=0,
                  pcotdelta_function=None, pcotdelta_parameter_list=None,
                  tbks_entry=None, slice_index=None,
                  project=False, irrep=None):
        """Build the K matrix on a single shell."""
        nP = self.qcis.fvs.nP
        three_scheme = self.qcis.tbis.three_scheme
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta
        use_pv_shift_prescription\
            = self.qcis.tbis.use_pv_shift_prescription[sc_ind]
        if use_pv_shift_prescription:
            pv_shift_parameters = self.qcis.tbis.pv_shift_parameters[sc_ind]
        else:
            pv_shift_parameters = None

        mask_slices, slice_entry\
            = shell_utils._get_masks_and_shells_for_k(
                self, E, L, tbks_entry, cindex, slice_index)
        Kshell = QCFunctions.getK_array(
            E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell,
            pcotdelta_function, pcotdelta_parameter_list, alpha, beta,
            qc_impl, three_scheme,
            use_pv_shift_prescription=use_pv_shift_prescription,
            pv_shift_parameters=pv_shift_parameters)

        if project:
            try:
                if nP@nP != 0:
                    ibest_always_zero = QC_IMPL_DEFAULTS['ibest_always_zero']
                    if 'ibest_always_zero' in self.qcis.fvs.qc_impl:
                        ibest_always_zero =\
                            self.qcis.fvs.qc_impl['ibest_always_zero']
                    if ibest_always_zero:
                        ibest = 0
                    else:
                        ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_ind][ibest])[mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "ibest is set to 0. This is a temporary fix."
                                  f"{bcolors.ENDC}")
                    ibest = 0
                    proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_ind][ibest][slice_index][irrep]
                    proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                return np.array([])
        if project:
            Kshell = proj_tmp_left@Kshell@proj_tmp_right
        return Kshell

    def get_value(self, E=5.0, L=5.0, pcotdelta_parameter_lists=None,
                  project=False, irrep=None):
        """Build the K matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
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
        sc_list_sorted = self.qcis.fcs.sc_list_sorted
        slices_by_three_masses = self.qcis.fcs.slices_by_three_masses
        three_slice_index = 0
        inslice_index = 0
        sc_index = slices_by_three_masses[three_slice_index][inslice_index]
        masses = sc_list_sorted[sc_index].masses_indexed
        [mspec, m2, m3] = masses
        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][tbks_sub_indices[0]]
            slices = tbks_entry.shells
        else:
            # ibest = self.qcis._get_ibest(E, L)
            ibest = 0
            warnings.warn(f"\n{bcolors.WARNING}"
                          "ibest is set to 0. This is a temporary fix."
                          f"{bcolors.ENDC}")
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
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
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        k_final_list = []
        for sc_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            ell_set = self.qcis.fcs.sc_list[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in K")
            ell = ell_set[0]
            pcotdelta_parameter_list = pcotdelta_parameter_lists[sc_ind]
            pcotdelta_function = self.qcis.fcs.sc_list[
                sc_ind].p_cot_deltas[0]
            for slice_index in range(len(slices)):
                k_tmp = self.get_shell(
                    E, L, mspec, m2, m3, cindex, sc_ind, ell,
                    pcotdelta_function, pcotdelta_parameter_list, tbks_entry,
                    slice_index, project, irrep)
                if len(k_tmp) != 0:
                    k_final_list = k_final_list+[k_tmp]
        return block_diag(*k_final_list)


class Kdf:
    """
    Class for the three-to-three K matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace

    At this stage only the asymmetric version is implemented, and only for
    zero total momentum.
    """
    def __init__(self, qcis=None):
        self.qcis = qcis

    def get_value(self, E, L, k3_params, project, irrep):
        nP = self.qcis.fvs.nP
        cindex_row = cindex_col = 0
        not_projecting = (irrep is None) and (project is False)
        projecting = not not_projecting
        irrep_not_in_keys = irrep not in self.qcis.proj_dict.keys()
        if projecting and irrep_not_in_keys:
            raise ValueError("irrep "+str(irrep)+" not in "
                             "qcis.proj_dict.keys()")
        tbks_entry, slices = self._get_entry_and_slices(E, L, nP)
        kdf_final = self._get_value_from_tbks(E, L, k3_params, project, irrep,
                                              cindex_col, cindex_row,
                                              tbks_entry, slices)
        return kdf_final

    def _get_entry_and_slices(self, E, L, nP):
        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 "length one.")
            tbks_entry = self.qcis.tbks_list[0][tbks_sub_indices[0]]
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            raise NotImplementedError("get_value within Kdf is not "
                                      "implemented for non-zero nP yet.")
        return tbks_entry, slices

    def _get_value_from_tbks(self, E, L, k3_params, project, irrep, cindex_col,
                             cindex_row, tbks_entry, slices):
        m1, m2, m3 = [1.0, 1.0, 1.0]
        warnings.warn(f"\n{bcolors.WARNING}"
                      "assuming m1 = m2 = m3 = 1.0 in Kdf"
                      f"{bcolors.ENDC}")
        kdf_final = []
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            kdf_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 "supported in Kdf")
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                col_ell_set = self.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     "supported in Kdf")
                ell2 = col_ell_set[0]
                kdf_inner = []
                for row_shell_index in range(len(slices)):
                    kdf_inner_row = []
                    for col_shell_index in range(len(slices)):
                        kdf_tmp = self.get_shell(E, L, k3_params,
                                                 m1, m2, m3,
                                                 cindex_row, cindex_col,
                                                 # only for non-zero nP
                                                 sc_row_ind, sc_col_ind,
                                                 ell1, ell2,
                                                 tbks_entry,
                                                 row_shell_index,
                                                 col_shell_index,
                                                 project, irrep)
                        kdf_inner_row.append(kdf_tmp)
                    kdf_inner.append(kdf_inner_row)
                kdf_inner = self._clean_shape(kdf_inner)
                kdf_block_tmp = np.block(kdf_inner)
                kdf_outer_row = kdf_outer_row+[kdf_block_tmp]
            kdf_final.append(kdf_outer_row)
        kdf_final = self._clean_shape(kdf_final)
        kdf_final = np.block(kdf_final)
        return kdf_final

    def get_shell(self, E=5.0, L=5.0, k3_params=None, m1=1.0, m2=1.0, m3=1.0,
                  cindex_row=None, cindex_col=None,  # only for non-zero nP
                  sc_index_row=None, sc_index_col=None, ell1=0, ell2=0,
                  tbks_entry=None, row_shell_index=None, col_shell_index=None,
                  project=False, irrep=None):
        """Build the Kdf matrix on a single shell."""
        nP = self.qcis.fvs.nP

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = shell_utils._get_masks_and_shells_for_nondiagonal(
                self, E, L, tbks_entry, cindex_row, cindex_col,
                row_shell_index, col_shell_index)
        if project:
            try:
                if nP@nP == 0:
                    proj_tmp_right, proj_tmp_left = self._nPzero_projectors(
                        sc_index_row, sc_index_col,
                        row_shell_index, col_shell_index, irrep)
                else:
                    proj_tmp_right, proj_tmp_left = self.\
                        _nP_nonzero_projectors(E, L,
                                               sc_index_row, sc_index_col,
                                               row_shell_index,
                                               col_shell_index,
                                               irrep,
                                               mask_row_shells,
                                               mask_col_shells)
            except KeyError:
                return np.array([])

        if len(k3_params) != 1:
            raise ValueError("k3_params must have length 1 for the version "
                             "of Kdf currently implemented.")

        if ell1 == ell2 == 1:
            Kdfshell = np.ones((proj_tmp_left.shape[1],
                                proj_tmp_right.shape[0]))*k3_params[0]
        else:
            Kdfshell = np.zeros((proj_tmp_left.shape[1],
                                 proj_tmp_right.shape[0]))
        if project:
            Kdfshell = proj_tmp_left@Kdfshell@proj_tmp_right
        return Kdfshell

    def _nPzero_projectors(self, sc_index_row, sc_index_col,
                           row_shell_index, col_shell_index, irrep):
        proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_index_col][0][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
        return proj_tmp_right, proj_tmp_left

    def _nP_nonzero_projectors(self, E, L, sc_index_row, sc_index_col,
                               row_shell_index, col_shell_index, irrep,
                               mask_row_shells, mask_col_shells):
        ibest = self.qcis._get_ibest(E, L)
        ibest = 0
        warnings.warn(f"\n{bcolors.WARNING}"
                      "ibest is set to 0. This is a temporary fix."
                      f"{bcolors.ENDC}")
        proj_tmp_right\
            = np.array(
                self.qcis.proj_dicts_by_sc_and_shellset[sc_index_col][ibest]
                )[mask_col_shells][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
            np.array(
                self.qcis.proj_dicts_by_sc_and_shellset[sc_index_row][ibest]
                     )[mask_row_shells][row_shell_index][irrep]).T)
        return proj_tmp_right, proj_tmp_left

    def _clean_shape(self, kdf_collection):
        rowsizes = [0]*len(kdf_collection)
        colsizes = [0]*len(kdf_collection)
        for i in range(len(kdf_collection)):
            for j in range(len(kdf_collection)):
                shtmp = kdf_collection[i][j].shape
                if shtmp != (0,):
                    if shtmp[0] > rowsizes[i]:
                        rowsizes[i] = shtmp[0]
                    if shtmp[1] > colsizes[j]:
                        colsizes[j] = shtmp[1]
        for i in range(len(kdf_collection)):
            for j in range(len(kdf_collection)):
                shtmp = kdf_collection[i][j].shape
                if shtmp == (0,) or shtmp == (0, 0):
                    kdf_collection[i][j].shape = (rowsizes[i], colsizes[j])
        return kdf_collection
