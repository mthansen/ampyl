#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# shell_utils.py
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
import warnings
from .constants import FOURPI2, TWOPI
from .constants import QC_IMPL_DEFAULTS


def _verify_irrep_is_known(qcis, irrep):
    """Raise if irrep appears in no shell projector dictionary.

    A key missing from one shell's dictionary is legitimate -- that
    shell simply contributes nothing to the irrep -- but a key missing
    from every dictionary is a typo'd or otherwise invalid irrep, which
    callers must not silently convert into an empty block.
    """
    known_irreps = set()
    for sc_dicts in qcis.proj_dicts_by_sc_and_shellset:
        for shellset_dicts in sc_dicts:
            for shell_dict in shellset_dicts:
                known_irreps.update(shell_dict.keys())
    if irrep not in known_irreps:
        raise ValueError(
            f"irrep {irrep} appears in no projector dictionary; "
            f"known irreps are {sorted(known_irreps, key=str)}")


def _get_masks_and_shells_for_k(k, E, L, tbks_entry, cindex, slice_index):
    nP = k.qcis.fvs.nP
    mask_slices = None
    three_slice_index = k.qcis.sc_to_three_slice[cindex]
    if nP@nP == 0:
        slice_entry = tbks_entry.shells[slice_index]
    else:
        sc_list_sorted = k.qcis.fcs.sc_list_sorted
        slices_by_three_masses = k.qcis.fcs.slices_by_three_masses
        inslice_index = 0
        sc_index = slices_by_three_masses[three_slice_index][inslice_index]
        sc = sc_list_sorted[sc_index]
        mspec = sc.spectator.mass
        kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
        kvec_arr = TWOPI*tbks_entry.nvec_arr/L
        omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
        Pvec = TWOPI*nP/L
        PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
        threshold = sc.first_dimer.mass + sc.second_dimer.mass
        zero_support_point = _get_zero_support_point(k, threshold)
        mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
        slices = tbks_entry.shells
        mask_slices = []
        for slice_entry in slices:
            mask_slices = mask_slices\
                + [mask[slice_entry[0]:slice_entry[1]].all()]
        slices = list(np.array(slices)[mask_slices])
        slice_entry = slices[slice_index]
    return mask_slices, slice_entry


def _get_masks_and_shells_for_nondiagonal(nondiagonal, E, L, tbks_entry,
                                          cindex_row, cindex_col,
                                          row_shell_index, col_shell_index,
                                          col_tbks_entry=None):
    if col_tbks_entry is None:
        col_tbks_entry = tbks_entry
    nP = nondiagonal.qcis.fvs.nP
    three_slice_index_row =\
        nondiagonal.qcis.sc_to_three_slice[cindex_row]
    three_slice_index_col =\
        nondiagonal.qcis.sc_to_three_slice[cindex_col]
    three_slice_index = three_slice_index_row
    if nP@nP == 0:
        mask_row_shells, mask_col_shells, row_shell, col_shell =\
            _mask_and_shell_helper_nPzero(
                nondiagonal, tbks_entry, row_shell_index, col_shell_index,
                col_tbks_entry)
    else:
        if three_slice_index_row != three_slice_index_col:
            raise NotImplementedError(
                "multi-slice nonzero-momentum G blocks are not supported")
        mask_row_shells, mask_col_shells, row_shell, col_shell =\
            _mask_and_shell_helper_nPnonzero(
                nondiagonal, E, nP, L, tbks_entry,
                row_shell_index, col_shell_index, three_slice_index)
    return mask_row_shells, mask_col_shells, row_shell, col_shell


def _mask_and_shell_helper_nPzero(nondiagonal, tbks_entry,
                                  row_shell_index, col_shell_index,
                                  col_tbks_entry=None):
    if col_tbks_entry is None:
        col_tbks_entry = tbks_entry
    mask_row_shells = None
    mask_col_shells = None
    row_shell = tbks_entry.shells[row_shell_index]
    col_shell = col_tbks_entry.shells[col_shell_index]
    return mask_row_shells, mask_col_shells, row_shell, col_shell


def _mask_and_shell_helper_nPnonzero(nondiagonal, E, nP, L, tbks_entry,
                                     row_shell_index, col_shell_index,
                                     three_slice_index):
    reduce_size = QC_IMPL_DEFAULTS['reduce_size']
    if 'reduce_size' in nondiagonal.qcis.fvs.qc_impl:
        reduce_size = nondiagonal.qcis.fvs.qc_impl['reduce_size']
    if reduce_size:
        sc_index = nondiagonal.qcis.fcs.slices_by_three_masses[0][0]
        sc = nondiagonal.qcis.fcs.sc_list_sorted[sc_index]
        mspec = sc.spectator.mass
        m2 = sc.first_dimer.mass
        m3 = sc.second_dimer.mass
        kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
        kvec_arr = TWOPI*tbks_entry.nvec_arr/L
        omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
        Pvec = TWOPI*nP/L
        PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
        threshold = m2+m3
        zero_support_point = _get_zero_support_point(nondiagonal, threshold)
        mask_row = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
        row_shells = tbks_entry.shells
        mask_row_shells = []
        for row_shell in row_shells:
            mask_row_shells = mask_row_shells\
                    + [mask_row[row_shell[0]:row_shell[1]].all()]
        row_shells = list(np.array(row_shells)[mask_row_shells])
        row_shell = list(row_shells[row_shell_index])
    else:
        row_shells = tbks_entry.shells
        mask_row_shells = len(row_shells)*[True]
        row_shell = list(row_shells[row_shell_index])

    if reduce_size:
        mask_col = mask_row
        col_shells = tbks_entry.shells
        mask_col_shells = []
        for col_shell in col_shells:
            mask_col_shells = mask_col_shells\
                    + [mask_col[col_shell[0]:col_shell[1]].all()]
        col_shells = list(np.array(col_shells)[mask_col_shells])
        col_shell = list(col_shells[col_shell_index])
    else:
        col_shells = tbks_entry.shells
        mask_col_shells = len(col_shells)*[True]
        col_shell = list(col_shells[col_shell_index])
    return mask_row_shells, mask_col_shells, row_shell, col_shell


def _get_masks_and_shells_for_f(f, E, L, tbks_entry, cindex, slice_index):
    nP = f.qcis.fvs.nP
    mask_slices = None
    # three_slice_index\
    #     = self.qcis._get_three_slice_index(cindex)
    if nP@nP == 0:
        slice_entry = tbks_entry.shells[slice_index]
    else:
        reduce_size = QC_IMPL_DEFAULTS['reduce_size']
        if 'reduce_size' in f.qcis.fvs.qc_impl:
            reduce_size = f.qcis.fvs.qc_impl['reduce_size']
        if reduce_size:
            sc_index = f.qcis.fcs.slices_by_three_masses[0][0]
            sc = f.qcis.fcs.sc_list_sorted[sc_index]
            mspec = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            threshold = m2+m3
            zero_support_point = _get_zero_support_point(f, threshold)
            mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
            slices = tbks_entry.shells
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        else:
            slice_entry = tbks_entry.shells[slice_index]
            mask_slices = [True]*len(tbks_entry.shells)
    return mask_slices, slice_entry


def _get_zero_support_point(qcmatrix, threshold):
    warnings.warn("The zero support point is currently hardcoded to be "
                  "the same for all QC implementations. This may not be "
                  "correct for all implementations.")
    if hasattr(qcmatrix, 'alpha') and hasattr(qcmatrix, 'beta'):
        alpha = qcmatrix.alpha
        beta = qcmatrix.beta
    else:
        alpha, beta = qcmatrix.qcis.tbis.scheme_data[0]
    return (1.0+alpha)*threshold**2/4.0-beta*((3.0-alpha)*threshold**2/4.0)
