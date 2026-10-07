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
from .constants import FOURPI2, TWOPI
from .constants import QC_IMPL_DEFAULTS


def two_particle_block_dim(project, irrep):
    """Return the block size of an s-wave two-particle channel.

    At zero total momentum the s-wave pair contributes a single state,
    which transforms in A1PLUS; for any other irrep the projected block
    is empty.
    """
    if (irrep is None) and (project is False):
        return 1
    irrep_name = irrep[0] if isinstance(irrep, (tuple, list)) else irrep
    return 1 if irrep_name == 'A1PLUS' else 0


def _verify_irrep_is_known(qcis, irrep):
    """Raise if irrep appears in no shell projector dictionary.

    A key missing from one shell's dictionary is legitimate -- that
    shell simply contributes nothing to the irrep -- but a key missing
    from every dictionary is a typo'd or otherwise invalid irrep, which
    callers must not silently convert into an empty block.
    """
    known_irreps = set()
    for sc_dicts in qcis.proj_dicts_by_sc_and_shellset:
        # zero momentum stores a flat per-shell table, nonzero momentum
        # one table per shell set
        if len(sc_dicts) > 0 and isinstance(sc_dicts[0], dict):
            shellset_dicts_list = [sc_dicts]
        else:
            shellset_dicts_list = sc_dicts
        for shellset_dicts in shellset_dicts_list:
            for shell_dict in shellset_dicts:
                known_irreps.update(shell_dict.keys())
    if irrep not in known_irreps:
        raise ValueError(
            f"irrep {irrep} appears in no projector dictionary; "
            f"known irreps are {sorted(known_irreps, key=str)}")


def _get_zero_support_point(alpha, beta, threshold):
    """Return the dimer sigma below which the cutoff H vanishes."""
    return (1.0+alpha)*threshold**2/4.0-beta*((3.0-alpha)*threshold**2/4.0)


def _get_active_shells(qcis, sc_index, E, L, tbks_entry):
    """Return the shell mask and the active shells of one channel.

    At nonzero total momentum with ``reduce_size``, a shell is active
    iff the dimer invariant mass squared exceeds the zero-support point
    of the cutoff on every momentum of the shell. The masses and the
    cutoff parameters are those of channel ``sc_index``, so channels
    sharing a TBKS entry can keep different shells. At zero total
    momentum the TBKS entry already selects the shells and the mask is
    ``None``.
    """
    nP = qcis.fvs.nP
    shells = tbks_entry.shells
    if nP@nP == 0:
        return None, shells
    reduce_size = QC_IMPL_DEFAULTS['reduce_size']
    if 'reduce_size' in qcis.fvs.qc_impl:
        reduce_size = qcis.fvs.qc_impl['reduce_size']
    if not reduce_size:
        return len(shells)*[True], shells
    sc = qcis.fcs.sc_list_sorted[sc_index]
    mspec = sc.spectator.mass
    threshold = sc.first_dimer.mass+sc.second_dimer.mass
    alpha, beta = qcis.tbis.scheme_data[sc_index]
    zero_support_point = _get_zero_support_point(alpha, beta, threshold)
    kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
    kvec_arr = TWOPI*tbks_entry.nvec_arr/L
    omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
    Pvec = TWOPI*nP/L
    PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
    mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
    mask_shells = []
    for shell in shells:
        mask_shells = mask_shells+[mask[shell[0]:shell[1]].all()]
    return mask_shells, list(np.array(shells)[mask_shells])


def _get_masks_and_shells_for_k(k, E, L, tbks_entry, cindex, slice_index):
    mask_slices, slices = _get_active_shells(k.qcis, cindex, E, L,
                                             tbks_entry)
    return mask_slices, slices[slice_index]


def _get_masks_and_shells_for_nondiagonal(nondiagonal, E, L, tbks_entry,
                                          cindex_row, cindex_col,
                                          row_shell_index, col_shell_index,
                                          col_tbks_entry=None):
    if col_tbks_entry is None:
        col_tbks_entry = tbks_entry
    nP = nondiagonal.qcis.fvs.nP
    if nP@nP == 0:
        return _mask_and_shell_helper_nPzero(
            nondiagonal, tbks_entry, row_shell_index, col_shell_index,
            col_tbks_entry)
    three_slice_index_row =\
        nondiagonal.qcis.sc_to_three_slice[cindex_row]
    three_slice_index_col =\
        nondiagonal.qcis.sc_to_three_slice[cindex_col]
    if three_slice_index_row != three_slice_index_col:
        raise NotImplementedError(
            "multi-slice nonzero-momentum G blocks are not supported")
    mask_row_shells, row_shells = _get_active_shells(
        nondiagonal.qcis, cindex_row, E, L, tbks_entry)
    mask_col_shells, col_shells = _get_active_shells(
        nondiagonal.qcis, cindex_col, E, L, col_tbks_entry)
    row_shell = list(row_shells[row_shell_index])
    col_shell = list(col_shells[col_shell_index])
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


def _get_masks_and_shells_for_f(f, E, L, tbks_entry, cindex, slice_index):
    mask_slices, slices = _get_active_shells(f.qcis, cindex, E, L,
                                             tbks_entry)
    return mask_slices, slices[slice_index]
