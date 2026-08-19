#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created May 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# interpolable_utils.py
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

import numpy as np
from copy import deepcopy
from itertools import product
from . import shell_utils
from . import check_utils
from .constants import QC_IMPL_DEFAULTS
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON10
from .constants import BAD_MIN_GUESS
from .constants import BAD_MAX_GUESS
from .constants import POLE_CUT
from . import qc_functions
import warnings
warnings.simplefilter("once")


def _interpolator_data_attrs(interpolable):
    return (
        'all_relevant_nvecSQ_lists',
        'interp_data_lists',
        'polefree_interp_data_lists',
        'cob_matrix_lists',
        'cob_matrix_key_lists',
        'matrix_dim_lists',
        'cob_list_lens',
        'interp_tensors',
        'interps',
        'pole_lists',
        'pole_mass_lists',
        'pole_textures_lists',
        'complement_textures_lists',
        'pole_residue_matrix_lists',
    )


def _get_interpolation_flag(interpolable, short_string, interpolate):
    interpolate_string = f'{short_string}_interpolate'
    if interpolate is None:
        interpolate = QC_IMPL_DEFAULTS[interpolate_string]
        if interpolate_string in interpolable.qcis.fvs.qc_impl:
            interpolate = interpolable.qcis.fvs.qc_impl[interpolate_string]
    return interpolate


def _canonicalize_pole_candidate(nvecSQs, masses):
    """Sort a pole candidate while preserving nvecSQ-mass alignment."""
    candidate = np.array(list(zip(nvecSQs, masses)), dtype=float)
    permuted = candidate[
        np.lexsort((candidate[:, 0], candidate[:, 1]))
    ]
    return tuple(permuted[:, 0].astype(int).tolist()), tuple(
        permuted[:, 1].tolist())


def _grids_and_interp(interpolable, Emin, Emax, Estep, Lmin, Lmax, Lstep,
                      project, irrep):
    L_grid = np.arange(Lmin, Lmax+EPSILON4, Lstep)
    E_grid = np.arange(Emin, Emax+EPSILON4, Estep)
    nP = interpolable.qcis.fvs.nP
    if nP@nP == 0:
        tbks_sub_indices = interpolable.qcis.get_tbks_sub_indices(Emax, Lmax)
        projected_axis_index = 1
        max_interp_dim = 0
        for sc_index in range(interpolable.qcis.n_channels):
            three_slice_index = interpolable.qcis.sc_to_three_slice[sc_index]
            tbks_sub_index = tbks_sub_indices[three_slice_index]
            # the zero-momentum projector table is flat over shells;
            # restrict it to the shells active in the matched space
            n_shells = len(interpolable.qcis.tbks_list[three_slice_index][
                tbks_sub_index].shells)
            proj_dict_list = interpolable.qcis.proj_dicts_by_sc_and_shellset[
                sc_index][:n_shells]
            for proj_dict in proj_dict_list:
                try:
                    projected_size =\
                        proj_dict[irrep].shape[projected_axis_index]
                    max_interp_dim += projected_size
                except KeyError:
                    continue
    else:
        max_interp_matrix_shape = (interpolable.get_value(E=Emax, L=Lmax,
                                                          project=project,
                                                          irrep=irrep)).shape
        max_interp_dim = max_interp_matrix_shape[0]
    interp_data_list = []
    for _ in range(max_interp_dim):
        interp_mat_row = []
        for _ in range(max_interp_dim):
            interp_mat_row.append([[BAD_MIN_GUESS, BAD_MAX_GUESS,
                                    BAD_MIN_GUESS, BAD_MAX_GUESS], [[]]])
        interp_data_list.append(interp_mat_row)
    return L_grid, E_grid, max_interp_dim, interp_data_list


def _new_interp_data_entry():
    return [[BAD_MIN_GUESS, BAD_MAX_GUESS,
             BAD_MIN_GUESS, BAD_MAX_GUESS], [[]]]


def _resize_interp_data_list(interp_data_list, target_dim):
    current_dim = len(interp_data_list)
    if current_dim >= target_dim:
        return interp_data_list
    for row in interp_data_list:
        for _ in range(target_dim-current_dim):
            row.append(_new_interp_data_entry())
    for _ in range(target_dim-current_dim):
        interp_data_row = []
        for _ in range(target_dim):
            interp_data_row.append(_new_interp_data_entry())
        interp_data_list.append(interp_data_row)
    return interp_data_list


def _get_dim_with_shell_index_all_scs(interpolable, irrep,
                                      tbks_sub_indices=None):
    # change-of-basis bookkeeping is only used at zero total momentum,
    # where the projector table is flat over shells (exact for every
    # truncation); the shell loop below is bounded by the matched
    # kinematic space
    assert interpolable.qcis.nPSQ == 0
    dim_with_shell_index_all_scs = []
    if tbks_sub_indices is None:
        tbks_sub_indices = [0]*interpolable.qcis.fcs.n_three_slices
    for spectator_channel_index in range(
            len(interpolable.qcis.fcs.sc_list_sorted)):
        dim_with_shell_index_single_sc = []
        three_slice_index = interpolable.qcis.sc_to_three_slice[
            spectator_channel_index]
        tbks_sub_index = tbks_sub_indices[three_slice_index]
        shells = interpolable.qcis.tbks_list[three_slice_index][
            tbks_sub_index].shells
        for shell_index in range(len(shells)):
            try:
                proj_candidate = interpolable.qcis.\
                    proj_dicts_by_sc_and_shellset[
                        spectator_channel_index][
                            shell_index][irrep]
                dim_with_shell_index_single_sc.\
                    append([(proj_candidate.shape)[1], shell_index])
            except KeyError:
                pass
        dim_with_shell_index_all_scs.\
            append(dim_with_shell_index_single_sc)
    return dim_with_shell_index_all_scs


def _get_cob_matrix_key_list(interpolable):
    """Return all TBKS sub-index tuples that need COB matrices."""
    tbks_sub_index_ranges = []
    for three_slice_index in range(interpolable.qcis.fcs.n_three_slices):
        tbks_sub_index_ranges.append(
            range(len(interpolable.qcis.tbks_list[three_slice_index])))
    return list(product(*tbks_sub_index_ranges))


def _get_final_set_for_change_of_basis(
        interpolable, dim_with_shell_index_all_scs):
    dim_shell_counter_all = []
    dim_counter = 0
    for dim_with_shell_index_for_sc in dim_with_shell_index_all_scs:
        dim_shell_counter = []
        for dim_with_shell_index in dim_with_shell_index_for_sc:
            counter_set = []
            for _ in range(dim_with_shell_index[0]):
                counter_set = counter_set+[dim_counter]
                dim_counter = dim_counter+1
            dim_shell_counter = dim_shell_counter\
                + [[dim_with_shell_index, counter_set]]
        dim_shell_counter_all = dim_shell_counter_all+[dim_shell_counter]
    return dim_shell_counter_all


def _get_cob_matrix_list(interpolable, final_set_for_change_of_basis,
                         cob_matrix_key_list=None):
    all_restacks = []
    for dim_shell_counter_all in final_set_for_change_of_basis:
        restack = []
        for shell_index in range(len(dim_shell_counter_all)):
            for dim_shell_counter in dim_shell_counter_all[shell_index]:
                restack.append([[dim_shell_counter[0][1],
                                 shell_index], dim_shell_counter[1]])
        all_restacks.append(sorted(restack))
    all_restacks_second = []
    for restack in all_restacks:
        second_restack = []
        for entry in restack:
            second_restack = second_restack+entry[1]
        all_restacks_second.append(second_restack)
    fixed_max_basis = (
        interpolable.qcis.fcs.n_three_slices > 1
        and cob_matrix_key_list is not None
        and len(cob_matrix_key_list) != 0
    )
    if not fixed_max_basis:
        cob_matrix_list = []
        for restack in all_restacks_second:
            cob_matrix_list.append((np.identity(len(restack))[restack]).T)
        return cob_matrix_list

    max_key = tuple([0]*interpolable.qcis.fcs.n_three_slices)
    max_basis_index = cob_matrix_key_list.index(max_key)
    max_restack = all_restacks[max_basis_index]
    max_basis_map = {}
    max_basis_dim = 0
    for label, counter_set in max_restack:
        for offset in range(len(counter_set)):
            max_basis_map[tuple(label+[offset])] = max_basis_dim
            max_basis_dim += 1

    cob_matrix_list = []
    for restack in all_restacks_second:
        cob_matrix_list.append(np.zeros((len(restack), max_basis_dim)))
    for matrix_index in range(len(cob_matrix_list)):
        active_restack = all_restacks[matrix_index]
        for label, counter_set in active_restack:
            for offset in range(len(counter_set)):
                max_basis_index = max_basis_map[tuple(label+[offset])]
                active_basis_index = counter_set[offset]
                cob_matrix_list[matrix_index][
                    active_basis_index, max_basis_index] = 1.
    return cob_matrix_list


def _get_cob_matrix_index(cob_matrix_key_list, tbks_sub_indices):
    tbks_sub_indices = tuple(tbks_sub_indices)
    try:
        return cob_matrix_key_list.index(tbks_sub_indices)
    except ValueError:
        return None


def _get_cob_matrix_for_value(interpolable, E, L, cob_matrix_list,
                              cob_matrix_key_list):
    if len(cob_matrix_list) == 0:
        return None
    tbks_sub_indices = interpolable.qcis.get_tbks_sub_indices(E, L)
    tbks_sub_indices = tuple(
        tbks_sub_indices[:interpolable.qcis.fcs.n_three_slices])
    cob_matrix_index = _get_cob_matrix_index(
        cob_matrix_key_list, tbks_sub_indices)
    if cob_matrix_index is None:
        return None
    return cob_matrix_list[cob_matrix_index]


def _update_mins_and_maxes(
        interpolable, interpolator_matrix, energy_vol_dat_index,
        i, j, interpolator_entry):
    [E, L, _] = interpolator_entry
    if E < interpolator_matrix[i][j][
            energy_vol_dat_index][0]:
        interpolator_matrix[i][j][
            energy_vol_dat_index][0] = E
    if E > interpolator_matrix[i][j][
            energy_vol_dat_index][1]:
        interpolator_matrix[i][j][
            energy_vol_dat_index][1] = E
    if L < interpolator_matrix[i][j][
            energy_vol_dat_index][2]:
        interpolator_matrix[i][j][
            energy_vol_dat_index][2] = L
    if L > interpolator_matrix[i][j][
            energy_vol_dat_index][3]:
        interpolator_matrix[i][j][
            energy_vol_dat_index][3] = L
    return interpolator_matrix


def _get_all_nvecSQs_by_shell(interpolable, E=5.0, L=5.0, project=False,
                              irrep=None):
    check_utils.check_value_within_qcis_bounds(interpolable, E, L)
    nP = interpolable.qcis.fvs.nP
    if (not ((irrep is None) and (project is False))
       and (not (irrep in interpolable.qcis.proj_dict.keys()))):
        raise ValueError("irrep "+str(irrep)+" not in "
                         + "qcis.proj_dict.keys()")
    if nP@nP == 0:
        if interpolable.qcis.verbosity >= 2:
            print('nP = [0 0 0] indexing')
        tbks_sub_indices = interpolable.qcis.get_tbks_sub_indices(E=E, L=L)
        tbks_entries = []
        slices_by_three_slice = []
        slot_offset = 1 if interpolable.qcis.n_two_channels > 0 else 0
        for three_slice_index in range(interpolable.qcis.fcs.n_three_slices):
            slot_index = three_slice_index+slot_offset
            tbks_entry = interpolable.qcis.tbks_list[slot_index][
                tbks_sub_indices[slot_index]]
            tbks_entries.append(tbks_entry)
            slices_by_three_slice.append(tbks_entry.shells)
        if interpolable.qcis.verbosity >= 2:
            print('tbks_sub_indices =', tbks_sub_indices)
    else:
        if interpolable.qcis.fcs.n_three_slices != 1:
            raise NotImplementedError(
                "multi-slice interpolation pole detection is implemented "
                "only for zero total momentum")
        if interpolable.qcis.verbosity >= 2:
            print('nP != [0 0 0] indexing')
        sc_index = interpolable.qcis.fcs.slices_by_three_masses[0][0]
        sc = interpolable.qcis.fcs.sc_list_sorted[sc_index]
        mspec = sc.spectator.mass
        m2 = sc.first_dimer.mass
        m3 = sc.second_dimer.mass
        ibest = interpolable.qcis.get_shellset_index(E, L)
        if len(interpolable.qcis.tbks_list) > 1:
            raise ValueError("get_value within G assumes tbks_list is "
                             + "length one.")
        tbks_entry = interpolable.qcis.tbks_list[0][ibest]
        kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
        kvec_arr = TWOPI*tbks_entry.nvec_arr/L
        omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
        Pvec = TWOPI*nP/L
        PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
        threshold = m2+m3
        zero_support_point = shell_utils._get_zero_support_point(
            interpolable, threshold)
        mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
        if interpolable.qcis.verbosity >= 2:
            print('mask =')
            print(mask)

        reduce_size = QC_IMPL_DEFAULTS['reduce_size']
        if 'reduce_size' in interpolable.qcis.fvs.qc_impl:
            reduce_size = interpolable.qcis.fvs.qc_impl['reduce_size']
        if reduce_size:
            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])
        else:
            slices = tbks_entry.shells
            mask_slices = [True]*len(slices)
        tbks_entries = [tbks_entry]
        slices_by_three_slice = [slices]

    nvecSQs_final = [[]]
    if interpolable.qcis.verbosity >= 2:
        print('iterating over spectator channels, slices')
    # two-particle channels carry no shell data; their pole candidates
    # are generated directly from the channel masses instead
    slot_offset = 1 if interpolable.qcis.n_two_channels > 0 else 0
    for sc_row_ind in range(len(interpolable.qcis.fcs.sc_list_sorted)):
        if interpolable.qcis.fcs.sc_list_sorted[
                sc_row_ind].fc.n_particles == 2:
            continue
        nvecSQs_outer_row = []
        row_ell_set = interpolable.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
        if len(row_ell_set) != 1:
            raise ValueError("only length-one ell_set currently "
                             + "supported in G")
        for sc_col_ind in range(len(interpolable.qcis.fcs.sc_list_sorted)):
            if interpolable.qcis.fcs.sc_list_sorted[
                    sc_col_ind].fc.n_particles == 2:
                continue
            if interpolable.qcis.verbosity >= 2:
                print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
            col_ell_set =\
                interpolable.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
            if len(col_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")

            row_three_slice = interpolable.qcis.sc_to_three_slice[
                sc_row_ind]-slot_offset
            col_three_slice = interpolable.qcis.sc_to_three_slice[
                sc_col_ind]-slot_offset
            row_slices = slices_by_three_slice[row_three_slice]
            col_slices = slices_by_three_slice[col_three_slice]
            row_tbks_entry = tbks_entries[row_three_slice]
            col_tbks_entry = tbks_entries[col_three_slice]
            nvecSQs_inner = [[]]
            for row_shell_index in range(len(row_slices)):
                nvecSQs_inner_row = []
                for col_shell_index in range(len(col_slices)):
                    nvecSQs_tmp = _get_shell_nvecSQs_projs(
                        interpolable, E, L, sc_row_ind, sc_col_ind,
                        # only for non-zero nP
                        sc_row_ind, sc_col_ind, row_tbks_entry,
                        row_shell_index, col_shell_index, project, irrep,
                        col_tbks_entry)
                    nvecSQs_inner_row = nvecSQs_inner_row+[nvecSQs_tmp]
                nvecSQs_inner = nvecSQs_inner+[nvecSQs_inner_row]
            nvecSQs_block_tmp = nvecSQs_inner[1:]
            nvecSQs_outer_row = nvecSQs_outer_row+[nvecSQs_block_tmp]
        nvecSQs_final = nvecSQs_final+[nvecSQs_outer_row]
    nvecSQs_final = nvecSQs_final[1:]
    return nvecSQs_final


def _get_shell_nvecSQs_projs(interpolable, E=5.0, L=5.0,
                             cindex_row=None, cindex_col=None,
                             # only for non-zero nP
                             sc_index_row=None, sc_index_col=None,
                             tbks_entry=None,
                             row_shell_index=None,
                             col_shell_index=None,
                             project=False, irrep=None,
                             col_tbks_entry=None):
    nP = interpolable.qcis.fvs.nP
    if col_tbks_entry is None:
        col_tbks_entry = tbks_entry

    mask_row_shells, mask_col_shells, row_shell, col_shell\
        = shell_utils._get_masks_and_shells_for_nondiagonal(
            interpolable, E, L, tbks_entry, cindex_row, cindex_col,
            row_shell_index, col_shell_index, col_tbks_entry)
    if col_tbks_entry is not tbks_entry:
        return []
    if project:
        try:
            if nP@nP != 0:
                ibest = interpolable.qcis.get_shellset_index(E, L)
                proj_tmp_right = np.array(interpolable.qcis
                                          .proj_dicts_by_sc_and_shellset[
                                              sc_index_col][ibest]
                                          )[mask_col_shells][
                                              col_shell_index][irrep]
                proj_tmp_left = np.conjugate((
                    np.array(interpolable.qcis.
                             proj_dicts_by_sc_and_shellset[
                                 sc_index_row][ibest]
                             )[mask_row_shells][row_shell_index][irrep]
                ).T)
            else:
                proj_tmp_right =\
                    interpolable.qcis.proj_dicts_by_sc_and_shellset[
                        sc_index_col][col_shell_index][irrep]
                proj_tmp_left = np.conjugate((
                    interpolable.qcis.proj_dicts_by_sc_and_shellset[
                        sc_index_row][row_shell_index][irrep]
                ).T)
        except KeyError:
            return np.array([])
    nvecSQ_mat_shells = qc_functions\
        .get_nvecSQ_mat_shells(tbks_entry, row_shell, col_shell)
    return [nvecSQ_mat_shells, proj_tmp_left, proj_tmp_right]


def _get_entry_window_and_keeps(interpolable, interp_data_matrix_entry,
                                all_pole_candidates):
    Lmin_tmp = BAD_MIN_GUESS
    Lmax_tmp = BAD_MAX_GUESS
    Emin_tmp = BAD_MIN_GUESS
    Emax_tmp = BAD_MAX_GUESS
    for single_interp_entry in interp_data_matrix_entry:
        energy_volume_set = single_interp_entry[:-1]
        [E_candidate, L_candidate] = energy_volume_set
        if E_candidate < Emin_tmp:
            Emin_tmp = E_candidate
        if E_candidate > Emax_tmp:
            Emax_tmp = E_candidate
        if L_candidate < Lmin_tmp:
            Lmin_tmp = L_candidate
        if L_candidate > Lmax_tmp:
            Lmax_tmp = L_candidate
    nvecSQs_keeps = []
    for nvecSQ_entry, masses in all_pole_candidates:
        n1vecSQ = nvecSQ_entry[0]
        n2vecSQ = nvecSQ_entry[1]
        n3vecSQ = nvecSQ_entry[2]
        m1, m2, m3 = masses
        removal_at_Lmin = get_pole_candidate(
            interpolable, Lmin_tmp, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3)
        removal_at_Lmax = get_pole_candidate(
            interpolable, Lmax_tmp, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3)
        if ((Emin_tmp < removal_at_Lmin < Emax_tmp)
           or (Emin_tmp < removal_at_Lmax < Emax_tmp)):
            nvecSQs_keeps.append([[n1vecSQ, n2vecSQ, n3vecSQ],
                                  [m1, m2, m3]])
    return Lmin_tmp, Lmax_tmp, nvecSQs_keeps


def _get_all_relevant_nvecSQs_list(interpolable, Emax, project, irrep,
                                   max_interp_dim, interp_data_list,
                                   cob_matrix_list, all_pole_candidates):
    interp_data_index = 1
    # First screen the pole candidates against each matrix entry's own
    # E/L data window and record the (E, L) points at which that entry
    # wants the matrix checked, so that each distinct point is evaluated
    # only once below rather than once per entry.
    eval_requests = {}
    for i in range(max_interp_dim):
        for j in range(max_interp_dim):
            interp_data_matrix_entry = interp_data_list[i][j][
                interp_data_index][1:]
            if len(interp_data_matrix_entry) == 0:
                continue
            Lmin_tmp, Lmax_tmp, nvecSQs_keeps = _get_entry_window_and_keeps(
                interpolable, interp_data_matrix_entry, all_pole_candidates)
            for nvecSQs_keep in nvecSQs_keeps:
                [n1vecSQ, n2vecSQ, n3vecSQ] = nvecSQs_keep[0]
                [m1, m2, m3] = nvecSQs_keep[1]
                Lvals_tmp = [Lmin_tmp+EPSILON4, Lmax_tmp-EPSILON4]
                for Ltmp in Lvals_tmp:
                    Etmp = _get_pole_candidate_eps(
                        interpolable, Ltmp, n1vecSQ, n2vecSQ, n3vecSQ,
                        m1, m2, m3)
                    if Etmp < Emax:
                        key = (Etmp, Ltmp)
                        if key not in eval_requests:
                            eval_requests[key] = []
                        eval_requests[key].append([i, j, nvecSQs_keep])
    all_relevant_nvecSQs = []
    cob_matrix_key_list = interpolable.cob_matrix_key_lists.get(irrep, [])
    for (Etmp, Ltmp), requests in eval_requests.items():
        try:
            matrix_tmp = interpolable.get_value(E=Etmp, L=Ltmp,
                                                project=project,
                                                irrep=irrep)
            cob_matrix = _get_cob_matrix_for_value(
                interpolable, Etmp, Ltmp, cob_matrix_list,
                cob_matrix_key_list)
            if cob_matrix is not None:
                matrix_tmp = (cob_matrix.T)@matrix_tmp@cob_matrix
        except IndexError:
            continue
        for [i, j, nvecSQs_keep] in requests:
            try:
                interpolable_value = matrix_tmp[i][j]
            except IndexError:
                continue
            near_pole_mag = np.abs(interpolable_value)
            pole_found = (near_pole_mag > POLE_CUT)
            relevant_entry = [i, j, nvecSQs_keep[0], nvecSQs_keep[1]]
            if pole_found and relevant_entry not in all_relevant_nvecSQs:
                all_relevant_nvecSQs.append(relevant_entry)
    return all_relevant_nvecSQs


def _get_pole_candidate_eps(interpolable, L, n1vecSQ, n2vecSQ, n3vecSQ,
                            m1, m2, m3):
    pole_candidate_eps = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
        + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
        + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)+EPSILON10
    return pole_candidate_eps


def _get_polefree_interp_data_list(interpolable, max_interp_dim,
                                   interp_data_list,
                                   interp_data_index,
                                   all_relevant_nvecSQs_list):
    polefree_interp_data_list = []
    for i in range(max_interp_dim):
        polefree_interp_data_row = []
        for j in range(max_interp_dim):
            matrix_entry_interp_data = interp_data_list[i][j][
                interp_data_index]
            relevant_poles = []
            for relevant_candidate in all_relevant_nvecSQs_list:
                if ((relevant_candidate[0] == i)
                   and (relevant_candidate[1] == j)):
                    relevant_poles.append(relevant_candidate)
            for entry_index in range(len(matrix_entry_interp_data)):
                dim_with_shell_index = matrix_entry_interp_data[
                    entry_index]
                [E, L, interpolable_value] = dim_with_shell_index
                for nvecSQs_set in relevant_poles:
                    nvecSQ = nvecSQs_set[2]
                    masses = nvecSQs_set[3]
                    n1vecSQ = nvecSQ[0]
                    n2vecSQ = nvecSQ[1]
                    n3vecSQ = nvecSQ[2]
                    m1, m2, m3 = masses
                    three_omega = get_pole_candidate(
                        interpolable, L, n1vecSQ, n2vecSQ, n3vecSQ,
                        m1, m2, m3)
                    pole_removal_factor = E-three_omega
                    interpolable_value\
                        = pole_removal_factor*interpolable_value
                matrix_entry_interp_data[entry_index]\
                    = [E, L, interpolable_value]
            entry = [interp_data_list[i][j][0],
                     matrix_entry_interp_data,
                     relevant_poles]
            polefree_interp_data_row.append(entry)
        polefree_interp_data_list.append(polefree_interp_data_row)
    return polefree_interp_data_list


def _get_pole_textures(interpolable, matrix_dim_list,
                       polefree_interp_data_list):
    pole_list = []
    pole_mass_list = []
    pole_textures_list = []
    complement_textures_list = []
    for index_tmp in range(len(matrix_dim_list)):
        pole_dict = {}
        pole_index_dict = {}
        poles = []
        pole_masses = []
        pole_textures = []
        matrix_dimension = matrix_dim_list[index_tmp]
        for i in range(matrix_dimension):
            for j in range(matrix_dimension):
                if (i >= len(polefree_interp_data_list)
                   or j >= len(polefree_interp_data_list[i])):
                    break
                for pole_data in (polefree_interp_data_list[i][j][2]):
                    pole_key = _canonicalize_pole_candidate(
                        pole_data[2], pole_data[3])
                    if pole_key not in pole_dict:
                        texture = np.zeros((matrix_dimension,
                                            matrix_dimension))
                        texture[i][j] = 1.
                        pole_dict[pole_key] = texture
                        pole_index_dict[pole_key] = len(poles)
                        poles.append(list(pole_key[0]))
                        pole_masses.append(list(pole_key[1]))
                        pole_textures.append(deepcopy(texture))
                    else:
                        if pole_dict[pole_key][i][j] != 0.:
                            continue
                        texture = np.zeros((matrix_dimension,
                                            matrix_dimension))
                        texture[i][j] = 1.
                        pole_dict[pole_key] += texture
                        pole_textures[pole_index_dict[pole_key]] += texture
        complement_textures = []
        for pole_texture in pole_textures:
            complement_texture = np.ones((matrix_dimension,
                                          matrix_dimension))\
                - pole_texture
            complement_textures.append(complement_texture)
        poles = np.array(poles)
        pole_masses = np.array(pole_masses)
        pole_textures = np.array(pole_textures)
        complement_textures = np.array(complement_textures)
        pole_list.append(poles)
        pole_mass_list.append(pole_masses)
        pole_textures_list.append(pole_textures)
        complement_textures_list.append(complement_textures)
    return pole_list, pole_mass_list, pole_textures_list, \
        complement_textures_list


def _get_pole_residue_matrix_list(interpolable, L, irrep, E_range=None,
                                  basis='standard'):
    pole_residue_matrix_list = []
    pole_list = interpolable.pole_lists[irrep]
    pole_mass_list = interpolable.pole_mass_lists[irrep]
    pole_textures_list = interpolable.pole_textures_lists[irrep]
    cob_matrix_list = getattr(interpolable, 'cob_matrix_lists', {}).get(
        irrep, [])
    if E_range is not None:
        E_range = [float(E_range[0]), float(E_range[1])]
        interp = interpolable.interps.get(irrep)
        interp_grid = getattr(interp, 'grid', None)
        if interp_grid is not None:
            E_range[0] = max(E_range[0], float(interp_grid[0][0]))
            E_range[1] = min(E_range[1], float(interp_grid[0][-1]))
    for sector_index in range(len(pole_list)):
        poles = pole_list[sector_index]
        pole_masses = pole_mass_list[sector_index]
        pole_textures = pole_textures_list[sector_index]
        cob_matrix = None
        if sector_index < len(cob_matrix_list):
            cob_matrix = cob_matrix_list[sector_index]
        residue_matrices = []
        for pole_index in range(len(poles)):
            E_pole = _get_pole_energy(poles[pole_index],
                                      pole_masses[pole_index], L)
            if (E_range is not None
               and (E_pole < E_range[0] or E_pole > E_range[1])):
                residue_matrix = np.zeros_like(pole_textures[pole_index],
                                               dtype=float)
                if cob_matrix is not None and basis == 'standard':
                    residue_matrix = np.zeros(
                        (cob_matrix.shape[0], cob_matrix.shape[0]))
                residue_matrices.append(residue_matrix)
                continue
            smooth_value = interpolable.interps[irrep]((E_pole, L))
            smooth_value = _pad_matrix_to_shape(
                smooth_value, pole_textures[pole_index].shape)
            residue_matrix = smooth_value*pole_textures[pole_index]
            for other_pole_index in range(len(poles)):
                if other_pole_index == pole_index:
                    continue
                other_E_pole = _get_pole_energy(
                    poles[other_pole_index],
                    pole_masses[other_pole_index],
                    L)
                denominator = E_pole-other_E_pole
                other_texture = pole_textures[other_pole_index]
                if np.abs(denominator) < EPSILON10:
                    overlap = pole_textures[pole_index]*other_texture
                    if np.any(overlap != 0.):
                        warnings.warn(
                            "coincident poles share at least one matrix "
                            "entry, so the strict simple residue is not "
                            "defined")
                    continue
                pole_factor_matrix =\
                    other_texture/denominator + (1.-other_texture)
                residue_matrix = residue_matrix*pole_factor_matrix
            if cob_matrix is not None and basis == 'standard':
                smooth_dim = cob_matrix.shape[1]
                residue_matrix = _pad_matrix_to_shape(
                    residue_matrix, (smooth_dim, smooth_dim))
                residue_matrix = cob_matrix@residue_matrix@cob_matrix.T
            residue_matrices.append(residue_matrix)
        pole_residue_matrix_list.append(np.array(residue_matrices))
    return pole_residue_matrix_list


def _get_pole_energy(pole, pole_masses, L):
    return np.sqrt(pole*FOURPI2/L**2 + pole_masses**2).sum()


def _pad_matrix_to_shape(matrix, shape):
    if matrix.shape == shape:
        return matrix
    matrix_tmp = np.zeros(shape, dtype=matrix.dtype)
    row_dim = min(shape[0], len(matrix))
    col_dim = min(shape[1], len(matrix.T))
    matrix_tmp[:row_dim, :col_dim] = matrix[:row_dim, :col_dim]
    return matrix_tmp


def _get_value_interpolated(interpolable, E, L, irrep):
    if 'use_cob_matrices' in interpolable.qcis.fvs.qc_impl:
        use_cob_matrices = interpolable.qcis.fvs.qc_impl['use_cob_matrices']
    else:
        use_cob_matrices = QC_IMPL_DEFAULTS['use_cob_matrices']
    cob_matrix_index = None
    if (interpolable.cob_list_lens != {} and use_cob_matrices
       and len(interpolable.cob_matrix_lists[irrep]) != 0):
        tbks_sub_indices = interpolable.qcis.get_tbks_sub_indices(E, L)
        tbks_sub_indices = tuple(
            tbks_sub_indices[:interpolable.qcis.fcs.n_three_slices])
        cob_matrix_index = _get_cob_matrix_index(
            interpolable.cob_matrix_key_lists[irrep], tbks_sub_indices)
    if len(interpolable.pole_lists[irrep]) == 0:
        pole_parts_smooth_basis = 1.
    else:
        if cob_matrix_index is None:
            poles = interpolable.pole_lists[irrep][0]
            pole_masses = interpolable.pole_mass_lists[irrep][0]
        else:
            poles = interpolable.pole_lists[irrep][cob_matrix_index]
            pole_masses = interpolable.pole_mass_lists[
                irrep][cob_matrix_index]
        if len(poles) == 0:
            pole_parts_smooth_basis = 1.
        else:
            if cob_matrix_index is None:
                pole_textures = interpolable.pole_textures_lists[irrep][0]
                complement_textures = interpolable.complement_textures_lists[
                    irrep][0]
            else:
                pole_textures = interpolable.pole_textures_lists[
                    irrep][cob_matrix_index]
                complement_textures =\
                    interpolable.complement_textures_lists[
                        irrep][cob_matrix_index]
            omegas =\
                np.sqrt(poles*FOURPI2/L**2 + pole_masses**2)
            pole_values = 1./(E-omegas.sum(1))
            pole_matrices =\
                np.multiply(pole_textures, pole_values[:, None, None])\
                + complement_textures
            pole_parts_smooth_basis = pole_matrices.prod(0)
    if cob_matrix_index is not None:
        cob_matrix = interpolable.cob_matrix_lists[irrep][cob_matrix_index]
    smooth_value = interpolable.interps[irrep]((E, L))
    if cob_matrix_index is not None:
        smooth_dim = cob_matrix.shape[1]
        if ((smooth_dim != len(smooth_value))
           or (smooth_dim != len(smooth_value.T))):
            smooth_value_tmp = np.zeros((smooth_dim, smooth_dim),
                                        dtype=smooth_value.dtype)
            row_dim = min(smooth_dim, len(smooth_value))
            col_dim = min(smooth_dim, len(smooth_value.T))
            smooth_value_tmp[:row_dim, :col_dim] =\
                smooth_value[:row_dim, :col_dim]
            smooth_value = smooth_value_tmp
        final_value_smooth_basis = smooth_value*pole_parts_smooth_basis
        final_value =\
            (cob_matrix)@final_value_smooth_basis@(cob_matrix.T)
    else:
        final_value = smooth_value*pole_parts_smooth_basis
    return final_value


def get_pole_candidate(interpolable, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
    """
    Evaluate the three-particle pole energy for fixed volume.

    Parameters
    ----------
    L : float
        Finite-volume length.
    n1vecSQ, n2vecSQ, n3vecSQ : int or float
        Squared finite-volume momenta for the three particles.
    m1, m2, m3 : float
        Particle masses.

    Returns
    -------
    float
        Sum of the three finite-volume single-particle energies.
    """
    pole_candidate = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
        + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
        + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)
    return pole_candidate
