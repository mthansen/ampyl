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
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import block_diag
from .constants import QC_IMPL_DEFAULTS
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON10
from .constants import BAD_MIN_GUESS
from .constants import BAD_MAX_GUESS
from .constants import POLE_CUT
from .constants import bcolors
from .functions import QCFunctions
from .spaces import QCIndexSpace
import warnings
warnings.simplefilter("once")


class Interpolable:
    """Class for preparing objects that can be interpolated."""

    def __init__(self, qcis=QCIndexSpace()):
        self.qcis = qcis
        three_scheme = self.qcis.tbis.three_scheme
        if (three_scheme == 'original pole')\
           or (three_scheme == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.all_relevant_nvecSQ_lists = {}
        self.interp_data_lists = {}
        self.polefree_interp_data_lists = {}
        self.cob_matrix_lists = {}
        self.interp_arrays = {}
        self.matrix_dim_lists = {}
        self.cob_list_lens = {}
        self.smart_interp_tensors = {}
        self.smart_interps = {}

    def build_interpolator(self, Emin, Emax, Estep,
                           Lmin, Lmax, Lstep, project, irrep):
        """
        Builds an interpolator.

        Constructs an interpolator by generating grids and matrices based on
        specified energy and volume ranges. The method determines the smooth
        basis, removes poles, and builds the interpolator functions. Relevant
        data is stored in the class for future use.

        :param Emin: minimum energy value
        :type Emin: float
        :param Emax: maximum energy value
        :type Emax: float
        :param Estep: energy step size
        :type Estep: float
        :param Lmin: minimum volume value
        :type Lmin: float
        :param Lmax: Maximum volume value
        :type Lmax: float
        :param Lstep: volume step size
        :type Lstep: float
        :param project: flag indicating whether to an irrep
        :type project: bool
        :param irrep: irrep identifier
        :type irrep: tuple
        """
        assert project
        # Generate grids and matrix structures
        L_grid, E_grid, max_interp_dim, interp_data_list = self\
            ._grids_and_interp(Emin, Emax, Estep, Lmin, Lmax, Lstep,
                               project, irrep)

        # Determine basis where entries are smooth
        use_cob_matrices = QC_IMPL_DEFAULTS['use_cob_matrices']
        if 'use_cob_matrices' in self.qcis.fvs.qc_impl:
            use_cob_matrices = QC_IMPL_DEFAULTS['use_cob_matrices']
        if use_cob_matrices:
            dim_with_shell_index_all_scs = self\
                ._get_dim_with_shell_index_all_scs(irrep)
            final_set_for_change_of_basis = self\
                ._get_final_set_for_change_of_basis(
                    dim_with_shell_index_all_scs)
            cob_matrix_list = self\
                ._get_cob_matrix_list(final_set_for_change_of_basis)
        else:
            cob_matrix_list = None

        # Populate interpolation data
        energy_volume_index = 0
        interp_data_index = 1
        for L in L_grid:
            for E in E_grid:
                matrix_tmp = self.get_value(E=E, L=L,
                                            project=project, irrep=irrep)
                for cob_matrix in cob_matrix_list:
                    try:
                        matrix_tmp = (cob_matrix.T)@matrix_tmp@cob_matrix
                    except ValueError:
                        pass
                for i in range(len(matrix_tmp)):
                    for j in range(len(matrix_tmp)):
                        interpolable_value = matrix_tmp[i][j]
                        if (not np.isnan(interpolable_value)
                           and (interpolable_value != 0.)):
                            interp_data_list[i][j][interp_data_index]\
                                = interp_data_list[i][j][interp_data_index]\
                                + [[E, L, interpolable_value]]
        for i in range(max_interp_dim):
            for j in range(max_interp_dim):
                interp_data_list[i][j][interp_data_index] =\
                    interp_data_list[i][j][interp_data_index][1:]
                if len(interp_data_list[i][j][interp_data_index]) == 0:
                    interp_data_list[i][j][energy_volume_index] = []
                else:
                    for interp_entry in\
                       interp_data_list[i][j][interp_data_index]:
                        interp_data_list = self\
                            ._update_mins_and_maxes(interp_data_list,
                                                    energy_volume_index,
                                                    i, j, interp_entry)
                    interp_data_list[i][j][energy_volume_index][0]\
                        = interp_data_list[i][j][energy_volume_index][0]\
                        - Lstep
                    interp_data_list[i][j][energy_volume_index][2]\
                        = interp_data_list[i][j][energy_volume_index][2]\
                        - Estep
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  f"Subtracting {Lstep} from Lmin and "
                                  f"{Estep} from Emin"
                                  f"{bcolors.ENDC}")

        # Identify all poles in projected entries
        nvecSQs_by_shell = self._get_all_nvecSQs_by_shell(E=Emax, L=Lmax,
                                                          project=project,
                                                          irrep=irrep)
        all_nvecSQs = self._get_all_nvecSQs(nvecSQs_by_shell)
        m1, m2, m3 = self._extract_masses()
        all_relevant_nvecSQs_list = self\
            ._get_all_relevant_nvecSQs_list(Emax, project, irrep,
                                            max_interp_dim,
                                            interp_data_list, cob_matrix_list,
                                            all_nvecSQs, m1, m2, m3)

        # Remove poles
        polefree_interp_data_list = self\
            ._get_polefree_interp_data_list(max_interp_dim,
                                            interp_data_list,
                                            interp_data_index, m1, m2, m3,
                                            all_relevant_nvecSQs_list)

        for i in range(max_interp_dim):
            for j in range(max_interp_dim):
                interp_data_entry_complete = []
                polefree_interp_data_entry_complete = []
                if len(interp_data_list[i][j][energy_volume_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = interp_data_list[i][j][energy_volume_index]
                    Lgrid_entry = np.arange(Lmin_entry, Lmax_entry+EPSILON4,
                                            Lstep)
                    Egrid_entry = np.arange(Emin_entry, Emax_entry+EPSILON4,
                                            Estep)
                    for L_loop in Lgrid_entry:
                        for E_loop in Egrid_entry:
                            not_found = True
                            for interp_loop_index in\
                                range(len(interp_data_list[i][j][
                                    interp_data_index])):
                                interp_entry = interp_data_list[i][j][
                                    interp_data_index][
                                        interp_loop_index]
                                E_candidate = interp_entry[0]
                                L_candidate = interp_entry[1]
                                if ((np.abs(E_candidate-E_loop)
                                    < EPSILON10)
                                    and (np.abs(L_candidate-L_loop)
                                         < EPSILON10)):
                                    not_found = False
                                    interp_data_entry_complete.\
                                        append(interp_entry)
                                    polefree_interp_entry\
                                        = polefree_interp_data_list[i][j][
                                            interp_data_index][
                                                interp_loop_index]
                                    polefree_interp_data_entry_complete.\
                                        append(polefree_interp_entry)
                            if not_found:
                                interp_data_entry_complete.\
                                    append([E_loop, L_loop, 0.])
                                polefree_interp_data_entry_complete.\
                                    append([E_loop, L_loop, 0.])
                    interp_data_list[i][j][interp_data_index]\
                        = interp_data_entry_complete
                    polefree_interp_data_list[i][j][interp_data_index]\
                        = polefree_interp_data_entry_complete

        # Build interpolator functions
        interp_array = []
        interp_tuple_array = []
        for i in range(max_interp_dim):
            interp_row = []
            interp_tuple_row = []
            for j in range(max_interp_dim):
                if len(interp_data_list[i][j][energy_volume_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = polefree_interp_data_list[i][j][
                            energy_volume_index]
                    L_grid_tmp\
                        = np.arange(Lmin_entry, Lmax_entry+EPSILON4, Lstep)
                    E_grid_tmp\
                        = np.arange(Emin_entry, Emax_entry+EPSILON4, Estep)
                    E_mesh_grid, L_mesh_grid\
                        = np.meshgrid(E_grid_tmp, L_grid_tmp)
                    g_pole_free_mesh_grid\
                        = (np.array(polefree_interp_data_list[i][j][
                            interp_data_index]).T)[2].\
                        reshape(L_mesh_grid.shape).T
                    try:
                        interp_entry =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='cubic')
                    except ValueError:
                        interp_entry =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='linear')
                    interp_row.append(interp_entry)
                    interp_tuple_row.append([E_grid_tmp, L_grid_tmp,
                                             g_pole_free_mesh_grid])
                else:
                    interp_row.append(None)
                    interp_tuple_row.append(None)
            interp_array.append(interp_row)
            interp_tuple_array.append(interp_tuple_row)
        interp_array = np.array(interp_array)
        interp_tuple_array = np.array(interp_tuple_array, dtype=object)

        # Get unique E and L sets
        E_grid_unique = []
        for i in range(len(interp_tuple_array)):
            for j in range(len(interp_tuple_array[i])):
                if interp_tuple_array[i][j] is not None:
                    E_grid_candidate = interp_tuple_array[i][j][0]
                    for E in E_grid_candidate:
                        if E not in E_grid_unique:
                            E_grid_unique.append(E)
        E_grid_unique = np.unique(np.sort(E_grid_unique).round(decimals=10))

        L_grid_unique = []
        for i in range(len(interp_tuple_array)):
            for j in range(len(interp_tuple_array[i])):
                if interp_tuple_array[i][j] is not None:
                    L_grid_candidate = interp_tuple_array[i][j][1]
                    for L in L_grid_candidate:
                        if L not in L_grid_unique:
                            L_grid_unique.append(L)
        L_grid_unique = np.unique(np.sort(L_grid_unique).round(decimals=10))

        # Build the rank 4 tensor
        L_grid_unique = L_grid_unique[1:]
        E_grid_unique = E_grid_unique[1:]
        warnings.warn(f"\n{bcolors.WARNING}"
                      "Discarding the first energy and volume points"
                      f"{bcolors.ENDC}")
        smart_interp_tensor = []
        for E in E_grid_unique:
            vol_rank = []
            for L in L_grid_unique:
                xi_rank = []
                for i in range(len(interp_tuple_array)):
                    xj_rank = []
                    for j in range(len(interp_tuple_array[i])):
                        if interp_tuple_array[i][j] is None:
                            xj_rank.append(0.0)
                        else:
                            en_bools =\
                                (np.abs(interp_tuple_array[i][j][0]-E)
                                 < EPSILON10)
                            vol_bools =\
                                (np.abs(interp_tuple_array[i][j][1]-L)
                                 < EPSILON10)
                            if (not en_bools.any()) or (not vol_bools.any()):
                                xj_rank.append(0.0)
                            else:
                                en_loc = np.where(en_bools)[0][0]
                                vol_loc = np.where(vol_bools)[0][0]
                                xj_rank.append(
                                    interp_tuple_array[i][j][2][
                                        en_loc][vol_loc])
                    xi_rank.append(xj_rank)
                vol_rank.append(xi_rank)
            smart_interp_tensor.append(vol_rank)
        smart_interp_tensor = np.array(smart_interp_tensor)
        try:
            smart_interp = RegularGridInterpolator((E_grid_unique,
                                                    L_grid_unique),
                                                   smart_interp_tensor,
                                                   method='cubic')
        except ValueError:
            smart_interp = RegularGridInterpolator((E_grid_unique,
                                                    L_grid_unique),
                                                   smart_interp_tensor,
                                                   method='linear')

        matrix_dim_list = []
        for cob_matrix in cob_matrix_list:
            matrix_dim_list.append(len(cob_matrix))

        # Add relevant data to self
        self.all_relevant_nvecSQ_lists[irrep] = all_relevant_nvecSQs_list
        self.polefree_interp_data_lists[irrep]\
            = polefree_interp_data_list
        self.interp_data_lists[irrep] = interp_data_list
        self.cob_matrix_lists[irrep] = cob_matrix_list
        self.cob_list_lens[irrep] = len(cob_matrix_list)
        self.interp_arrays[irrep] = interp_array
        self.matrix_dim_lists[irrep] = matrix_dim_list
        self.smart_interp_tensors[irrep] = smart_interp_tensor
        self.smart_interps[irrep] = smart_interp

    def _grids_and_interp(self, Emin, Emax, Estep, Lmin, Lmax, Lstep,
                          project, irrep):
        L_grid = np.arange(Lmin, Lmax+EPSILON4, Lstep)
        E_grid = np.arange(Emin, Emax+EPSILON4, Estep)
        max_interp_matrix_shape = (self.get_value(E=Emax, L=Lmax,
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

    def _get_dim_with_shell_index_all_scs(self, irrep):
        dim_with_shell_index_all_scs = []
        for spectator_channel_index in range(
             len(self.qcis.fcs.sc_list_sorted)):
            dim_with_shell_index_single_sc = []
            ell_set = self\
                .qcis.fcs.sc_list_sorted[spectator_channel_index].ell_set
            ang_mom_dim = 0
            for ell in ell_set:
                ang_mom_dim = ang_mom_dim+(2*ell+1)
            for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
                shell = self.qcis.tbks_list[0][0].shells[shell_index]
                try:
                    transposed_proj_dict = self.qcis.sc_proj_dicts[
                        spectator_channel_index][irrep][
                        ang_mom_dim*shell[0]:ang_mom_dim*shell[1]].T
                    support_rows = []
                    for row_index in range(len(transposed_proj_dict)):
                        row = transposed_proj_dict[row_index]
                        if (not (row@row < EPSILON10)):
                            support_rows = support_rows\
                                + [row_index]
                    proj_candidate = transposed_proj_dict[support_rows].T
                    # only purpose of the following is to trigger KeyError
                    self.qcis.sc_proj_dicts_by_shell[
                        spectator_channel_index][0][shell_index][irrep]
                    dim_with_shell_index_single_sc.\
                        append([(proj_candidate.shape)[1], shell_index])
                except KeyError:
                    pass
            dim_with_shell_index_all_scs.\
                append(dim_with_shell_index_single_sc)
        return dim_with_shell_index_all_scs

    def _get_final_set_for_change_of_basis(self, dim_with_shell_index_all_scs):
        final_set_for_change_of_basis = [[]]
        for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
            dim_shell_counter_all = [[]]
            dim_counter = 0
            for dim_with_shell_index_for_sc in dim_with_shell_index_all_scs:
                dim_shell_counter = [[]]
                for dim_with_shell_index in dim_with_shell_index_for_sc:
                    if dim_with_shell_index[1] <= shell_index:
                        counter_set = []
                        for _ in range(dim_with_shell_index[0]):
                            counter_set = counter_set+[dim_counter]
                            dim_counter = dim_counter+1
                        dim_shell_counter = dim_shell_counter\
                            + [[dim_with_shell_index, counter_set]]
                dim_shell_counter_all = dim_shell_counter_all\
                    + [dim_shell_counter[1:]]
            final_set_for_change_of_basis = final_set_for_change_of_basis\
                + [dim_shell_counter_all[1:]]
        final_set_for_change_of_basis = final_set_for_change_of_basis[1:]
        return final_set_for_change_of_basis

    def _get_cob_matrix_list(self, final_set_for_change_of_basis):
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
        cob_matrix_list = []
        for restack in all_restacks_second:
            cob_matrix_list.append((np.identity(len(restack))[restack]).T)
        return cob_matrix_list

    def _update_mins_and_maxes(self, interpolator_matrix, energy_vol_dat_index,
                               i, j, interpolator_entry):
        [E, L, _] = interpolator_entry
        if L < interpolator_matrix[i][j][
                                energy_vol_dat_index][0]:
            interpolator_matrix[i][j][
                                energy_vol_dat_index][0] = L
        if L > interpolator_matrix[i][j][
                                energy_vol_dat_index][1]:
            interpolator_matrix[i][j][
                                energy_vol_dat_index][1] = L
        if E < interpolator_matrix[i][j][
                                energy_vol_dat_index][2]:
            interpolator_matrix[i][j][
                                energy_vol_dat_index][2] = E
        if E > interpolator_matrix[i][j][
                                energy_vol_dat_index][3]:
            interpolator_matrix[i][j][
                                energy_vol_dat_index][3] = E
        return interpolator_matrix

    def _get_all_nvecSQs_by_shell(self, E=5.0, L=5.0, project=False,
                                  irrep=None):
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if (not ((irrep is None) and (project is False))
           and (not (irrep in self.qcis.proj_dict.keys()))):
            raise ValueError("irrep "+str(irrep)+" not in "
                             + "qcis.proj_dict.keys()")

        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        m1 = masses[0]

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
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        nvecSQs_final = [[]]
        if self.qcis.verbosity >= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            nvecSQs_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                if self.qcis.verbosity >= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     + "supported in G")

                nvecSQs_inner = [[]]
                for row_shell_index in range(len(slices)):
                    nvecSQs_inner_row = []
                    for col_shell_index in range(len(slices)):
                        nvecSQs_tmp = self\
                            ._get_shell_nvecSQs_projs(E, L,
                                                      cindex_row, cindex_col,
                                                      # only for non-zero P
                                                      sc_row_ind, sc_col_ind,
                                                      tbks_entry,
                                                      row_shell_index,
                                                      col_shell_index,
                                                      project, irrep)
                        nvecSQs_inner_row = nvecSQs_inner_row+[nvecSQs_tmp]
                    nvecSQs_inner = nvecSQs_inner+[nvecSQs_inner_row]
                nvecSQs_block_tmp = nvecSQs_inner[1:]
                nvecSQs_outer_row = nvecSQs_outer_row+[nvecSQs_block_tmp]
            nvecSQs_final = nvecSQs_final+[nvecSQs_outer_row]

        nvecSQs_final = nvecSQs_final[1:]
        return nvecSQs_final

    def _get_shell_nvecSQs_projs(self, E=5.0, L=5.0,
                                 cindex_row=None, cindex_col=None,
                                 # only for non-zero P
                                 sc_index_row=None, sc_index_col=None,
                                 tbks_entry=None,
                                 row_shell_index=None,
                                 col_shell_index=None,
                                 project=False, irrep=None):
        nP = self.qcis.nP

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
                                         row_shell_index, col_shell_index)
        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(self.qcis
                                              .sc_proj_dicts_by_shell[
                                                  sc_index_col][ibest]
                                              )[mask_col_shells][
                                                  col_shell_index][irrep]
                    proj_tmp_left = np.conjugate((
                        np.array(self.qcis.
                                 sc_proj_dicts_by_shell[sc_index_row][ibest]
                                 )[mask_row_shells][row_shell_index][irrep]
                    ).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_index_col][0][col_shell_index][irrep]
                    proj_tmp_left = np.conjugate((
                        self.qcis.sc_proj_dicts_by_shell[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
            except KeyError:
                return np.array([])
        nvecSQ_mat_shells = QCFunctions\
            .get_nvecSQ_mat_shells(tbks_entry, row_shell, col_shell)
        return [nvecSQ_mat_shells, proj_tmp_left, proj_tmp_right]

    def _get_masks_and_shells(self, E, nP, L, tbks_entry,
                              cindex_row, cindex_col,
                              row_shell_index, col_shell_index):
        three_slice_index_row\
            = self.qcis._get_three_slice_index(cindex_row)
        three_slice_index_col\
            = self.qcis._get_three_slice_index(cindex_col)
        if not (three_slice_index_row == three_slice_index_col == 0):
            raise ValueError("only one mass slice is supported in G")
        three_slice_index = three_slice_index_row
        if nP@nP == 0:
            mask_row_shells, mask_col_shells, row_shell, col_shell\
                = self._mask_and_shell_helper_nPzero(tbks_entry,
                                                     row_shell_index,
                                                     col_shell_index)
        else:
            mask_row_shells, mask_col_shells, row_shell, col_shell = self.\
                _mask_and_shell_helper_nPnonzero(E, nP, L, tbks_entry,
                                                 row_shell_index,
                                                 col_shell_index,
                                                 three_slice_index)
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def _mask_and_shell_helper_nPzero(self, tbks_entry, row_shell_index,
                                      col_shell_index):
        mask_row_shells = None
        mask_col_shells = None
        row_shell = tbks_entry.shells[row_shell_index]
        col_shell = tbks_entry.shells[col_shell_index]
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def _mask_and_shell_helper_nPnonzero(self, E, nP, L, tbks_entry,
                                         row_shell_index, col_shell_index,
                                         three_slice_index):
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[three_slice_index][0]].\
            masses_indexed
        m_spec = masses[0]
        kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
        kvec_arr = TWOPI*tbks_entry.nvec_arr/L
        omk_arr = np.sqrt(m_spec**2+kvecSQ_arr)
        Pvec = TWOPI*nP/L
        PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
        mask_row = (E-omk_arr)**2-PmkSQ_arr > 0.0
        row_shells = tbks_entry.shells
        mask_row_shells = []
        for row_shell in row_shells:
            mask_row_shells = mask_row_shells\
                    + [mask_row[row_shell[0]:row_shell[1]].all()]
        row_shells = list(np.array(row_shells)[mask_row_shells])
        row_shell = list(row_shells[row_shell_index])

        mask_col = mask_row
        col_shells = tbks_entry.shells
        mask_col_shells = []
        for col_shell in col_shells:
            mask_col_shells = mask_col_shells\
                    + [mask_col[col_shell[0]:col_shell[1]].all()]
        col_shells = list(np.array(col_shells)[mask_col_shells])
        col_shell = list(col_shells[col_shell_index])
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def _get_all_nvecSQs(self, nvecSQs_by_shell):
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
        return all_nvecSQs

    def _extract_masses(self):
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses
        return m1, m2, m3

    def _get_all_relevant_nvecSQs_list(self, Emax, project, irrep,
                                       max_interp_dim, interp_data_list,
                                       cob_matrix_list, all_nvecSQs,
                                       m1, m2, m3):
        interp_data_index = 1
        all_relevant_nvecSQs = []
        for i in range(max_interp_dim):
            for j in range(max_interp_dim):
                interp_data_matrix_entry = interp_data_list[i][j][
                    interp_data_index][1:]
                if len(interp_data_matrix_entry) != 0:
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
                    nvecSQs_all_keeps = [[]]
                    for nvecSQ_entry in all_nvecSQs:
                        n1vecSQ = nvecSQ_entry[0]
                        n2vecSQ = nvecSQ_entry[1]
                        n3vecSQ = nvecSQ_entry[2]
                        removal_at_Lmin = self\
                            .get_pole_candidate(Lmin_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        removal_at_Lmax = self\
                            .get_pole_candidate(Lmax_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        if ((Emin_tmp < removal_at_Lmin < Emax_tmp)
                           or (Emin_tmp < removal_at_Lmax < Emax_tmp)):
                            nvecSQs_all_keeps = nvecSQs_all_keeps\
                                + [[n1vecSQ, n2vecSQ, n3vecSQ]]
                    nvecSQs_all_keeps = nvecSQs_all_keeps[1:]
                    for nvecSQs_keep in nvecSQs_all_keeps:
                        [n1vecSQ, n2vecSQ, n3vecSQ] = nvecSQs_keep
                        Lvals_tmp = [Lmin_tmp+EPSILON4, Lmax_tmp-EPSILON4]
                        for Ltmp in Lvals_tmp:
                            Etmp = self\
                                ._get_pole_candidate_eps(Ltmp,
                                                         n1vecSQ, n2vecSQ,
                                                         n3vecSQ, m1, m2, m3)
                            if Etmp < Emax:
                                try:
                                    matrix_tmp = self\
                                        .get_value(E=Etmp, L=Ltmp,
                                                   project=project,
                                                   irrep=irrep)
                                    for cob_matrix in cob_matrix_list:
                                        try:
                                            matrix_tmp =\
                                                (cob_matrix.T)@matrix_tmp\
                                                @ cob_matrix
                                        except ValueError:
                                            pass
                                    interpolable_value = matrix_tmp[i][j]
                                    near_pole_mag = np.abs(interpolable_value)
                                    pole_found = (near_pole_mag > POLE_CUT)
                                    if (pole_found and
                                        ([i, j, nvecSQs_keep] not in
                                         all_relevant_nvecSQs)):
                                        all_relevant_nvecSQs.\
                                            append([i, j, nvecSQs_keep])
                                except IndexError:
                                    pass
        return all_relevant_nvecSQs

    def _get_pole_candidate_eps(self, L, n1vecSQ, n2vecSQ, n3vecSQ,
                                m1, m2, m3):
        pole_candidate_eps = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                           + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                           + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)+EPSILON10
        return pole_candidate_eps

    def _get_polefree_interp_data_list(self, max_interp_dim,
                                       interp_data_list,
                                       interp_data_index, m1, m2, m3,
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
                        n1vecSQ = nvecSQ[0]
                        n2vecSQ = nvecSQ[1]
                        n3vecSQ = nvecSQ[2]
                        three_omega = self\
                            .get_pole_candidate(L, n1vecSQ, n2vecSQ, n3vecSQ,
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


class G(Interpolable):
    r"""
    Class for the finite-volume G matrix.

    The G matrix is responsible for finite-volume effects arising from switches
        in the scattering pair.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    """

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the G matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP

        g_interpolate = QC_IMPL_DEFAULTS['g_interpolate']
        g_smart_interpolate = QC_IMPL_DEFAULTS['g_smart_interpolate']
        if 'g_interpolate' in self.qcis.fvs.qc_impl:
            g_interpolate = self.qcis.fvs.qc_impl['g_interpolate']
        if 'g_smart_interpolate' in self.qcis.fvs.qc_impl:
            g_smart_interpolate = self.qcis.fvs.qc_impl[
                'g_smart_interpolate']
        if g_interpolate and g_smart_interpolate:
            raise ValueError("g_interpolate and g_smart_interpolate "
                             "cannot both be True")
        if g_interpolate:
            g_final = self._get_value_interpolated(E, L, irrep)
            return g_final
        elif g_smart_interpolate:
            g_final = self._get_value_smart_interpolated(E, L, irrep)
            return g_final

        if self.qcis.verbosity >= 2:
            self._g_verbose_a(E, L, nP)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if self.qcis.verbosity >= 2:
            print('representatives of three_slice:')
            print('    cindex_row =', cindex_row,
                  ', cindex_col =', cindex_col)

        if (not ((irrep is None) and (project is False))
           and (not (irrep in self.qcis.proj_dict.keys()))):
            raise ValueError("irrep "+str(irrep)+" not in "
                             + "qcis.proj_dict.keys()")

        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses
        tbks_entry, slices = self._get_entry_and_slices(E, L, nP, m1)
        g_final = self._get_value_not_interpolated(E, L, project, irrep,
                                                   cindex_col, cindex_row,
                                                   m1, m2, m3,
                                                   tbks_entry, slices)
        return g_final

    def _get_value_interpolated(self, E, L, irrep):
        g_final_smooth_basis = []
        cob_list_len = self.cob_list_lens[irrep]
        matrix_dimension = self.\
            matrix_dim_lists[irrep][cob_list_len-self.
                                    qcis.get_tbks_sub_indices(E, L)[0]-1]
        m1, m2, m3 = self._extract_masses()
        for i in range(matrix_dimension):
            g_row_tmp = []
            for j in range(matrix_dimension):
                interp_tmp = self.interp_arrays[irrep][i][j]
                if interp_tmp is not None:
                    value_tmp = float(interp_tmp((E, L)))
                    for pole_data in (self.polefree_interp_data_lists[
                       irrep][i][j][2]):
                        factor_tmp = E-self.\
                            get_pole_candidate(L, *pole_data[2], m1, m2, m3)
                        value_tmp = value_tmp/factor_tmp
                    g_row_tmp = g_row_tmp+[value_tmp]
                else:
                    g_row_tmp = g_row_tmp+[0.]
            g_final_smooth_basis.append(g_row_tmp)
        g_final_smooth_basis = np.array(g_final_smooth_basis)
        cob_matrix =\
            self.cob_matrix_lists[irrep][cob_list_len
                                         - self.qcis.
                                         get_tbks_sub_indices(E, L)[0]-1]
        g_final =\
            (cob_matrix)@g_final_smooth_basis@(cob_matrix.T)
        return g_final

    def _get_value_smart_interpolated(self, E, L, irrep):
        pole_parts_smooth_basis = []
        cob_list_len = self.cob_list_lens[irrep]
        matrix_dimension = self.\
            matrix_dim_lists[irrep][cob_list_len-self.
                                    qcis.get_tbks_sub_indices(E, L)[0]-1]
        m1, m2, m3 = self._extract_masses()
        for i in range(matrix_dimension):
            pole_row_tmp = []
            for j in range(matrix_dimension):
                value_tmp = 1.
                for pole_data in (self.polefree_interp_data_lists[
                   irrep][i][j][2]):
                    factor_tmp = E-self.\
                        get_pole_candidate(L, *pole_data[2], m1, m2, m3)
                    value_tmp = value_tmp/factor_tmp
                pole_row_tmp = pole_row_tmp+[value_tmp]
            pole_parts_smooth_basis.append(pole_row_tmp)
        pole_parts_smooth_basis = np.array(pole_parts_smooth_basis)
        cob_matrix =\
            self.cob_matrix_lists[irrep][cob_list_len
                                         - self.qcis.
                                         get_tbks_sub_indices(E, L)[0]-1]
        g_smooth = self.smart_interps[irrep]((E, L))
        warnings.warn(f"\n{bcolors.WARNING}"
                      "Resizing g_smooth to match the dimension of the "
                      "cob_matrix matrix. This is a temporary fix."
                      f"{bcolors.ENDC}")
        g_smooth = g_smooth[:len(cob_matrix)]
        g_smooth_T = (g_smooth.T)[:len(cob_matrix)]
        g_smooth = g_smooth_T.T
        g_final_smooth_basis = g_smooth*pole_parts_smooth_basis
        g_final =\
            (cob_matrix)@g_final_smooth_basis@(cob_matrix.T)
        return g_final

    def get_pole_candidate(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                       + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                       + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)
        return pole_candidate

    def _g_verbose_a(self, E, L, nP):
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

    def _get_entry_and_slices(self, E, L, nP, m1):
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
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)
            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])
        return tbks_entry, slices

    def _get_value_not_interpolated(self, E, L, project, irrep, cindex_col,
                                    cindex_row, m1, m2, m3,
                                    tbks_entry, slices):
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
                                               # only for non-zero P
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
                  cindex_row=None, cindex_col=None,  # only for non-zero P
                  sc_index_row=None, sc_index_col=None,
                  ell1=0, ell2=0,
                  g_rescale=1.0, tbks_entry=None,
                  row_shell_index=None,
                  col_shell_index=None,
                  project=False, irrep=None):
        """Build the G matrix on a single shell."""
        three_scheme = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
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
        ibest = self.qcis._get_ibest(E, L)
        proj_tmp_right = np.array(self.qcis.sc_proj_dicts_by_shell[
            sc_index_col][ibest])[mask_col_shells][
                col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
            np.array(self.qcis.
                     sc_proj_dicts_by_shell[sc_index_row][ibest]
                     )[mask_row_shells][row_shell_index][irrep]).T)
        return proj_tmp_right, proj_tmp_left

    def _nPzero_projectors(self, sc_index_row, sc_index_col,
                           row_shell_index, col_shell_index, irrep):
        proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_index_col][0][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
                        self.qcis.sc_proj_dicts_by_shell[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
        return proj_tmp_right, proj_tmp_left

    def _clean_shape(self, g_collection):
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


class F:
    """
    Class for the finite-volume F matrix.

    Warning: It is up to the user to select values of alphaKSS and C1cut that
    lead to a sufficient estimate of the F matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    :param alphaKSS: damping factor entering the zeta functions
    :type alphaKSS: float
    :param C1cut: hard cutoff used in the zeta functions
    :type C1cut: int
    """

    def __init__(self, qcis=None, alphaKSS=1.0, C1cut=3):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS

    def _get_masks_and_shells(self, E, nP, L, tbks_entry,
                              cindex, slice_index):
        mask_slices = None
        three_slice_index\
            = self.qcis._get_three_slice_index(cindex)

        if nP@nP == 0:
            slice_entry = tbks_entry.shells[slice_index]
        else:
            masses = self.qcis.fcs.sc_list_sorted[
                self.qcis.fcs.slices_by_three_masses[three_slice_index][0]]\
                .masses_indexed
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            slices = tbks_entry.shells
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        return mask_slices, slice_entry

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell1=0, ell2=0, tbks_entry=None,
                  slice_index=None, project=False, irrep=None,
                  mask=None):
        """Build the F matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta
        C1cut = self.C1cut
        alphaKSS = self.alphaKSS

        mask_slices, slice_entry\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Fshell = QCFunctions.getF_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell1, ell2,
                                        alpha, beta,
                                        C1cut, alphaKSS,
                                        qc_impl, ts)
        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][ibest])[mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][0][slice_index][irrep]
                    proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                return np.array([])
        if project:
            Fshell = proj_tmp_left@Fshell@proj_tmp_right

        return Fshell

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the F matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        three_slice_index = 0
        cindex = 0
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within F assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            mask = None
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within F assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[three_slice_index][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
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
        f_final_list = []
        for sc_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            ell_set = self.qcis.fcs.sc_list_sorted[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in F")
            ell1 = ell_set[0]
            ell2 = ell1
            for slice_index in range(len(slices)):
                if self.qcis.verbosity >= 2:
                    print('get_shell is receiving:')
                    print(E, L,
                          m1, m2, m3)
                    print(f'cindex = {cindex}\n'
                          f'sc_ind = {sc_ind}\n'
                          f'ell1 = {ell1}\n'
                          f'ell2 = {ell2}\n')
                    print(tbks_entry)
                    print('slice_index = '+str(slice_index))
                    print(project, irrep)
                f_tmp = self.get_shell(E, L,
                                       m1, m2, m3,
                                       cindex,  # only for non-zero P
                                       sc_ind,
                                       ell1, ell2,
                                       tbks_entry,
                                       slice_index,
                                       project, irrep,
                                       mask)
                if len(f_tmp) != 0:
                    f_final_list = f_final_list+[f_tmp]
        return block_diag(*f_final_list)


class FplusG(Interpolable):
    r"""
    Class for F+G (typically with interpolation).
    """

    def __init__(self, qcis=QCIndexSpace(), alphaKSS=1.0, C1cut=3):
        super().__init__(qcis)
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS
        self.f = F(qcis=qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=qcis)

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the F plus G matrix in a shell-based way."""
        g_interpolate = QC_IMPL_DEFAULTS['g_interpolate']
        if 'g_interpolate' in self.qcis.fvs.qc_impl:
            g_interpolate = self.qcis.fvs.qc_impl['g_interpolate']
        if not project:
            raise ValueError("FplusG().get_value() should only be called "
                             + "with project==True")
        if not g_interpolate:
            return self.g.get_value(E=E, L=L, project=project, irrep=irrep)\
                + self.f.get_value(E=E, L=L, project=project, irrep=irrep)

        f_plus_g_final_smooth_basis = []
        cob_list_len = self.cob_list_lens[irrep]
        matrix_dimension = self.\
            matrix_dim_lists[irrep][
                cob_list_len-self.qcis.get_tbks_sub_indices(E, L)[0]-1]
        m1, m2, m3 = self._extract_masses()
        for i in range(matrix_dimension):
            f_plus_g_row_tmp = []
            for j in range(matrix_dimension):
                interp_tmp = self.interp_arrays[irrep][i][j]
                if interp_tmp is not None:
                    value_tmp = float(interp_tmp((E, L)))
                    for pole_data in (self.
                                      polefree_interp_data_lists[
                                          irrep][i][j][2]):
                        factor_tmp = E-self.\
                            get_pole_candidate(L, *pole_data[2], m1, m2, m3)
                        value_tmp = value_tmp/factor_tmp
                    f_plus_g_row_tmp = f_plus_g_row_tmp+[value_tmp]
                else:
                    f_plus_g_row_tmp = f_plus_g_row_tmp+[0.]
            f_plus_g_final_smooth_basis.append(f_plus_g_row_tmp)
        f_plus_g_final_smooth_basis = np.array(f_plus_g_final_smooth_basis)
        cob_matrix = self.\
            cob_matrix_lists[irrep][
                cob_list_len-self.qcis.get_tbks_sub_indices(E, L)[0]-1]
        f_plus_g_matrix_tmp_rotated\
            = (cob_matrix)@f_plus_g_final_smooth_basis@(cob_matrix.T)
        return f_plus_g_matrix_tmp_rotated

    def get_pole_candidate(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                       + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                       + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)
        return pole_candidate

    def _get_all_nvecSQs(self, nvecSQs_by_shell):
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
