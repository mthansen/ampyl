#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# interpolable.py
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
from copy import deepcopy
from . import check_utils
from . import interpolable_utils
from .constants import QC_IMPL_DEFAULTS
from .constants import EPSILON4
from .constants import EPSILON10
from .constants import bcolors
from .spaces import QCIndexSpace
import warnings
warnings.simplefilter("once")


class Interpolable:
    """
    Base class for matrix objects that can be interpolated.

    Subclasses provide ``_get_value_not_interpolated`` and pole-detection
    details, while this base class handles grid construction, pole removal,
    change-of-basis bookkeeping, and storage of one or more interpolators.

    Parameters
    ----------
    qcis : QCIndexSpace, optional
        Quantization-condition index space defining channels, finite-volume
        setup, and three-body interaction data.

    Attributes
    ----------
    all_relevant_nvecSQ_lists : dict
        Pole-candidate momentum-squared data by irrep.
    interp_data_lists, polefree_interp_data_lists : dict
        Raw and pole-removed interpolation data by irrep.
    cob_matrix_lists : dict
        Change-of-basis matrices by irrep.
    interps : dict
        Matrix-valued interpolators by irrep.
    interpolators : list of dict
        Snapshots of interpolation data built by ``build_interpolator``.
    interpolator_names : dict
        Mapping from user-provided names to stored interpolator IDs.
    active_interpolator_id : int or None
        ID of the interpolation data currently loaded on the object.
    """

    def __init__(self, qcis=QCIndexSpace()):
        """
        Initialize interpolation storage for a QC index space.

        Parameters
        ----------
        qcis : QCIndexSpace, optional
            Quantization-condition index space used by matrix evaluations.
        """
        self.qcis = qcis
        self.all_relevant_nvecSQ_lists = {}
        self.interp_data_lists = {}
        self.polefree_interp_data_lists = {}
        self.cob_matrix_lists = {}
        self.matrix_dim_lists = {}
        self.cob_list_lens = {}
        self.interp_tensors = {}
        self.interps = {}
        self.pole_lists = {}
        self.pole_textures_lists = {}
        self.complement_textures_lists = {}
        self.interpolators = []
        self.interpolator_names = {}
        self.active_interpolator_id = None

    def build_interpolator(self, Emin, Emax, Estep, Lmin, Lmax, Lstep,
                           project, irrep, name=None):
        """
        Build and store interpolation data over an energy-volume grid.

        Constructs an interpolator by generating grids and matrices based on
        specified energy and volume ranges. The method determines the smooth
        basis, removes poles, and builds the interpolator functions. Relevant
        data is stored in the class for future use.

        Parameters
        ----------
        Emin, Emax : float
            Minimum and maximum energies in the interpolation grid.
        Estep : float
            Energy grid spacing.
        Lmin, Lmax : float
            Minimum and maximum volumes in the interpolation grid.
        Lstep : float
            Volume grid spacing.
        project : bool
            Whether to project onto an irrep. This method currently requires
            ``True``.
        irrep : tuple
            Irrep key used by the projection dictionaries.
        name : str, optional
            Human-readable name for the stored interpolator.

        Raises
        ------
        AssertionError
            If projection is disabled, or if nonzero-momentum interpolation is
            requested with unsupported QC implementation options.
        ValueError
            If ``name`` duplicates an existing interpolator name.
        """
        assert project
        nP = self.qcis.fvs.nP
        if nP@nP != 0:
            use_cob_matrices = QC_IMPL_DEFAULTS['use_cob_matrices']
            if 'use_cob_matrices' in self.qcis.fvs.qc_impl:
                use_cob_matrices = self.qcis.fvs.qc_impl['use_cob_matrices']
            assert use_cob_matrices is False
            reduce_size = QC_IMPL_DEFAULTS['reduce_size']
            if 'reduce_size' in self.qcis.fvs.qc_impl:
                reduce_size = self.qcis.fvs.qc_impl['reduce_size']
            assert reduce_size is False

        # Generate grids and interp structure
        L_grid, E_grid, max_interp_dim, interp_data_list =\
            interpolable_utils._grids_and_interp(
                self, Emin, Emax, Estep, Lmin, Lmax, Lstep, project, irrep)

        # Determine basis where entries are smooth
        use_cob_matrices = QC_IMPL_DEFAULTS['use_cob_matrices']
        if 'use_cob_matrices' in self.qcis.fvs.qc_impl:
            use_cob_matrices = self.qcis.fvs.qc_impl['use_cob_matrices']
        if use_cob_matrices and self.qcis.fcs.n_three_slices > 1:
            warnings.warn(f"\n{bcolors.WARNING}"
                          "Change-of-basis interpolation matrices are not "
                          "supported for multiple three-body mass slices yet. "
                          "This needs to be added. Proceeding without COB "
                          "matrices."
                          f"{bcolors.ENDC}", stacklevel=2)
            use_cob_matrices = False
        if use_cob_matrices:
            dim_with_shell_index_all_scs =\
                interpolable_utils._get_dim_with_shell_index_all_scs(
                    self, irrep)
            final_set_for_change_of_basis =\
                interpolable_utils._get_final_set_for_change_of_basis(
                    self, dim_with_shell_index_all_scs)
            cob_matrix_list = interpolable_utils._get_cob_matrix_list(
                self, final_set_for_change_of_basis)
        else:
            cob_matrix_list = []

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
                        populate_interp_zeros =\
                            QC_IMPL_DEFAULTS['populate_interp_zeros']
                        if 'populate_interp_zeros' in self.qcis.fvs.qc_impl:
                            populate_interp_zeros = self.qcis.fvs.qc_impl[
                                'populate_interp_zeros']
                        if not populate_interp_zeros:
                            if (not np.isnan(interpolable_value)
                               and (interpolable_value != 0.)):
                                interp_data_list[i][j][interp_data_index]\
                                    = interp_data_list[i][j][
                                        interp_data_index]\
                                    + [[E, L, interpolable_value]]
                        else:
                            if np.isnan(interpolable_value):
                                interpolable_value = 0.
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
                        interp_data_list =\
                            interpolable_utils._update_mins_and_maxes(
                                self, interp_data_list, energy_volume_index,
                                i, j, interp_entry)

        # Identify all poles in projected entries
        nvecSQs_by_shell = interpolable_utils._get_all_nvecSQs_by_shell(
            self, E=Emax, L=Lmax, project=project, irrep=irrep)
        all_nvecSQs = self._get_all_nvecSQs_for_pole_detection(
            nvecSQs_by_shell
        )
        sc_index = self.qcis.fcs.slices_by_three_masses[0][0]
        sc = self.qcis.fcs.sc_list_sorted[sc_index]
        m1 = sc.spectator.mass
        m2 = sc.first_dimer.mass
        m3 = sc.second_dimer.mass
        all_relevant_nvecSQs_list =\
            interpolable_utils._get_all_relevant_nvecSQs_list(
                self, Emax, project, irrep, max_interp_dim, interp_data_list,
                cob_matrix_list, all_nvecSQs, m1, m2, m3)

        # Remove poles
        polefree_interp_data_list =\
            interpolable_utils._get_polefree_interp_data_list(
                self, max_interp_dim, interp_data_list, interp_data_index,
                m1, m2, m3, all_relevant_nvecSQs_list)

        for i in range(max_interp_dim):
            for j in range(max_interp_dim):
                interp_data_entry_complete = []
                polefree_interp_data_entry_complete = []
                if len(interp_data_list[i][j][energy_volume_index]) == 4:
                    [Emin_entry, Emax_entry, Lmin_entry, Lmax_entry]\
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
                    [Emin_entry, Emax_entry, Lmin_entry, Lmax_entry]\
                        = polefree_interp_data_list[i][j][
                            energy_volume_index]
                    L_grid_tmp\
                        = np.arange(Lmin_entry, Lmax_entry+EPSILON4, Lstep)
                    E_grid_tmp\
                        = np.arange(Emin_entry, Emax_entry+EPSILON4, Estep)
                    E_mesh_grid, L_mesh_grid\
                        = np.meshgrid(E_grid_tmp, L_grid_tmp)
                    data_index = 2
                    pole_free_mesh_grid\
                        = (np.array(polefree_interp_data_list[i][j][
                            interp_data_index]).T)[data_index].\
                        reshape(L_mesh_grid.shape).T
                    try:
                        interp_entry =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    pole_free_mesh_grid,
                                                    method='cubic')
                    except ValueError:
                        interp_entry =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    pole_free_mesh_grid,
                                                    method='linear')
                    interp_row.append(interp_entry)
                    interp_tuple_row.append([E_grid_tmp, L_grid_tmp,
                                             pole_free_mesh_grid])
                else:
                    interp_row.append(None)
                    interp_tuple_row.append(None)
            interp_array.append(interp_row)
            interp_tuple_array.append(interp_tuple_row)
        interp_array = np.array(interp_array)
        try:
            interp_tuple_array = np.array(interp_tuple_array, dtype=object)
        except ValueError:
            warnings.warn(f"\n{bcolors.WARNING}"
                          "casting interp_tuple_array to be a numpy array of "
                          "objects failed. Problem is that numpy is trying to "
                          "broadcast input array from shape (n,n) into shape "
                          "(n,). To resolve, loop over the first two ranks."
                          f"{bcolors.ENDC}")
            shape_tmp = (len(interp_tuple_array), len(interp_tuple_array))
            interp_tuple_array_tmp = np.zeros(shape_tmp,
                                              dtype=object)
            for i in range(shape_tmp[0]):
                for j in range(shape_tmp[1]):
                    interp_tuple_array_tmp[i][j] = interp_tuple_array[i][j]
            interp_tuple_array = interp_tuple_array_tmp

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

        if len(cob_matrix_list) == 0:
            matrix_dim_index = 2
            matrix_dim_list = [smart_interp_tensor.shape[matrix_dim_index]]
        else:
            matrix_dim_list = []
            for cob_matrix in cob_matrix_list:
                matrix_dim_list.append(len(cob_matrix))

        smart_poles_list, smart_textures_list, complement_textures_list =\
            interpolable_utils._get_smart_poles(
                self, matrix_dim_list, polefree_interp_data_list)

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
        self.smart_poles_lists[irrep] = smart_poles_list
        self.smart_textures_lists[irrep] = smart_textures_list
        self.complement_textures_lists[irrep] = complement_textures_list
        self._store_interpolator(name=name)

    def _store_interpolator(self, name=None):
        """Store the current interpolation data as an addressable entry."""
        interpolator_id = len(self.interpolators)
        interpolator_name = str(interpolator_id) if name is None else str(name)
        if interpolator_name in self.interpolator_names:
            raise ValueError(f"interpolator name '{interpolator_name}' "
                             "is already in use")
        data_attrs = interpolable_utils._interpolator_data_attrs(self)
        self.interpolators.append({
            attr: deepcopy(getattr(self, attr)) for attr in data_attrs
        })
        self.interpolator_names[interpolator_name] = interpolator_id
        self.active_interpolator_id = interpolator_id
        return interpolator_id

    def _load_interpolator(self, interpolator_id=None,
                           interpolator_name=None):
        """Activate interpolation data by integer ID or string name."""
        if interpolator_name is not None:
            if interpolator_name not in self.interpolator_names:
                raise KeyError(f"unknown interpolator name "
                               f"'{interpolator_name}'")
            interpolator_id = self.interpolator_names[interpolator_name]
        if interpolator_id is None:
            interpolator_id = 0
        if isinstance(interpolator_id, str):
            if interpolator_id not in self.interpolator_names:
                raise KeyError(f"unknown interpolator name "
                               f"'{interpolator_id}'")
            interpolator_id = self.interpolator_names[interpolator_id]
        if not isinstance(interpolator_id, int):
            raise TypeError("interpolator_id must be an int or string")
        data = self.interpolators[interpolator_id]
        for attr, value in data.items():
            setattr(self, attr, value)
        self.active_interpolator_id = interpolator_id

    def _get_all_nvecSQs_for_pole_detection(self, nvecSQs_by_shell):
        return []

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None,
                  short_string='g', interpolate=None, smart_interpolate=None,
                  interpolator_id=None, interpolator_name=None):
        """
        Evaluate the matrix directly or from stored interpolation data.

        Parameters
        ----------
        E : float, optional
            Energy at which to evaluate the matrix.
        L : float, optional
            Volume at which to evaluate the matrix.
        project : bool, optional
            Whether direct evaluation should project onto ``irrep``.
        irrep : tuple, optional
            Irrep key for projected or interpolated evaluations.
        short_string : str, optional
            Prefix used to look up QC implementation flags such as
            ``'<short_string>_interpolate'``.
        interpolate : bool, optional
            Force entrywise interpolation. If ``None``, the corresponding
            ``QC_IMPL_DEFAULTS`` or ``qcis.fvs.qc_impl`` flag is used.
        smart_interpolate : bool, optional
            Force matrix-valued smart interpolation. If ``None``, the
            corresponding implementation flag is used.
        interpolator_id : int or str, optional
            Stored interpolator ID, or name, to activate before evaluation.
        interpolator_name : str, optional
            Stored interpolator name to activate before evaluation.

        Returns
        -------
        numpy.ndarray or None
            Evaluated matrix. The base class returns ``None`` for direct
            non-interpolated evaluation; subclasses override that path.

        Raises
        ------
        ValueError
            If ``E`` exceeds ``qcis.Emax``, if ``L`` exceeds ``qcis.Lmax``, or
            if both interpolation modes are enabled.
        KeyError
            If the requested stored interpolator name is unknown.
        TypeError
            If the requested interpolator ID has an unsupported type.
        """
        check_utils.check_value_within_qcis_bounds(self, E, L)
        interpolate, smart_interpolate =\
            interpolable_utils._get_interpolation_flags(
                self, short_string, interpolate, smart_interpolate)
        if interpolate or smart_interpolate:
            self._load_interpolator(interpolator_id=interpolator_id,
                                    interpolator_name=interpolator_name)
        if smart_interpolate:
            final_value = interpolable_utils._get_value_smart_interpolated(
                self, E, L, irrep)
            return final_value
        if interpolate:
            final_value = interpolable_utils._get_value_interpolated(
                self, E, L, irrep)
            return final_value
        final_value = self._get_value_not_interpolated(E, L, project, irrep)
        return final_value

    def get_pole_candidate(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        return interpolable_utils.get_pole_candidate(
            self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3)

    def _get_value_not_interpolated(self, E, L, project, irrep):
        return None
