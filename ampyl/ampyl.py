#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# ampyl.py
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
from scipy.optimize import root_scalar
from . import shell_utils
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON5
from .constants import EPSILON6
from .constants import EPSILON10
from .constants import EPSILON30
from .constants import QC_IMPL_DEFAULTS
from .constants import DEFAULT_CUTS
from .constants import QC_DICT_DEFAULTS
from .constants import MINMAXOFFSET
from .constants import DEFAULT_EMIN
from .constants import DEFAULT_LMIN
from .constants import bcolors
from .functions import QCFunctions
from .cuts import G
from .cuts import F
from .cuts import FplusG
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
            = shell_utils.get_masks_and_shells_for_k(self, E, L, tbks_entry,
                                                     cindex, slice_index)
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
            zero_support_point = self._get_zero_support_point(threshold)
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

    def _get_zero_support_point(self, threshold):
        alpha = self.alpha
        beta = self.beta
        return (1.0+alpha)*threshold**2/4.0-beta*((3.0-alpha)*threshold**2/4.0)


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
            = shell_utils.get_masks_and_shells_for_kdf(
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

    def _mask_and_shell_helper_nPzero(self, tbks_entry, row_shell_index,
                                      col_shell_index):
        mask_row_shells = None
        mask_col_shells = None
        row_shell = tbks_entry.shells[row_shell_index]
        col_shell = tbks_entry.shells[col_shell_index]
        return mask_row_shells, mask_col_shells, row_shell, col_shell

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


class QC:
    r"""
    QC: A class for handling the quantization condition (QC) in finite-volume
    lattice calculations. This class provides methods for computing QC values
    and finding roots.

    Warning: It is up to the user to select values of alphaKSS and C1cut that
    lead to a sufficient estimate of the F matrix.

    Attributes:
        qcis (QCIndexSpace): The quantization-condition index space, specifying
            data for the class.
        f (F): The F matrix, derived from the quantization condition.
        g (G): The G matrix, derived from the quantization condition.
        fplusg (FplusG): The sum of F and G matrices.
        k (K): The K matrix, representing the two-particle interaction.
        verbosity (int): The verbosity level for logging and debugging.

    Methods:
        get_value(E, L, qc_dict):
            Computes a QC value based on the specified parameters and version.

        get_roots_from_range(E_range, L, qc_dict, ni_functions,
                             cuts=DEFAULT_CUTS):
            Finds roots of the QC within a specified energy range.

        get_all_energies(qc_dict, dL=0.1):
            Computes all energy levels for a given box length and step size.

        simple_try_at_fixed_L(E_bracket, L, qc_dict):
            Simplified method for finding a single root of the QC at a fixed
            box length.

        get_roots_for_Erange_and_LdL(E_range, L, qc_dict, ni_functions,
                                     qc_dict, cuts=DEFAULT_CUTS):
            Computes the roots of the QC for a given energy range and box size,
            considering non-interacting energy levels and specified
            breakpoints.
    """

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0, verbosity=0):
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.fplusg = FplusG(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.k = K(qcis=self.qcis)
        self.kdf = Kdf(qcis=self.qcis)
        self._verbosity = verbosity
        self.verbosity = verbosity

    @property
    def verbosity(self):
        """Verbosity of the QC."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the QC."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

    def get_value(self, E, L, qc_dict):
        r"""
        Compute a value based on the specified parameters and version.

        This method calculates a value using various mathematical operations
        and matrix manipulations. The behavior of the computation depends on
        the `version` parameter, which determines the specific formula or
        algorithm to be used. The method supports multiple versions, each
        corresponding to a different computation strategy.

        Parameters:
            E (float): The energy value. This is a required parameter.
            L (float): The box length. This is a required parameter.
            qc_dict (dict): A dictionary containing the following keys:
                - 'k_params' (list): Parameters for the K-matrices, which
                    splits into:
                    - pcotdelta_parameter_lists (list): Lists of parameters
                        for the K-matrix.
                    - k3_params (list): Parameters for the K3 matrix.
                    This is a required parameter.
                - 'project' (bool): Whether to project the function, defaults
                    to False.
                - 'irrep' (tuple): Irreducible representation information,
                    defaults to None.
                - 'version' (str): Version identifier, defaults to
                    'kdf_zero_1+'. Supported versions include:
                    - 'kdf_zero_1+'
                    - 'f3'
                    - 'kdf_zero_k2_inv'
                    - 'kdf_zero_f+g_inv'
                    - '1+Kdf_F3'
                    - 'kdf+f3inv'
                    - 'detF3inverse'
                    - 'kdf_zero_1+_fgcombo'
                    - 'kdf_zero_detf3inv_asym_fgcombo'
                    - 'kdf+f3inv_asym_fgcombo'
                    - 'kdf_zero_1+_FinverseF3'
                - 'rescale' (float): Rescaling factor, defaults to 1.0.
                - 'shift' (float): Shift value, defaults to 0.0.

        Returns:
            float: The computed value based on the specified parameters and
            version.

        Raises:
            TypeError: If any of the required parameters (`E`, `L`, `k_params`,
                or `irrep` when `project` is True) are missing.

        Notes:
            - The method performs matrix operations and may issue warnings
                if matrix dimensions are mismatched. Temporary fixes are
                applied in such cases by padding or zeroing matrices.
            - The computation involves various components such as `F`, `G`,
                `FplusG`, and `K`, which are derived from other methods or
                interpolations.
        """
        if not isinstance(E, float):
            raise TypeError("E must be a float")
        if not isinstance(L, float):
            raise TypeError("L must be a float")

        qc_dict = self.validate_qc_dict(qc_dict)
        k_params = qc_dict['k_params']
        project = qc_dict['project']
        irrep = qc_dict['irrep']
        version = qc_dict['version']
        rescale = qc_dict['rescale']
        shift = qc_dict['shift']

        [pcotdelta_parameter_lists, k3_params] = k_params

        K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                             project, irrep)*rescale

        createF = (version == '1+Kdf_F3'
                   or version == 'kdf+f3inv'
                   or version == 'f3'
                   or version == 'detF3inverse'
                   or version == 'kdf_zero_1+'
                   or version == 'kdf_zero_k2_inv'
                   or version == 'kdf_zero_f+g_inv'
                   or version == 'kdf_zero_1+_FinverseF3')
        if createF:
            f_smart_interpolate = QC_IMPL_DEFAULTS['f_smart_interpolate']
            if 'f_smart_interpolate' in self.qcis.fvs.qc_impl:
                f_smart_interpolate =\
                    self.qcis.fvs.qc_impl['f_smart_interpolate']

            if f_smart_interpolate:
                warnings.warn(f"\n{bcolors.WARNING}"
                              "f_smart_interpolate is not yet supported. "
                              "Using f instead."
                              f"{bcolors.ENDC}")
                F = self.f.get_value(E, L, project, irrep,
                                     short_string='f')/rescale
            else:
                F = self.f.get_value(E, L, project, irrep,
                                     short_string='f')/rescale

            if len(F) > len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "F and K have different shapes, and F is "
                              "larger. Padding K with extra entries. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                padded_K = np.zeros_like(F)
                padded_K[:len(K), :len(K)] = K
                K = padded_K
            elif len(F) < len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "F and K have different shapes, and F is "
                              "smaller. Setting F to zero. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                F = np.zeros(K.shape)

        createFplusG = (version == '1+Kdf_F3'
                        or version == 'kdf+f3inv'
                        or version == 'f3'
                        or version == 'detF3inverse'
                        or version == 'kdf_zero_1+_fgcombo'
                        or version == 'kdf_zero_detf3inv_asym_fgcombo'
                        or version == 'kdf+f3inv_asym_fgcombo')
        if createFplusG:
            FplusG = self.fplusg.get_value(E, L, project, irrep,
                                           short_string='fplusg')/rescale

            if len(FplusG) > len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "FplusG and K have different shapes, and "
                              "FplusG is larger. "
                              "Padding K with extra entries. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                padded_K = np.zeros_like(FplusG)
                padded_K[:len(K), :len(K)] = K
                K = padded_K
            elif len(FplusG) < len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "FplusG and K have different shapes, and "
                              "FplusG is smaller. "
                              "Setting FplusG to zero. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                FplusG = np.zeros(K.shape)

        createKdf = (version == '1+Kdf_F3'
                     or version == 'kdf+f3inv'
                     or version == 'kdf+f3inv_asym_fgcombo')
        if createKdf:
            Kdf = self.kdf.get_value(E, L, k3_params, project, irrep)*rescale

        createG = (version == 'kdf_zero_1+'
                   or version == 'kdf_zero_k2_inv'
                   or version == 'kdf_zero_f+g_inv'
                   or version == 'kdf_zero_1+_FinverseF3')
        if createG:
            G = self.g.get_value(E, L, project, irrep,
                                 short_string='g')/rescale

            if len(G) > len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "G and K have different shapes, and G is "
                              "larger. Padding K with extra entries. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                padded_K = np.zeros_like(G)
                padded_K[:len(K), :len(K)] = K
                K = padded_K
            elif len(G) < len(K):
                warnings.warn(f"\n{bcolors.WARNING}"
                              "G and K have different shapes, and G is "
                              "smaller. Setting G to zero. "
                              "This is a temporary fix."
                              f"{bcolors.ENDC}")
                G = np.zeros(K.shape)

        if version == '1+Kdf_F3':
            raise NotImplementedError(
                "version '1+Kdf_F3' is not implemented yet.")

        if version == 'kdf+f3inv':
            raise NotImplementedError("kdf+f3inv is not implemented yet")

        if version == 'f3':
            return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3

        if version == 'detF3inverse':
            F3 = (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3
            return 1./np.linalg.det(F3)

        if version == 'kdf_zero_1+_fgcombo':
            id_mat = np.identity(len(FplusG))
            return np.linalg.det(id_mat+(FplusG)@K)-shift

        if version == 'kdf_zero_detf3inv_asym_fgcombo':
            id_mat = np.identity(len(FplusG))
            H = id_mat+FplusG@K
            detH = np.linalg.det(H)
            if np.abs(detH) < EPSILON30:
                Hinverse = id_mat/(EPSILON30)
            else:
                Hinverse = np.linalg.inv(H)
            F3 = (FplusG - FplusG@K@Hinverse@FplusG)/L**3
            return 1./np.linalg.det(F3)

        if version == 'kdf+f3inv_asym_fgcombo':
            id_mat = np.identity(len(FplusG))
            H = id_mat+FplusG@K
            detH = np.linalg.det(H)
            if np.abs(detH) < EPSILON30:
                Hinverse = id_mat/(EPSILON30)
            else:
                Hinverse = np.linalg.inv(H)
            F3 = (FplusG - FplusG@K@Hinverse@FplusG)/L**3
            detF3 = np.linalg.det(F3)
            if np.abs(detF3) < EPSILON30:
                F3inv = id_mat/(EPSILON30)
            else:
                F3inv = np.linalg.inv(F3)
            return np.linalg.det(F3inv+Kdf)

        if version == 'kdf_zero_1+':
            id_mat = np.identity(len(G))
            return np.linalg.det(id_mat+(F+G)@K)

        if version == 'kdf_zero_k2_inv':
            return np.linalg.det(np.linalg.inv(K)+(F+G))

        if version == 'kdf_zero_f+g_inv':
            return np.linalg.det(np.linalg.inv(F+G)+K)

        if version == 'kdf_zero_1+_FinverseF3':
            id_mat = np.identity(len(G))
            block_inv = np.linalg.inv(np.linalg.inv(K)+F+G)
            matrix_in_det = id_mat-3.*block_inv@F
            inverse_det = 1./np.linalg.det(matrix_in_det)
            return inverse_det

    def validate_qc_dict(self, qc_dict):
        if not isinstance(qc_dict, dict):
            raise TypeError("qc_dict must be a dictionary")
        key_is_required = {
            'k_params': True,
            'project': False,
            'irrep': False,
            'version': False,
            'rescale': False,
            'shift': False
        }
        for key in key_is_required:
            if key not in qc_dict and key_is_required[key]:
                raise ValueError(f"qc_dict must contain the key '{key}'")
            if key not in qc_dict:
                qc_dict[key] = QC_DICT_DEFAULTS[key]
        expected_types = {
            'k_params': list,
            'project': bool,
            'irrep': (tuple, type(None)),
            'version': str,
            'rescale': float,
            'shift': float
        }
        for key, expected_type in expected_types.items():
            if not isinstance(qc_dict[key], expected_type):
                raise TypeError(f"qc_dict['{key}'] must be of type "
                                f"{expected_type}")
        if qc_dict['project'] and not isinstance(qc_dict['irrep'], tuple):
            raise TypeError("qc_dict['irrep'] must be a tuple")
        if qc_dict['project'] and len(qc_dict['irrep']) != 2:
            raise ValueError("qc_dict['irrep'] must be a tuple of length 2")
        if qc_dict['project'] and not isinstance(qc_dict['irrep'][0], str):
            raise TypeError("qc_dict['irrep'][0] must be a string")
        if qc_dict['project'] and not isinstance(qc_dict['irrep'][1], int):
            raise TypeError("qc_dict['irrep'][1] must be an int")
        return qc_dict

    def get_all_energies(self, qc_dict, dL=0.1):
        version, irrep = self.get_version_and_irrep(qc_dict)
        E_range, L, L_vals, Lmin, Lmax, Emax, Emin =\
            self.extract_EL_set(version, irrep, dL)
        ni_functions = self.get_ni_functions(irrep)
        all_E_vals = self.get_roots_for_Erange_and_LdL(
            E_range, L, dL, ni_functions, qc_dict)
        interp_E_vals, interp_L_vals =\
            self.build_interpolated_E_vals(all_E_vals, L_vals)

        for i in range(len(interp_E_vals)):
            for j in range(len(interp_E_vals[i])):
                Ltmp = interp_L_vals[i][j]
                Etmp = interp_E_vals[i][j]
                E_vals_tmp = np.array(interp_E_vals[i])
                L_vals_tmp = np.array(interp_L_vals[i])
                if j != 0 and j != len(E_vals_tmp)-1:
                    E_vals_tmp = np.delete(E_vals_tmp, j)
                    L_vals_tmp = np.delete(L_vals_tmp, j)
                sorted_indices = np.argsort(L_vals_tmp)
                Etmp = np.interp(Ltmp, L_vals_tmp[sorted_indices],
                                 E_vals_tmp[sorted_indices])
                interp_E_vals[i][j] = Etmp
                Eupdate = np.nan
                bracket_shift = EPSILON10
                while np.isnan(Eupdate) and bracket_shift < 1.e-1:
                    E_range = [Etmp-bracket_shift, Etmp+bracket_shift]
                    cuts_a = np.logspace(-8, -2, 4)
                    cuts_b = np.linspace(0.011, 0.989, 10)
                    cuts_c = 1.-np.logspace(-8, -2, 4)
                    cuts = np.concatenate((cuts_a, cuts_b, cuts_c))
                    cuts = np.sort(cuts)
                    E_set = self.get_roots_from_range(
                        E_range, Ltmp, qc_dict, ni_functions, cuts=cuts)
                    if len(E_set) == 1:
                        Eupdate = E_set[0]
                    elif len(E_set) > 1:
                        index = np.abs(E_set - Etmp).argmin()
                        Eupdate = E_set[index]
                        warnings.warn(f"\n{bcolors.WARNING}"
                                      f"multiple solutions found for L = {L},"
                                      f"differences are {np.abs(E_set - Etmp)}"
                                      f"{bcolors.ENDC}")
                    bracket_shift = bracket_shift*10.
                if np.isnan(Eupdate):
                    bracket_shift = 1.e-10
                    while np.isnan(Eupdate) and bracket_shift < 3.e-1:
                        E_bracket = [Etmp-bracket_shift, Etmp+bracket_shift]
                        Eupdate = self.simple_try_at_fixed_L(
                            E_bracket, Ltmp, qc_dict)
                        bracket_shift = bracket_shift*5.
                        if np.isnan(Eupdate):
                            warnings.warn(f"\n{bcolors.WARNING}"
                                          "failed to find solution for "
                                          f"L = {L}"
                                          f"{bcolors.ENDC}")
                        else:
                            interp_E_vals[i][j] = Eupdate
                        self.qcis.fvs.qc_impl['fplusg_smart_interpolate']\
                            = True
                else:
                    interp_E_vals[i][j] = Eupdate
                self.qcis.fvs.qc_impl['fplusg_smart_interpolate'] = True

        while Lmin+np.abs(dL) <= L <= Lmax-np.abs(dL):
            L = L+dL
            for i in range(len(interp_E_vals)):
                degree = min(len(interp_L_vals[i])-1, 3)
                if len(interp_L_vals[i]) > 9:
                    fit = np.polyfit(interp_L_vals[i][-9:],
                                     interp_E_vals[i][-9:],
                                     degree)
                else:
                    fit = np.polyfit(interp_L_vals[i], interp_E_vals[i],
                                     degree)
                line = np.poly1d(fit)
                E_guess = line(L)
                dE = 1.e-6
                E_val = []
                while len(E_val) == 0 and dE < 1.e-1:
                    print(f'E_guess = {E_guess}, dE = {dE}')
                    E_range = [E_guess-dE, E_guess+dE]
                    if (E_guess+dE > Emax or E_guess-dE > Emax or
                       E_guess-dE < Emin or E_guess+dE < Emin):
                        warnings.warn('E_guess+-dE out of bounds')
                        dE = 1.0
                        continue
                    cuts = np.linspace(0.1, 0.9, 3)
                    E_val = self.get_roots_from_range(
                        E_range, L, qc_dict, ni_functions, cuts=cuts)
                    print(f'E_val = {E_val}')
                    dE = dE*10.
                if len(E_val) == 1:
                    print(f'Unique solution found with dE = {dE}')
                    print(f'L = {L}, E = {E_val[0]}')
                    interp_E_vals[i].append(E_val[0])
                    interp_L_vals[i].append(L)
                elif len(E_val) > 1:
                    index = np.abs(E_val - E_guess).argmin()
                    Eupdate = E_val[index]
                    interp_E_vals[i].append(Eupdate)
                    interp_L_vals[i].append(L)
                    warnings.warn(f'Multiple solutions found for L = {L}.\n'
                                  f'Differences are {np.abs(E_set - Etmp)}')

                    self.qcis.fvs.qc_impl['fplusg_smart_interpolate'] = True
        return interp_L_vals, interp_E_vals

    def get_version_and_irrep(self, qc_dict):
        project = qc_dict['project']
        if not project:
            raise ValueError("project must be True")
        irrep = qc_dict['irrep']
        version = qc_dict['version']
        return version, irrep

    def extract_EL_set(self, version, irrep, dL):
        if (version in ['kdf_zero_1+_fgcombo',
                        'kdf_zero_detf3inv_asym_fgcombo',
                        'kdf+f3inv_asym_fgcombo']
           and self.qcis.fvs.qc_impl['fplusg_smart_interpolate']):

            Emin_interp, Emax_interp, Lmin_interp, Lmax_interp =\
                self.fplusg.interp_data_lists[irrep][0][0][0]

            Emin = Emin_interp + MINMAXOFFSET
            Emax = Emax_interp - MINMAXOFFSET
            Lmin = Lmin_interp + MINMAXOFFSET
            Lmax = Lmax_interp - MINMAXOFFSET
        else:
            Emin = DEFAULT_EMIN
            Emax = self.qcis.Emax
            Lmin = DEFAULT_LMIN
            Lmax = self.qcis.Lmax
        if dL > 0.:
            L = Lmin+dL+EPSILON4
        else:
            L = Lmax+dL-EPSILON4
        E_range = [Emin, Emax]
        L_vals = [L-dL, L]
        return E_range, L, L_vals, Lmin, Lmax, Emax, Emin

    def get_ni_functions(self, irrep):
        ni_functions = []
        for ni_function_channel in self.qcis.nonint_functions:
            ni_functions.extend(ni_function_channel[irrep])
        return ni_functions

    def get_roots_for_Erange_and_LdL(self, E_range, L, dL, ni_functions,
                                     qc_dict, cuts=DEFAULT_CUTS):
        L_values = [L-dL, L]
        E_sets = []
        for Ltmp in L_values:
            E_set = self.get_roots_from_range(E_range, Ltmp, qc_dict,
                                              ni_functions, cuts=cuts)
            E_sets.append(E_set)
        return E_sets

    def get_roots_from_range(self, E_range, L, qc_dict, ni_functions,
                             cuts=DEFAULT_CUTS):
        """
        Compute the roots of the QC within a specified energy range.

        This method calculates the roots of the QC for a given energy range
        and box size `L`, considering non-interacting energy levels and
        specified breakpoints. It uses a dictionary of parameters to define the
        QC.

        Args:
            E_range (list): A list of two floats specifying the energy range
                [E_min, E_max] within which to search for roots.
            L (float): The box size parameter.
            qc_dict (dict): See `get_value` method for details.
            ni_functions (list): A list of functions that compute
                non-interacting energy levels for a given `L`.
            cuts (list, optional): A list of floats specifying the fractional
                positions within each range to add additional breakpoints.
                Defaults to `DEFAULT_CUTS`.

        Returns:
            list: A list of roots found within the specified energy range.

        Raises:
            TypeError: If `E_range` is not a list of two floats.
            TypeError: If `L` is not a float.
            TypeError: If `qc_dict` is not a dictionary or is missing
                        required keys.
            TypeError: If the types of values in `qc_dict` do not match the
                        expected types.

        Notes:
            - The method identifies non-interacting energy levels within the
                specified range and uses them to define subranges for root
                finding.
            - The `simple_try_at_fixed_L` method is used to find roots within
                each subrange.
            - Roots that are `np.nan` are excluded from the results.
        """
        if not isinstance(E_range, list) or len(E_range) != 2 or \
                not all(isinstance(E, float) for E in E_range):
            raise TypeError("E_range must be a list of two floats")
        if not isinstance(L, float):
            raise TypeError("L must be a float")
        self.validate_qc_dict(qc_dict)
        nonint_energies = []
        for ni_function in ni_functions:
            nonint_energies.append(ni_function(L))
        nonint_energies = np.array(nonint_energies)
        nonint_in_range = nonint_energies[E_range[0] < nonint_energies]
        nonint_in_range = nonint_in_range[nonint_in_range < E_range[1]]
        breakpoints = np.concatenate(([E_range[0]], nonint_in_range,
                                      [E_range[1]]))
        breakpoints = np.sort(breakpoints)
        differences = np.diff(breakpoints)
        cuts = np.sort(cuts)
        all_breakpoints = []
        for i in range(len(differences)):
            for cut in cuts:
                all_breakpoints.append(breakpoints[i]+cut*differences[i])
        all_breakpoints.append(breakpoints[-1])
        all_breakpoints = np.array(all_breakpoints)
        all_ranges = []
        for i in range(len(all_breakpoints)-1):
            candidate_range = [all_breakpoints[i], all_breakpoints[i+1]]
            no_nonint_in_candidate = True
            for nonint_energy in nonint_in_range:
                if candidate_range[0] < nonint_energy < candidate_range[1]:
                    no_nonint_in_candidate = False
            if no_nonint_in_candidate:
                all_ranges.append([all_breakpoints[i], all_breakpoints[i+1]])
        all_roots = []
        for E_bracket in all_ranges:
            root = self.simple_try_at_fixed_L(E_bracket, L, qc_dict)
            if root is not np.nan:
                all_roots.append(root)
        return all_roots

    def simple_try_at_fixed_L(self, E_bracket, L, qc_dict):
        try:
            root = root_scalar(self.get_value,
                               args=(L, qc_dict),
                               bracket=E_bracket).root
            qc_ratio = np.abs(self.get_value(root, L, qc_dict)
                              / self.get_value(root+EPSILON6, L, qc_dict))
            if qc_ratio < EPSILON5:
                return root
            warnings.warn("Root was found but QC at the root is not "
                          "sufficiently close to zero, returning NaN.")
            return np.nan
        except ValueError:
            warnings.warn("Root not found and ValueError was raised either by "
                          "root_scalar or by get_value. Returning NaN.")
            return np.nan
        warnings.warn("Root not found, not sure why. Returning NaN.")
        return np.nan

    def build_interpolated_E_vals(self, all_E_vals, L_vals, n_interp_points=4):
        cleaned = []
        for E_vals in all_E_vals:
            unique_vals = []
            for val in E_vals:
                if not any(np.isclose(val, seen) for seen in unique_vals):
                    unique_vals.append(val)
            cleaned.append(sorted(unique_vals))
        n = min(len(vals) for vals in cleaned)
        trimmed = [vals[:n] for vals in cleaned]
        for vals in trimmed:
            assert np.all(np.diff(vals) >= 0)
        grouped_E_vals = np.array(trimmed).T
        target_L_vals = np.linspace(*L_vals, n_interp_points)
        source_L_vals = np.array([target_L_vals[0], target_L_vals[-1]])
        interp_E_vals = [
            np.interp(target_L_vals, source_L_vals, E_pair).tolist()
            for E_pair in grouped_E_vals
        ]
        interp_L_vals = [target_L_vals.copy().tolist()
                         for _ in range(len(interp_E_vals))]
        return interp_E_vals, interp_L_vals
