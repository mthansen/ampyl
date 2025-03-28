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
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON10
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
from .cuts import Finterp
from .cuts import Kdf
import warnings
from copy import deepcopy
warnings.simplefilter("once")


class K:
    """
    Class for the two-to-two k matrix.

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

    def _get_masks_and_shells(self, E, L, tbks_entry, cindex, slice_index):
        nP = self.qcis.fvs.nP
        mask_slices = None
        three_slice_index = self.qcis.sc_to_three_slice[cindex]
        if nP@nP == 0:
            slice_entry = tbks_entry.shells[slice_index]
        else:
            sc_list_sorted = self.qcis.fcs.sc_list_sorted
            slices_by_three_masses = self.qcis.fcs.slices_by_three_masses
            inslice_index = 0
            sc_index = slices_by_three_masses[three_slice_index][inslice_index]
            masses = sc_list_sorted[sc_index].masses_indexed
            spec_index = 0
            mspec = masses[spec_index]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            scatterer_a_index = 1
            scatterer_b_index = 2
            threshold = masses[scatterer_a_index] + masses[scatterer_b_index]
            zero_support_point = self._get_zero_support_point(threshold)
            mask = (E-omk_arr)**2-PmkSQ_arr > zero_support_point
            slices = tbks_entry.shells
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        return mask_slices, slice_entry

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

        mask_slices, slice_entry\
            = self._get_masks_and_shells(E, L, tbks_entry, cindex, slice_index)
        Kshell = QCFunctions.getK_array(
            E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell,
            pcotdelta_function, pcotdelta_parameter_list, alpha, beta,
            qc_impl, three_scheme)
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
        finterp (Finterp): Interpolated version of the F matrix.
        g (G): The G matrix, derived from the quantization condition.
        kdf (Kdf): The Kdf matrix, representing the three-particle interaction.
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
                    - 'kdf_zero_1+_asym_fgcombo'
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
                F = self.finterp.get_value(E, L, project, irrep,
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
                        or version == 'kdf_zero_1+_asym_fgcombo')
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
            Kdf = self.kdf.get_value(E, L, k3_params,
                                     project, irrep,
                                     short_string='kdf')*rescale

            id_mat = np.identity(len(Kdf))

            F3 = (F/3 - F@K@np.linalg.inv(id_mat+(FplusG)@K)@F)/L**3
            return np.linalg.det(id_mat + Kdf@F3)

        if version == 'kdf+f3inv':
            Kdf = self.kdf.get_value(E, L, k3_params,
                                     project, irrep,
                                     short_string='kdf')*rescale

            F3 = (F/3 - F@np.linalg.inv(np.linalg.inv(FplusG)+K)@F)/L**3

            F3inv = np.linalg.inv(F3)
            return Kdf + F3inv

        if version == 'f3':
            return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3

        if version == 'detF3inverse':
            F3 = (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3
            return 1./np.linalg.det(F3)

        if version == 'kdf_zero_1+_fgcombo':
            id_mat = np.identity(len(FplusG))
            return np.linalg.det(id_mat+(FplusG)@K)-shift

        if version == 'kdf_zero_1+_asym_fgcombo':
            id_mat = np.identity(len(FplusG))
            H = id_mat+FplusG@K
            detH = np.linalg.det(H)
            if np.abs(detH) < EPSILON10:
                Hinverse = id_mat/(EPSILON10)
            else:
                Hinverse = np.linalg.inv(H)
            F3 = FplusG - FplusG@K@Hinverse@FplusG
            return 1./np.linalg.det(F3)

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

