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
from scipy.optimize import root_scalar
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
from .cuts import G
from .cuts import F
from .cuts import FplusG
from .k_matrices import K
from .k_matrices import Kdf
import warnings
warnings.simplefilter("once")


class QCMatrixBuilder:
    """Build the matrices needed by each QC version."""

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0):
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.fplusg = FplusG(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.k = K(qcis=self.qcis)
        self.kdf = Kdf(qcis=self.qcis)

    def build(self, E, L, qc_dict):
        k_params = qc_dict['k_params']
        project = qc_dict['project']
        irrep = qc_dict['irrep']
        version = qc_dict['version']
        rescale = qc_dict['rescale']

        [pcotdelta_parameter_lists, k3_params] = k_params

        K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                             project, irrep)*rescale
        matrices = {'K': K}

        if self._creates_f(version):
            F = self._get_f_matrix(E, L, project, irrep, rescale)
            matrices['K'], matrices['F'] = self._match_matrix_to_k(F, K, 'F')
            K = matrices['K']

        if self._creates_fplusg(version):
            FplusG = self.fplusg.get_value(
                E, L, project, irrep, short_string='fplusg')/rescale
            matrices['K'], matrices['FplusG'] = self._match_matrix_to_k(
                FplusG, K, 'FplusG')
            K = matrices['K']

        if self._creates_kdf(version):
            matrices['Kdf'] = self.kdf.get_value(
                E, L, k3_params, project, irrep)*rescale

        if self._creates_g(version):
            G = self.g.get_value(E, L, project, irrep,
                                 short_string='g')/rescale
            matrices['K'], matrices['G'] = self._match_matrix_to_k(G, K, 'G')

        return matrices

    def _get_f_matrix(self, E, L, project, irrep, rescale):
        f_smart_interpolate = QC_IMPL_DEFAULTS['f_smart_interpolate']
        if 'f_smart_interpolate' in self.qcis.fvs.qc_impl:
            f_smart_interpolate = self.qcis.fvs.qc_impl[
                'f_smart_interpolate']
        if f_smart_interpolate:
            warnings.warn(f"\n{bcolors.WARNING}"
                          "f_smart_interpolate is not yet supported. "
                          "Using f instead."
                          f"{bcolors.ENDC}")
        return self.f.get_value(E, L, project, irrep,
                                short_string='f')/rescale

    def _match_matrix_to_k(self, matrix, K, matrix_name):
        if len(matrix) > len(K):
            warnings.warn(f"\n{bcolors.WARNING}"
                          f"{matrix_name} and K have different shapes, and "
                          f"{matrix_name} is larger. Padding K with extra "
                          "entries. This is a temporary fix."
                          f"{bcolors.ENDC}")
            padded_K = np.zeros_like(matrix)
            padded_K[:len(K), :len(K)] = K
            K = padded_K
        elif len(matrix) < len(K):
            warnings.warn(f"\n{bcolors.WARNING}"
                          f"{matrix_name} and K have different shapes, and "
                          f"{matrix_name} is smaller. Setting {matrix_name} "
                          "to zero. This is a temporary fix."
                          f"{bcolors.ENDC}")
            matrix = np.zeros(K.shape)
        return K, matrix

    def _creates_f(self, version):
        return version in [
            '1+Kdf_F3',
            'kdf+f3inv',
            'f3',
            'detF3inverse',
            'kdf_zero_1+',
            'kdf_zero_k2_inv',
            'kdf_zero_f+g_inv',
            'kdf_zero_1+_FinverseF3'
        ]

    def _creates_fplusg(self, version):
        return version in [
            '1+Kdf_F3',
            'kdf+f3inv',
            'f3',
            'detF3inverse',
            'kdf_zero_1+_fgcombo',
            'kdf_zero_detf3inv_asym_fgcombo',
            'kdf+f3inv_asym_fgcombo'
        ]

    def _creates_kdf(self, version):
        return version in [
            '1+Kdf_F3',
            'kdf+f3inv',
            'kdf+f3inv_asym_fgcombo'
        ]

    def _creates_g(self, version):
        return version in [
            'kdf_zero_1+',
            'kdf_zero_k2_inv',
            'kdf_zero_f+g_inv',
            'kdf_zero_1+_FinverseF3'
        ]


class QCVersionEvaluator:
    """Evaluate QC formulas once the needed matrices have been built."""

    def evaluate(self, L, qc_dict, matrices):
        version = qc_dict['version']
        shift = qc_dict['shift']
        K = matrices['K']
        F = matrices.get('F')
        FplusG = matrices.get('FplusG')
        Kdf = matrices.get('Kdf')
        G = matrices.get('G')

        if version == '1+Kdf_F3':
            raise NotImplementedError(
                "version '1+Kdf_F3' is not implemented yet.")

        if version == 'kdf+f3inv':
            raise NotImplementedError("kdf+f3inv is not implemented yet")

        if version == 'f3':
            return self._get_symmetric_f3(F, FplusG, K, L)

        if version == 'detF3inverse':
            F3 = self._get_symmetric_f3(F, FplusG, K, L)
            return 1./np.linalg.det(F3)

        if version == 'kdf_zero_1+_fgcombo':
            id_mat = np.identity(len(FplusG))
            return np.linalg.det(id_mat+(FplusG)@K)-shift

        if version == 'kdf_zero_detf3inv_asym_fgcombo':
            F3 = self._get_asymmetric_f3(FplusG, K, L)
            return 1./np.linalg.det(F3)

        if version == 'kdf+f3inv_asym_fgcombo':
            F3 = self._get_asymmetric_f3(FplusG, K, L)
            id_mat = np.identity(len(FplusG))
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

    def _get_symmetric_f3(self, F, FplusG, K, L):
        return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3

    def _get_asymmetric_f3(self, FplusG, K, L):
        id_mat = np.identity(len(FplusG))
        H = id_mat+FplusG@K
        detH = np.linalg.det(H)
        if np.abs(detH) < EPSILON30:
            Hinverse = id_mat/(EPSILON30)
        else:
            Hinverse = np.linalg.inv(H)
        return (FplusG - FplusG@K@Hinverse@FplusG)/L**3


class QCEnergySolver:
    """Find QC roots and trace energy levels."""

    def __init__(self, qc):
        self.qc = qc

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
                        self.qc.qcis.fvs.qc_impl[
                            'fplusg_smart_interpolate'] = True
                else:
                    interp_E_vals[i][j] = Eupdate
                self.qc.qcis.fvs.qc_impl['fplusg_smart_interpolate'] = True

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

                    self.qc.qcis.fvs.qc_impl[
                        'fplusg_smart_interpolate'] = True
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
           and self.qc.qcis.fvs.qc_impl['fplusg_smart_interpolate']):

            Emin_interp, Emax_interp, Lmin_interp, Lmax_interp =\
                self.qc.fplusg.interp_data_lists[irrep][0][0][0]

            Emin = Emin_interp + MINMAXOFFSET
            Emax = Emax_interp - MINMAXOFFSET
            Lmin = Lmin_interp + MINMAXOFFSET
            Lmax = Lmax_interp - MINMAXOFFSET
        else:
            Emin = DEFAULT_EMIN
            Emax = self.qc.qcis.Emax
            Lmin = DEFAULT_LMIN
            Lmax = self.qc.qcis.Lmax
        if dL > 0.:
            L = Lmin+dL+EPSILON4
        else:
            L = Lmax+dL-EPSILON4
        E_range = [Emin, Emax]
        L_vals = [L-dL, L]
        return E_range, L, L_vals, Lmin, Lmax, Emax, Emin

    def get_ni_functions(self, irrep):
        ni_functions = []
        for ni_function_channel in self.qc.qcis.nonint_functions:
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
        if not isinstance(E_range, list) or len(E_range) != 2 or \
                not all(isinstance(E, float) for E in E_range):
            raise TypeError("E_range must be a list of two floats")
        if not isinstance(L, float):
            raise TypeError("L must be a float")
        self.qc.validate_qc_dict(qc_dict)
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
            root = root_scalar(self.qc.get_value,
                               args=(L, qc_dict),
                               bracket=E_bracket).root
            qc_ratio = np.abs(self.qc.get_value(root, L, qc_dict)
                              / self.qc.get_value(root+EPSILON6,
                                                  L, qc_dict))
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

    def build_interpolated_E_vals(self, all_E_vals, L_vals,
                                  n_interp_points=4):
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
        self.matrix_builder = QCMatrixBuilder(qcis=self.qcis,
                                              C1cut=C1cut,
                                              alphaKSS=alphaKSS)
        self.version_evaluator = QCVersionEvaluator()
        self.energy_solver = QCEnergySolver(self)
        self.f = self.matrix_builder.f
        self.g = self.matrix_builder.g
        self.fplusg = self.matrix_builder.fplusg
        self.k = self.matrix_builder.k
        self.kdf = self.matrix_builder.kdf
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
        Evaluate the selected quantization-condition expression.

        :param E: energy value
        :type E: float
        :param L: box length
        :type L: float
        :param qc_dict: options for the QC evaluation. Must include
            ``'k_params'`` and may override ``'project'``, ``'irrep'``,
            ``'version'``, ``'rescale'``, and ``'shift'``.
        :type qc_dict: dict
        :return: value of the selected QC version
        :rtype: float or numpy.ndarray

        Supported versions are handled by :class:`QCVersionEvaluator`.
        """
        self._validate_energy_and_volume(E, L)
        qc_dict = self.validate_qc_dict(qc_dict)
        matrices = self.matrix_builder.build(E, L, qc_dict)
        return self.version_evaluator.evaluate(L, qc_dict, matrices)

    def _validate_energy_and_volume(self, E, L):
        if not isinstance(E, float):
            raise TypeError("E must be a float")
        if not isinstance(L, float):
            raise TypeError("L must be a float")

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
        return self.energy_solver.get_all_energies(qc_dict, dL)

    def get_version_and_irrep(self, qc_dict):
        return self.energy_solver.get_version_and_irrep(qc_dict)

    def extract_EL_set(self, version, irrep, dL):
        return self.energy_solver.extract_EL_set(version, irrep, dL)

    def get_ni_functions(self, irrep):
        return self.energy_solver.get_ni_functions(irrep)

    def get_roots_for_Erange_and_LdL(self, E_range, L, dL, ni_functions,
                                     qc_dict, cuts=DEFAULT_CUTS):
        return self.energy_solver.get_roots_for_Erange_and_LdL(
            E_range, L, dL, ni_functions, qc_dict, cuts)

    def get_roots_from_range(self, E_range, L, qc_dict, ni_functions,
                             cuts=DEFAULT_CUTS):
        return self.energy_solver.get_roots_from_range(
            E_range, L, qc_dict, ni_functions, cuts)

    def simple_try_at_fixed_L(self, E_bracket, L, qc_dict):
        return self.energy_solver.simple_try_at_fixed_L(
            E_bracket, L, qc_dict)

    def build_interpolated_E_vals(self, all_E_vals, L_vals, n_interp_points=4):
        return self.energy_solver.build_interpolated_E_vals(
            all_E_vals, L_vals, n_interp_points)
