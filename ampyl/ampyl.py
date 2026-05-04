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
from .constants import EPSILON30
from .constants import QC_IMPL_DEFAULTS
from .constants import QC_DICT_DEFAULTS
from .constants import bcolors
from .cuts import G
from .cuts import F
from .cuts import FplusG
from . import fv_spectrum_utils
from .k_matrices import K
from .k_matrices import Kdf
import warnings
warnings.simplefilter("once")


class IdentifiedObjectList:
    """Store objects with integer IDs and optional string names."""

    def __init__(self):
        self._items = []
        self._names = {}

    def add(self, item, name=None):
        """Append an item and attach ``id``/``name`` attributes."""
        item_id = len(self._items)
        item_name = str(item_id) if name is None else str(name)
        if item_name in self._names:
            raise ValueError(f"name '{item_name}' is already in use")
        if item is not None:
            item.id = item_id
            item.name = item_name
        self._items.append(item)
        self._names[item_name] = item_id
        return item

    def get(self, identifier=0):
        """Return an item by integer ID or string name."""
        if isinstance(identifier, str):
            if identifier not in self._names:
                raise KeyError(f"unknown name '{identifier}'")
            identifier = self._names[identifier]
        if not isinstance(identifier, int):
            raise TypeError("identifier must be an int or string")
        return self._items[identifier]

    def __getitem__(self, identifier):
        return self.get(identifier)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


class EvaluationPolicy:
    """Select QC options from inclusive E/L ranges."""

    _RANGE_KEYS = ('Lmin', 'Lmax', 'Emin', 'Emax')

    def __init__(self, elements):
        if isinstance(elements, dict):
            elements = [elements]
        if not isinstance(elements, list) or len(elements) == 0:
            raise TypeError("policy must be a non-empty dict or list")
        self.elements = [
            self._validate_element(element_id, element)
            for element_id, element in enumerate(elements)
        ]
        default_element = self.elements[-1]
        if any(default_element[key] is not None for key in self._RANGE_KEYS):
            raise ValueError("the last policy element must use None bounds")

    @classmethod
    def from_qc_dict(cls, qc_dict):
        """Build a policy from ``qc_dict['policy']`` or legacy version data."""
        if 'policy' in qc_dict:
            policy = qc_dict['policy']
            if isinstance(policy, cls):
                return policy
            return cls(policy)
        return cls([{
            'version': qc_dict['version'],
            'Lmin': None,
            'Lmax': None,
            'Emin': None,
            'Emax': None,
        }])

    def select(self, E, L):
        """Return the first matching non-default element, then the default."""
        for element in self.elements[:-1]:
            if self._matches(element, E, L):
                return element
        return self.elements[-1]

    def _validate_element(self, element_id, element):
        if not isinstance(element, dict):
            raise TypeError("each policy element must be a dictionary")
        if 'version' not in element:
            raise ValueError("each policy element must contain 'version'")
        normalized = dict(element)
        normalized['id'] = element_id
        for key in self._RANGE_KEYS:
            normalized.setdefault(key, None)
        return normalized

    def _matches(self, element, E, L):
        if any(element[key] is None for key in self._RANGE_KEYS):
            return False
        return (element['Lmin'] <= L <= element['Lmax']
                and element['Emin'] <= E <= element['Emax'])


class QCMatrixBuilder:
    """Build the matrix inputs needed to evaluate QC expressions."""

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0):
        """Initialize the matrix builder for a QC index space."""
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.fplusg = FplusG(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.k = K(qcis=self.qcis)
        self.kdf = Kdf(qcis=self.qcis)

    def build(self, E, L, qc_dict):
        """Build the matrices required for the selected QC version."""
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
        """Return the F matrix for the requested kinematics."""
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
        """Adjust a matrix shape so it can be combined with K."""
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
        """Return whether a QC version requires the F matrix."""
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
        """Return whether a QC version requires the F+G matrix."""
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
        """Return whether a QC version requires the Kdf matrix."""
        return version in [
            '1+Kdf_F3',
            'kdf+f3inv',
            'kdf+f3inv_asym_fgcombo'
        ]

    def _creates_g(self, version):
        """Return whether a QC version requires the G matrix."""
        return version in [
            'kdf_zero_1+',
            'kdf_zero_k2_inv',
            'kdf_zero_f+g_inv',
            'kdf_zero_1+_FinverseF3'
        ]


class QCVersionEvaluator:
    """Evaluate QC formulas from a prepared set of matrices."""

    def evaluate(self, L, qc_dict, matrices):
        """Evaluate the selected QC expression."""
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
        """Return the symmetric F3 matrix."""
        return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+FplusG) @ F)/L**3

    def _get_asymmetric_f3(self, FplusG, K, L):
        """Return the asymmetric F3 matrix."""
        id_mat = np.identity(len(FplusG))
        H = id_mat+FplusG@K
        detH = np.linalg.det(H)
        if np.abs(detH) < EPSILON30:
            Hinverse = id_mat/(EPSILON30)
        else:
            Hinverse = np.linalg.inv(H)
        return (FplusG - FplusG@K@Hinverse@FplusG)/L**3


class QC:
    """Represent a finite-volume quantization condition.

    The class owns the matrix-building and QC-evaluation machinery for a
    populated QC index space. Use :class:`FVSpectrum` to trace finite-volume
    energy levels from a QC instance.
    """

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0, verbosity=0):
        """Initialize a QC evaluator and its matrix components."""
        self.qcis = qcis
        self.matrix_builder = QCMatrixBuilder(qcis=self.qcis,
                                              C1cut=C1cut,
                                              alphaKSS=alphaKSS)
        self.version_evaluator = QCVersionEvaluator()
        self.f = self.matrix_builder.f
        self.g = self.matrix_builder.g
        self.fplusg = self.matrix_builder.fplusg
        self.k = self.matrix_builder.k
        self.kdf = self.matrix_builder.kdf
        self._verbosity = verbosity
        self.verbosity = verbosity

    @property
    def verbosity(self):
        """int: Verbosity level for QC-related diagnostics."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity level."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

    def get_value(self, E, L, qc_dict):
        """Evaluate the selected quantization-condition expression.

        Parameters
        ----------
        E : float
            Energy value.
        L : float
            Box length.
        qc_dict : dict
            QC evaluation options. Must include ``'k_params'`` and may
            override ``'project'``, ``'irrep'``, ``'version'``,
            ``'rescale'``, and ``'shift'``.

        Returns
        -------
        float or numpy.ndarray
            Value of the selected QC version.
        """
        self._validate_energy_and_volume(E, L)
        qc_dict = self.validate_qc_dict(qc_dict)
        matrices = self.matrix_builder.build(E, L, qc_dict)
        return self.version_evaluator.evaluate(L, qc_dict, matrices)

    def _validate_energy_and_volume(self, E, L):
        """Validate scalar energy and volume inputs."""
        if not isinstance(E, float):
            raise TypeError("E must be a float")
        if not isinstance(L, float):
            raise TypeError("L must be a float")

    def validate_qc_dict(self, qc_dict):
        """Validate and fill defaults in a QC evaluation dictionary."""
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


class FVSpectrum:
    """Trace finite-volume spectra for a QC instance."""

    def __init__(self, qc):
        """Initialize a finite-volume spectrum solver from a QC instance."""
        self.qc = qc

    def get_all_energies(self, qc_dict, dL=0.1):
        """Track all interpolated energy levels across a range of volumes.

        Parameters
        ----------
        qc_dict : dict
            QC evaluation options for the spectrum calculation.
        dL : float, optional
            Step size in box length.

        Returns
        -------
        tuple[list[list[float]], list[list[float]]]
            Interpolated volume values and their corresponding energy levels.
        """
        version, irrep = fv_spectrum_utils._get_version_and_irrep(qc_dict)
        solver_state = fv_spectrum_utils._initialize_energy_scan(
            self, version, irrep, qc_dict, dL)
        fv_spectrum_utils._refine_interpolated_energies(
            self,
            solver_state['interp_E_vals'],
            solver_state['interp_L_vals'],
            qc_dict,
            solver_state['ni_functions']
        )
        fv_spectrum_utils._extend_energy_levels(
            self,
            solver_state['L'],
            solver_state['Lmin'],
            solver_state['Lmax'],
            solver_state['Emin'],
            solver_state['Emax'],
            dL,
            qc_dict,
            solver_state['ni_functions'],
            solver_state['interp_E_vals'],
            solver_state['interp_L_vals']
        )
        interp_L_vals = solver_state['interp_L_vals']
        interp_E_vals = solver_state['interp_E_vals']
        return interp_L_vals, interp_E_vals

    def get_roots_from_range(self, E_range, L, qc_dict, ni_functions,
                             cuts=fv_spectrum_utils.DEFAULT_CUTS):
        """Return QC roots in an energy window at fixed volume.

        Parameters
        ----------
        E_range : list[float]
            Two-element energy interval to search.
        L : float
            Box length.
        qc_dict : dict
            QC evaluation options.
        ni_functions : list[callable]
            Noninteracting energy functions used to split the search interval.
        cuts : array-like, optional
            Fractional cut positions used when subdividing each bracket.

        Returns
        -------
        list[float]
            Roots found in the requested interval.
        """
        return fv_spectrum_utils._get_roots_from_range(
            self, E_range, L, qc_dict, ni_functions, cuts)
