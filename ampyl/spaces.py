#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# spaces.py
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
from copy import deepcopy
import functools
from .groups import Groups
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON10
from .constants import EPSILON20
from .constants import DELTA_L_FOR_GRID
from .constants import DELTA_E_FOR_GRID
from .constants import L_GRID_SHIFT
from .constants import E_GRID_SHIFT
from .constants import ISO_PROJECTORS
from .constants import CAL_C_ISO
from .constants import QC_IMPL_DEFAULTS
from .constants import bcolors
from .flavor import FlavorChannel
from .flavor import FlavorChannelSpace
import warnings
warnings.simplefilter("once")

PRINT_THRESHOLD_DEFAULT = np.get_printoptions()['threshold']


class FiniteVolumeSetup:
    """
    Class used to represent a finite-volume setup.

    :param formalism: formalism used (Currently, only ``'RFT'`` is supported.)
    :type formalism: str
    :param nP: total momentum in the finite-volume frame
    :type nP: numpy.ndarray of ints with shape (3,)
    :param qc_impl: implementation details of the quantization condition
    :type qc_impl: dict with keys from QC_IMPL_DEFAULTS

    :ivar irrep_set: set of irreps relevant for the finite-volume setup
    :vartype irrep_set: list
    :ivar nPSQ: squared magnitude of the total momentum
    :vartype nPSQ: int
    :ivar nPmag: magnitude of the total momentum
    :vartype nPmag: float

    :raises ValueError: If `nP` is not a numpy.ndarray of shape (3,) or if it
        is not populated with integers. If `qc_impl` is not a dictionary or if
        its keys are not from QC_IMPL_DEFAULTS. If `qc_impl` values are not of
        the correct type.
    """

    def __init__(self, formalism='RFT', nP=np.array([0, 0, 0]), qc_impl={}):
        self.formalism = formalism
        self.qc_impl = qc_impl
        self.nP = nP
        self.set_irreps()

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def qc_impl(self):
        """
        Implementation of the quantization condition.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.
        """
        return self._qc_impl

    @qc_impl.setter
    def qc_impl(self, qc_impl):
        """Set the implementation of the quantization condition."""
        if not isinstance(qc_impl, dict):
            raise ValueError("qc_impl must be a dict")
        for key in qc_impl:
            if key not in QC_IMPL_DEFAULTS:
                raise ValueError("key", key, "not recognized")
        for key in QC_IMPL_DEFAULTS:
            if (key in qc_impl
               and (not isinstance(qc_impl[key],
                                   type(QC_IMPL_DEFAULTS[key])))):
                raise ValueError(f"qc_impl entry {key} mest be a "
                                 f"{type(QC_IMPL_DEFAULTS[key])}")
        self._qc_impl = qc_impl

    def set_irreps(self):
        """Set the irreps relevant for the finite-volume setup."""
        if (self._nP == np.array([0, 0, 0])).all():
            self.A1PLUS = 'A1PLUS'
            self.A2PLUS = 'A2PLUS'
            self.T1PLUS = 'T1PLUS'
            self.T2PLUS = 'T2PLUS'
            self.EPLUS = 'EPLUS'
            self.A1MINUS = 'A1MINUS'
            self.A2MINUS = 'A2MINUS'
            self.T1MINUS = 'T1MINUS'
            self.T2MINUS = 'T2MINUS'
            self.EMINUS = 'EMINUS'
            self.irrep_set = [self.A1PLUS, self.A2PLUS, self.EPLUS,
                              self.T1PLUS, self.T2PLUS, self.A1MINUS,
                              self.A2MINUS, self.EMINUS, self.T1MINUS,
                              self.T2MINUS]
        elif (self._nP == np.array([0, 0, 1])).all():
            self.A1 = 'A1'
            self.A2 = 'A2'
            self.B1 = 'B1'
            self.B2 = 'B2'
            self.E = 'E2'
            self.irrep_set = [self.A1, self.A2, self.B1, self.B2, self.E]
        elif (self._nP == np.array([0, 1, 1])).all():
            self.A1 = 'A1'
            self.A2 = 'A2'
            self.B1 = 'B1'
            self.B2 = 'B2'
            self.irrep_set = [self.A1, self.A2, self.B1, self.B2]
        else:
            raise ValueError("unsupported value of nP in irreps: "
                             + str(self._nP))

    def __str__(self):
        """Return a string representation of the FiniteVolumeSetup object."""
        finite_volume_setup_str =\
            f"FiniteVolumeSetup using the {self.formalism}:\n"
        finite_volume_setup_str += f"    nP = {self._nP},\n"
        finite_volume_setup_str += f"    qc_impl = {self.qc_impl},\n"
        finite_volume_setup_str += f"    irrep_set = {self.irrep_set},\n"
        return finite_volume_setup_str[:-2]+"."


class ThreeBodyInteractionScheme:
    """
    Class used to represent all details of the three-body interaction.

    :param fcs: flavor-channel space needed to define the domain of Kdf
    :type fcs: :class:`FlavorChannelSpace` object
    :param Emin: minimum energy extent of the subthreshold region
    :type Emin: float
    :param three_scheme: three-body interaction scheme (Currently only
        ``'relativistic pole'`` is supported.)
    :type three_scheme: str
    :param scheme_data: data needed to define the three-body interaction
        (Currently only ``[alpha, beta]`` is supported, where alpha dictates
        the width of the smooth cutoff function and beta dictates the
        position.)
    :type scheme_data: list
    :param kdf_functions: list of functions used to define the three-body
        interaction for each pair of FlavorChannels
    :type kdf_functions: list of callables
    """

    def __init__(self, fcs=None, Emin=0.0, three_scheme='relativistic pole',
                 scheme_data=[-1.0, 0.0], kdf_functions=None):
        self.Emin = Emin
        if fcs is None:
            self.fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        else:
            self.fcs = fcs
        self.three_scheme = three_scheme
        self.scheme_data = scheme_data
        if kdf_functions is None:
            self.kdf_functions = []
            for fc in self.fcs.fc_list:
                self.kdf_functions = self.kdf_functions+[self.kdf_iso_constant]
        else:
            self.kdf_functions = kdf_functions

    def with_str(str_func):
        """Change print behavior of a function."""
        def wrapper(f):
            class FuncType:
                def __call__(self, *args, **kwargs):
                    return f(*args, **kwargs)

                def __str__(self):
                    return str_func()
            return functools.wraps(f)(FuncType())
        return wrapper

    def kdf_iso_constant_str():
        """Print behavior for kdf_iso_constant."""
        return "kdf_iso_constant"

    @with_str(kdf_iso_constant_str)
    def kdf_iso_constant(beta_0):
        """Kdf: constant form."""
        return beta_0

    def __str__(self):
        """Return a string representation of the ThreeBodyInteractionScheme."""
        three_body_interaction_scheme_str =\
            "ThreeBodyInteractionScheme with the following data:\n"
        three_body_interaction_scheme_str += f"    Emin = {self.Emin},\n"
        three_body_interaction_scheme_str +=\
            f"    three_scheme = {self.three_scheme},\n"
        three_body_interaction_scheme_str +=\
            f"    [alpha, beta] = {self.scheme_data},\n"
        three_body_interaction_scheme_str +=\
            "    kdf_functions as follows:\n"
        for i in range(len(self.fcs.fc_list)):
            three_body_interaction_scheme_str +=\
                f"        {self.kdf_functions[i]} for\n"
            three_body_interaction_scheme_str +=\
                "        "+str(self.fcs.fc_list[i]).replace(
                    "   ", "            ")[:-1]+",\n"
        return three_body_interaction_scheme_str[:-2]+"."


class ThreeBodyKinematicSpace:
    """
    Class encoding the spectator-momentum kinematics.

    :param nP: total momentum in the finite-volume frame
    :type nP: numpy.ndarray of ints with shape (3,)
    :param nvec_arr: array of nvecs
    :type nvec_arr: numpy.ndarray
    :param build_shell_acc: whether to build the data to accelerate the
        evaluations by shell
    :type build_shell_acc: bool
    :param verbosity: verbosity level
    :type verbosity: int
    :ivar nPSQ: total momentum squared in the finite-volume frame
    :vartype nPSQ: int
    :ivar nPmag: magnitude of the total momentum in the finite-volume frame
    :vartype nPmag: float
    :ivar shells: list of shells
    :vartype shells: list
    :ivar nvecSQ_arr: array of nvec^2
    :vartype nvecSQ_arr: numpy.ndarray
    :ivar nP_minus_nvec_arr: array of nP - nvecs
    :vartype nP_minus_nvec_arr: numpy.ndarray
    :ivar nP_minus_nvec_SQ_arr: array of (nP - nvecs)^2
    :vartype nP_minus_nvec_SQ_arr: numpy.ndarray
    :ivar nvecmag_arr: array of |nvec|
    :vartype nvecmag_arr: numpy.ndarray
    :ivar nP_minus_nvec_mag_arr: array of |nP - nvec|
    :vartype nP_minus_nvec_mag_arr: numpy.ndarray

    :ivar n1vec_mat: matrix of n1vecs
    :vartype n1vec_mat: numpy.ndarray
    :ivar n2vec_mat: matrix of n2vecs
    :vartype n2vec_mat: numpy.ndarray
    :ivar n3vec_mat: matrix of n3vecs
    :vartype n3vec_mat: numpy.ndarray
    :ivar n1vecSQ_mat: matrix of n1vec^2
    :vartype n1vecSQ_mat: numpy.ndarray
    :ivar n2vecSQ_mat: matrix of n2vec^2
    :vartype n2vecSQ_mat: numpy.ndarray
    :ivar n3vecSQ_mat: matrix of n3vec^2
    :vartype n3vecSQ_mat: numpy.ndarray
    :ivar nP_minus_n1vec_mat: matrix of nP - n1vecs
    :vartype nP_minus_n1vec_mat: numpy.ndarray
    :ivar nP_minus_n2vec_mat: matrix of nP - n2vecs
    :vartype nP_minus_n2vec_mat: numpy.ndarray
    :ivar n1vec_arr_all_shells: array of n1vecs for all shells
    :vartype n1vec_arr_all_shells: numpy.ndarray
    :ivar n1vecSQ_arr_all_shells: array of n1vec^2 for all shells
    :vartype n1vecSQ_arr_all_shells: numpy.ndarray
    :ivar n2vec_arr_all_shells: array of n2vecs for all shells
    :vartype n2vec_arr_all_shells: numpy.ndarray
    :ivar n2vecSQ_arr_all_shells: array of n2vec^2 for all shells
    :vartype n2vecSQ_arr_all_shells: numpy.ndarray
    :ivar n1vec_mat_all_shells: matrix of n1vecs for all shells
    :vartype n1vec_mat_all_shells: numpy.ndarray
    :ivar n1vecSQ_mat_all_shells: matrix of n1vec^2 for all shells
    :vartype n1vecSQ_mat_all_shells: numpy.ndarray
    :ivar n2vec_mat_all_shells: matrix of n2vecs for all shells
    :vartype n2vec_mat_all_shells: numpy.ndarray
    :ivar n2vecSQ_mat_all_shells: matrix of n2vec^2 for all shells
    :vartype n2vecSQ_mat_all_shells: numpy.ndarray
    :ivar n3vec_mat_all_shells: matrix of n3vecs for all shells
    :vartype n3vec_mat_all_shells: numpy.ndarray
    :ivar n3vecSQ_mat_all_shells: matrix of n3vec^2 for all shells
    :vartype n3vecSQ_mat_all_shells: numpy.ndarray
    """

    def __init__(self, nP=np.array([0, 0, 0]), nvec_arr=np.array([]),
                 build_shell_acc=True, verbosity=0):
        self.build_shell_acc = build_shell_acc
        self.nP = nP
        self.nvec_arr = nvec_arr
        self.verbosity = verbosity

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def nvec_arr(self):
        """Array of spectator-momentum vectors in the finite-volume frame."""
        return self._nvec_arr

    @nvec_arr.setter
    def nvec_arr(self, nvec_arr):
        """Set the nvec_arr attribute."""
        if self.build_shell_acc:
            if len(nvec_arr) == 0:
                self._nvec_arr = nvec_arr
            else:
                nvec_arr_first_sort = self._get_first_sort(nvec_arr)
                self._nvec_arr, self.shells\
                    = self._get_shell_sort(nvec_arr_first_sort)
                self._populate_nvec_simple_derivatives()
                self._populate_nvec_matrices()
                self._populate_nvec_shells()
        else:
            self._nvec_arr = nvec_arr

    def _get_first_sort(self, nvec_arr):
        nvecSQ_arr = (nvec_arr*nvec_arr).sum(1)
        nP_minus_nvec_arr = self._nP - nvec_arr
        nP_minus_nvec_SQ_arr = (nP_minus_nvec_arr
                                * nP_minus_nvec_arr).sum(1)
        arrtmp = np.concatenate(
            (
                nvecSQ_arr.reshape((1, len(nvec_arr))),
                nP_minus_nvec_SQ_arr.reshape((1, len(nvec_arr))),
                nvec_arr.T
             )
            ).T
        for i in range(5):
            arrtmp = arrtmp[arrtmp[:, 4-i].argsort(kind='mergesort')]
        return (arrtmp.T)[2:].T

    def _get_shell_sort(self, nvec_arr_first_sort):
        group = Groups(ell_max=0)
        little_group = group.get_little_group(self._nP)
        nvec_arr_copy = np.copy(nvec_arr_first_sort)
        shell_dict_nvec_arr = self\
            ._build_shell_dict_nvec_arr(little_group, nvec_arr_copy)
        nvec_arr_shell_sort, shells = self\
            ._build_shell_sort_with_counter(shell_dict_nvec_arr)
        return nvec_arr_shell_sort, shells

    def _build_shell_dict_nvec_arr(self, little_group, nvec_arr_copy):
        shell_dict_nvec_arr = {}
        shell_dict_index = 0
        while len(nvec_arr_copy) > 0:
            nvec_tmp = nvec_arr_copy[0].reshape((3, 1))
            nvec_rotations = (little_group*nvec_tmp).sum(1)
            nvec_rotations_unique = np.unique(nvec_rotations, axis=0)
            shell_dict_nvec_arr[shell_dict_index] = nvec_rotations_unique
            shell_dict_index = shell_dict_index+1
            mins = np.minimum(nvec_rotations_unique.min(0),
                              nvec_arr_copy.min(0))
            nvec_rotations_shifted = nvec_rotations_unique-mins
            nvec_arr_shifted = nvec_arr_copy-mins
            dims = np.maximum(nvec_rotations_shifted.max(0),
                              nvec_arr_shifted.max(0))+1
            nvec_arr_shifted_purged = nvec_arr_shifted[~np.in1d(
                np.ravel_multi_index(
                    nvec_arr_shifted.T, dims
                    ),
                np.ravel_multi_index(
                    nvec_rotations_shifted.T, dims
                    )
                )]
            nvec_arr_copy = nvec_arr_shifted_purged+mins
        return shell_dict_nvec_arr

    def _build_shell_sort_with_counter(self, shell_dict_nvec_arr):
        nvec_arr_shell_sort = None
        shells = []
        shells_counter = 0
        for i in range(len(shell_dict_nvec_arr)):
            if nvec_arr_shell_sort is None:
                nvec_arr_shell_sort = shell_dict_nvec_arr[i]
            else:
                nvec_arr_shell_sort = np.concatenate((nvec_arr_shell_sort,
                                                     shell_dict_nvec_arr[i]))
            shells.append([shells_counter,
                           shells_counter+len(shell_dict_nvec_arr[i])])
            shells_counter = shells_counter+len(shell_dict_nvec_arr[i])
        return nvec_arr_shell_sort, shells

    def _populate_nvec_simple_derivatives(self):
        self.nvecSQ_arr = (self._nvec_arr*self._nvec_arr).sum(1)
        self.nP_minus_nvec_arr = self.nP - self._nvec_arr
        self.nP_minus_nvec_SQ_arr = (self.nP_minus_nvec_arr
                                     * self.nP_minus_nvec_arr).sum(1)
        self.nvecmag_arr = np.sqrt(self.nvecSQ_arr)
        self.nP_minus_nvec_mag_arr = np.sqrt(self.nP_minus_nvec_SQ_arr)

    def _populate_nvec_matrices(self):
        self.n1vec_mat = (np.tile(self._nvec_arr,
                                  (len(self._nvec_arr), 1))).reshape(
                                      (len(self._nvec_arr),
                                       len(self._nvec_arr),
                                       3))
        self.n2vec_mat = np.transpose(self.n1vec_mat, (1, 0, 2))
        self.n3vec_mat = self.nP-self.n1vec_mat-self.n2vec_mat
        self.n1vecSQ_mat = (self.n1vec_mat*self.n1vec_mat).sum(2)
        self.n2vecSQ_mat = (self.n2vec_mat*self.n2vec_mat).sum(2)
        self.n3vecSQ_mat = (self.n3vec_mat*self.n3vec_mat).sum(2)
        self.nP_minus_n1vec_mat = self.nP - self.n1vec_mat
        self.nP_minus_n2vec_mat = self.nP - self.n2vec_mat

    def _populate_nvec_shells(self):
        n1vec_arr_all_shells = []
        n1vecSQ_arr_all_shells = []
        n2vec_arr_all_shells = []
        n2vecSQ_arr_all_shells = []
        n1vec_mat_all_shells = []
        n2vec_mat_all_shells = []
        n3vec_mat_all_shells = []
        n1vecSQ_mat_all_shells = []
        n2vecSQ_mat_all_shells = []
        n3vecSQ_mat_all_shells = []

        for row_shell in self.shells:
            n1vec_arr_row_shells, \
                n1vecSQ_arr_row_shells, \
                n2vec_arr_row_shells, \
                n2vecSQ_arr_row_shells, \
                n1vec_mat_row_shells, \
                n2vec_mat_row_shells, \
                n3vec_mat_row_shells, \
                n1vecSQ_mat_row_shells, \
                n2vecSQ_mat_row_shells, \
                n3vecSQ_mat_row_shells\
                = self._build_row_shells(row_shell)
            n1vec_arr_all_shells.append(n1vec_arr_row_shells)
            n1vecSQ_arr_all_shells.append(n1vecSQ_arr_row_shells)
            n2vec_arr_all_shells.append(n2vec_arr_row_shells)
            n2vecSQ_arr_all_shells.append(n2vecSQ_arr_row_shells)
            n1vec_mat_all_shells.append(n1vec_mat_row_shells)
            n2vec_mat_all_shells.append(n2vec_mat_row_shells)
            n3vec_mat_all_shells.append(n3vec_mat_row_shells)
            n1vecSQ_mat_all_shells.append(n1vecSQ_mat_row_shells)
            n2vecSQ_mat_all_shells.append(n2vecSQ_mat_row_shells)
            n3vecSQ_mat_all_shells.append(n3vecSQ_mat_row_shells)
            self.n1vec_arr_all_shells = n1vec_arr_all_shells
            self.n1vecSQ_arr_all_shells = n1vecSQ_arr_all_shells
            self.n2vec_arr_all_shells = n2vec_arr_all_shells
            self.n2vecSQ_arr_all_shells = n2vecSQ_arr_all_shells
            self.n1vec_mat_all_shells = n1vec_mat_all_shells
            self.n2vec_mat_all_shells = n2vec_mat_all_shells
            self.n3vec_mat_all_shells = n3vec_mat_all_shells
            self.n1vecSQ_mat_all_shells = n1vecSQ_mat_all_shells
            self.n2vecSQ_mat_all_shells = n2vecSQ_mat_all_shells
            self.n3vecSQ_mat_all_shells = n3vecSQ_mat_all_shells

    def _build_row_shells(self, row_shell):
        n1vec_arr_row_shells = []
        n1vecSQ_arr_row_shells = []
        n2vec_arr_row_shells = []
        n2vecSQ_arr_row_shells = []
        n1vec_mat_row_shells = []
        n2vec_mat_row_shells = []
        n3vec_mat_row_shells = []
        n1vecSQ_mat_row_shells = []
        n2vecSQ_mat_row_shells = []
        n3vecSQ_mat_row_shells = []
        for col_shell in self.shells:
            n1vec_arr_shell, \
                n1vecSQ_arr_shell, \
                n2vec_arr_shell, \
                n2vecSQ_arr_shell, \
                n1vec_mat_shell, \
                n2vec_mat_shell, \
                n3vec_mat_shell, \
                n1vecSQ_mat_shell, \
                n2vecSQ_mat_shell, \
                n3vecSQ_mat_shell\
                = self._slice_and_swap(row_shell, col_shell)
            n1vec_arr_row_shells.append(n1vec_arr_shell)
            n1vecSQ_arr_row_shells.append(n1vecSQ_arr_shell)
            n2vec_arr_row_shells.append(n2vec_arr_shell)
            n2vecSQ_arr_row_shells.append(n2vecSQ_arr_shell)
            n1vec_mat_row_shells.append(n1vec_mat_shell)
            n2vec_mat_row_shells.append(n2vec_mat_shell)
            n3vec_mat_row_shells.append(n3vec_mat_shell)
            n1vecSQ_mat_row_shells.append(n1vecSQ_mat_shell)
            n2vecSQ_mat_row_shells.append(n2vecSQ_mat_shell)
            n3vecSQ_mat_row_shells.append(n3vecSQ_mat_shell)
        return n1vec_arr_row_shells, n1vecSQ_arr_row_shells, \
            n2vec_arr_row_shells, n2vecSQ_arr_row_shells, \
            n1vec_mat_row_shells, n2vec_mat_row_shells, n3vec_mat_row_shells, \
            n1vecSQ_mat_row_shells, n2vecSQ_mat_row_shells, \
            n3vecSQ_mat_row_shells

    def _slice_and_swap(self, row_shell, col_shell):
        n1vec_arr_shell = self.nvec_arr[
                            row_shell[0]:row_shell[1]]
        n1vecSQ_arr_shell = self.nvecSQ_arr[
                            row_shell[0]:row_shell[1]]
        n2vec_arr_shell = self.nvec_arr[
                            col_shell[0]:col_shell[1]]
        n2vecSQ_arr_shell = self.nvecSQ_arr[
                            col_shell[0]:col_shell[1]]
        # Awkward swap here
        n1vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n2vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        n2vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n1vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        n3vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n3vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        n1vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n2vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        n2vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n1vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        n3vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n3vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

        return n1vec_arr_shell, n1vecSQ_arr_shell, n2vec_arr_shell, \
            n2vecSQ_arr_shell, n1vec_mat_shell, n2vec_mat_shell, \
            n3vec_mat_shell, n1vecSQ_mat_shell, n2vecSQ_mat_shell, \
            n3vecSQ_mat_shell

    def __str__(self):
        """Return a string representation of the ThreeBodyKinematicSpace."""
        np.set_printoptions(threshold=10)
        three_body_kinematic_space_str =\
            "ThreeBodyKinematicSpace with the following data:\n"
        three_body_kinematic_space_str += "    nvec_arr="\
            + str(self.nvec_arr).replace("\n", "\n             ")+",\n"
        np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)
        return three_body_kinematic_space_str[:-2]+"."


class QCIndexSpace:
    """
    Class representing the quantization condition index space.

    :param fcs: flavor-channel space
    :type fcs: :class:`FlavorChannelSpace` object
    :param fvs: finite-volume setup
    :type fvs: :class:`FiniteVolumeSetup` object
    :param tbis: three-body interaction scheme
    :type tbis: :class:`ThreeBodyInteractionScheme` object
    :param Emax: maximum energy, used to build the spectator index space
    :type Emax: float
    :param Lmax: maximum volume, used to build the spectator index space
    :type Lmax: float
    :param verbosity: verbosity level
    :type verbosity: int

    :ivar nP: total momentum in the finite-volume frame
    :vartype nP: numpy.ndarray of ints with shape (3,)
    :ivar nPSQ: squared total momentum in the finite-volume frame
    :vartype nPSQ: int
    :ivar nPmag: magnitude of the total momentum in the finite-volume frame
    :vartype nPmag: float
    :ivar group: relevant symmetry group
    :vartype group: Group
    :ivar Evals: energy values used to build the grid for non-zero nP
    :vartype Evals: numpy.ndarray
    :ivar Lvals: volume values used to build the grid for non-zero nP
    :vartype Lvals: numpy.ndarray
    :ivar param_structure: structure of the list used to input the quantization
        condition parameters
    :vartype param_structure: list
    :ivar ell_sets: list of sets of angular momenta
    :vartype ell_sets: list
    :ivar ellm_sets: list of sets of angular momenta and their m components
    :vartype ellm_sets: list
    :ivar proj_dict: dictionary of projection matrices
    :vartype proj_dict: dict
    :ivar non_int_proj_dict: dictionary of projection matrices for the
        non-interacting states
    :vartype non_int_proj_dict: dict
    :ivar n_channels: number of channels
    :vartype n_channels: int
    :ivar n_two_channels: number of two-particle channels
    :vartype n_two_channels: int
    :ivar n_three_channels: number of three-particle channels
    :vartype n_three_channels: int
    :ivar tbks_list: list of three-body kinematic spaces
    :vartype tbks_list: list of ThreeBodyKinematicSpace
    :ivar kellm_spaces: list of spectator + angular-momentum spaces
    :vartype kellm_spaces: list
    :ivar kellm_shells: list of spectator + angular-momentum spaces,
        organized by shell
    :vartype kellm_shells: list
    :ivar sc_proj_dicts: list of projection dictionaries by spectator channel
    :vartype sc_proj_dicts: list
    :ivar sc_proj_dicts_by_shell: list of projection dictionaries by spectator
        channel, organized by momentum shell
    :vartype sc_proj_dicts_by_shell: list
    :ivar nvecset_arr: nvecs
    :vartype nvecset_arr: numpy.ndarray
    :ivar nvecset_SQs: nvecs^2
    :vartype nvecset_SQs: numpy.ndarray
    :ivar nvecset_reps: nvec representatives
    :vartype nvecset_reps: numpy.ndarray
    :ivar nvecset_SQreps: nvec^2 representatives
    :vartype nvecset_SQreps: numpy.ndarray
    :ivar nvecset_inds: indices for the nvecs
    :vartype nvecset_inds: numpy.ndarray
    :ivar nvecset_counts: counts for the nvecs
    :vartype nvecset_counts: numpy.ndarray
    :ivar nvecset_batched: nvecs organized by batch
    :vartype nvecset_batched: numpy.ndarray
    :ivar nvecset_ident: nvecs for identical particles
    :vartype nvecset_ident: numpy.ndarray
    :ivar nvecset_ident_SQs: nvecs^2 for identical particles
    :vartype nvecset_ident_SQs: numpy.ndarray
    :ivar nvecset_ident_reps: nvec representatives for identical particles
    :vartype nvecset_ident_reps: numpy.ndarray
    :ivar nvecset_ident_SQreps: nvec^2 representatives for identical particles
    :vartype nvecset_ident_SQreps: numpy.ndarray
    :ivar nvecset_ident_inds: indices for the nvecs for identical particles
    :vartype nvecset_ident_inds: numpy.ndarray
    :ivar nvecset_ident_counts: counts for the nvecs for identical particles
    :vartype nvecset_ident_counts: numpy.ndarray
    :ivar nvecset_ident_batched: nvecs organized by batch for identical
        particles
    :vartype nvecset_ident_batched: numpy.ndarray
    """

    def __init__(self, fcs=None, fvs=None, tbis=None,
                 Emax=5.0, Lmax=5.0,
                 deltaE=DELTA_E_FOR_GRID, deltaL=DELTA_L_FOR_GRID,
                 verbosity=0):
        self.verbosity = verbosity
        self.Emax = Emax
        self.Lmax = Lmax
        self.deltaE = deltaE
        self.deltaL = deltaL

        if fcs is None:
            if verbosity == 2:
                print('setting the flavor-channel space, None was passed')
            self._fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        else:
            if verbosity == 2:
                print('setting the flavor-channel space')
            self._fcs = fcs

        if fvs is None:
            if verbosity == 2:
                print('setting the finite-volume setup, None was passed')
            self.fvs = FiniteVolumeSetup()
        else:
            if verbosity == 2:
                print('setting the finite-volume setup')
            self.fvs = fvs

        if tbis is None:
            if verbosity == 2:
                print('setting the three-body interaction scheme, '
                      + 'None was passed')
            self.tbis = ThreeBodyInteractionScheme()
        else:
            if verbosity == 2:
                print('setting the three-body interaction scheme')
            self.tbis = tbis

        self._nP = self.fvs.nP
        if verbosity == 2:
            print('setting nP to '+str(self.fvs.nP))
        self.nP = self.fvs.nP
        self.fcs = self._fcs

    def populate(self):
        """Populate the index space."""
        ell_max = 4
        half_spin = False
        for sc in self.fcs.sc_list_sorted:
            if np.max(sc.ell_set) > ell_max:
                ell_max = np.max(sc.ell_set)
        for nic in self.fcs.ni_list:
            spins = nic.spins
            for spin in spins:
                spin_int = int(spin)
                if (np.abs(spin-spin_int) > EPSILON10
                   and np.abs(spin-0.5) < EPSILON10):
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Spin half detected; certain objects may "
                                  "not be supported"
                                  f"{bcolors.ENDC}", stacklevel=2)
                    half_spin = True
                elif np.abs(spin-spin_int) > EPSILON10:
                    raise ValueError("only integer spin and (partially spin "
                                     "half) currently supported")
            maxspin = int(np.max(nic.spins))
            if maxspin > ell_max:
                ell_max = maxspin
        self.group = Groups(ell_max=ell_max, half_spin=half_spin)
        self.half_spin = half_spin

        if self.nPSQ != 0:
            if self.verbosity == 2:
                print('nPSQ is nonzero, will use grid')
            [self.Evals, self.Lvals] = self._get_grid_nonzero_nP(self.Emax,
                                                                 self.Lmax,
                                                                 self.deltaE,
                                                                 self.deltaL)
            if self.verbosity == 2:
                print('Lvals =', self.Lvals)
                print('Evals =', self.Evals)
        else:
            self.Lvals = None
            self.Evals = None

        parametrization_structure = []
        two_param_struc_tmp = []
        for sc in self.fcs.sc_list_sorted:
            tmp_entry = []
            for n_params_tmp in sc.n_params_set:
                tmp_entry.append([0.0]*n_params_tmp)
            two_param_struc_tmp.append(tmp_entry)
        parametrization_structure.append(two_param_struc_tmp)
        three_param_struc_tmp = []
        for _ in self.tbis.kdf_functions:
            three_param_struc_tmp = three_param_struc_tmp+[0.0]
        parametrization_structure.append(three_param_struc_tmp)
        self.param_structure = parametrization_structure

        self.populate_all_nvec_arr()
        self.ell_sets = self._get_ell_sets()
        self.populate_all_kellm_spaces()
        self.populate_all_proj_dicts()
        self.proj_dict = self.group.get_full_proj_dict(qcis=self)
        self.populate_all_nonint_data()
        self.populate_nonint_proj_dict()
        self.populate_nonint_multiplicities()

    def populate_nonint_proj_dict(self):
        """Populate the non-interacting projection dictionary."""
        nonint_proj_dict = []
        for nic_index in range(len(self.fcs.ni_list)):
            n_particles = self.fcs.ni_list[nic_index].n_particles
            two_particles = (n_particles == 2)
            three_particles = (n_particles == 3)
            first_spin = self.fcs.ni_list[nic_index].spins[0]
            if two_particles:
                isospin_channel = self.fcs.ni_list[nic_index].isospin_channel
                nonint_proj_dict\
                    .append(self.group.get_noninttwo_proj_dict(
                        qcis=self, nic_index=nic_index,
                        isospin_channel=isospin_channel))
            elif three_particles and first_spin == 0.:
                nonint_proj_dict.append(
                    self.group.get_nonint_proj_dict(qcis=self,
                                                    nic_index=nic_index))
            elif three_particles and first_spin == 1.:
                nonint_proj_dict.append(
                    self.group.
                    get_nonint_proj_dict_vector(qcis=self,
                                                nic_index=nic_index))
            elif three_particles and self.half_spin:
                nonint_proj_dict.append(
                    self.group.get_nonint_proj_dict_half(qcis=self,
                                                         nic_index=nic_index))
            else:
                raise ValueError("only two and three particles with certain "
                                 "spin combinations are supported by "
                                 "nonint_proj_dict")
        self.nonint_proj_dict = nonint_proj_dict

    def _get_grid_nonzero_nP(self, Emax, Lmax, deltaE, deltaL):
        Lmin = np.mod(Lmax-L_GRID_SHIFT, deltaL)+L_GRID_SHIFT
        Emin = np.mod(Emax-E_GRID_SHIFT, deltaE)+E_GRID_SHIFT
        Lvals = np.arange(Lmin, Lmax+EPSILON4, deltaL)
        Evals = np.arange(Emin, Emax+EPSILON4, deltaE)
        if np.abs(Lvals[-1] - Lmax) > EPSILON20:
            Lvals = np.append(Lvals, Lmax)
        if np.abs(Evals[-1] - Emax) > EPSILON20:
            Evals = np.append(Evals, Emax)
        Lvals = Lvals[::-1]
        Evals = Evals[::-1]
        return [Evals, Lvals]

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def fcs(self):
        """FlavorChannelSpace object."""
        return self._fcs

    @fcs.setter
    def fcs(self, fcs):
        """Set the FlavorChannelSpace object."""
        self.n_channels = len(fcs.sc_list_sorted)
        n_two_channels = 0
        n_three_channels = 0
        for sc in fcs.sc_list_sorted:
            if np.sum(sc.masses_indexed) > self.Emax:
                raise ValueError("QCIndexSpace includes channel with "
                                 + "threshold exceeding Emax")
            if sc.fc.n_particles == 2:
                n_two_channels += 1
            elif sc.fc.n_particles == 3:
                n_three_channels += 1
            else:
                raise ValueError("QCIndexSpace currently only supports "
                                 + "two- and three-particle channels")
        self.n_two_channels = n_two_channels
        self.n_three_channels = n_three_channels

        tbks_list = []
        if self.n_two_channels > 0:
            tbks_list.append(ThreeBodyKinematicSpace(nP=self.nP))
        for _ in range(self.fcs.n_three_slices):
            tbks_list.append(ThreeBodyKinematicSpace(nP=self.nP))
        self.tbks_list = tbks_list
        self._fcs = fcs

    def _get_nPspecmax(self, three_slice_index):
        sc = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]
        m_spec = sc.masses_indexed[0]
        Emax = self.Emax
        EmaxSQ = Emax**2
        nPSQ = self.nPSQ
        Lmax = self.Lmax
        EminSQ = self.tbis.Emin**2
        if (EminSQ != 0.0):
            if nPSQ == 0:
                nPspecmax = (Lmax*np.sqrt(
                    Emax**4+(EminSQ-m_spec**2)**2-2.*Emax**2*(EminSQ+m_spec**2)
                    ))/(2.*Emax*TWOPI)
                return nPspecmax
            else:
                raise ValueError("simultaneous nonzero nP and Emin not"
                                 + " supported")
        else:
            if nPSQ == 0:
                nPspecmax = Lmax*(EmaxSQ-m_spec**2)/(2.0*TWOPI*Emax)
                return nPspecmax
            else:
                nPmag = np.sqrt(nPSQ)
                nPspecmax = (FOURPI2*nPmag*(
                    Lmax**2*(EmaxSQ+m_spec**2)-FOURPI2*nPSQ
                    )+np.sqrt(EmaxSQ*FOURPI2*Lmax**2*(
                        Lmax**2*(-EmaxSQ+m_spec**2)+FOURPI2*nPSQ
                        )**2))/(2.*FOURPI2*(EmaxSQ*Lmax**2-FOURPI2*nPSQ))
                return nPspecmax

    def populate_nvec_arr_slot(self, slot_index, three_particle_channel=True):
        """Populate a given nvec_arr slot."""
        if three_particle_channel:
            if self.n_two_channels > 0:
                three_slice_index = slot_index-1
            else:
                three_slice_index = slot_index
            if self.nPSQ == 0:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                if self.verbosity >= 2:
                    print(f"populating nvec array, three_slice_index = "
                          f"{three_slice_index}")
                self._populate_slot_zero_momentum(slot_index, nPspecmax)
                if self.half_spin:
                    self._populate_spin_zero_momentum(slot_index, nPspecmax)
            else:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                self._populate_slot_nonzero_momentum(slot_index,
                                                     three_slice_index,
                                                     nPspecmax)
                if self.half_spin:
                    raise ValueError("half spin not yet supported for "
                                     "nonzero nP")
        elif not three_particle_channel and not self.half_spin:
            nPspecmax = EPSILON4
            self._populate_slot_zero_momentum(slot_index, nPspecmax)
        else:
            raise ValueError("half spin not yet supported for two-particle "
                             "channels")

    def _populate_slot_nonzero_momentum(self, slot_index, three_slice_index,
                                        nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is a list, "
                      "taking first entry")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is not a list")
        rng = range(-int(nPspecmax), int(nPspecmax)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        Lmax = self.Lmax
        Emax = self.Emax
        deltaE = self.deltaE
        deltaL = self.deltaL
        masses = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]\
            .masses_indexed
        m_spec = masses[0]
        nP = self.nP
        [Evals, Lvals] = self._get_grid_nonzero_nP(Emax, Lmax, deltaE, deltaL)
        for Ltmp in Lvals:
            for Etmp in Evals:
                self._populate_EL_iteration(slot_index, tbks_tmp, nvec_arr,
                                            m_spec, nP, Ltmp, Etmp)
        self.tbks_list[slot_index] = self.tbks_list[slot_index][:-1]

    def _populate_EL_iteration(self, slot_index, tbks_tmp, nvec_arr, m_spec,
                               nP, Ltmp, Etmp):
        E2CMSQ = (Etmp-np.sqrt(m_spec**2+FOURPI2/Ltmp**2
                               * ((nvec_arr**2)
                                  .sum(axis=1))))**2\
            - FOURPI2/Ltmp**2*((nP-nvec_arr)**2).sum(axis=1)
        carr = E2CMSQ < 0.0
        E2CMSQ = E2CMSQ.reshape((len(E2CMSQ), 1))
        E2nvec_arr = np.concatenate((E2CMSQ, nvec_arr), axis=1)
        E2nvec_arr = np.delete(E2nvec_arr, np.where(carr), axis=0)
        nvec_arr_tmp = ((E2nvec_arr.T)[1:]).T
        nvec_arr_tmp = nvec_arr_tmp.astype(np.int64)
        if self.verbosity >= 2:
            print(f"L = {np.round(Ltmp, 10)}, "
                  f"E = {np.round(Etmp, 10)}")
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr_tmp
        if self.verbosity >= 2:
            print(self.tbks_list[slot_index][-1])
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = self.tbks_list[slot_index]\
            + [tbks_copy]

    def _populate_slot_zero_momentum(self, slot_index, nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is a list, "
                      "taking first entry")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is not a list")
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        nPspec = nPspecmax
        if self.verbosity >= 2:
            print("populating up to nPspecmax = "+str(nPspecmax))
        while nPspec > 0:
            nPspec = self._populate_nP_iteration(slot_index, tbks_tmp, nPspec)
        self.tbks_list[slot_index] = self.tbks_list[slot_index][:-1]

    def _populate_spin_zero_momentum(self, slot_index, nPspecmax):
        warnings.warn(f"\n{bcolors.WARNING}"
                      f"Populate for spin not yet implemented"
                      f"{bcolors.ENDC}", stacklevel=2)
        pass

    def _populate_nP_iteration(self, slot_index, tbks_tmp, nPspec):
        if self.verbosity >= 2:
            print("nPspec**2 = ", int(nPspec**2))
        rng = range(-int(nPspec), int(nPspec)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        carr = (nvec_arr*nvec_arr).sum(1) > nPspec**2
        nvec_arr = np.delete(nvec_arr, np.where(carr), axis=0)
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr
        if self.verbosity >= 2:
            print(self.tbks_list[slot_index][-1])
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] =\
            self.tbks_list[slot_index]+[tbks_copy]
        nPspecSQ = nPspec**2-1.0
        if nPspecSQ >= 0.0:
            nPspec = np.sqrt(nPspecSQ)
        else:
            nPspec = -1.0
        return nPspec

    def populate_all_nvec_arr(self):
        """Populate all nvec_arr slots."""
        if self.n_two_channels > 0:
            slot_index = 0
            self.populate_nvec_arr_slot(slot_index,
                                        three_particle_channel=False)
        for three_slice_index in range(self.fcs.n_three_slices):
            if self.n_two_channels > 0:
                slot_index = three_slice_index+1
            else:
                slot_index = three_slice_index
            self.populate_nvec_arr_slot(slot_index)

    def _get_ell_sets(self):
        ell_sets = [[]]
        for sc_index in range(self.n_channels):
            ell_set = self.fcs.sc_list_sorted[sc_index].ell_set
            ell_sets = ell_sets+[ell_set]
        return ell_sets[1:]

    @property
    def ell_sets(self):
        """Angular-momentum value sets."""
        return self._ell_sets

    @ell_sets.setter
    def ell_sets(self, ell_sets):
        """Set angular-momentum value sets."""
        self._ell_sets = ell_sets
        ellm_sets = []
        for ell_set in ell_sets:
            ellm_set = []
            for ell in ell_set:
                for mazi in range(-ell, ell+1):
                    ellm_set.append((ell, mazi))
            ellm_sets.append(ellm_set)
        self.ellm_sets = ellm_sets

    def _get_three_slice_index(self, sc_index):
        three_channel_max = self.fcs.slices_by_three_masses[-1][-1]-1
        if ((self.n_two_channels == 0) and (sc_index > three_channel_max)):
            raise ValueError(f"using cindex = {sc_index} with three_slices = "
                             f"{self.fcs.slices_by_three_masses} "
                             f"and (no two-particle channels) "
                             f"is not allowed")
        slice_index = 0
        for three_slice in self.fcs.slices_by_three_masses:
            if sc_index > three_slice[1]:
                slice_index = slice_index+1
        if self.n_two_channels > 0:
            slice_index = slice_index+1
        return slice_index

    def populate_all_kellm_spaces(self):
        """Populate all kellm spaces."""
        if self.verbosity >= 2:
            print("populating kellm spaces")
            print(self.n_channels, "channels to populate")
        kellm_shells = [[]]
        kellm_spaces = [[]]
        for cindex in range(self.n_channels):
            if cindex < self.n_two_channels:
                slot_index = 0
            else:
                cindex_shift = cindex-self.n_two_channels
                slot_index = -1
                for k in range(len(self.fcs.slices_by_three_masses)):
                    three_slice = self.fcs.slices_by_three_masses[k]
                    if three_slice[0] <= cindex_shift < three_slice[1]:
                        slot_index = k
                if self.n_two_channels > 0:
                    slot_index = slot_index+1
            tbks_list_tmp = self.tbks_list[slot_index]
            ellm_set = self.ellm_sets[cindex]
            kellm_shells_single = [[]]
            kellm_spaces_single = [[]]
            for tbks_tmp in tbks_list_tmp:
                nvec_arr = tbks_tmp.nvec_arr
                kellm_shell = (len(ellm_set)*np.array(tbks_tmp.shells
                                                      )).tolist()
                kellm_shells_single = kellm_shells_single+[kellm_shell]
                ellm_set_extended = np.tile(ellm_set, (len(nvec_arr), 1))
                nvec_arr_extended = np.repeat(nvec_arr, len(ellm_set),
                                              axis=0)
                kellm_space = np.concatenate((nvec_arr_extended,
                                              ellm_set_extended),
                                             axis=1)
                kellm_spaces_single = kellm_spaces_single+[kellm_space]
            kellm_shells = kellm_shells+[kellm_shells_single[1:]]
            kellm_spaces = kellm_spaces+[kellm_spaces_single[1:]]
        self.kellm_spaces = kellm_spaces[1:]
        self.kellm_shells = kellm_shells[1:]
        if self.verbosity >= 2:
            print("result for kellm spaces")
            print("location: channel index, nvec-space index")
            np.set_printoptions(threshold=20)
            for i in range(len(self.kellm_spaces)):
                for j in range(len(self.kellm_spaces[i])):
                    print('location:', i, j)
                    print(self.kellm_spaces[i][j])
            np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)

    def populate_all_proj_dicts(self):
        """Populate all projector dictionaries."""
        group = self.group
        sc_proj_dicts = []
        sc_proj_dicts_by_shell = [[]]
        if self.verbosity >= 2:
            print("getting the dict for following qcis:")
            print(self)
        for sc_index in range(self.n_channels):
            proj_dict = group.get_channel_proj_dict(qcis=self,
                                                    sc_index=sc_index)
            sc_proj_dicts = sc_proj_dicts+[proj_dict]
            sc_proj_dict_channel_by_shell = [[]]
            for kellm_shell_index in range(len(self.kellm_shells[sc_index])):
                kellm_shell_set = self.kellm_shells[sc_index][
                    kellm_shell_index]
                sc_proj_dict_shell_set = []
                for kellm_shell in kellm_shell_set:
                    sc_proj_dict_shell_set = sc_proj_dict_shell_set\
                        + [group.get_shell_proj_dict(
                            qcis=self,
                            cindex=sc_index,
                            kellm_shell=kellm_shell,
                            shell_index=kellm_shell_index)]
                sc_proj_dict_channel_by_shell = sc_proj_dict_channel_by_shell\
                    + [sc_proj_dict_shell_set]
            sc_proj_dict_channel_by_shell = sc_proj_dict_channel_by_shell[1:]
            sc_proj_dicts_by_shell = sc_proj_dicts_by_shell\
                + [sc_proj_dict_channel_by_shell]
        self.sc_proj_dicts = sc_proj_dicts
        self.sc_proj_dicts_by_shell = sc_proj_dicts_by_shell[1:]

    def get_tbks_sub_indices(self, E, L):
        """Get the indices of the relevant three-body kinematics spaces."""
        if E > self.Emax:
            raise ValueError("get_tbks_sub_indices called with E > Emax")
        if L > self.Lmax:
            raise ValueError("get_tbks_sub_indices called with L > Lmax")
        if self.nPSQ != 0:
            tbks_sub_indices =\
                self._get_tbks_sub_indices_nonzero_mom(E, L)
            return tbks_sub_indices
        tbks_sub_indices = self._get_tbks_sub_indices_zero_mom(E, L)
        return tbks_sub_indices

    def _get_tbks_sub_indices_zero_mom(self, E, L):
        tbks_sub_indices = [0]*len(self.tbks_list)
        for slice_index in range(self.fcs.n_three_slices):
            sc_index = self.fcs.slices_by_three_masses[slice_index][0]
            nPspecmax = self._get_nPspecmax(sc_index)
            sc = self.fcs.sc_list_sorted[sc_index]
            m_spec = sc.fc.masses[sc.indexing[0]]
            ESQ = E**2
            nPSQ = self.nPSQ
            EminSQ = self.tbis.Emin**2
            if (EminSQ != 0.0):
                if nPSQ == 0:
                    nPspecnew = (L*np.sqrt(
                        E**4+(EminSQ-m_spec**2)**2-2.*E**2*(EminSQ+m_spec**2)
                        ))/(2.*E*TWOPI)
                else:
                    raise ValueError("nonzero nP and Emin not supported")
            else:
                if nPSQ == 0:
                    nPspecnew = L*(ESQ-m_spec**2)/(2.0*TWOPI*E)
                else:
                    nPmag = np.sqrt(nPSQ)
                    nPspecnew = (FOURPI2*nPmag*(
                        L**2*(ESQ+m_spec**2)-FOURPI2*nPSQ
                        )+np.sqrt(ESQ*FOURPI2*L**2*(
                            L**2*(-ESQ+m_spec**2)+FOURPI2*nPSQ
                            )**2))/(2.*FOURPI2*(ESQ*L**2-FOURPI2*nPSQ))
            nPmaxintSQ = int(nPspecmax**2)
            nPnewintSQ = int(nPspecnew**2)
            tbks_sub_indices[sc_index] = nPmaxintSQ - nPnewintSQ
        return tbks_sub_indices

    def _get_tbks_sub_indices_nonzero_mom(self, E, L):
        tbks_sub_indices = [0]*len(self.tbks_list)
        for slice_index in range(self.fcs.n_three_slices):
            sc_index = self.fcs.slices_by_three_masses[slice_index][0]
            sc = self.fcs.sc_list_sorted[sc_index]
            m_spec = sc.masses_indexed[0]
            nP = self.nP
            tbkstmp_set = self.tbks_list[sc_index]
            still_searching = True
            i = 0
            while still_searching:
                try:
                    tbkstmp = tbkstmp_set[i]
                    nvec_arr = tbkstmp.nvec_arr
                    E2CMSQfull = (E-np.sqrt(m_spec**2
                                            + FOURPI2/L**2
                                            * ((nvec_arr**2).sum(axis=1))))**2\
                        - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
                    still_searching = not (np.sort(E2CMSQfull) > 0.0).all()
                    i += 1
                except IndexError:
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Crude search inside of "
                                  "get_tbks_sub_indices failed; taking last "
                                  "index before failure"
                                  f"{bcolors.ENDC}", stacklevel=2)
                    break
            i -= 1
            tbkstmp = tbkstmp_set[i]
            nvec_arr = tbkstmp.nvec_arr
            E2CMSQfull = (E-np.sqrt(m_spec**2
                                    + FOURPI2/L**2
                                    * ((nvec_arr**2).sum(axis=1))))**2\
                - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
            tbks_sub_indices[sc_index] = i
        warnings.warn(f"\n{bcolors.WARNING}"
                      f"get_tbks_sub_indices is being called with "
                      f"non_zero nP; this can lead to shells being "
                      f"missed! result is = {str(tbks_sub_indices)}"
                      f"{bcolors.ENDC}", stacklevel=2)
        return tbks_sub_indices

    def _load_ni_data_three(self, fc):
        [m1, m2, m3] = fc.masses
        Emax = self.Emax
        nP = self.nP
        Lmax = self.Lmax
        pSQ = 0.
        for m in [m1, m2, m3]:
            pSQ_tmp = ((Emax-m)**2-(nP@nP)*(TWOPI/Lmax)**2)/4.-m**2
            if pSQ_tmp > pSQ:
                pSQ = pSQ_tmp
                omp = np.sqrt(pSQ+m**2)
        beta = np.sqrt(nP@nP)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = int(p_cutoff*Lmax/TWOPI)+1
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return [m1, m2, m3, Emax, nP, Lmax, nvec_cutoff, nvecs]

    def _get_nvecset_arr_three(self, nvecset_arr, nmin, nmax,
                               m1, m2, m3, Emax, nP, Lmax, n1, n2):
        n3 = nP-n1-n2
        n1SQ = n1@n1
        n2SQ = n2@n2
        n3SQ = n3@n3
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2)), *(list(n3))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_arr = nvecset_arr+[[n1, n2, n3]]
        return [nvecset_arr, nmin, nmax]

    def _square_and_sort_three(self, nvecset_arr, nmin, nmax,
                               m1, m2, m3, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            n1 = nvecset_arr[i][0]
            n2 = nvecset_arr[i][1]
            n3 = nvecset_arr[i][2]
            n1SQ = n1@n1
            n2SQ = n2@n2
            n3SQ = n3@n3
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            n3_as_num = (n3[2]-nmin)\
                + (n3[1]-nmin)*numsys+(n3[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num,
                                                    n3_as_num]]
            nvecset_SQs = nvecset_SQs+[[n1SQ, n2SQ, n3SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_SQs = np.array(nvecset_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(4):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
        nvecset_arr = nvecset_arr[re_indexing]
        nvecset_SQs = nvecset_SQs[re_indexing]
        return [nvecset_arr, nvecset_SQs]

    def _get_nvecset_ident_three(self, nvecset_arr, nvecset_SQs):
        nvecset_ident = []
        nvecset_ident_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            [n1, n2, n3] = nvecset_arr[i]
            candidates = [np.array([n1, n2, n3]),
                          np.array([n2, n3, n1]),
                          np.array([n3, n1, n2]),
                          np.array([n3, n2, n1]),
                          np.array([n2, n1, n3]),
                          np.array([n1, n3, n2])]

            include_entry = True

            for candidate in candidates:
                for nvecset_tmp_entry in nvecset_ident:
                    nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                    include_entry = include_entry\
                        and (not ((candidate == nvecset_tmp_entry)
                                  .all()))
            if include_entry:
                nvecset_ident = nvecset_ident+[[n1, n2, n3]]
                nvecset_ident_SQs = nvecset_ident_SQs+[nvecset_SQs[i]]
        nvecset_ident = np.array(nvecset_ident)
        nvecset_ident_SQs = np.array(nvecset_ident_SQs)
        return [nvecset_ident, nvecset_ident_SQs]

    def _reps_and_batches_three(self, nvecset_arr, nvecset_SQs,
                                nvecset_ident, nvecset_ident_SQs,
                                nP):
        nvecset_reps = [nvecset_arr[0]]
        nvecset_ident_reps = deepcopy([nvecset_ident[0]])
        nvecset_SQreps = [nvecset_SQs[0]]
        nvecset_ident_SQreps = deepcopy([nvecset_ident_SQs[0]])
        nvecset_inds = [0]
        nvecset_ident_inds = deepcopy([0])
        nvecset_counts = deepcopy([0])
        nvecset_ident_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_arr)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_reps)):
                        n_included = nvecset_reps[k]
                        if (nvecset_arr[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_counts[k] = nvecset_counts[k]+1
            if not already_included:
                nvecset_reps = nvecset_reps+[nvecset_arr[j]]
                nvecset_SQreps = nvecset_SQreps+[nvecset_SQs[j]]
                nvecset_inds = nvecset_inds+[j]
                nvecset_counts = nvecset_counts+[1]

        for j in range(len(nvecset_ident)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_ident_reps)):
                        n_included = nvecset_ident_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2, n3] = nvecset_ident[j]@g_elem
                        candidates = [np.array([n1, n2, n3]),
                                      np.array([n2, n3, n1]),
                                      np.array([n3, n1, n2]),
                                      np.array([n3, n2, n1]),
                                      np.array([n2, n1, n3]),
                                      np.array([n1, n3, n2])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_ident_counts[k]\
                                = nvecset_ident_counts[k]+1
            if not already_included:
                nvecset_ident_reps = nvecset_ident_reps\
                    + [nvecset_ident[j]]
                nvecset_ident_SQreps = nvecset_ident_SQreps\
                    + [nvecset_ident_SQs[j]]
                nvecset_ident_inds = nvecset_ident_inds+[j]
                nvecset_ident_counts = nvecset_ident_counts+[1]

        nvecset_batched = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_arr)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_arr[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_batched[k], np.int64):
                        nvecset_batched[k] = [nvecset_arr[j]]
                    else:
                        nvecset_batched[k] = nvecset_batched[k]\
                            + [nvecset_arr[j]]

        nvecset_ident_batched\
            = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_ident)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_ident[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_ident_batched[k], np.int64):
                        nvecset_ident_batched[k] = [nvecset_ident[j]]
                    else:
                        nvecset_ident_batched[k]\
                            = nvecset_ident_batched[k]\
                            + [nvecset_ident[j]]

        for j in range(len(nvecset_batched)):
            nvecset_batched[j] = np.array(nvecset_batched[j])

        for j in range(len(nvecset_ident_batched)):
            nvecset_ident_batched[j]\
                = np.array(nvecset_ident_batched[j])
        return [nvecset_reps, nvecset_ident_reps,
                nvecset_SQreps, nvecset_ident_SQreps,
                nvecset_inds, nvecset_ident_inds,
                nvecset_counts, nvecset_ident_counts,
                nvecset_batched, nvecset_ident_batched]

    def _load_ni_data_two(self, fc):
        Emax = self.Emax
        nP = self.nP
        nPSQ = nP@nP
        Lmax = self.Lmax

        [m1, m2] = fc.masses
        ECMSQ = Emax**2-FOURPI2*nPSQ/Lmax**2
        pSQ = (ECMSQ**2-2.0*ECMSQ*m1**2
               + m1**4-2.0*ECMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*ECMSQ)
        mmax = np.max([m1, m2])
        omp = np.sqrt(pSQ+mmax**2)
        beta = np.sqrt(nPSQ)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = int(p_cutoff*Lmax/TWOPI)+1
        warnings.warn(f"\n{bcolors.WARNING}"
                      "nvec_cutoff was increased by one. "
                      "This needs to be checked."
                      f"{bcolors.ENDC}", stacklevel=2)
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]

    def _get_nvecset_arr_two(self, nvecset_arr, nmin, nmax, m1, m2,
                             Emax, nP, Lmax, n1):
        n2 = nP-n1
        n1SQ = n1@n1
        n2SQ = n2@n2
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_arr = nvecset_arr+[[n1, n2]]
        return [nvecset_arr, nmin, nmax]

    def _square_and_sort_two(self, nvecset_arr, nmin, nmax,
                             m1, m2, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            n1 = nvecset_arr[i][0]
            n2 = nvecset_arr[i][1]
            n1SQ = n1@n1
            n2SQ = n2@n2
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num]]
            nvecset_SQs = nvecset_SQs+[[n1SQ, n2SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_SQs = np.array(nvecset_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(3):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
        nvecset_arr = nvecset_arr[re_indexing]
        nvecset_SQs = nvecset_SQs[re_indexing]
        return [nvecset_arr, nvecset_SQs]

    def _get_nvecset_ident_two(self, nvecset_arr, nvecset_SQs):
        nvecset_ident = []
        nvecset_ident_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            [n1, n2] = nvecset_arr[i]
            candidates = [np.array([n1, n2]),
                          np.array([n2, n1])]
            include_entry = True
            for candidate in candidates:
                for nvecset_tmp_entry in nvecset_ident:
                    nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                    include_entry = include_entry\
                        and (not ((candidate == nvecset_tmp_entry)
                                  .all()))
            if include_entry:
                nvecset_ident = nvecset_ident+[[n1, n2]]
                nvecset_ident_SQs = nvecset_ident_SQs+[nvecset_SQs[i]]
        nvecset_ident = np.array(nvecset_ident)
        nvecset_ident_SQs = np.array(nvecset_ident_SQs)
        return [nvecset_ident, nvecset_ident_SQs]

    def _reps_and_batches_two(self, nvecset_arr, nvecset_SQs, nvecset_ident,
                              nvecset_ident_SQs, nP):
        nvecset_reps = [nvecset_arr[0]]
        nvecset_ident_reps = deepcopy([nvecset_ident[0]])
        nvecset_SQreps = [nvecset_SQs[0]]
        nvecset_ident_SQreps = deepcopy([nvecset_ident_SQs[0]])
        nvecset_inds = [0]
        nvecset_ident_inds = deepcopy([0])
        nvecset_counts = deepcopy([0])
        nvecset_ident_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_arr)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_reps)):
                        n_included = nvecset_reps[k]
                        if (nvecset_arr[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_counts[k] = nvecset_counts[k]+1
            if not already_included:
                nvecset_reps = nvecset_reps+[nvecset_arr[j]]
                nvecset_SQreps = nvecset_SQreps+[nvecset_SQs[j]]
                nvecset_inds = nvecset_inds+[j]
                nvecset_counts = nvecset_counts+[1]

        for j in range(len(nvecset_ident)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_ident_reps)):
                        n_included = nvecset_ident_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2] = nvecset_ident[j]@g_elem
                        candidates = [np.array([n1, n2]),
                                      np.array([n2, n1])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_ident_counts[k]\
                                = nvecset_ident_counts[k]+1
            if not already_included:
                nvecset_ident_reps = nvecset_ident_reps\
                    + [nvecset_ident[j]]
                nvecset_ident_SQreps = nvecset_ident_SQreps\
                    + [nvecset_ident_SQs[j]]
                nvecset_ident_inds = nvecset_ident_inds+[j]
                nvecset_ident_counts = nvecset_ident_counts+[1]

        nvecset_batched = list(np.arange(len(nvecset_reps)))
        for j in range(len(nvecset_arr)):
            for k in range(len(nvecset_reps)):
                include_entry = False
                n_rep = nvecset_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_arr[j]@g_elem
                    candidates = [np.array([n1, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_batched[k], np.int64):
                        nvecset_batched[k] = [nvecset_arr[j]]
                    else:
                        nvecset_batched[k] = nvecset_batched[k]\
                            + [nvecset_arr[j]]

        nvecset_ident_batched\
            = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_ident)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_ident[j]@g_elem
                    candidates = [np.array([n1, n2]),
                                  np.array([n2, n1])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_ident_batched[k], np.int64):
                        nvecset_ident_batched[k] = [nvecset_ident[j]]
                    else:
                        nvecset_ident_batched[k]\
                            = nvecset_ident_batched[k]\
                            + [nvecset_ident[j]]

        for j in range(len(nvecset_batched)):
            nvecset_batched[j] = np.array(nvecset_batched[j])

        for j in range(len(nvecset_ident_batched)):
            nvecset_ident_batched[j]\
                = np.array(nvecset_ident_batched[j])
        return [nvecset_reps, nvecset_ident_reps,
                nvecset_SQreps, nvecset_ident_SQreps,
                nvecset_inds, nvecset_ident_inds,
                nvecset_counts, nvecset_ident_counts,
                nvecset_batched, nvecset_ident_batched]

    def populate_all_nonint_data(self):
        """Populate all non-interacting data."""
        nvecset_arr_all = []
        nvecset_SQs_all = []
        nvecset_reps_all = []
        nvecset_SQreps_all = []
        nvecset_inds_all = []
        nvecset_counts_all = []
        nvecset_batched_all = []
        nvecset_ident_all = []
        nvecset_ident_SQs_all = []
        nvecset_ident_reps_all = []
        nvecset_ident_SQreps_all = []
        nvecset_ident_inds_all = []
        nvecset_ident_counts_all = []
        nvecset_ident_batched_all = []
        ni_list = self.fcs.ni_list
        for fc in ni_list:
            if fc.n_particles == 3:
                [m1, m2, m3, Emax, nP, Lmax, nvec_cutoff, nvecs]\
                    = self._load_ni_data_three(fc)
                nvecset_arr = []
                nmin = nvec_cutoff
                nmax = nvec_cutoff
                for n1 in nvecs:
                    for n2 in nvecs:
                        [nvecset_arr, nmin, nmax]\
                            = self._get_nvecset_arr_three(nvecset_arr,
                                                          nmin, nmax,
                                                          m1, m2, m3,
                                                          Emax, nP, Lmax,
                                                          n1, n2)
                nvecset_arr = np.array(nvecset_arr)
                [nvecset_arr, nvecset_SQs]\
                    = self._square_and_sort_three(nvecset_arr, nmin, nmax,
                                                  m1, m2, m3, Lmax)
                [nvecset_ident, nvecset_ident_SQs]\
                    = self._get_nvecset_ident_three(nvecset_arr, nvecset_SQs)
                [nvecset_reps, nvecset_ident_reps,
                 nvecset_SQreps, nvecset_ident_SQreps,
                 nvecset_inds, nvecset_ident_inds,
                 nvecset_counts, nvecset_ident_counts,
                 nvecset_batched, nvecset_ident_batched]\
                    = self._reps_and_batches_three(nvecset_arr, nvecset_SQs,
                                                   nvecset_ident,
                                                   nvecset_ident_SQs,
                                                   nP)
            else:
                [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]\
                    = self._load_ni_data_two(fc)
                nvecset_arr = []
                nmin = nvec_cutoff
                nmax = nvec_cutoff
                for n1 in nvecs:
                    [nvecset_arr, nmin, nmax]\
                        = self._get_nvecset_arr_two(nvecset_arr, nmin, nmax,
                                                    m1, m2, Emax, nP, Lmax, n1)
                nvecset_arr = np.array(nvecset_arr)
                [nvecset_arr, nvecset_SQs]\
                    = self._square_and_sort_two(nvecset_arr, nmin, nmax,
                                                m1, m2, Lmax)
                [nvecset_ident, nvecset_ident_SQs]\
                    = self._get_nvecset_ident_two(nvecset_arr, nvecset_SQs)
                [nvecset_reps, nvecset_ident_reps,
                 nvecset_SQreps, nvecset_ident_SQreps,
                 nvecset_inds, nvecset_ident_inds,
                 nvecset_counts, nvecset_ident_counts,
                 nvecset_batched, nvecset_ident_batched]\
                    = self._reps_and_batches_two(nvecset_arr, nvecset_SQs,
                                                 nvecset_ident,
                                                 nvecset_ident_SQs, nP)

            nvecset_arr_all = nvecset_arr_all+[nvecset_arr]
            nvecset_SQs_all = nvecset_SQs_all+[nvecset_SQs]
            nvecset_reps_all = nvecset_reps_all+[nvecset_reps]
            nvecset_SQreps_all = nvecset_SQreps_all+[nvecset_SQreps]
            nvecset_inds_all = nvecset_inds_all+[nvecset_inds]
            nvecset_counts_all = nvecset_counts_all+[nvecset_counts]
            nvecset_batched_all = nvecset_batched_all+[nvecset_batched]

            nvecset_ident_all = nvecset_ident_all\
                + [nvecset_ident]
            nvecset_ident_SQs_all = nvecset_ident_SQs_all+[nvecset_ident_SQs]
            nvecset_ident_reps_all = nvecset_ident_reps_all\
                + [nvecset_ident_reps]
            nvecset_ident_SQreps_all = nvecset_ident_SQreps_all\
                + [nvecset_ident_SQreps]
            nvecset_ident_inds_all = nvecset_ident_inds_all\
                + [nvecset_ident_inds]
            nvecset_ident_counts_all = nvecset_ident_counts_all\
                + [nvecset_ident_counts]
            nvecset_ident_batched_all = nvecset_ident_batched_all\
                + [nvecset_ident_batched]
        self.nvecset_arr = nvecset_arr_all
        self.nvecset_SQs = nvecset_SQs_all
        self.nvecset_reps = nvecset_reps_all
        self.nvecset_SQreps = nvecset_SQreps_all
        self.nvecset_inds = nvecset_inds_all
        self.nvecset_counts = nvecset_counts_all
        self.nvecset_batched = nvecset_batched_all

        self.nvecset_ident = nvecset_ident_all
        self.nvecset_ident_SQs = nvecset_ident_SQs_all
        self.nvecset_ident_reps = nvecset_ident_reps_all
        self.nvecset_ident_SQreps = nvecset_ident_SQreps_all
        self.nvecset_ident_inds = nvecset_ident_inds_all
        self.nvecset_ident_counts = nvecset_ident_counts_all
        self.nvecset_ident_batched = nvecset_ident_batched_all

    @staticmethod
    def count_by_isospin(flavor_basis):
        """Count by isospin."""
        iso_basis = CAL_C_ISO@flavor_basis

        iso_basis_normalized = []
        for entry in iso_basis:
            if entry@entry != 0.:
                entry_norm = entry/np.sqrt(entry@entry)
                iso_basis_normalized = iso_basis_normalized+[entry_norm]
            else:
                iso_basis_normalized = iso_basis_normalized+[entry]
        iso_basis_normalized = np.array(iso_basis_normalized)

        iso_basis_broken = []
        for iso_projector in ISO_PROJECTORS:
            iso_basis_broken_entry = iso_projector@iso_basis_normalized
            iso_basis_broken = iso_basis_broken+[iso_basis_broken_entry]

        iso_basis_broken_collapsed = []
        counts = [0, 0, 0, 0]
        for k in range(len(iso_basis_broken)):
            iso_basis_broken_entry = iso_basis_broken[k]
            reduced_entry = deepcopy(iso_basis_broken_entry)
            for i in range(len(reduced_entry)):
                reduced_entry_line = reduced_entry[i]
                if reduced_entry_line@reduced_entry_line != 0.:
                    for j in range(len(reduced_entry)):
                        if ((j > i) and
                           (reduced_entry_line@reduced_entry[j] != 0.)):
                            reduced_entry[j] = reduced_entry[j]\
                                - reduced_entry_line\
                                * (reduced_entry_line@reduced_entry[j])
            collapsed_entry = []
            for reduced_entry_line in reduced_entry:
                if reduced_entry_line@reduced_entry_line > EPSILON20:
                    collapsed_entry = collapsed_entry+[reduced_entry_line]
                    counts[k] = counts[k]+1
            collapsed_entry = np.array(collapsed_entry)
            iso_basis_broken_collapsed = iso_basis_broken_collapsed\
                + [collapsed_entry]
        return counts, iso_basis_broken_collapsed

    def populate_nonint_multiplicities(self):
        """Populate the non-interacting multiplicities."""
        if len(self.fcs.fc_list) == 0 or self.fcs.fc_list[0].isospin is None:
            self.nonint_multiplicities = None
            return
        nPSQ = self.nPSQ
        isospin_int = int(self.fcs.fc_list[0].isospin)
        group = self.group
        if nPSQ == 0:
            group_str = 'OhP_'
        elif nPSQ == 1:
            group_str = 'Dic4_'
        elif nPSQ == 2:
            group_str = 'Dic2_'
        else:
            raise ValueError("nPSQ not supported")
        nonint_multiplicities = []
        for cindex in range(len(self.fcs.ni_list)):
            nonint_multis_channel_dict = {}
            for key_best_irreps in self.proj_dict['best_irreps']:
                irrep = key_best_irreps[0]
                irrep_dim = group.chardict[group_str+irrep].shape[0]
                nonint_proj_dict_entry = self.nonint_proj_dict[cindex]
                if cindex == 0:
                    n_shells = len(self.nvecset_ident_SQreps[cindex])
                else:
                    n_shells = len(self.nvecset_SQreps[cindex])
                channel_multis_summary_list = []
                for shell_index in range(n_shells):
                    for key in nonint_proj_dict_entry[(shell_index,
                                                       isospin_int)]:
                        if key == key_best_irreps:
                            if cindex == 0:
                                nSQs = self.nvecset_ident_SQreps[
                                    cindex][shell_index]
                            else:
                                nSQs = self.nvecset_SQreps[
                                    cindex][shell_index]
                            multi = int(
                                nonint_proj_dict_entry[
                                    (shell_index, isospin_int)][key].shape[1]
                                / irrep_dim)
                            entry = [*nSQs, nPSQ, multi]
                            channel_multis_summary_list.append(entry)
                nonint_multis_channel_dict[key_best_irreps]\
                    = channel_multis_summary_list
            nonint_multiplicities.append(nonint_multis_channel_dict)
        self.nonint_multiplicities = nonint_multiplicities

    def _get_ibest(self, E, L):
        """Only for non-zero P."""
        Lvals = self.Lvals
        Evals = self.Evals
        i = 0
        ibest = 0
        for Ltmp in Lvals:
            for Etmp in Evals:
                if (Etmp > E) and (Ltmp > L):
                    ibest = i
                i = i+1
        if self.verbosity == 2:
            print('Lvals  =', Lvals)
            print('Evals  =', Evals)
            print('ibest =', ibest, '=', np.mod(ibest,
                                                len(Evals)),
                  '+',
                  int(ibest/len(Evals)), '* len(Evals)')
            print('so Lmaxtmp =',
                  np.round(Lvals[
                      int(ibest/len(Evals))], 10),
                  'and Emaxtmp =',
                  np.round(Evals[
                      np.mod(ibest, len(Evals))], 10))
        return ibest

    def default_k_params(self):
        """Get the default k-matrix parameters."""
        pcotdelta_parameter_list = [[]]
        for sc in self.fcs.sc_list_sorted:
            for n_params in sc.n_params_set:
                pcotdelta_parameter_list = pcotdelta_parameter_list\
                    + [[0.0]*n_params]
        pcotdelta_parameter_list = pcotdelta_parameter_list[1:]
        k3_params = [0.0]
        return [pcotdelta_parameter_list, k3_params]

    def __str__(self):
        """Return a string representation of the QCIndexSpace object."""
        qc_index_space_str = "QCIndexSpace containing:\n"
        qc_index_space_str += "    "\
            + str(self.fcs).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    "\
            + str(self.fvs).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    "\
            + str(self.tbis).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    Parameter input structure:\n"
        qc_index_space_str += "        "\
            + str(self.param_structure)+"\n\n"
        for tbkstmp in self.tbks_list:
            qc_index_space_str += "    "\
                + str(tbkstmp[0]).replace("\n", "\n    ")+"\n"
        return qc_index_space_str[:-1]
