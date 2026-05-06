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
from .groups import Groups
from .constants import TWOPI
from .constants import FOURPI2
from .constants import EPSILON4
from .constants import EPSILON10
from .constants import EPSILON20
from .constants import DELTA_L_NPNZ_GRID
from .constants import DELTA_E_NPNZ_GRID
from .constants import L_GRID_SHIFT
from .constants import E_GRID_SHIFT
from .constants import ISO_PROJECTORS
from .constants import CAL_C_ISO
from .constants import bcolors
from .flavor import FlavorChannel
from .flavor import FlavorChannelSpace
from .schemes import FiniteVolumeSetup
from .schemes import ThreeBodyInteractionScheme
import warnings
warnings.simplefilter("once")

PRINT_THRESHOLD_DEFAULT = np.get_printoptions()['threshold']


class ThreeBodyKinematicSpace:
    """
    Encode spectator-momentum kinematics for a three-body channel.

    Parameters
    ----------
    nP : numpy.ndarray of int, shape (3,), optional
        Total momentum in finite-volume units.
    nvec_arr : numpy.ndarray, optional
        Spectator momentum vectors. Each row is a three-component integer
        vector.
    build_shell_acc : bool, optional
        Whether to sort vectors into little-group shells and build shell-level
        arrays used to accelerate matrix construction.
    verbosity : int, optional
        Verbosity level for diagnostic output.

    Attributes
    ----------
    nPSQ : int
        Squared total momentum, ``nP @ nP``.
    nPmag : float
        Magnitude of the total momentum.
    shells : list of list of int
        Start and stop indices for shell blocks in the sorted ``nvec_arr``.
        Present when ``build_shell_acc`` is true and ``nvec_arr`` is nonempty.
    nvecSQ_arr, nP_minus_nvec_SQ_arr : numpy.ndarray
        Squared norms of spectator vectors and complementary pair momenta.
    n1vec_mat, n2vec_mat, n3vec_mat : numpy.ndarray
        Pairwise matrices of spectator, second-particle, and third-particle
        momentum vectors.
    n1vecSQ_mat, n2vecSQ_mat, n3vecSQ_mat : numpy.ndarray
        Squared norms corresponding to ``n1vec_mat``, ``n2vec_mat``, and
        ``n3vec_mat``.
    *_all_shells : list
        Shell-sliced versions of the vector and squared-vector arrays used by
        shell-resolved matrix builders.

    Raises
    ------
    ValueError
        If ``nP`` is not an integer ``numpy.ndarray`` with shape ``(3,)``.
    """

    def __init__(self, nP=np.array([0, 0, 0]), nvec_arr=np.array([]),
                 build_shell_acc=True, verbosity=0):
        """
        Initialize a spectator-momentum space.

        Parameters
        ----------
        nP : numpy.ndarray of int, shape (3,), optional
            Total finite-volume momentum.
        nvec_arr : numpy.ndarray, optional
            Initial spectator momentum vectors.
        build_shell_acc : bool, optional
            Whether to precompute shell-accelerated arrays.
        verbosity : int, optional
            Verbosity level.
        """
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
            nvec_arr_shifted_purged = nvec_arr_shifted[~np.isin(
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
    Represent the full quantization-condition index space.

    ``QCIndexSpace`` combines flavor channels, finite-volume symmetry data,
    three-body interaction choices, spectator momentum spaces, projection
    dictionaries, and non-interacting reference levels into one object consumed
    by the matrix builders.

    Parameters
    ----------
    fcs : FlavorChannelSpace, optional
        Flavor-channel space. If omitted, a default three-particle channel
        space is created.
    fvs : FiniteVolumeSetup, optional
        Finite-volume setup. If omitted, the default setup is used.
    tbis : ThreeBodyInteractionScheme, optional
        Three-body interaction scheme. If omitted, the default scheme is used.
    Emax : float, optional
        Maximum energy used when building spectator and non-interacting index
        spaces.
    Lmax : float, optional
        Maximum volume used when building spectator and non-interacting index
        spaces.
    deltaE_nPnz, deltaL_nPnz : float, optional
        Grid spacings used for nonzero total momentum.
    verbosity : int, optional
        Verbosity level for diagnostic output.

    Attributes
    ----------
    nP : numpy.ndarray of int, shape (3,)
        Total finite-volume momentum inherited from ``fvs``.
    nPSQ : int
        Squared total momentum.
    group : Groups
        Symmetry-group helper for the selected maximum angular momentum and
        spin sector.
    Evals, Lvals : numpy.ndarray or None
        Energy and volume grids used for nonzero total momentum.
    param_structure : list
        Nested structure describing the expected QC parameter input.
    ell_sets, ellm_sets : list
        Angular-momentum sets and expanded ``(ell, m)`` sets by spectator
        channel.
    proj_dict : dict
        Projection matrices for the full QC space.
    nonint_proj_dict : list
        Projection dictionaries for non-interacting channels.
    tbks_list : list of ThreeBodyKinematicSpace
        Spectator-momentum spaces by two-body/three-body mass slice.
    kellm_spaces, kellm_shells : list
        Spectator-momentum plus angular-momentum spaces, and their shell
        block boundaries.
    nvecset_ab_*, nvecset_aa_*, nvecset_abc_*, nvecset_aab_*,
    nvecset_aaa_* : list
        Non-interacting momentum sets, representatives, counts, and batched
        group orbits for two-particle distinguishable/identical labels and
        three-particle distinguishable/partially-identical/fully-identical
        labels respectively.

    Raises
    ------
    ValueError
        If a channel threshold exceeds ``Emax``, if unsupported particle
        counts are present, or if momentum/spin combinations are unsupported.
    """

    def __init__(self, fcs=None, fvs=None, tbis=None, Emax=5., Lmax=5.,
                 deltaE_nPnz=DELTA_E_NPNZ_GRID, deltaL_nPnz=DELTA_L_NPNZ_GRID,
                 verbosity=0):
        """
        Initialize a quantization-condition index space.

        Parameters
        ----------
        fcs : FlavorChannelSpace, optional
            Flavor-channel space to index.
        fvs : FiniteVolumeSetup, optional
            Finite-volume setup.
        tbis : ThreeBodyInteractionScheme, optional
            Three-body interaction scheme.
        Emax : float, optional
            Maximum energy for index construction.
        Lmax : float, optional
            Maximum volume for index construction.
        deltaE_nPnz, deltaL_nPnz : float, optional
            Nonzero-momentum grid spacings.
        verbosity : int, optional
            Verbosity level.
        """
        self._verbosity = verbosity
        self.verbosity = verbosity
        self.Emax = Emax
        self.Lmax = Lmax
        self.deltaE_nPnz = deltaE_nPnz
        self.deltaL_nPnz = deltaL_nPnz

        if fcs is None:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the flavor-channel space, None was passed"
                      f"{bcolors.ENDC}")
            self._fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        else:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the flavor-channel space"
                      f"{bcolors.ENDC}")
            self._fcs = fcs

        if fvs is None:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the finite-volume setup, None was passed"
                      f"{bcolors.ENDC}")
            self.fvs = FiniteVolumeSetup()
        else:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the finite-volume setup"
                      f"{bcolors.ENDC}")
            self.fvs = fvs

        if tbis is None:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the three-body interaction scheme, None was "
                      "passed"
                      f"{bcolors.ENDC}")
            self.tbis = ThreeBodyInteractionScheme()
        else:
            if verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "Setting the three-body interaction scheme"
                      f"{bcolors.ENDC}")
            self.tbis = tbis

        self._nP = self.fvs.nP
        self.nP = self.fvs.nP
        if verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  f"Setting the total momentum to nP = {self._nP}"
                  f"{bcolors.ENDC}")
        self.fcs = self._fcs

    @property
    def verbosity(self):
        """Verbosity of the QCIndexSpace."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the QCIndexSpace."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

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
            if np.sum([particle.mass for particle in sc.fc.particles])\
               > self.Emax:
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

    def populate(self):
        """
        Populate all derived index-space data.

        This builds symmetry groups, nonzero-momentum grids, parameter
        structures, spectator-momentum spaces, angular-momentum spaces,
        projection dictionaries, and non-interacting reference data.
        """
        ell_max, spin_half = self.get_ell_and_spin()
        self.group = Groups(ell_max=ell_max, spin_half=spin_half)
        self.spin_half = spin_half
        self.Evals, self.Lvals = self.get_grid_nPnonzero()
        self.param_structure = self.get_param_structure()
        self.populate_all_nvec_arr()
        self.ell_sets = self._get_ell_sets()
        self.sc_to_three_slice = self._get_sc_to_three_slice()
        if not self.spin_half:
            self.populate_all_kellm_spaces()
            self.populate_all_proj_dicts()
            self.proj_dict = self.group.get_full_proj_dict(qcis=self)
        self.populate_all_nonint_data()
        self.populate_nonint_proj_dict()
        self.populate_nonint_multiplicities()
        self.populate_nonint_functions()

    def get_ell_and_spin(self):
        """
        Determine the maximum angular momentum and spin sector.

        Returns
        -------
        ell_max : int
            Maximum angular momentum needed by interacting and
            non-interacting channels.
        spin_half : bool
            Whether a spin-half channel is present.

        Raises
        ------
        ValueError
            If a non-integer spin other than one-half is encountered.
        """
        ell_max = 4
        spin_half = False
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
                    spin_half = True
                elif np.abs(spin-spin_int) > EPSILON10:
                    raise ValueError("only integer spin and (partially spin "
                                     "half) currently supported")
            max_spin = int(np.max(nic.spins))
            if max_spin > ell_max:
                ell_max = max_spin
        return ell_max, spin_half

    def get_grid_nPnonzero(self):
        """
        Return the interpolation grid used for nonzero total momentum.

        Returns
        -------
        Evals : numpy.ndarray or None
            Energy grid, or ``None`` when ``nP`` is zero.
        Lvals : numpy.ndarray or None
            Volume grid, or ``None`` when ``nP`` is zero.
        """
        if self.nPSQ != 0:
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "nPSQ is nonzero, grid will be used"
                      f"{bcolors.ENDC}")
            [Evals, Lvals] = self._get_grid_nPnonzero(self.Emax, self.Lmax,
                                                      self.deltaE_nPnz,
                                                      self.deltaL_nPnz)
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      f"Grid for non-zero nP:\n"
                      f"Evals = {Evals}\n"
                      f"Lvals = {Lvals}")
        else:
            Evals = None
            Lvals = None
        return Evals, Lvals

    def _get_grid_nPnonzero(self, Emax, Lmax, deltaE, deltaL):
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

    def get_param_structure(self):
        """
        Build the nested parameter structure expected by QC evaluators.

        Returns
        -------
        list
            Two-entry list containing two-body p-cot-delta parameter blocks
            followed by three-body ``Kdf`` parameters.
        """
        param_structure = []
        two_param_structure = []
        for sc in self.fcs.sc_list_sorted:
            param_entry = []
            for n_params_tmp in sc.n_params_set:
                param_entry.append([0.]*n_params_tmp)
            two_param_structure.append(param_entry)
        param_structure.append(two_param_structure)
        three_param_structure = [0.]*len(self.tbis.kdf_functions)
        param_structure.append(three_param_structure)
        return param_structure

    def populate_all_nvec_arr(self):
        """
        Populate spectator-momentum arrays for every kinematic slice.

        The first slot is reserved for two-particle channels when present.
        Remaining slots correspond to three-particle mass slices.
        """
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

    def populate_nvec_arr_slot(self, slot_index, three_particle_channel=True):
        """
        Populate a single spectator-momentum slot.

        Parameters
        ----------
        slot_index : int
            Index into ``tbks_list`` to populate.
        three_particle_channel : bool, optional
            Whether the slot corresponds to a three-particle channel. If
            false, the slot is treated as a two-particle channel.

        Raises
        ------
        ValueError
            If an unsupported spin or momentum configuration is requested.
        """
        if three_particle_channel:
            if self.n_two_channels > 0:
                three_slice_index = slot_index-1
            else:
                three_slice_index = slot_index
            if self.nPSQ == 0:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                if self.verbosity >= 2:
                    print(f"{bcolors.OKGREEN}"
                          "Populating nvec array, three_slice_index = "
                          f"{three_slice_index}"
                          f"{bcolors.ENDC}")
                self._populate_slot_zero_momentum(slot_index, nPspecmax)
                if self.spin_half:
                    self._populate_spin_zero_momentum(slot_index, nPspecmax)
            else:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                self._populate_slot_nonzero_momentum(slot_index,
                                                     three_slice_index,
                                                     nPspecmax)
                if self.spin_half:
                    raise ValueError("half spin not yet supported for "
                                     "nonzero nP")
        elif not three_particle_channel and not self.spin_half:
            nPspecmax = EPSILON4
            self._populate_slot_zero_momentum(slot_index, nPspecmax)
        else:
            raise ValueError("half spin not yet supported for two-particle "
                             "channels")

    def _populate_spin_zero_momentum(self, slot_index, nPspecmax):
        warnings.warn(f"\n{bcolors.WARNING}"
                      f"Populate for spin not yet implemented"
                      f"{bcolors.ENDC}", stacklevel=2)
        pass

    def _populate_slot_zero_momentum(self, slot_index, nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "self.tbks_list[slot_index] is a list, "
                      "taking first entry"
                      f"{bcolors.ENDC}")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "self.tbks_list[slot_index] is not a list"
                      f"{bcolors.ENDC}")
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        nPspec = nPspecmax
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  "Populating nvec array, slot_index = "
                  f"{slot_index}, nPspecmax = {nPspecmax}"
                  f"{bcolors.ENDC}")
        while nPspec > 0:
            nPspec = self._populate_nP_iteration(slot_index, tbks_tmp, nPspec)
        self.tbks_list[slot_index] = self.tbks_list[slot_index][:-1]

    def _populate_slot_nonzero_momentum(self, slot_index, three_slice_index,
                                        nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "self.tbks_list[slot_index] is a list, "
                      "taking first entry"
                      f"{bcolors.ENDC}")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print(f"{bcolors.OKGREEN}"
                      "self.tbks_list[slot_index] is not a list"
                      f"{bcolors.ENDC}")
        rng = range(-int(nPspecmax), int(nPspecmax)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        Lmax = self.Lmax
        Emax = self.Emax
        deltaE = self.deltaE_nPnz
        deltaL = self.deltaL_nPnz
        sc = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]
        m_spec = sc.spectator.mass
        nP = self.fvs.nP
        [Evals, Lvals] = self._get_grid_nPnonzero(Emax, Lmax, deltaE, deltaL)
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
            print(f"{bcolors.OKGREEN}"
                  "Populating nvec array, L = "
                  f"{np.round(Ltmp, 10)}, E = {np.round(Etmp, 10)}"
                  f"{bcolors.ENDC}")
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr_tmp
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  f"Values of ThreeBodyKinematicSpace at slot {slot_index}:\n"
                  f"{self.tbks_list[slot_index][-1]}"
                  f"{bcolors.ENDC}")
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = self.tbks_list[slot_index]\
            + [tbks_copy]

    def _get_ell_sets(self):
        ell_sets = [[]]
        for sc_index in range(self.n_channels):
            ell_set = self.fcs.sc_list_sorted[sc_index].ell_set
            ell_sets = ell_sets+[ell_set]
        return ell_sets[1:]

    def _get_sc_to_three_slice(self):
        """Get the spectator channel to three-slice mapping."""
        last_loc = -1
        offset = 1
        three_channel_max =\
            self.fcs.slices_by_three_masses[last_loc][last_loc]-offset
        no_two_channels = self.n_two_channels == 0

        sc_to_three_slice = []
        for sc_index in range(self.n_channels):
            sc_too_high = sc_index > three_channel_max
            if no_two_channels and sc_too_high:
                raise ValueError(f"using sc_index = {sc_index} with "
                                 f"three_slices = "
                                 f"{self.fcs.slices_by_three_masses} "
                                 f"and (no two-particle channels) "
                                 f"is not allowed")
            if sc_index < self.n_two_channels:
                three_slice_index = 0
            else:
                sc_index_shift = sc_index-self.n_two_channels
                three_slice_index = -1
                for k in range(len(self.fcs.slices_by_three_masses)):
                    three_slice = self.fcs.slices_by_three_masses[k]
                    if three_slice[0] <= sc_index_shift < three_slice[1]:
                        three_slice_index = k
                if self.n_two_channels > 0:
                    three_slice_index = three_slice_index+1
            sc_to_three_slice.append(three_slice_index)
        return sc_to_three_slice

    def populate_all_kellm_spaces(self):
        """
        Populate spectator-momentum plus angular-momentum spaces.

        Creates ``kellm_spaces`` and shell boundaries ``kellm_shells`` for each
        spectator channel and each precomputed kinematic space.
        """
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  f"Populating kellm spaces\n"
                  f"{self.n_channels} channels to populate"
                  f"{bcolors.ENDC}")
        kellm_shells = []
        kellm_spaces = []
        for sc_index in range(self.n_channels):
            three_slice_index = self.sc_to_three_slice[sc_index]
            tbks_fixed_masses_list = self.tbks_list[three_slice_index]
            ellm_set = self.ellm_sets[sc_index]
            kellm_shells_entry = []
            kellm_spaces_entry = []
            for tbks_entry in tbks_fixed_masses_list:
                nvec_arr = tbks_entry.nvec_arr
                kellm_shell = (len(ellm_set)*np.array(tbks_entry.shells
                                                      )).tolist()
                kellm_shells_entry.append(kellm_shell)
                ellm_set_extended = np.tile(ellm_set, (len(nvec_arr), 1))
                nvec_arr_extended = np.repeat(nvec_arr, len(ellm_set),
                                              axis=0)
                kellm_space = np.concatenate((nvec_arr_extended,
                                              ellm_set_extended),
                                             axis=1)
                kellm_spaces_entry.append(kellm_space)
            kellm_shells.append(kellm_shells_entry)
            kellm_spaces.append(kellm_spaces_entry)
        self.kellm_spaces = kellm_spaces
        self.kellm_shells = kellm_shells
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  "Result for kellm spaces:\n"
                  "location: channel index, nvec-space index"
                  f"{bcolors.ENDC}")
            np.set_printoptions(threshold=20)
            for i in range(len(self.kellm_spaces)):
                for j in range(len(self.kellm_spaces[i])):
                    print(f"{bcolors.OKGREEN}")
                    print('location:', i, j)
                    print(self.kellm_spaces[i][j])
                    print(f"{bcolors.ENDC}")
            np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)

    def populate_all_proj_dicts(self):
        """
        Populate interacting-channel projection dictionaries.

        The resulting dictionaries are organized by spectator channel and, for
        shell-resolved projections, by kinematic shell set.
        """
        group = self.group
        proj_dicts_by_sc = []
        proj_dicts_by_sc_and_shellset = []
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}\n"
                  f"Getting the dict for following qcis:")
            print(self)
            print(f"{self}{bcolors.ENDC}")
        for sc_index in range(self.n_channels):
            fixed_sc_proj_dict = group.get_fixed_sc_proj_dict(
                qcis=self, sc_index=sc_index)
            proj_dicts_by_sc.append(fixed_sc_proj_dict)
            fixed_sc_proj_dicts_by_shellset = []
            for kellm_shell_index in range(len(self.kellm_shells[sc_index])):
                kellm_shell_set = self.kellm_shells[sc_index][
                    kellm_shell_index]
                fixed_sc_and_shellset_proj_dict = []
                for kellm_shell in kellm_shell_set:
                    fixed_sc_and_shellset_proj_dict.append(
                        group.get_fixed_sc_and_shell_proj_dict(
                            qcis=self, sc_index=sc_index,
                            kellm_shell=kellm_shell,
                            kellm_shell_index=kellm_shell_index))
                fixed_sc_proj_dicts_by_shellset.append(
                    fixed_sc_and_shellset_proj_dict)
            proj_dicts_by_sc_and_shellset.append(
                fixed_sc_proj_dicts_by_shellset)
        self.proj_dicts_by_sc = proj_dicts_by_sc
        self.proj_dicts_by_sc_and_shellset = proj_dicts_by_sc_and_shellset

    def populate_all_nonint_data(self):
        """
        Populate non-interacting momentum sets and group-orbit data.

        Builds two-particle ``ab``/``aa`` and three-particle
        ``abc``/``aab``/``aaa`` momentum sets for every
        non-interacting channel, along with representatives, counts, indices,
        and batched orbit data.
        """
        nonint_channel_data = [
            self._get_nonint_channel_data(fc)
            for fc in self.fcs.ni_list
        ]
        self._populate_ab_nonint_data(nonint_channel_data)
        self._populate_aa_nonint_data(nonint_channel_data)
        self._populate_abc_nonint_data(nonint_channel_data)
        self._populate_aab_nonint_data(nonint_channel_data)
        self._populate_aaa_nonint_data(nonint_channel_data)

    def _populate_ab_nonint_data(self, nonint_channel_data):
        """Populate distinguishable two-particle non-interacting data."""
        self._populate_nonint_data_fields(
            nonint_channel_data,
            ['nvecset_ab', 'nvecset_ab_SQs', 'nvecset_ab_reps',
             'nvecset_ab_SQreps', 'nvecset_ab_inds', 'nvecset_ab_counts',
             'nvecset_ab_batched'])

    def _populate_aa_nonint_data(self, nonint_channel_data):
        """Populate identical two-particle non-interacting data."""
        self._populate_nonint_data_fields(
            nonint_channel_data,
            ['nvecset_aa', 'nvecset_aa_SQs', 'nvecset_aa_reps',
             'nvecset_aa_SQreps', 'nvecset_aa_inds', 'nvecset_aa_counts',
             'nvecset_aa_batched'])

    def _populate_abc_nonint_data(self, nonint_channel_data):
        """Populate fully distinguishable three-particle data."""
        self._populate_nonint_data_fields(
            nonint_channel_data,
            ['nvecset_abc', 'nvecset_abc_SQs', 'nvecset_abc_reps',
             'nvecset_abc_SQreps', 'nvecset_abc_inds', 'nvecset_abc_counts',
             'nvecset_abc_batched'])

    def _populate_aab_nonint_data(self, nonint_channel_data):
        """Populate data with particles 0 and 1 treated as identical."""
        self._populate_nonint_data_fields(
            nonint_channel_data,
            ['nvecset_aab', 'nvecset_aab_SQs',
             'nvecset_aab_reps', 'nvecset_aab_SQreps',
             'nvecset_aab_inds', 'nvecset_aab_counts',
             'nvecset_aab_batched'])

    def _populate_aaa_nonint_data(self, nonint_channel_data):
        """Populate fully identical non-interacting data."""
        self._populate_nonint_data_fields(
            nonint_channel_data,
            ['nvecset_aaa', 'nvecset_aaa_SQs',
             'nvecset_aaa_reps', 'nvecset_aaa_SQreps',
             'nvecset_aaa_inds', 'nvecset_aaa_counts',
             'nvecset_aaa_batched'])

    def _populate_nonint_data_fields(self, nonint_channel_data, field_names):
        for field_name in field_names:
            setattr(self, field_name, [
                channel_data.get(field_name)
                for channel_data in nonint_channel_data
            ])

    def _nonint_channel_particle_label(self, cindex):
        fc = self.fcs.ni_list[cindex]
        flavors = fc.flavors
        if fc.n_particles == 2:
            if flavors[0] == flavors[1]:
                return 'aa'
            return 'ab'
        if fc.n_particles != 3:
            raise ValueError("only two- and three-particle non-interacting "
                             "channels are supported")
        flavors = self.fcs.ni_list[cindex].flavors
        if len(set(flavors)) == 1:
            return 'aaa'
        if len(set(flavors)) == 2:
            return 'aab'
        return 'abc'

    def _get_nonint_channel_data(self, fc):
        if fc.n_particles == 3:
            return self._get_nonint_channel_data_three(fc)
        if fc.n_particles == 2:
            return self._get_nonint_channel_data_two(fc)
        raise ValueError("only two- and three-particle non-interacting "
                         "channels are supported")

    def _get_nonint_channel_data_three(self, fc):
        nvecset_abc, nvecset_abc_SQs, nP = self._get_nonint_nvecsets_three(fc)
        nvecset_aab, nvecset_aab_SQs = self._get_nvecset_aab_three(
            nvecset_abc, nvecset_abc_SQs)
        [nvecset_aab_reps, nvecset_aab_SQreps,
         nvecset_aab_inds, nvecset_aab_counts,
         nvecset_aab_batched] = self._reps_and_batches_permutations(
             nvecset_aab, nvecset_aab_SQs, nP,
             self._aab_three_permutations())

        nvecset_aaa, nvecset_aaa_SQs = self._get_nvecset_aaa_three(
            nvecset_abc, nvecset_abc_SQs)
        [nvecset_abc_reps, nvecset_aaa_reps,
         nvecset_abc_SQreps, nvecset_aaa_SQreps,
         nvecset_abc_inds, nvecset_aaa_inds,
         nvecset_abc_counts, nvecset_aaa_counts,
         nvecset_abc_batched, nvecset_aaa_batched] =\
            self._reps_and_batches_three(
                nvecset_abc, nvecset_abc_SQs, nvecset_aaa,
                nvecset_aaa_SQs, nP)

        return {
            'nvecset_abc': nvecset_abc,
            'nvecset_abc_SQs': nvecset_abc_SQs,
            'nvecset_abc_reps': nvecset_abc_reps,
            'nvecset_abc_SQreps': nvecset_abc_SQreps,
            'nvecset_abc_inds': nvecset_abc_inds,
            'nvecset_abc_counts': nvecset_abc_counts,
            'nvecset_abc_batched': nvecset_abc_batched,
            'nvecset_aab': nvecset_aab,
            'nvecset_aab_SQs': nvecset_aab_SQs,
            'nvecset_aab_reps': nvecset_aab_reps,
            'nvecset_aab_SQreps': nvecset_aab_SQreps,
            'nvecset_aab_inds': nvecset_aab_inds,
            'nvecset_aab_counts': nvecset_aab_counts,
            'nvecset_aab_batched': nvecset_aab_batched,
            'nvecset_aaa': nvecset_aaa,
            'nvecset_aaa_SQs': nvecset_aaa_SQs,
            'nvecset_aaa_reps': nvecset_aaa_reps,
            'nvecset_aaa_SQreps': nvecset_aaa_SQreps,
            'nvecset_aaa_inds': nvecset_aaa_inds,
            'nvecset_aaa_counts': nvecset_aaa_counts,
            'nvecset_aaa_batched': nvecset_aaa_batched,
        }

    def _get_nonint_channel_data_two(self, fc):
        nvecset_ab, nvecset_ab_SQs, nP = self._get_nonint_nvecsets_two(fc)
        nvecset_aa, nvecset_aa_SQs = self._get_nvecset_aa_two(
            nvecset_ab, nvecset_ab_SQs)
        [nvecset_ab_reps, nvecset_aa_reps,
         nvecset_ab_SQreps, nvecset_aa_SQreps,
         nvecset_ab_inds, nvecset_aa_inds,
         nvecset_ab_counts, nvecset_aa_counts,
         nvecset_ab_batched, nvecset_aa_batched] =\
            self._reps_and_batches_two(
                nvecset_ab, nvecset_ab_SQs, nvecset_aa,
                nvecset_aa_SQs, nP)

        return {
            'nvecset_ab': nvecset_ab,
            'nvecset_ab_SQs': nvecset_ab_SQs,
            'nvecset_ab_reps': nvecset_ab_reps,
            'nvecset_ab_SQreps': nvecset_ab_SQreps,
            'nvecset_ab_inds': nvecset_ab_inds,
            'nvecset_ab_counts': nvecset_ab_counts,
            'nvecset_ab_batched': nvecset_ab_batched,
            'nvecset_aa': nvecset_aa,
            'nvecset_aa_SQs': nvecset_aa_SQs,
            'nvecset_aa_reps': nvecset_aa_reps,
            'nvecset_aa_SQreps': nvecset_aa_SQreps,
            'nvecset_aa_inds': nvecset_aa_inds,
            'nvecset_aa_counts': nvecset_aa_counts,
            'nvecset_aa_batched': nvecset_aa_batched,
        }

    def _get_nonint_nvecsets_three(self, fc):
        [m1, m2, m3, Emax, nP, Lmax, nvec_cutoff, nvecs]\
            = self._load_ni_data_three(fc)
        nvecset_abc = []
        nmin = nvec_cutoff
        nmax = nvec_cutoff
        for n1 in nvecs:
            for n2 in nvecs:
                [nvecset_abc, nmin, nmax]\
                    = self._get_nvecset_abc_three(nvecset_abc, nmin, nmax,
                                                  m1, m2, m3, Emax, nP,
                                                  Lmax, n1, n2)
        nvecset_abc = np.array(nvecset_abc)
        [nvecset_abc, nvecset_abc_SQs] = self._square_and_sort_three(
            nvecset_abc, nmin, nmax, m1, m2, m3, Lmax)
        return nvecset_abc, nvecset_abc_SQs, nP

    def _get_nonint_nvecsets_two(self, fc):
        [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]\
            = self._load_ni_data_two(fc)
        nvecset_ab = []
        nmin = nvec_cutoff
        nmax = nvec_cutoff
        for n1 in nvecs:
            [nvecset_ab, nmin, nmax]\
                = self._get_nvecset_ab_two(nvecset_ab, nmin, nmax,
                                            m1, m2, Emax, nP, Lmax, n1)
        nvecset_ab = np.array(nvecset_ab)
        [nvecset_ab, nvecset_ab_SQs] = self._square_and_sort_two(
            nvecset_ab, nmin, nmax, m1, m2, Lmax)
        return nvecset_ab, nvecset_ab_SQs, nP

    def populate_nonint_proj_dict(self):
        """
        Populate projection dictionaries for non-interacting levels.

        Raises
        ------
        ValueError
            If a non-interacting channel has an unsupported particle count or
            spin combination.
        """
        nonint_proj_dict = []
        for nic_index in range(len(self.fcs.ni_list)):
            n_particles = self.fcs.ni_list[nic_index].n_particles
            two_particles = (n_particles == 2)
            three_particles = (n_particles == 3)
            first_spin = self.fcs.ni_list[nic_index].spins[0]
            if two_particles:
                isospin_channel = self.fcs.ni_list[nic_index].isospin_channel
                nonint_proj_dict\
                    .append(self.group.get_proj_nonint_two_particles_dict(
                        qcis=self, nic_index=nic_index,
                        isospin_channel=isospin_channel))
            elif three_particles and first_spin == 0.:
                nonint_proj_dict.append(
                    self.group.get_proj_nonint_three_pions_dict(
                        qcis=self, nic_index=nic_index))
            elif three_particles and first_spin == 1.:
                nonint_proj_dict.append(
                    self.group.
                    get_proj_nonint_three_spinning_dict(
                        qcis=self, nic_index=nic_index))
            elif three_particles and self.spin_half:
                nonint_proj_dict.append(
                    self.group.get_proj_nonint_three_spinning_dict(
                        qcis=self, nic_index=nic_index))
            else:
                raise ValueError("only two and three particles with certain "
                                 "spin combinations are supported by "
                                 "nonint_proj_dict")
        self.nonint_proj_dict = nonint_proj_dict

    def populate_nonint_multiplicities(self):
        """
        Populate non-interacting irrep multiplicity summaries.

        The generated entries collect shell quantum numbers, total momentum
        squared, and multiplicities for each best-irrep key.
        """
        if len(self.fcs.fc_list) == 0:
            self.nonint_multiplicities = None
            return
        if (len(self.fcs.fc_list) == 1
           and not self.fcs.fc_list[0].isospin_channel):
            isospin_int = 0
        elif self.fcs.fc_list[0].isospin_channel:
            isospin_int = int(self.fcs.fc_list[0].isospin)
        nPSQ = self.nPSQ
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
                particle_label = self._nonint_channel_particle_label(cindex)
                nvecset_label_SQreps = getattr(
                    self, f'nvecset_{particle_label}_SQreps')
                n_shells = len(nvecset_label_SQreps[cindex])
                channel_multis_summary_list = []
                for shell_index in range(n_shells):
                    for key in nonint_proj_dict_entry[(shell_index,
                                                       isospin_int)]:
                        if key == key_best_irreps:
                            nSQs = nvecset_label_SQreps[cindex][shell_index]
                            multi = int(
                                nonint_proj_dict_entry[
                                    (shell_index, isospin_int)][key].shape[1]
                                / irrep_dim)
                            entry = [*nSQs, nPSQ, multi]
                            channel_multis_summary_list.append(entry)
                            # if cindex == 1:
                            #     entry = [nSQs[1], nSQs[0], nPSQ, multi]
                            #     channel_multis_summary_list.append(entry)
                            #     warnings.warn(f"\n{bcolors.WARNING}"
                            #                   "Assuming a non-degenerate "
                            #                   "two-particle channel."
                            #                   f"{bcolors.ENDC}",
                            #                   stacklevel=2)
                nonint_multis_channel_dict[key_best_irreps]\
                    = channel_multis_summary_list
            nonint_multiplicities.append(nonint_multis_channel_dict)
        self.nonint_multiplicities = nonint_multiplicities

    def populate_nonint_functions(self):
        """
        Build non-interacting energy functions by channel and irrep.

        The resulting ``nonint_functions`` entries are callables of ``L`` that
        return the corresponding non-interacting energy level.

        Raises
        ------
        ValueError
            If a multiplicity entry has an unsupported shape.
        """
        nonint_functions = []
        for nonint_channel_mult_dict in self.nonint_multiplicities:
            nonint_channel_functions_dict = {}
            for key in nonint_channel_mult_dict:
                nonint_channel_functions_dict[key] = []
                for nonint_channel_mult in nonint_channel_mult_dict[key]:
                    if len(nonint_channel_mult) == 5:
                        nonint_function = self._get_nonint_function_three(
                            nonint_channel_mult)
                    elif len(nonint_channel_mult) == 4:
                        nonint_function = self._get_nonint_function_two(
                            nonint_channel_mult)
                    else:
                        raise ValueError("nonint_function not supported")
                    nonint_channel_functions_dict[key].append(nonint_function)
            nonint_functions.append(nonint_channel_functions_dict)
        self.nonint_functions = nonint_functions

    def _get_nonint_function_three(self, nonint_channel_mult):
        nSQ1, nSQ2, nSQ3, _, _ = nonint_channel_mult
        mSQ1 = 1.
        mSQ2 = 1.
        mSQ3 = 1.

        def nonint_function(L):
            """
            Evaluate the three-particle non-interacting energy.

            Parameters
            ----------
            L : float
                Finite-volume length.

            Returns
            -------
            float
                Sum of the three single-particle finite-volume energies.
            """
            omega1 = np.sqrt(mSQ1+FOURPI2*nSQ1/L**2)
            omega2 = np.sqrt(mSQ2+FOURPI2*nSQ2/L**2)
            omega3 = np.sqrt(mSQ3+FOURPI2*nSQ3/L**2)
            return omega1+omega2+omega3
        return nonint_function

    def _get_nonint_function_two(self, nonint_channel_mult):
        nSQ1, nSQ2, _, _ = nonint_channel_mult
        mSQ1 = 2.2**2
        mSQ2 = 1.

        def nonint_function(L):
            """
            Evaluate the two-particle non-interacting energy.

            Parameters
            ----------
            L : float
                Finite-volume length.

            Returns
            -------
            float
                Sum of the two single-particle finite-volume energies.
            """
            omega1 = np.sqrt(mSQ1+FOURPI2*nSQ1/L**2)
            omega2 = np.sqrt(mSQ2+FOURPI2*nSQ2/L**2)
            return omega1+omega2
        return nonint_function

    def _get_ESQmin(self, three_slice_index):
        sc_index = self.fcs.slices_by_three_masses[three_slice_index][0]
        return self.tbis.ESQmins[sc_index]

    def _get_nPspecmax(self, three_slice_index):
        sc = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]
        m_spec = sc.spectator.mass
        Emax = self.Emax
        EmaxSQ = Emax**2
        nPSQ = self.nPSQ
        Lmax = self.Lmax
        ESQmin = self._get_ESQmin(three_slice_index)
        if (ESQmin != 0.0):
            if nPSQ == 0:
                nPspecmax = (Lmax*np.sqrt(
                    Emax**4+(ESQmin-m_spec**2)**2-2.*Emax**2*(ESQmin+m_spec**2)
                    ))/(2.*Emax*TWOPI)
                return nPspecmax
            else:
                raise ValueError("simultaneous nonzero nP and ESQmin not"
                                 " supported")
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

    def _populate_nP_iteration(self, slot_index, tbks_tmp, nPspec):
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  "Populating nvec array, nPspec**2 = "
                  f"{int(nPspec**2)}"
                  f"{bcolors.ENDC}")
        rng = range(-int(nPspec), int(nPspec)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        carr = (nvec_arr*nvec_arr).sum(1) > nPspec**2
        nvec_arr = np.delete(nvec_arr, np.where(carr), axis=0)
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}"
                  f"Values of ThreeBodyKinematicSpace at slot {slot_index}:\n"
                  f"{self.tbks_list[slot_index][-1]}"
                  f"{bcolors.ENDC}")
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

    def get_tbks_sub_indices(self, E, L):
        """
        Get kinematic-space indices relevant at a given energy and volume.

        Parameters
        ----------
        E : float
            Energy at which the QC matrix will be evaluated.
        L : float
            Volume at which the QC matrix will be evaluated.

        Returns
        -------
        list of int
            Indices into each ``tbks_list`` entry selecting the precomputed
            kinematic space appropriate for ``E`` and ``L``.

        Raises
        ------
        ValueError
            If ``E`` exceeds ``Emax`` or ``L`` exceeds ``Lmax``.
        """
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
            nPspecmax = self._get_nPspecmax(slice_index)
            sc = self.fcs.sc_list_sorted[sc_index]
            m_spec = sc.fc.masses[sc.indexing[0]]
            ESQ = E**2
            nPSQ = self.nPSQ
            ESQmin = self._get_ESQmin(slice_index)
            if (ESQmin != 0.0):
                if nPSQ == 0:
                    nPspecnew = (L*np.sqrt(
                        E**4+(ESQmin-m_spec**2)**2-2.*E**2*(ESQmin+m_spec**2)
                        ))/(2.*E*TWOPI)
                else:
                    raise ValueError("nonzero nP and Emin not supported")
            else:
                if nPSQ == 0:
                    if E == 0.0:
                        nPspecnew = 0.0
                        warnings.warn(f"\n{bcolors.WARNING}"
                                      "E = 0.0 in get_tbks_sub_indices; "
                                      "setting nspecnew to 0.0"
                                      f"{bcolors.ENDC}", stacklevel=2)
                    else:
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
            m_spec = sc.spectator.mass
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
        PSQ = (nP@nP)*FOURPI2/Lmax**2
        ECMSQ_max = Emax**2-PSQ

        pSQ = 0.
        m_pairs = [[m1, m2], [m1, m3], [m2, m3]]
        for m_pair in m_pairs:
            ma, mb = m_pair
            if ma == mb:
                pSQ_tmp = ECMSQ_max/4.-ma**2
                m_max = ma
            else:
                pSQ_tmp = (ECMSQ_max**2
                           + (ma**2 - mb**2)**2
                           - 2*ECMSQ_max*(ma**2 + mb**2))/(4.*ECMSQ_max)
                m_max = max(ma, mb)
            if pSQ_tmp > pSQ:
                pSQ = pSQ_tmp
                omp = np.sqrt(pSQ+m_max**2)

        beta = np.sqrt(nP@nP)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = p_cutoff*Lmax/TWOPI
        nvec_int_cutoff = int(nvec_cutoff)+1
        rng = range(-nvec_int_cutoff, nvec_int_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        carr = (nvecs*nvecs).sum(1) > nvec_cutoff**2
        nvecs = np.delete(nvecs, np.where(carr), axis=0)
        return [m1, m2, m3, Emax, nP, Lmax, nvec_int_cutoff, nvecs]

    def _get_nvecset_abc_three(self, nvecset_abc, nmin, nmax,
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
            nvecset_abc = nvecset_abc+[[n1, n2, n3]]
        return [nvecset_abc, nmin, nmax]

    def _square_and_sort_three(self, nvecset_abc, nmin, nmax,
                               m1, m2, m3, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_abc_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            n1 = nvecset_abc[i][0]
            n2 = nvecset_abc[i][1]
            n3 = nvecset_abc[i][2]
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
            nvecset_abc_SQs = nvecset_abc_SQs+[[n1SQ, n2SQ, n3SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_abc_SQs = np.array(nvecset_abc_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(4):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
        nvecset_abc = nvecset_abc[re_indexing]
        nvecset_abc_SQs = nvecset_abc_SQs[re_indexing]
        return [nvecset_abc, nvecset_abc_SQs]

    def _get_nvecset_aaa_three(self, nvecset_abc, nvecset_abc_SQs):
        nvecset_aaa = []
        nvecset_aaa_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            [n1, n2, n3] = nvecset_abc[i]
            candidates = self._permuted_three_candidates(
                n1, n2, n3, self._aaa_three_permutations())
            include_entry = self._include_symmetrized_entry(
                candidates, nvecset_aaa)
            if include_entry:
                nvecset_aaa = nvecset_aaa+[[n1, n2, n3]]
                nvecset_aaa_SQs = nvecset_aaa_SQs+[nvecset_abc_SQs[i]]
        nvecset_aaa = np.array(nvecset_aaa)
        nvecset_aaa_SQs = np.array(nvecset_aaa_SQs)
        return [nvecset_aaa, nvecset_aaa_SQs]

    def _get_nvecset_aab_three(self, nvecset_abc, nvecset_abc_SQs):
        nvecset_aab = []
        nvecset_aab_SQs = deepcopy([])
        for i in range(len(nvecset_abc)):
            [n1, n2, n3] = nvecset_abc[i]
            candidates = self._permuted_three_candidates(
                n1, n2, n3, self._aab_three_permutations())
            include_entry = self._include_symmetrized_entry(
                candidates, nvecset_aab)
            if include_entry:
                nvecset_aab = nvecset_aab+[[n1, n2, n3]]
                nvecset_aab_SQs = nvecset_aab_SQs+[nvecset_abc_SQs[i]]
        nvecset_aab = np.array(nvecset_aab)
        nvecset_aab_SQs = np.array(nvecset_aab_SQs)
        return [nvecset_aab, nvecset_aab_SQs]

    @staticmethod
    def _aaa_three_permutations():
        return [(0, 1, 2), (1, 2, 0), (2, 0, 1),
                (2, 1, 0), (1, 0, 2), (0, 2, 1)]

    @staticmethod
    def _aab_three_permutations():
        return [(0, 1, 2), (1, 0, 2)]

    @staticmethod
    def _aaa_two_permutations():
        return [(0, 1), (1, 0)]

    @staticmethod
    def _permuted_three_candidates(n1, n2, n3, permutations):
        nvecs = [n1, n2, n3]
        return [np.array([nvecs[index] for index in permutation])
                for permutation in permutations]

    @staticmethod
    def _permuted_two_candidates(n1, n2, permutations):
        nvecs = [n1, n2]
        return [np.array([nvecs[index] for index in permutation])
                for permutation in permutations]

    @staticmethod
    def _include_symmetrized_entry(candidates, nvecset):
        include_entry = True
        for candidate in candidates:
            for nvecset_tmp_entry in nvecset:
                nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                include_entry = include_entry\
                    and (not ((candidate == nvecset_tmp_entry).all()))
        return include_entry
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

    @staticmethod
    def count_by_isospin(flavor_basis):
        """
        Count independent flavor-basis vectors by isospin projector.

        Parameters
        ----------
        flavor_basis : numpy.ndarray
            Flavor basis vectors to decompose into isospin sectors.

        Returns
        -------
        counts : list of int
            Number of independent vectors in each isospin sector.
        iso_basis_broken_collapsed : list of numpy.ndarray
            Reduced basis vectors after applying each isospin projector.
        """
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
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}")
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
            print(f"{bcolors.ENDC}")
        return ibest

    def default_k_params(self):
        """
        Get default two- and three-body K-matrix parameters.

        Returns
        -------
        list
            ``[pcotdelta_parameter_list, k3_params]`` with zeros matching the
            current channel parameter structure.
        """
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
