#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# k_matrices.py
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
from . import shell_utils
from . import check_utils
from .constants import TWOPI
from .constants import FOURPI2
from . import qc_functions
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

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell=0,
                  pcotdelta_function=None, pcotdelta_parameter_list=None,
                  tbks_entry=None, slice_index=None,
                  project=False, irrep=None):
        """Build the K matrix on a single shell."""
        nP = self.qcis.fvs.nP
        three_scheme = self.qcis.tbis.three_scheme
        qc_impl = self.qcis.fvs.qc_impl
        alpha, beta = self.qcis.tbis.scheme_data[sc_ind]
        use_pv_shift_prescription\
            = self.qcis.tbis.use_pv_shift_prescription[sc_ind]
        if use_pv_shift_prescription:
            pv_shift_parameters = self.qcis.tbis.pv_shift_parameters[sc_ind]
        else:
            pv_shift_parameters = None
        sc = self.qcis.fcs.sc_list_sorted[sc_ind]
        dimer_symmetry_factor = 1.0
        if sc.first_dimer != sc.second_dimer:
            dimer_symmetry_factor = 2.0

        mask_slices, slice_entry\
            = shell_utils._get_masks_and_shells_for_k(
                self, E, L, tbks_entry, cindex, slice_index)
        Kshell = qc_functions.getK_array(
            E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell,
            pcotdelta_function, pcotdelta_parameter_list, alpha, beta,
            qc_impl, three_scheme,
            use_pv_shift_prescription=use_pv_shift_prescription,
            pv_shift_parameters=pv_shift_parameters,
            dimer_symmetry_factor=dimer_symmetry_factor)

        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis.get_shellset_index(E, L)
                    proj_tmp_right = np.array(
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_ind][ibest])[mask_slices][slice_index][irrep]
                else:
                    proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_ind][slice_index][irrep]
                proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                shell_utils._verify_irrep_is_known(self.qcis, irrep)
                return np.array([])
        if project:
            Kshell = proj_tmp_left@Kshell@proj_tmp_right
        return Kshell

    def get_value(self, E=5.0, L=5.0, pcotdelta_parameter_lists=None,
                  project=False, irrep=None):
        """Build the K matrix in a shell-based way."""
        check_utils.check_value_within_qcis_bounds(self, E, L)
        nP = self.qcis.fvs.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)
        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
        else:
            if self.qcis.fcs.n_three_slices != 1:
                raise NotImplementedError(
                    "multi-slice K is implemented only for zero total "
                    "momentum")
            ibest = self.qcis.get_shellset_index(E, L)
            sc = self.qcis.fcs.sc_list_sorted[0]
            mspec = sc.spectator.mass
            m2 = sc.first_dimer.mass
            m3 = sc.second_dimer.mass
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            threshold = m2+m3
            zero_support_point = shell_utils._get_zero_support_point(
                self, threshold)
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
            sc = self.qcis.fcs.sc_list_sorted[sc_ind]
            # the pair sector lives in Ktwo; K is three-particle only
            if sc.fc.n_particles == 2:
                continue
            ell_set = sc.ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in K")
            ell = ell_set[0]
            pcotdelta_parameter_list = pcotdelta_parameter_lists[sc_ind]
            pcotdelta_function = sc.p_cot_deltas[0]
            if nP@nP == 0:
                three_slice_index = self.qcis.sc_to_three_slice[sc_ind]
                tbks_entry = self.qcis.tbks_list[three_slice_index][
                    tbks_sub_indices[three_slice_index]]
                slices = tbks_entry.shells
                mspec = sc.spectator.mass
                m2 = sc.first_dimer.mass
                m3 = sc.second_dimer.mass
            cindex = sc_ind
            for slice_index in range(len(slices)):
                k_tmp = self.get_shell(
                    E, L, mspec, m2, m3, cindex, sc_ind, ell,
                    pcotdelta_function, pcotdelta_parameter_list, tbks_entry,
                    slice_index, project, irrep)
                if len(k_tmp) != 0:
                    k_final_list = k_final_list+[k_tmp]
        if len(k_final_list) == 0:
            return np.zeros((0, 0))
        return block_diag(*k_final_list)


def _get_two_particle_k_shell(qcis, E, L, sc, sc_ind,
                              pcotdelta_parameter_lists, project, irrep):
    """Build the 1x1 K block for one two-particle channel."""
    nP = qcis.fvs.nP
    if nP@nP != 0:
        raise NotImplementedError(
            "two-particle channels are implemented only for zero total "
            "momentum")
    if list(sc.ell_set) != [0]:
        raise NotImplementedError(
            "two-particle channels currently support only "
            "ell_set == [0]")
    if shell_utils.two_particle_block_dim(project, irrep) == 0:
        return np.array([])
    m1, m2 = sc.fc.masses
    alpha, beta = qcis.tbis.scheme_data[sc_ind]
    # mspec = 0 reduces the spectator kinematics to the genuine
    # two-particle system; the hermitian 2*omega_spec normalization
    # belongs to the three-particle sector and must not multiply the
    # pair block
    qc_impl = dict(qcis.fvs.qc_impl)
    qc_impl['hermitian'] = False
    Ktwo_entry = qc_functions.getK_single_entry(
        pcotdelta_function=sc.p_cot_deltas[0],
        pcotdelta_parameter_list=pcotdelta_parameter_lists[sc_ind],
        E=E, nP=nP, npspec=np.array([0, 0, 0]), L=L,
        m1=m1, m2=m2, mspec=0.0, alpha=alpha, beta=beta,
        ell=0, qc_impl=qc_impl)
    return np.array([[Ktwo_entry]])


class Ktwo:
    """Two-particle sector of the two-body K matrix.

    Block diagonal over the two-particle channels of the space, in
    ``sc_list_sorted`` order, matching the layout of :class:`Ftwo`.

    :param qcis: quantization-condition index space, specifying all data
        for the class
    :type qcis: QCIndexSpace
    """

    def __init__(self, qcis=None):
        self.qcis = qcis

    def get_value(self, E=5.0, L=5.0, pcotdelta_parameter_lists=None,
                  project=False, irrep=None):
        """Build the two-particle K matrix channel by channel."""
        check_utils.check_value_within_qcis_bounds(self, E, L)
        blocks = []
        for sc_ind, sc in enumerate(self.qcis.fcs.sc_list_sorted):
            if sc.fc.n_particles != 2:
                continue
            k_tmp = _get_two_particle_k_shell(
                self.qcis, E, L, sc, sc_ind,
                pcotdelta_parameter_lists, project, irrep)
            if len(k_tmp) != 0:
                blocks.append(k_tmp)
        if len(blocks) == 0:
            return np.zeros((0, 0))
        return block_diag(*blocks)


class Kdf:
    """Three-to-three K matrix, with a single constant parameter.

    The matrix is laid out exactly like :class:`ampyl.cuts.G` and the
    three-particle block of :class:`K`: block rows and columns run over
    the three-particle entries of ``fcs.sc_list_sorted``, and within a
    channel over that channel's own spectator shells. Channels whose
    spectator masses differ live in different three-particle slices and
    therefore carry different shell sets; row and column blocks are
    taken from their own slice, so a space with more than one slice is
    supported.

    Only a single three-body parameter is implemented: ``k3_params``
    has length one and the matrix is that constant on every block whose
    row and column angular momenta agree, and zero on every block where
    they differ. Blocks connecting different spectator channels carry
    the same constant, because the term does not depend on which
    particle is called the spectator.

    At ``ell = 0`` that constant block is the isotropic term
    ``Kdf3[k, 0, 0; p, 0, 0] = K_iso``. It is rank one, so after
    projection it survives only in the trivial irrep, which is the
    expected behaviour of an isotropic three-body term at zero total
    momentum.

    At ``ell > 0`` the constant block is what the implementation
    carried before the ``ell = 0`` case was added, and the three-pion
    benchmark in ``tests/test_qc.py`` fixes it. It is one ansatz among
    several: a term that is a scalar under rotations would be
    proportional to ``delta_{m m'}`` within a wave rather than constant
    across it. That choice is deliberately left unchanged here.

    :param qcis: quantization-condition index space, specifying all data
        for the class
    :type qcis: QCIndexSpace

    Only zero total momentum is supported.
    """

    def __init__(self, qcis=None):
        self.qcis = qcis

    def get_value(self, E, L, k3_params, project, irrep):
        """Build the Kdf matrix for the whole index space.

        Parameters
        ----------
        E : float
            Total energy.
        L : float
            Box length.
        k3_params : list[float]
            Three-body parameters; length one, the isotropic term.
        project : bool
            Whether to project onto ``irrep``.
        irrep : tuple
            Target irrep when projecting.

        Returns
        -------
        numpy.ndarray
            Kdf matrix, laid out to match F, G and F+G.
        """
        nP = self.qcis.fvs.nP
        not_projecting = (irrep is None) and (project is False)
        projecting = not not_projecting
        irrep_not_in_keys = irrep not in self.qcis.proj_dict.keys()
        if projecting and irrep_not_in_keys:
            raise ValueError("irrep "+str(irrep)+" not in "
                             "qcis.proj_dict.keys()")
        tbks_entries, slices_by_three_slice\
            = self._get_entry_and_slices(E, L, nP)
        return self._get_value_from_tbks(E, L, k3_params, project, irrep,
                                         tbks_entries, slices_by_three_slice)

    def _get_entry_and_slices(self, E, L, nP):
        """Return one TBKS entry and shell list per three-particle slice."""
        if nP@nP != 0:
            raise NotImplementedError("get_value within Kdf is not "
                                      "implemented for non-zero nP yet.")
        tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
        # the pair sector owns slot 0 of tbks_list when it is present, so
        # three-slice i lives in slot i + 1; the same convention as G
        slot_offset = 1 if self.qcis.n_two_channels > 0 else 0
        tbks_entries = []
        slices_by_three_slice = []
        for three_slice_index in range(self.qcis.fcs.n_three_slices):
            slot_index = three_slice_index+slot_offset
            tbks_entry = self.qcis.tbks_list[slot_index][
                tbks_sub_indices[slot_index]]
            tbks_entries.append(tbks_entry)
            slices_by_three_slice.append(tbks_entry.shells)
        if self.qcis.verbosity >= 2:
            print('tbks_sub_indices =', tbks_sub_indices)
            print('slices =', slices_by_three_slice)
        return tbks_entries, slices_by_three_slice

    def _single_ell(self, sc_index):
        """Return the one angular momentum of a spectator channel."""
        ell_set = self.qcis.fcs.sc_list_sorted[sc_index].ell_set
        if len(ell_set) != 1:
            raise ValueError("only length-one ell_set currently "
                             "supported in Kdf")
        return ell_set[0]

    def _channel_masses(self, sc_index):
        """Return the spectator and dimer masses of a channel."""
        sc = self.qcis.fcs.sc_list_sorted[sc_index]
        return (sc.spectator.mass, sc.first_dimer.mass,
                sc.second_dimer.mass)

    def _get_value_from_tbks(self, E, L, k3_params, project, irrep,
                             tbks_entries, slices_by_three_slice):
        """Assemble the full Kdf matrix from the TBKS entries."""
        # the pair sector lives in Ktwo; Kdf is three-particle only
        slot_offset = 1 if self.qcis.n_two_channels > 0 else 0
        sc_list = self.qcis.fcs.sc_list_sorted
        kdf_final = []
        for sc_row_ind in range(len(sc_list)):
            if sc_list[sc_row_ind].fc.n_particles == 2:
                continue
            kdf_outer_row = []
            ell1 = self._single_ell(sc_row_ind)
            m1, m2, m3 = self._channel_masses(sc_row_ind)
            row_three_slice = self.qcis.sc_to_three_slice[
                sc_row_ind]-slot_offset
            row_tbks_entry = tbks_entries[row_three_slice]
            row_slices = slices_by_three_slice[row_three_slice]
            for sc_col_ind in range(len(sc_list)):
                if sc_list[sc_col_ind].fc.n_particles == 2:
                    continue
                ell2 = self._single_ell(sc_col_ind)
                col_three_slice = self.qcis.sc_to_three_slice[
                    sc_col_ind]-slot_offset
                col_tbks_entry = tbks_entries[col_three_slice]
                col_slices = slices_by_three_slice[col_three_slice]
                kdf_inner = []
                for row_shell_index in range(len(row_slices)):
                    kdf_inner_row = []
                    for col_shell_index in range(len(col_slices)):
                        kdf_tmp = self.get_shell(
                            E, L, k3_params,
                            m1, m2, m3,
                            sc_row_ind, sc_col_ind,
                            sc_row_ind, sc_col_ind,
                            ell1, ell2,
                            row_tbks_entry,
                            row_shell_index,
                            col_shell_index,
                            project, irrep,
                            col_tbks_entry=col_tbks_entry)
                        kdf_inner_row.append(kdf_tmp)
                    kdf_inner.append(kdf_inner_row)
                kdf_inner = self._clean_shape(kdf_inner)
                kdf_outer_row = kdf_outer_row+[np.block(kdf_inner)]
            kdf_final.append(kdf_outer_row)
        if len(kdf_final) == 0:
            return np.zeros((0, 0))
        kdf_final = self._clean_shape(kdf_final)
        return np.block(kdf_final)

    def get_shell(self, E=5.0, L=5.0, k3_params=None, m1=1.0, m2=1.0, m3=1.0,
                  cindex_row=None, cindex_col=None,  # only for non-zero nP
                  sc_index_row=None, sc_index_col=None, ell1=0, ell2=0,
                  tbks_entry=None, row_shell_index=None, col_shell_index=None,
                  project=False, irrep=None, col_tbks_entry=None):
        """Build the Kdf matrix block for a single pair of shells.

        Parameters
        ----------
        E : float, optional
            Total energy.
        L : float, optional
            Box length.
        k3_params : list[float], optional
            Three-body parameters; length one, the isotropic term.
        m1, m2, m3 : float, optional
            Spectator and dimer masses of the row channel. The isotropic
            term does not depend on them; they are carried so that a
            kinematic Kdf3 can use them without a signature change.
        cindex_row, cindex_col : int, optional
            Spectator-channel indices used to reach the three-slice, for
            nonzero total momentum.
        sc_index_row, sc_index_col : int, optional
            Spectator-channel indices.
        ell1, ell2 : int, optional
            Partial-wave indices of the row and column channels.
        tbks_entry : object, optional
            TBKS entry defining the row channel's shells.
        row_shell_index, col_shell_index : int, optional
            Shell indices within the row and column channels.
        project : bool, optional
            Whether to project onto ``irrep``.
        irrep : tuple, optional
            Target irrep when projecting.
        col_tbks_entry : object, optional
            TBKS entry defining the column channel's shells. Defaults to
            ``tbks_entry``, which is correct only when both channels sit
            in the same three-particle slice.

        Returns
        -------
        numpy.ndarray
            Kdf block for the requested shells.
        """
        nP = self.qcis.fvs.nP

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = shell_utils._get_masks_and_shells_for_nondiagonal(
                self, E, L, tbks_entry, cindex_row, cindex_col,
                row_shell_index, col_shell_index,
                col_tbks_entry=col_tbks_entry)
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
                shell_utils._verify_irrep_is_known(self.qcis, irrep)
                return np.array([])

        if len(k3_params) != 1:
            raise ValueError("k3_params must have length 1 for the version "
                             "of Kdf currently implemented.")

        row_dim = (row_shell[1]-row_shell[0])*(2*ell1+1)
        col_dim = (col_shell[1]-col_shell[0])*(2*ell2+1)
        # a constant block on each partial wave, and nothing off the
        # ell diagonal. At ell = 0 that is the isotropic term. The
        # ell > 0 blocks keep the behaviour the gate had before it was
        # widened, which the three-pion benchmark in tests/test_qc.py
        # fixes; see the class docstring
        if ell1 == ell2:
            Kdfshell = np.ones((row_dim, col_dim))*k3_params[0]
        else:
            Kdfshell = np.zeros((row_dim, col_dim))
        if project:
            Kdfshell = proj_tmp_left@Kdfshell@proj_tmp_right
        return Kdfshell

    def _nPzero_projectors(self, sc_index_row, sc_index_col,
                           row_shell_index, col_shell_index, irrep):
        proj_tmp_right = self.qcis.proj_dicts_by_sc_and_shellset[
                        sc_index_col][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
                        self.qcis.proj_dicts_by_sc_and_shellset[
                            sc_index_row][row_shell_index][irrep]
                        ).T)
        return proj_tmp_right, proj_tmp_left

    def _nP_nonzero_projectors(self, E, L, sc_index_row, sc_index_col,
                               row_shell_index, col_shell_index, irrep,
                               mask_row_shells, mask_col_shells):
        ibest = self.qcis.get_shellset_index(E, L)
        proj_tmp_right\
            = np.array(
                self.qcis.proj_dicts_by_sc_and_shellset[sc_index_col][ibest]
                )[mask_col_shells][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
            np.array(
                self.qcis.proj_dicts_by_sc_and_shellset[sc_index_row][ibest]
                     )[mask_row_shells][row_shell_index][irrep]).T)
        return proj_tmp_right, proj_tmp_left

    def _clean_shape(self, kdf_collection):
        """Pad empty blocks so a nested Kdf collection can be assembled.

        The collection is rectangular rather than square whenever the
        row and column channels carry different numbers of shells, so
        the column count is taken from the rows themselves.
        """
        nrows = len(kdf_collection)
        ncols = max((len(row) for row in kdf_collection), default=0)
        rowsizes = [0]*nrows
        colsizes = [0]*ncols
        for i in range(nrows):
            for j in range(len(kdf_collection[i])):
                shtmp = kdf_collection[i][j].shape
                if shtmp != (0,):
                    if shtmp[0] > rowsizes[i]:
                        rowsizes[i] = shtmp[0]
                    if shtmp[1] > colsizes[j]:
                        colsizes[j] = shtmp[1]
        for i in range(nrows):
            for j in range(len(kdf_collection[i])):
                shtmp = kdf_collection[i][j].shape
                if shtmp == (0,) or shtmp == (0, 0):
                    kdf_collection[i][j] = np.zeros((rowsizes[i],
                                                     colsizes[j]))
        return kdf_collection
