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
from .functions import QCFunctions
from .functions import BKFunctions
from .flavor import Particle
from .flavor import FlavorChannel
from .flavor import SpectatorChannel
from .flavor import FlavorChannelSpace
from .spaces import FiniteVolumeSetup
from .spaces import ThreeBodyInteractionScheme
from .spaces import ThreeBodyKinematicSpace
from .spaces import QCIndexSpace
from .cuts import G
from .cuts import F
from .cuts import FplusG
import warnings
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
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data

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
                  cindex=None, sc_ind=None, ell=0,
                  pcotdelta_function=None, pcotdelta_parameter_list=None,
                  tbks_entry=None,
                  slice_index=None, project=False, irrep=None):
        """Build the K matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta

        mask_slices, slice_entry\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Kshell = QCFunctions.getK_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell,
                                        pcotdelta_function,
                                        pcotdelta_parameter_list,
                                        alpha, beta,
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
            Kshell = proj_tmp_left@Kshell@proj_tmp_right

        return Kshell

    def get_value(self, E=5.0, L=5.0, pcotdelta_parameter_lists=None,
                  project=False, irrep=None):
        """Build the K matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        # n_two_channels = self.qcis.n_two_channels
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
            self.qcis.fcs.slices_by_three_masses[three_slice_index][0]]\
            .masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
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
                k_tmp = self.get_shell(E, L, m1, m2, m3,
                                       cindex,  # only for non-zero P
                                       sc_ind, ell,
                                       pcotdelta_function,
                                       pcotdelta_parameter_list,
                                       tbks_entry,
                                       slice_index, project, irrep)
                if len(k_tmp) != 0:
                    k_final_list = k_final_list+[k_tmp]
        return block_diag(*k_final_list)


class QC:
    """
    Class for the quantization condition.

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

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0):
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.fplusg = FplusG(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.k = K(qcis=self.qcis)

    def get_value(self, E=None, L=None, k_params=None, project=True,
                  irrep=None, version='kdf_zero_1+',
                  rescale=1.0):
        r"""
        Get value.

        version is drawn from the following:
            'kdf_zero_1+' (defaul)
            'f3'
            'kdf_zero_k2_inv'
            'kdf_zero_f+g_inv'
        """
        if E is None:
            raise TypeError("missing required argument 'E' (float)")
        if L is None:
            raise TypeError("missing required argument 'L' (float)")
        if k_params is None:
            raise TypeError("missing required argument 'k_params'")
        if irrep is None:
            raise TypeError("missing required argument 'irrep'")

        if version == 'f3':
            [pcotdelta_parameter_lists, k3_params] = k_params
            F = self.f.get_value(E, L, project, irrep)/rescale
            G = self.g.get_value(E, L, project, irrep)/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale
            return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+F+G) @ F)

        if (len(version) >= 8) and (version[:8] == 'kdf_zero'):
            [pcotdelta_parameter_lists, k3_params] = k_params
            if version == 'kdf_zero_1+_fgcombo':
                FplusG = self.fplusg.get_value(E, L, project, irrep)/rescale
                K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                     project, irrep)*rescale
                ident_tmp = np.identity(len(FplusG))
                return np.linalg.det(ident_tmp+(FplusG)@K)

            F = self.f.get_value(E, L, project, irrep)/rescale
            G = self.g.get_value(E, L, project, irrep)/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale

            if version == 'kdf_zero_1+':
                ident_tmp = np.identity(len(G))
                return np.linalg.det(ident_tmp+(F+G)@K)

            if version == 'kdf_zero_k2_inv':
                return np.linalg.det(np.linalg.inv(K)+(F+G))

            if version == 'kdf_zero_f+g_inv':
                return np.linalg.det(np.linalg.inv(F+G)+K)

    def get_qc_curve(self, Emin=None, Emax=None, Estep=None, L=None,
                     k_params=None, project=True, irrep=None,
                     version='kdf_zero_1+_fgcombo', rescale=1.0):
        E_values = np.arange(Emin, Emax, Estep)
        qc_values = []
        for E in E_values:
            qc_value = self.get_value(E=E, L=L, k_params=k_params,
                                      project=project, irrep=irrep,
                                      version=version, rescale=1.0)
            qc_values = qc_values+[qc_value]
        qc_values = np.array(qc_values)
        return E_values, qc_values

    def get_root(self, Elower=None, Eupper=None, L=None,
                 k_params=None, project=True, irrep=None,
                 version='kdf_zero_1+_fgcombo', rescale=1.0):
        root = root_scalar(self.get_value,
                           args=(L, k_params, project, irrep, version,
                                 rescale),
                           bracket=[Elower, Eupper]).root
        return root

    def search_range(self, Emin=None, Emax=None, Estep=None, cutoff=None,
                     L=None, k_params=None, project=True, irrep=None,
                     version='kdf_zero_1+_fgcombo', rescale=1.0):
        E_values = []
        qc_values = []
        for E in np.arange(Emin, Emax, Estep):
            qc = self.get_value(E, L, k_params, project, irrep, version,
                                rescale)
            if np.abs(qc) < cutoff:
                E_values = E_values+[E]
                qc_values = qc_values+[qc]
        E_values = np.array(E_values)
        qc_values = np.array(qc_values)
        return E_values, qc_values

    def get_energy_curve(self, Lstart=None, Lfinish=None, deltaL=None,
                         Estart=None, Eshift=None, dE=None,
                         k_params=None, project=True, irrep=None,
                         version='kdf_zero_1+_fgcombo', rescale=1.0):
        L_vals = np.arange(Lstart, Lfinish, deltaL)
        Estart = Estart+Eshift
        Evalsfinal = []
        Lvalsfinal = []
        for L_val in L_vals:
            try:
                rs = root_scalar(self.get_value,
                                 args=(L_val,
                                       k_params,
                                       project,
                                       irrep,
                                       version,
                                       rescale),
                                 bracket=[Estart-dE, Estart+dE]).root
                dE = np.abs(rs-Estart)*5
                Evalsfinal = Evalsfinal+[rs]
                Lvalsfinal = Lvalsfinal+[L_val]
                Estart = rs
            except ValueError:
                return np.array(Lvalsfinal), np.array(Evalsfinal)
        return np.array(Lvalsfinal), np.array(Evalsfinal)
