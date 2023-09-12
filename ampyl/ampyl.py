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
from .constants import bcolors
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
                    ibest = 0
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "ibest is set to 0. This is a temporary fix."
                                  f"{bcolors.ENDC}")
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
                  rescale=1.0, shift=0.):
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
                if len(FplusG) > len(K):
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "FplusG and K have different shapes, and "
                                  "FplusG is larger. "
                                  "Padding K with extra entries. "
                                  "This is a temporary fix."
                                  f"{bcolors.ENDC}")
                    K_tmp = np.zeros_like(FplusG)
                    K_tmp[:len(K), :len(K)] = K
                    K = K_tmp
                elif len(FplusG) < len(K):
                    FplusG = np.zeros(K.shape)
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "FplusG and K have different shapes, and "
                                  "FplusG is smaller. "
                                  "Setting FplusG to zero. "
                                  "This is a temporary fix."
                                  f"{bcolors.ENDC}")
                ident_tmp = np.identity(len(FplusG))
                return np.linalg.det(ident_tmp+(FplusG)@K)-shift

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

            if version == 'kdf_zero_1+_FinverseF3':
                ident_tmp = np.identity(len(G))
                block_inv = np.linalg.inv(np.linalg.inv(K)+F+G)
                matrix_in_det = ident_tmp-3.*block_inv@F
                inverse_det = 1./np.linalg.det(matrix_in_det)
                return inverse_det
    def get_roots_at_Lmax(self, Emax_for_roots=None, L_for_roots=None,
                          n_steps=10, k_params=None, project=True, irrep=None,
                          version='kdf_zero_1+_fgcombo', rescale=1.0,
                          also_search_brackets=False):
        L = L_for_roots
        nonint_energies = self\
            ._get_nonint_energies(Emax_for_roots, k_params, irrep, L)
        brackets = self._get_brackets(Emax_for_roots, nonint_energies)
        roots = []
        for bracket in brackets:
            roots += self._try_to_find_roots_at_Lmax(L, n_steps, bracket,
                                                     k_params, project, irrep,
                                                     version, rescale)
        roots = np.array(roots)
        roots_rounded = np.round(roots, 10)
        roots_unique_rounded = np.unique(roots_rounded)
        roots_unique = []
        for unique_root in roots_unique_rounded:
            locations = np.where(roots_rounded == unique_root)[0]
            already_included = False
            for root in roots[locations]:
                qc_value = self.get_value(E=root, L=L, k_params=k_params,
                                          project=project, irrep=irrep,
                                          version=version, rescale=rescale)
                if np.abs(qc_value) < EPSILON10 and not already_included:
                    roots_unique.append(root)
                    already_included = True
        if not also_search_brackets:
            return roots
        roots_down_shift = []
        for bracket in brackets:
            neg_shift = -0.1
            roots_down_shift += self.\
                _try_to_find_roots_at_Lmax(L, n_steps, bracket,
                                           k_params, project, irrep,
                                           version, rescale, shift=neg_shift)
        roots_down_shift = np.array(roots_down_shift)
        roots_down_shift = np.unique(np.round(roots_down_shift, 15))
        roots_up_shift = []
        for bracket in brackets:
            pos_shift = 0.1
            roots_up_shift += self.\
                _try_to_find_roots_at_Lmax(L, n_steps, bracket,
                                           k_params, project, irrep,
                                           version, rescale, shift=pos_shift)
        roots_up_shift = np.array(roots_up_shift)
        roots_up_shift = np.unique(np.round(roots_up_shift, 15))
        return roots, roots_down_shift, roots_up_shift

    def _get_brackets(self, Emax_for_roots, nonint_energies,
                      Emin_for_roots=2.001):
        brackets = []
        brackets.append([Emin_for_roots, nonint_energies[0]])
        i = -1
        for i in range(len(nonint_energies)-1):
            brackets.append([nonint_energies[i], nonint_energies[i+1]])
        brackets.append([nonint_energies[i+1], Emax_for_roots])
        return brackets

    def _get_nonint_energies(self, Emax_for_roots, k_params, irrep, L):
        multis_pipipi = self.qcis.nonint_multiplicities[0][irrep]
        multis_rhopi = self.qcis.nonint_multiplicities[1][irrep]
        nonint_Evals = []
        for multi in multis_pipipi:
            pSQs = FOURPI2*np.array(multi[:3])/L**2
            m_array = np.array(self.fplusg._extract_masses())
            omegas = np.sqrt(m_array**2+pSQs)
            E_nonint = omegas.sum()
            if E_nonint < Emax_for_roots:
                nonint_Evals.append(E_nonint)
        for multi in multis_rhopi:
            pSQs = FOURPI2*np.array(multi[:2])/L**2
            mpi = self.fplusg._extract_masses()[0]
            mrho = k_params[0][0][1]
            m_array = np.array([mpi, mrho])
            omegas = np.sqrt(m_array**2+pSQs)
            E_nonint = omegas.sum()
            if E_nonint < Emax_for_roots:
                nonint_Evals.append(E_nonint)
        nonint_Evals = np.sort(np.array(nonint_Evals))
        return nonint_Evals

    def _try_to_find_roots_at_Lmax(self, Lmax=6., n_steps=10, bracket=None,
                                   k_params=None, project=True, irrep=None,
                                   version='kdf_zero_1+_fgcombo',
                                   rescale=1.0, shift=0.):
        Emin = bracket[0]+EPSILON4
        Emax = bracket[1]-EPSILON4
        Eslices = np.linspace(Emin, Emax, n_steps)
        roots = []
        for i in range(len(Eslices)-1):
            E1 = Eslices[i]
            E2 = Eslices[i+1]
            try:
                root = root_scalar(self.get_value,
                                   args=(Lmax, k_params, project, irrep,
                                         version, rescale, shift),
                                   bracket=[E1, E2]).root
                roots = roots+[root]
            except ValueError:
                continue
        return roots

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

    def get_energy_curve(self, L_start=None, L_finish=None, deltaL_target=None,
                         Estart=None, dE=None,
                         k_params=None, project=True, irrep=None,
                         version='kdf_zero_1+_fgcombo', rescale=1.0,
                         shift_upper=0.01, shift_lower=-0.01,
                         max_iterations=500):
        Evals_final = []
        deltaL_original_target = deltaL_target
        lower_Evals_final = []
        upper_Evals_final = []
        upper_shifts = []
        lower_shifts = []
        Lvals_final = []
        E_lower = Estart-dE
        E_upper = Estart+dE
        L_val = L_start
        iteration = 0
        while L_val > L_finish:
            iteration += 1
            if iteration > max_iterations:
                return np.array(Lvals_final), np.array(Evals_final),\
                    np.array(lower_Evals_final), np.array(upper_Evals_final),\
                    np.array(upper_shifts), np.array(lower_shifts)
            try:
                rs, rs_upper, rs_lower = self.\
                    _get_root_set(k_params, project, irrep, version, rescale,
                                  E_lower, E_upper, L_val,
                                  shift_upper, shift_lower)
                [E_lower, E_upper] = np.sort([rs_lower, rs_upper])
                lower_Evals_final = lower_Evals_final+[E_lower]
                upper_Evals_final = upper_Evals_final+[E_upper]
                Evals_final = Evals_final+[rs]
                Lvals_final = Lvals_final+[L_val]
                upper_shifts = upper_shifts+[shift_upper]
                lower_shifts = lower_shifts+[shift_lower]
                if np.abs(deltaL_target) < np.abs(deltaL_original_target):
                    deltaL_target = deltaL_target*4.
                if shift_upper < 0.1:
                    shift_upper = shift_upper*2.
                    shift_lower = shift_lower*2.
                L_next = L_val+deltaL_target
                if len(Evals_final) > 1:
                    E_lower = np.interp(L_next, Lvals_final, lower_Evals_final)
                    E_upper = np.interp(L_next, Lvals_final, upper_Evals_final)
                L_val = L_next
            except ValueError:
                if np.abs(deltaL_target) > 1.e-8:
                    L_val = L_val-deltaL_target
                    deltaL_target = deltaL_target/2.
                    L_val = L_val+deltaL_target
                    continue
                elif shift_upper > 1.e-3:
                    shift_upper = shift_upper/2.
                    shift_lower = shift_lower/2.
                    continue
                else:
                    return np.array(Lvals_final), np.array(Evals_final)
        return np.array(Lvals_final), np.array(Evals_final),\
            np.array(lower_Evals_final), np.array(upper_Evals_final),\
            np.array(upper_shifts), np.array(lower_shifts)

    def _get_root_set(self, k_params, project, irrep, version,
                      rescale, E_lower, E_upper, L_val,
                      shift_upper, shift_lower):
        rs = root_scalar(self.get_value,
                         args=(L_val, k_params, project, irrep, version,
                               rescale),
                         bracket=[E_lower, E_upper]).root
        rs_upper = root_scalar(self.get_value,
                               args=(L_val, k_params, project, irrep, version,
                                     rescale, shift_upper),
                               bracket=[E_lower, E_upper]).root
        rs_lower = root_scalar(self.get_value,
                               args=(L_val, k_params, project, irrep, version,
                                     rescale, shift_lower),
                               bracket=[E_lower, E_upper]).root
        return rs, rs_upper, rs_lower
