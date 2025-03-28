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
        if irrep is None and project:
            raise TypeError("missing required argument 'irrep'")

        if version == 'f3':
            [pcotdelta_parameter_lists, k3_params] = k_params
            F = self.f.get_value(E, L, project, irrep,
                                 short_string='f')/rescale
            G = self.g.get_value(E, L, project, irrep,
                                 short_string='g')/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale
            return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+F+G) @ F)/L**3

        if (len(version) >= 8) and (version[:8] == 'kdf_zero'):
            [pcotdelta_parameter_lists, k3_params] = k_params
            if version == 'kdf_zero_1+_fgcombo':
                FplusG = self.fplusg.get_value(E, L, project, irrep,
                                               short_string='fplusg')/rescale
                K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                     project, irrep)*rescale
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
                id_mat = np.identity(len(FplusG))
                return np.linalg.det(id_mat+(FplusG)@K)-shift

            F = self.f.get_value(E, L, project, irrep,
                                 short_string='f')/rescale
            G = self.g.get_value(E, L, project, irrep,
                                 short_string='g')/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale

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

    def generate_summary(self, Elower=None, Eupper=None, L=None,
                         k_params=None, project=True, irrep=None,
                         version='kdf_zero_1+_fgcombo', rescale=1.0):
        wf = open(f'summary_L{L:.1f}_irrep{irrep[0]}_'
                  f'Elower{Elower:.1f}_Eupper{Eupper:.1f}.txt', 'w')
        root = root_scalar(self.get_value,
                           args=(L, k_params, project, irrep, version,
                                 rescale),
                           bracket=[Elower, Eupper]).root
        wf.write('Summary of a quantization condition solution:\n\n'
                 'Following inputs were given:\n')
        wf.write(f'Elower = {Elower}\n'
                 f'Eupper = {Eupper}\n'
                 f'L = {L}\n'
                 f'k_params = {k_params}\n'
                 f'project = {project}\n'
                 f'irrep = {irrep}\n'
                 f'version = {version}\n'
                 f'rescale = {rescale}\n'
                 f'g_smart_interpolate = '
                 f'{self.qcis.fvs.qc_impl["g_smart_interpolate"]}\n'
                 f'f_smart_interpolate = '
                 f'{self.qcis.fvs.qc_impl["f_smart_interpolate"]}\n'
                 f'fplusg_smart_interpolate = '
                 f'{self.qcis.fvs.qc_impl["fplusg_smart_interpolate"]}\n\n')

        wf.write('A solution was found at\n'
                 f'E = {root}\n\n')

        qc_value = self.get_value(E=root, L=L, k_params=k_params,
                                  project=project, irrep=irrep,
                                  version=version, rescale=rescale)

        wf.write('Value of the QC near the solution:\n')
        for i in range(5):
            scaler = 0.98+0.01*i
            E = root*scaler
            qc_value = self.get_value(E=E, L=L, k_params=k_params,
                                      project=project, irrep=irrep,
                                      version=version, rescale=rescale)
            wf.write(f'qc(E = {scaler:.2f}sol) = {qc_value}\n')

        G = self.g.get_value(E=root, L=L, project=project, irrep=irrep,
                             short_string='g')
        wf.write(f'\nShape of projected G-matrix at the solution: {G.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(G))
        wf.write(f'Eigenvalues of projected G-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of projected G-matrix at the solution:\n{G}\n\n')

        F = self.f.get_value(E=root, L=L, project=project, irrep=irrep,
                             short_string='f')
        wf.write(f'Shape of projected F-matrix at the solution: {F.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(F))
        wf.write(f'Eigenvalues of projected F-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of projected F-matrix at the solution:\n{F}\n\n')

        K = self.k.get_value(E=root, L=L,
                             pcotdelta_parameter_lists=k_params[0],
                             project=project, irrep=irrep)
        wf.write(f'Shape of projected K-matrix at the solution: {K.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(K))
        wf.write(f'Eigenvalues of projected K-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of projected K-matrix at the solution:\n{K}\n\n')

        ident_tmp = np.identity(len(F))
        qc_matrix = (ident_tmp+(F+G)@K)
        wf.write(f'Shape of the projected I+(F+G)K at the solution: '
                 f'{qc_matrix.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(qc_matrix))
        wf.write('Eigenvalues of the projected I+(F+G)K at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write('Value of the projected I+(F+G)K at the solution:\n'
                 f'{qc_matrix}\n\n')

        G = self.g.get_value(E=root, L=L, short_string='g')
        wf.write(f'Shape of unprojected G-matrix at the solution: {G.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(G))
        wf.write(f'Eigenvalues of unprojected G-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of unprojected G-matrix at the solution:\n{G}\n\n')

        F = self.f.get_value(E=root, L=L, short_string='f')
        wf.write(f'Shape of unprojected F-matrix at the solution: {F.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(F))
        wf.write(f'Eigenvalues of unprojected F-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of unprojected F-matrix at the solution:\n{F}\n\n')

        K = self.k.get_value(E=root, L=L,
                             pcotdelta_parameter_lists=k_params[0])
        wf.write(f'Shape of unprojected K-matrix at the solution: {K.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(K))
        wf.write(f'Eigenvalues of unprojected K-matrix at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write(f'Value of unprojected K-matrix at the solution:\n{K}\n\n')

        ident_tmp = np.identity(len(F))
        qc_matrix = (ident_tmp+(F+G)@K)
        wf.write(f'Shape of the unprojected I+(F+G)K at the solution: '
                 f'{qc_matrix.shape}\n')
        eigenvalues = np.sort(np.linalg.eigvals(qc_matrix))
        wf.write('Eigenvalues of the unprojected I+(F+G)K at the solution:\n'
                 f'{eigenvalues}\n\n')
        wf.write('Value of the unprojected I+(F+G)K at the solution:\n'
                 f'{qc_matrix}\n')

        wf.close()


    def get_roots_at_fixed_L(self, Emax_for_roots=None, L_for_roots=None,
                             n_steps=10, k_params=None, project=True,
                             irrep=None, version='kdf_zero_1+_fgcombo',
                             rescale=1.0, also_search_brackets=False):
        L = L_for_roots
        nonint_energies = self\
            ._get_nonint_energies(Emax_for_roots, k_params, irrep, L)
        brackets = self._get_brackets(Emax_for_roots, nonint_energies)
        roots = []
        for bracket in brackets:
            roots += self._try_to_find_roots_at_fixed_L(L, n_steps, bracket,
                                                        k_params, project,
                                                        irrep, version,
                                                        rescale)
        roots = np.array(roots)
        roots_unique = np.array([])
        cutoff_for_unique = EPSILON3
        for root in roots:
            if len(roots_unique) == 0:
                qc_value = self.get_value(E=root, L=L, k_params=k_params,
                                          project=project, irrep=irrep,
                                          version=version, rescale=rescale)
                if np.abs(qc_value) < EPSILON10:
                    roots_unique = np.append(roots_unique, root)
            else:
                distances = np.abs(roots_unique-root)
                if np.min(distances) > cutoff_for_unique:
                    qc_value = self.get_value(E=root, L=L, k_params=k_params,
                                              project=project, irrep=irrep,
                                              version=version, rescale=rescale)
                    if np.abs(qc_value) < EPSILON10:
                        roots_unique = np.append(roots_unique, root)
        if not also_search_brackets:
            return roots_unique
        roots_down_shift = []
        for bracket in brackets:
            neg_shift = -0.1
            roots_down_shift += self.\
                _try_to_find_roots_at_fixed_L(L, n_steps, bracket,
                                              k_params, project, irrep,
                                              version, rescale,
                                              shift=neg_shift)
        roots_down_shift = np.array(roots_down_shift)
        roots_down_shift = np.unique(np.round(roots_down_shift, 15))
        roots_up_shift = []
        for bracket in brackets:
            pos_shift = 0.1
            roots_up_shift += self.\
                _try_to_find_roots_at_fixed_L(L, n_steps, bracket,
                                              k_params, project, irrep,
                                              version, rescale,
                                              shift=pos_shift)
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

    def pipipi_nonint(self, multi, L, Emax_for_roots, nonint_Evals):
        pSQs = FOURPI2*np.array(multi[:3])/L**2
        m_array = np.array(self.fplusg._extract_masses())
        omegas = np.sqrt(m_array**2+pSQs)
        E_nonint = omegas.sum()
        if E_nonint < Emax_for_roots:
            nonint_Evals.append(E_nonint)
        return nonint_Evals

    def rhopi_nonint(self, multi, L, mrho, Emax_for_roots, nonint_Evals):
        pSQs = FOURPI2*np.array(multi[:2])/L**2
        mpi = self.fplusg._extract_masses()[0]
        m_array = np.array([mpi, mrho])
        omegas = np.sqrt(m_array**2+pSQs)
        E_nonint = omegas.sum()
        if E_nonint < Emax_for_roots:
            nonint_Evals.append(E_nonint)
        return nonint_Evals

    def sigmapi_nonint(self, multi, L, msigma, Emax_for_roots, nonint_Evals):
        pSQs = FOURPI2*np.array(multi[:2])/L**2
        mpi = self.fplusg._extract_masses()[0]
        m_array = np.array([mpi, msigma])
        omegas = np.sqrt(m_array**2+pSQs)
        E_nonint = omegas.sum()
        if E_nonint < Emax_for_roots:
            nonint_Evals.append(E_nonint)
        return nonint_Evals

    def correct_irrep_row(self, irrep):
        """
        Corrects the row index of an irreducible representation (irrep)
        based on the available non-interacting multiplicities.
        """
        qcis = self.qcis
        irrep_name = irrep[0]
        irrep_row = irrep[1]
        ni_zero_keys = qcis.nonint_multiplicities[0].keys()
        for i in range(3):
            irrep_row = (irrep_row + i) % 3
            if (irrep_name, i) in ni_zero_keys:
                irrep = (irrep_name, i)
        return irrep

    def _get_nonint_energies(self, Emax_for_roots, k_params, irrep, L):
        irrep = self.correct_irrep_row(irrep)
        if self.verbosity >= 2:
            print(f'{bcolors.OKGREEN}'
                  f'irrep {irrep} found in nonint_multiplicities.\n'
                  'Proceeding with this irrep.'
                  f'{bcolors.ENDC}')
        fc_list = self.qcis.fcs.fc_list
        if len(fc_list) == 0:
            raise ValueError("No flavor channels in the QCIndexSpace in QC")
        isospin = self.qcis.fcs.fc_list[0].isospin
        for fc in self.qcis.fcs.fc_list:
            if fc.isospin != isospin:
                raise ValueError("isospin is not the same for all channels")
        if isospin == 2 or isospin == 0:
            multis_pipipi = self.qcis.nonint_multiplicities[0][irrep]
            multis_rhopi = self.qcis.nonint_multiplicities[1][irrep]
            if self.verbosity >= 2:
                print(f'{bcolors.OKGREEN}'
                      f'multis_pipipi: {multis_pipipi},\n'
                      f'multis_rhopi: {multis_rhopi}'
                      f'{bcolors.ENDC}')
            nonint_Evals = []
            for multi in multis_pipipi:
                nonint_Evals = self.pipipi_nonint(multi, L, Emax_for_roots,
                                                  nonint_Evals)
            for multi in multis_rhopi:
                mrho = k_params[0][0][1]
                nonint_Evals = self.rhopi_nonint(multi, L, mrho,
                                                 Emax_for_roots, nonint_Evals)
            nonint_Evals = np.sort(np.array(nonint_Evals))
        elif isospin == 1:
            multis_pipipi = self.qcis.nonint_multiplicities[0][irrep]
            multis_rhopi = self.qcis.nonint_multiplicities[1][irrep]
            multis_sigmapi = self.qcis.nonint_multiplicities[2][irrep]
            if self.verbosity >= 2:
                print(f'{bcolors.OKGREEN}'
                      f'multis_pipipi: {multis_pipipi},\n'
                      f'multis_rhopi: {multis_rhopi},\n'
                      f'multis_sigmapi: {multis_sigmapi}'
                      f'{bcolors.ENDC}')
            nonint_Evals = []
            for multi in multis_pipipi:
                nonint_Evals = self.pipipi_nonint(multi, L, Emax_for_roots,
                                                  nonint_Evals)
            for multi in multis_rhopi:
                mrho = k_params[0][1][1]
                nonint_Evals = self.rhopi_nonint(multi, L, mrho,
                                                 Emax_for_roots, nonint_Evals)
            for multi in multis_sigmapi:
                msigma = k_params[0][0][1]
                nonint_Evals = self.sigmapi_nonint(multi, L, msigma,
                                                   Emax_for_roots,
                                                   nonint_Evals)
            nonint_Evals = np.sort(np.array(nonint_Evals))
        else:
            raise ValueError("isospin is not 0, 1, or 2")
        return nonint_Evals

    def _try_to_find_roots_at_fixed_L(self, Lmax=6., n_steps=10, bracket=None,
                                      k_params=None, project=True, irrep=None,
                                      version='kdf_zero_1+_fgcombo',
                                      rescale=1.0, shift=0.):
        bracket_min = bracket[0]
        bracket_max = bracket[1]
        Emin = bracket_min+EPSILON3
        Emax = bracket_max-EPSILON3
        Eslices_low = np.array([bracket_min+EPSILON10,
                                bracket_min+EPSILON8,
                                bracket_min+EPSILON5])
        Eslices_mid = np.linspace(Emin, Emax, n_steps)
        Eslices_high = np.array([bracket_max-EPSILON5,
                                 bracket_max-EPSILON8,
                                 bracket_max-EPSILON10])
        Eslices = np.concatenate((Eslices_low, Eslices_mid, Eslices_high))
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

    def extrapolate_E_range(self, Lvals_final, Evals_final, lower_Evals_final,
                            upper_Evals_final, L_next):
        if len(Evals_final) == 2:
            degree = 1
        elif len(Evals_final) == 3:
            degree = 2
        elif len(Evals_final) >= 4:
            degree = 3
        fit_lower = np.polyfit(Lvals_final, lower_Evals_final, degree)
        curve_lower = np.poly1d(fit_lower)
        E_lower = curve_lower(L_next)
        fit_upper = np.polyfit(Lvals_final, upper_Evals_final, degree)
        curve_upper = np.poly1d(fit_upper)
        E_upper = curve_upper(L_next)
        return E_lower, E_upper

    def get_energy_curve(self, L_start=None, L_finish=None, deltaL_target=None,
                         Estart=None, dE=None, initial_dEdL=-1.,
                         k_params=None, project=True, irrep=None,
                         version='kdf_zero_1+_fgcombo', rescale=1.0,
                         shift_upper=0.01, shift_lower=-0.01,
                         max_iterations=500, percent_extension=0.05):
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
        while ((deltaL_target < 0. and L_val > L_finish) or
               (deltaL_target > 0. and L_val < L_finish)):
            iteration += 1
            if iteration > max_iterations:
                return np.array(Lvals_final), np.array(Evals_final), \
                    np.array(lower_Evals_final), np.array(upper_Evals_final), \
                    np.array(upper_shifts), np.array(lower_shifts)
            try:
                rs, rs_upper, rs_lower = self.\
                    _get_root_set(k_params, project, irrep, version, rescale,
                                  E_lower, E_upper, L_val,
                                  shift_upper, shift_lower, percent_extension)
                [E_lower, E_upper] = np.sort([rs_lower, rs_upper])
                lower_Evals_final = lower_Evals_final+[E_lower]
                upper_Evals_final = upper_Evals_final+[E_upper]
                Evals_final = Evals_final+[rs]
                Lvals_final = Lvals_final+[L_val]
                upper_shifts = upper_shifts+[shift_upper]
                lower_shifts = lower_shifts+[shift_lower]
                if np.abs(deltaL_target) < np.abs(deltaL_original_target):
                    deltaL_target = deltaL_target*4.
                L_next = L_val+deltaL_target
                if len(Evals_final) > 1:
                    E_lower, E_upper =\
                        self.extrapolate_E_range(Lvals_final,
                                                 Evals_final,
                                                 lower_Evals_final,
                                                 upper_Evals_final,
                                                 L_next)
                else:
                    dE_for_next_L = deltaL_target*initial_dEdL
                    E_lower = E_lower+dE_for_next_L
                    E_upper = E_upper+dE_for_next_L
                L_val = L_next
            except ValueError:
                if np.abs(deltaL_target) > 1.e-8:
                    L_val = L_val-deltaL_target
                    deltaL_target = deltaL_target/2.
                    L_val = L_val+deltaL_target
                    if len(Evals_final) > 1:
                        E_lower, E_upper =\
                            self.extrapolate_E_range(Lvals_final, Evals_final,
                                                     lower_Evals_final,
                                                     upper_Evals_final,
                                                     L_val)
                    continue
                else:
                    return np.array(Lvals_final), np.array(Evals_final)
        return np.array(Lvals_final), np.array(Evals_final), \
            np.array(lower_Evals_final), np.array(upper_Evals_final), \
            np.array(upper_shifts), np.array(lower_shifts)

    def _get_root_set(self, k_params, project, irrep, version,
                      rescale, E_lower, E_upper, L_val,
                      shift_upper, shift_lower, percent_extension):
        rs = root_scalar(self.get_value,
                         args=(L_val, k_params, project, irrep, version,
                               rescale),
                         bracket=[E_lower, E_upper]).root
        E_ext = percent_extension*np.abs(E_upper-E_lower)
        rs_upper = root_scalar(self.get_value,
                               args=(L_val, k_params, project, irrep, version,
                                     rescale, shift_upper),
                               bracket=[E_lower-E_ext, E_upper+E_ext]).root
        rs_lower = root_scalar(self.get_value,
                               args=(L_val, k_params, project, irrep, version,
                                     rescale, shift_lower),
                               bracket=[E_lower-E_ext, E_upper+E_ext]).root
        return rs, rs_upper, rs_lower
