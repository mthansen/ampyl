#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# qc_functions.py
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
from sympy.physics.quantum.cg import CG
from sympy import S
from scipy.special import sph_harm
from scipy.special import erfi
from scipy.special import erf
from scipy.linalg import block_diag
import functools

PI = np.pi
TWOPI = 2.*PI
FOURPI2 = 4.0*PI**2
ROOT4PI = np.sqrt(4.*PI)
EPSILON = 1.0e-15
QC_IMPL_DEFAULTS = {'hermitian': True,
                    'real harmonics': True,
                    'Zinterp': False,
                    'YYCG': False}


class bcolors:
    """Colors."""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class BKFunctions:
    """Class collecting basic kinematic functions."""

    @staticmethod
    def J_slow(z=0.5):
        r"""
        Underlying cutoff function.

        Smooth cutoff function ranging from \\[J(z) = 0.0\\] for \\[z < 0.0\\]
        to \\[J(z) = 1.0\\] for \\[z > 1.0\\].

        Parameters
        ----------
        z (float or np.ndarray)

        Returns
        -------
        (float or np.ndarray)

        """
        if isinstance(z, np.ndarray):
            J_array = np.array([])
            for z_val in z:
                if z_val <= 0.0:
                    J_array = np.append(J_array, 0.0)
                elif z_val >= 1.0:
                    J_array = np.append(J_array, 1.0)
                else:
                    J_array = np.append(J_array,
                                        np.exp(
                                            -1.0/z_val*np.exp(-1.0/(1.0-z_val))
                                            ))
            return J_array
        if isinstance(z, float):
            if z <= 0:
                return 0.0
            if z >= 1.0:
                return 1.0
            return np.exp(-1.0/z*np.exp(-1.0/(1.0-z)))
        raise ValueError('z must be a float or np.ndarray')

    @staticmethod
    def J(z=np.array([0.5])):
        r"""
        Underlying cutoff function.

        Smooth cutoff function ranging from \\[J(z) = 0.0\\] for \\[z < 0.0\\]
        to \\[J(z) = 1.0\\] for \\[z > 1.0\\].

        Parameters
        ----------
        z (float or np.ndarray)

        Returns
        -------
        (float or np.ndarray)

        """
        if isinstance(z, np.ndarray):
            J_array = np.zeros_like(z)
            mask1 = 0.0 < z
            mask2 = z < 1.0
            mask = (
                np.concatenate((mask1, mask2)).reshape((2, len(mask1))).T
                ).all(axis=1)
            mask_one = 1.0 <= z
            J_array[mask_one] = 1.0
            J_array[mask] = np.exp(-1.0/z[mask]*np.exp(-1.0/(1.0-z[mask])))
            return J_array
        if isinstance(z, float):
            if z <= 0:
                return 0.0
            if z >= 1.0:
                return 1.0
            return np.exp(-1.0/z*np.exp(-1.0/(1.0-z)))
        raise ValueError('z must be a float or np.ndarray')

    @staticmethod
    def H(E2CMSQ=9.0, threshold=2.0, alpha=-1.0, beta=0.0, J_slow=False):
        r"""
        Kinematic-based cutoff function.

        Smooth cutoff function built from \\[J(z)\\], ranging from
        \\[H = 0.0\\] for squared two-particle CMF energies below a chosen
        value to \\[H = 1.0\\] above threshold.

        Parameters
        ----------
        E2CMSQ (float): squared two-particle center-of-mass energy
        threshold (float): value of two-particle threshold
        alpha (float): width parameter (alpha = -1.0 is standard)
        beta (float): shift parameter (beta = 0.0 is standard)

        Returns
        -------
        (float or np.ndarray)
        """
        z = (E2CMSQ-(1.0+alpha)*threshold**2/4.0)\
            / ((3.0-alpha)*threshold**2/4.0)+beta
        if J_slow:
            return BKFunctions.J_slow(z)
        return BKFunctions.J(z)

    @staticmethod
    def phase_space(E2CMSQ=9.0, omk=1.0):
        r"""
        Two-body phase space with \\[2\omega_k\\] included.

        Parameters
        ----------
        E2CMSQ (float or np.ndarray): squared two-particle center-of-mass
            energy
        omk (float or np.ndarray): time-component of four-vector k

        Returns
        -------
        (float or np.ndarray)

        """
        return 1.0/(16.0*PI*np.sqrt(E2CMSQ))/(2.0*omk)

    @staticmethod
    def phase_space_alt(omk=1.0, m=1.0):
        r"""
        Two-body phase space, alternative definition.

        Parameters
        ----------
        omk (float or np.ndarray): time-component of four-vector k
        m (float or np.ndarray): mass

        Returns
        -------
        (float or np.ndarray)

        """
        return 1.0/(32.0*PI*m)/(2.0*omk)

    @staticmethod
    def q_one_minus_H(E2CMSQ=9.0, m1=1.0, m2=1.0, alpha=-1.0, beta=0.0,
                      J_slow=False):
        r"""
        Extra term for relating K2 to M2: \\[|q|*(1-H(\\cdots))\\].

        Parameters
        ----------
        E2CMSQ (float): squared two-particle center-of-mass energy
        m1 (float or np.ndarray): first mass
        m2 (float or np.ndarray): second mass
        alpha (float): width parameter (alpha = -1.0 is standard)
        beta (float): shift parameter (beta = 0.0 is standard)

        Returns
        -------
        (float or np.ndarray)
        """
        threshold = m1+m2
        if m1 == m2:
            qCMSQ = E2CMSQ/4.0-m1**2
        else:
            qCMSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
                     + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
                / (4.0*E2CMSQ)
        qCM = np.sqrt(np.abs(qCMSQ))
        return qCM*(1.0-BKFunctions.H(E2CMSQ, threshold, alpha, beta, J_slow))

    @staticmethod
    def cart_sph_harm(ell=0, mazi=0,
                      nvec_arr=np.array([[1.0, 2.0, 3.0]])):
        r"""
        Cartesian spherical harmonics.

        Includes multiplicative \\[\sqrt{4 \pi}\\] such that \\[Y_{00}==1\\].

        Parameters
        ----------
        ell (int): orbital angular momentum
        mazi (int): azimuthal component
        nvec_arr (np.ndarray): array of three-vectors
        (Also relies on global EPSILON parameter.)

        Returns
        -------
        (np.ndarray)
        """
        nxs = (nvec_arr.T)[0]
        nys = (nvec_arr.T)[1]
        nzs = (nvec_arr.T)[2]
        nmags = np.sqrt((nvec_arr**2).sum(1))
        thetas = np.arccos(nzs/(nmags+EPSILON))
        phis = np.arctan(nys/(nxs+EPSILON))+(1.0-np.sign(nxs+EPSILON))*PI/2.0
        return ROOT4PI*(nmags**ell)*sph_harm(mazi, ell, phis, thetas)

    @staticmethod
    def cart_sph_harm_real(ell=0, mazi=0,
                           nvec_arr=np.array([[1.0, 2.0, 3.0]])):
        r"""
        Real cartesian spherical harmonics.

        Includes multiplicative \\[\sqrt{4 \pi}\\] such that \\[Y_{00}==1\\].

        Parameters
        ----------
        ell (int): orbital angular momentum
        mazi (int): azimuthal component
        nvec_arr (np.ndarray): array of three-vectors
        (Also relies on global EPSILON parameter.)

        Returns
        -------
        (np.ndarray)
        """
        if mazi == 0:
            return BKFunctions.cart_sph_harm(ell, mazi, nvec_arr).real
        if mazi < 0:
            return (np.sqrt(2.0)*(-1.0)**mazi)\
                * BKFunctions.cart_sph_harm(ell, np.abs(mazi), nvec_arr).imag
        if mazi > 0:
            return (np.sqrt(2.0)*(-1.0)**mazi)\
                * BKFunctions.cart_sph_harm(ell, mazi, nvec_arr).real

    @staticmethod
    def recombine_YY(ell1, mazi1, ell2, mazi2):
        """
        Recombine.

        (ell2, mazi2) belong to the conjugated harmonic.
        """
        mazi = mazi1-mazi2
        ell_min = np.max([np.abs(ell1-ell2), np.abs(mazi)])
        ell_max = ell1+ell2
        recombine_set = [[]]
        for ell in range(ell_min, ell_max+1):
            coeff = ((-1.)**mazi2)\
                    * np.sqrt((2.*ell1+1.)*(2.*ell2+1.)
                              / (4.*np.pi*(2.*ell+1.)))\
                    * (CG(ell1, mazi1, ell2, -mazi2, ell, mazi).doit())\
                    * (CG(ell1, 0, ell2, 0, ell, 0).doit())
            coeff = float(coeff.evalf())
            recombine_set = recombine_set+[[ell, mazi, coeff]]
        return recombine_set[1:]

    @staticmethod
    def recombine_YY_real(ell1, mazi1, ell2, mazi2):
        """
        Recombine.

        All harmonics real.
        """
        if mazi1 < 0 and mazi2 < 0:
            #
            # [  1j/sqrt(2)*(Y(ell1, m1)-((-1)^m1)*Y(ell1, -m1))  ]
            #     * [  1j/sqrt(2)*(Y(ell2, m2)-((-1)^m2)*Y(ell2, -m2))  ]
            #
            # -0.5                  * Y(ell1, m1)*Y(ell2, m2)
            # +0.5*((-1)^m2)        * Y(ell1, m1)*Y(ell2, -m2)
            # +0.5*((-1)^m1)        * Y(ell1, -m1))*Y(ell2, m2)
            # -0.5*(((-1)^(m1+m2))) * Y(ell1, -m1))*Y(ell2, -m2)
            #
            foil_set = [[ell1, mazi1, ell2, mazi2, -0.5],
                        [ell1, mazi1, ell2, -mazi2, 0.5*((-1.)**mazi2)],
                        [ell1, -mazi1, ell2, mazi2, 0.5*((-1.)**mazi1)],
                        [ell1, -mazi1, ell2, -mazi2,
                         -0.5*((-1.)**(mazi1+mazi2))]]
        elif mazi1 < 0 and mazi2 > 0:
            #
            # [  1j/sqrt(2)*(Y(ell1, m1)-((-1)^m1)*Y(ell1, -m1))  ]
            #     * [  1./sqrt(2)*(((-1)^m2)*Y(ell2, m2)+Y(ell2, -m2))  ]
            #
            # 1j*0.5*((-1)^m2)       * Y(ell1, m1)*Y(ell2, m2)
            # 1j*0.5                 * Y(ell1, m1)*Y(ell2, -m2)
            # -1j*0.5*((-1)^(m1+m2)) * Y(ell1, -m1))*Y(ell2, m2)
            # -1j*0.5*((-1)^m1)      * Y(ell1, -m1))*Y(ell2, -m2)
            #
            foil_set = [[ell1, mazi1, ell2, mazi2, 1j*0.5*((-1.)**mazi2)],
                        [ell1, mazi1, ell2, -mazi2, 1j*0.5],
                        [ell1, -mazi1, ell2, mazi2,
                         -1j*0.5*((-1.)**(mazi1+mazi2))],
                        [ell1, -mazi1, ell2, -mazi2, -1j*0.5*((-1.)**mazi1)]]
        elif mazi1 > 0 and mazi2 < 0:
            foil_set = [[ell1, mazi1, ell2, mazi2, 1j*0.5*((-1.)**mazi1)],
                        [ell1, mazi1, ell2, -mazi2,
                         -1j*0.5*((-1.)**(mazi1+mazi2))],
                        [ell1, -mazi1, ell2, mazi2, 1j*0.5],
                        [ell1, -mazi1, ell2, -mazi2, -1j*0.5*((-1.)**mazi2)]]
        elif mazi1 > 0 and mazi2 > 0:
            #
            # [  1./sqrt(2)*(((-1)^m1)*Y(ell1, m1)+Y(ell1, -m1))  ]
            #     * [  1./sqrt(2)*(((-1)^m2)*Y(ell2, m2)+Y(ell2, -m2))  ]
            #
            # 0.5*((-1)^(m1+m2)) * Y(ell1, m1)*Y(ell2, m2)
            # 0.5*((-1)^m1)      * Y(ell1, m1)*Y(ell2, -m2)
            # 0.5*((-1)^m2)      * Y(ell1, -m1))*Y(ell2, m2)
            # 0.5                * Y(ell1, -m1))*Y(ell2, -m2)
            #
            foil_set = [[ell1, mazi1, ell2, mazi2, 0.5*((-1.)**(mazi1+mazi2))],
                        [ell1, mazi1, ell2, -mazi2, 0.5*((-1.)**mazi1)],
                        [ell1, -mazi1, ell2, mazi2, 0.5*((-1.)**mazi2)],
                        [ell1, -mazi1, ell2, -mazi2, 0.5]]
        elif mazi1 == 0 and mazi2 < 0:
            # [1j/sqrt(2)*(Y(ell2, m2)-((-1)^m2)*Y(ell2, -m2))]
            foil_set = [[ell1, mazi1, ell2, mazi2, 1j/np.sqrt(2.)],
                        [ell1, mazi1, ell2, -mazi2,
                         -1j/np.sqrt(2.)*((-1.)**mazi2)]]
        elif mazi2 == 0 and mazi1 < 0:
            foil_set = [[ell1, mazi1, ell2, mazi2, 1j/np.sqrt(2.)],
                        [ell1, -mazi1, ell2, mazi2,
                         -1j/np.sqrt(2.)*((-1.)**mazi1)]]
        elif mazi1 == 0 and mazi2 > 0:
            # [1./sqrt(2)*(((-1)^m2)*Y(ell2, m2)+Y(ell2, -m2))]
            foil_set = [[ell1, mazi1, ell2, mazi2,
                         1./np.sqrt(2.)*((-1.)**mazi2)],
                        [ell1, mazi1, ell2, -mazi2, 1./np.sqrt(2.)]]
        elif mazi2 == 0 and mazi1 > 0:
            # [1./sqrt(2)*(((-1)^m2)*Y(ell2, m2)+Y(ell2, -m2))]
            foil_set = [[ell1, mazi1, ell2, mazi2,
                         1./np.sqrt(2.)*((-1.)**mazi1)],
                        [ell1, -mazi1, ell2, mazi2, 1./np.sqrt(2.)]]
        elif mazi1 == 0 and mazi2 == 0:
            foil_set = [[ell1, mazi1, ell2, mazi2, 1.]]
        else:
            raise ValueError('Values for (mazi1, mazi2) not understood')

        reco_list = [[]]
        for entry in foil_set:
            [ell1, mazi1, ell2, mazi2, first_coeff] = entry
            mazi = mazi1+mazi2
            reco = [[]]
            ell_min = np.max([np.abs(ell1-ell2), np.abs(mazi)])
            ell_max = ell1+ell2
            for ell in range(ell_min, ell_max+1):
                tmp = np.sqrt((2.*ell1+1.)*(2.*ell2+1.)
                              / (4.*np.pi*(2.*ell+1.)))\
                    * (CG(ell1, mazi1, ell2, mazi2, ell, mazi).doit())\
                    * (CG(ell1, 0, ell2, 0, ell, 0).doit())
                second_coeff = float(tmp.evalf())
                final_coeff = first_coeff*second_coeff
                reco = reco+[[ell, mazi, final_coeff]]
            reco = reco[1:]
            reco_list = reco_list+reco
        reco_list = reco_list[1:]

        reco_dict = {}
        for entry in reco_list:
            if (entry[0], entry[1]) not in reco_dict.keys():
                reco_dict[(entry[0], entry[1])] = 0.0
        for entry in reco_list:
            [ell, mazi, coeff] = entry
            if mazi < 0:
                reco_dict[(ell, mazi)] = reco_dict[(ell, mazi)]\
                    - 1j*coeff/np.sqrt(2.)
                reco_dict[(ell, -mazi)] = reco_dict[(ell, -mazi)]\
                    + coeff/np.sqrt(2.)
            elif mazi > 0:
                reco_dict[(ell, mazi)] = reco_dict[(ell, mazi)]\
                    + ((-1.)**mazi)*coeff/np.sqrt(2.)
                reco_dict[(ell, -mazi)] = reco_dict[(ell, -mazi)]\
                    + 1j*((-1.)**mazi)*coeff/np.sqrt(2.)
            elif mazi == 0:
                reco_dict[(ell, mazi)] = reco_dict[(ell, mazi)]+coeff
            else:
                raise ValueError('Values for mazi not understood')

        final_reco_list = [[]]
        for key in reco_dict:
            if np.abs(reco_dict[key]) > 1.e-10:
                final_reco_list = final_reco_list+[[key[0], key[1],
                                                    reco_dict[key]]]
        final_reco_list = final_reco_list[1:]
        return final_reco_list

    @staticmethod
    def calY(ell=0, mazi=0, nvec_arr=np.array([[1.0, 2.0, 3.0]]),
             q=1.0, qc_impl={}):
        r"""
        Caligraphic spherical harmonics.

        Parameters
        ----------
        ell (int): orbital angular momentum
        mazi (int): azimuthal component
        nvec_arr (np.ndarray): array of three-vectors
        q (float): on-shell back-to-back momentum magnitude.
        qc_impl (dict): organization, determining exact definition.

        (Also relies on global EPSILON parameter.)

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.

        Returns
        -------
        (np.ndarray)
        """
        if (('real harmonics' not in qc_impl.keys())
           or (qc_impl['real harmonics'])):
            Y = BKFunctions.cart_sph_harm_real(ell, mazi, nvec_arr)
        else:
            Y = BKFunctions.cart_sph_harm(ell, mazi, nvec_arr)
        calY = Y/np.abs(q**ell)
        return calY

    @staticmethod
    def standard_boost(beta_vec=np.array([0.0, 0.0, 0.0]),
                       four_momentum=np.array([1.0, 0.0, 0.0, 0.0])):
        r"""
        Boost of the four momentum with the given beta.

        Designed to be applied as a single boost to a single four-vector.

        Parameters
        ----------
        beta_vec (np.ndarray): boost velocity
        four_momentum (np.ndarray): four-momentum to be boosted

        Returns
        -------
        (np.ndarray)
        """
        betaSQ = beta_vec@beta_vec
        if betaSQ == 0.0:
            return four_momentum
        if betaSQ < 0.0 or betaSQ >= 1.0:
            return np.array(4*[0.0])
        beta = np.sqrt(betaSQ)
        beta_hat = beta_vec/beta
        gamma = np.sqrt(1.0/(1.0-betaSQ))
        momentum_spatial_vec = four_momentum[1:]
        momentum_par_component = momentum_spatial_vec@beta_hat
        momentum_par_vec = momentum_par_component*beta_hat
        momentum_perp_vec = momentum_spatial_vec - momentum_par_vec
        boost_matrix = np.array([[gamma, beta*gamma], [beta*gamma, gamma]])
        momentum_par_unboosted = np.array([[four_momentum[0]],
                                           [momentum_par_component]])
        momentum_par_boosted = boost_matrix@momentum_par_unboosted
        momentum_spatial_vec_boosted = momentum_perp_vec\
            + momentum_par_boosted[1][0]*beta_hat
        four_momentum_boosted = np.array([momentum_par_boosted[0][0],
                                          momentum_spatial_vec_boosted[0],
                                          momentum_spatial_vec_boosted[1],
                                          momentum_spatial_vec_boosted[2]])
        return four_momentum_boosted

    @staticmethod
    def standard_boost_array(beta_vec=np.array([[[0.0, 0.0, 0.0]]]),
                             four_momentum=np.array([[[1.0, 0.0, 0.0, 0.0]]])):
        r"""
        Boost of the four momentum with the given beta.

        Designed to be applied as an array of boosts to an array of
        four-vectors.

        Parameters
        ----------
        beta_vec (np.ndarray): boost velocity
        four_momentum (np.ndarray): four-momentum to be boosted

        Returns
        -------
        (np.ndarray)
        """
        betaSQ = (beta_vec*beta_vec).sum(2)
        if not (np.abs(betaSQ - betaSQ[0][0]*np.ones_like(betaSQ))
                < EPSILON).all():
            raise ValueError('betaSQ in standard_boost_array is not '
                             + 'proportional to a matrix of ones')
        if betaSQ[0][0] == 0.0:
            return four_momentum
        if betaSQ[0][0] < 0.0:
            return np.zeros(four_momentum.shape)
        beta = np.sqrt(betaSQ)
        beta_hat = beta_vec/np.repeat(beta, 3, axis=1).reshape(beta_vec.shape)
        gamma = np.sqrt(1.0/(1.0-betaSQ))
        momentum_spatial_vec = four_momentum[:, :, 1:]
        momentum_par_component = (momentum_spatial_vec*beta_hat).sum(2)
        momentum_par_vec = np.repeat(momentum_par_component, 3, axis=1
                                     ).reshape(beta_hat.shape)*beta_hat
        momentum_perp_vec = momentum_spatial_vec-momentum_par_vec

        boost_matrix = np.transpose(np.array([[gamma, beta*gamma],
                                              [beta*gamma, gamma]]),
                                    axes=(2, 3, 0, 1))
        momentum_par_unboosted = np.concatenate(
            (four_momentum[:, :, 0].reshape(
                four_momentum[:, :, 0].shape+(1,)
                ),
             momentum_par_component.reshape(
                 four_momentum[:, :, 0].shape+(1,)
                 )),
            axis=2)

        momentum_par_boosted = np.einsum('ijkl,ijl->ijk', boost_matrix,
                                         momentum_par_unboosted)

        momentum_spatial_vec_boosted = momentum_perp_vec\
            + np.repeat(
                momentum_par_boosted[:, :, 1], 3, axis=1
                ).reshape(beta_hat.shape)*beta_hat

        four_momentum_boosted = np.concatenate(
            (momentum_par_boosted[:, :, 0].reshape(
                momentum_par_boosted[:, :, 0].shape+(1,)
                ),
                momentum_spatial_vec_boosted),
            axis=2)
        return four_momentum_boosted


class QCFunctions:
    """Class collecting quantization-condition functions."""

    @staticmethod
    def __helperG_single_entry(E, nP, L, np1spec, np2spec, m1, m2, m3):
        Pvec = TWOPI*nP/L
        p1spec_vec = TWOPI*np1spec/L  # called \vec p in 1408.5933
        p2spec_vec = TWOPI*np2spec/L  # called \vec k in 1408.5933
        p1specSQ = p1spec_vec@p1spec_vec
        p2specSQ = p2spec_vec@p2spec_vec
        omega_p1spec = np.sqrt(m3**2+p1specSQ)
        omega_p2spec = np.sqrt(m1**2+p2specSQ)
        # Following is called \vec \beta_p in 1408.5933
        beta_for1 = -(Pvec-p1spec_vec)/(E-omega_p1spec)
        # Following is called \vec \beta_k in 1408.5933
        beta_for2 = -(Pvec-p2spec_vec)/(E-omega_p2spec)

        # Following is called k^\mu in 1408.5933
        fourmom_for1 = np.concatenate(([omega_p2spec], p2spec_vec))
        # Following is called \vec k^* in 1408.5933
        vecstar_for1 = BKFunctions.standard_boost(beta_for1, fourmom_for1)[1:]
        # Following is called p^\mu in 1408.5933
        fourmom_for2 = np.concatenate(([omega_p1spec], p1spec_vec))
        # Following is called \vec p^* in 1408.5933
        vecstar_for2 = BKFunctions.standard_boost(beta_for2, fourmom_for2)[1:]

        E2CMSQ_for1 = (E-omega_p1spec)**2-(Pvec-p1spec_vec)@(Pvec-p1spec_vec)
        E2CMSQ_for2 = (E-omega_p2spec)**2-(Pvec-p2spec_vec)@(Pvec-p2spec_vec)

        if m1 == m2:
            qSQ_for1 = E2CMSQ_for1/4.0-m1**2
        else:
            qSQ_for1 = (E2CMSQ_for1**2-2.0*E2CMSQ_for1*m1**2
                        + m1**4-2.0*E2CMSQ_for1*m2**2-2.0*m1**2*m2**2+m2**4)\
                    / (4.0*E2CMSQ_for1)
        if m2 == m3:
            qSQ_for2 = E2CMSQ_for2/4.0-m3**2
        else:
            qSQ_for2 = (E2CMSQ_for2**2-2.0*E2CMSQ_for2*m3**2
                        + m3**4-2.0*E2CMSQ_for2*m2**2-2.0*m3**2*m2**2+m2**4)\
                    / (4.0*E2CMSQ_for2)

        q_for1 = np.sqrt(qSQ_for1+0.*1j)
        q_for2 = np.sqrt(qSQ_for2+0.*1j)
        if np.abs(q_for1.imag) < EPSILON:
            q_for1 = q_for1.real
        if np.abs(q_for2.imag) < EPSILON:
            q_for2 = q_for2.real
        return [vecstar_for1, vecstar_for2, E2CMSQ_for1,
                E2CMSQ_for2, q_for1, q_for2]

    @staticmethod
    def getG_single_entry(E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                          np1spec=np.array([0, 0, 0]),
                          np2spec=np.array([0, 0, 0]),
                          ell1=0, mazi1=0, ell2=0, mazi2=0,
                          m1=1.0, m2=1.0, m3=1.0,
                          alpha=-1.0, beta=0.0,
                          J_slow=False,
                          three_scheme='relativistic pole',
                          qc_impl={}):
        """
        Evaluate single entry of G.

        .......................................................


                  @@@  --(p1=k=p2spec)--(m1)-------
        p1spec    @@@@                                 p2spec
        ell1      @@  ---(p2=P-p1-p3)---(m2)--  @@@    ell2
        mazi1                                  @@@@    mazi2
                  -------(p3=p=p1spec)--(m3)--  @@@


        .......................................................

        Parameters
        ----------
        E (float): energy
        nP (np.ndarray): dimensionless momentum
        L (float): box-length
        np1spec (np.ndarray): first spectator
        np2spec (np.ndarray): second spectator
        ell1 (int): first angular momentum
        mazi1 (int): first azimuthal
        ell2 (int): second angular momentum
        mazi2 (int): second azimuthal
        m1 (float): first mass
        m2 (float): second mass
        m3 (float): third mass
        alpha (float): first shape parameter
        beta (float): second shape parameter
        J_slow (boolean): switch for version of J
        three_scheme (str): scheme for three-body interaction
        qc_impl (dict): scheme for organizing quantization condition

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.

        Returns
        -------
        (float)
        """
        [vecstar_for1, vecstar_for2, E2CMSQ_for1, E2CMSQ_for2, q_for1, q_for2]\
            = QCFunctions.__helperG_single_entry(E, nP, L,
                                                 np1spec, np2spec, m1, m2, m3)

        calY1 = BKFunctions.calY(ell1, mazi1,
                                 vecstar_for1.reshape((1, 3)),
                                 q_for1, qc_impl)[0]
        calY2 = BKFunctions.calY(ell2, mazi2,
                                 vecstar_for2.reshape((1, 3)),
                                 q_for2, qc_impl)[0]
        calY2conj = np.conjugate(calY2)

        HH = BKFunctions.H(E2CMSQ_for1, m1+m2, alpha, beta, J_slow)\
            * BKFunctions.H(E2CMSQ_for2, m2+m3, alpha, beta, J_slow)

        Pvec = TWOPI*nP/L
        p1vec = TWOPI*np2spec/L
        p3vec = TWOPI*np1spec/L
        p2vec = Pvec-p1vec-p3vec
        omega1 = np.sqrt(m1**2+p1vec@p1vec)
        omega2 = np.sqrt(m2**2+p2vec@p2vec)
        omega3 = np.sqrt(m3**2+p3vec@p3vec)
        if three_scheme == 'original pole':
            simple_factor = 1.0/(4.0*omega2*omega1*L**3)
        elif three_scheme == 'relativistic pole':
            simple_factor = 1.0/(2.0*omega1*L**3*(E-omega1-omega3+omega2))
        else:
            raise ValueError('three_scheme not recognized')

        if (('hermitian' not in qc_impl.keys())
           or (qc_impl['hermitian'])):
            simple_factor = simple_factor/(2.0*omega3*L**3)

        pole_factor = 1.0/(E-omega1-omega2-omega3)
        return calY1*calY2conj*HH*simple_factor*pole_factor

    @staticmethod
    def get_nvec_data(tbks_entry, row_slicing, col_slicing):
        """Get the data for the nvecs."""
        n1vec_arr_slice = tbks_entry.nvec_arr[row_slicing[0]:row_slicing[1]]
        n1vecSQ_arr_slice = tbks_entry.nvecSQ_arr[
            row_slicing[0]:row_slicing[1]]

        n2vec_arr_slice = tbks_entry.nvec_arr[col_slicing[0]:col_slicing[1]]
        n2vecSQ_arr_slice = tbks_entry.nvecSQ_arr[
            col_slicing[0]:col_slicing[1]]

        # Awkward swap here
        n1vec_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n2vec_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        n2vec_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n1vec_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        n3vec_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n3vec_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        n1vecSQ_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n2vecSQ_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        n2vecSQ_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n1vecSQ_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        n3vecSQ_mat_slice = np.swapaxes(
            np.swapaxes(
                ((tbks_entry.n3vecSQ_mat)[
                    row_slicing[0]:row_slicing[1]]),
                0, 1
                )[col_slicing[0]:col_slicing[1]],
            0, 1
            )

        return [n1vec_arr_slice, n1vecSQ_arr_slice,
                n2vec_arr_slice, n2vecSQ_arr_slice,
                n1vec_mat_slice, n2vec_mat_slice, n3vec_mat_slice,
                n1vecSQ_mat_slice, n2vecSQ_mat_slice, n3vecSQ_mat_slice]

    @staticmethod
    def __helperG_array(E, nP, L, m1, m2, m3,
                        tbks_entry,
                        row_slicing,
                        col_slicing):
        [n1vec_arr_slice, n1vecSQ_arr_slice,
         n2vec_arr_slice, n2vecSQ_arr_slice,
         n1vec_mat_slice, n2vec_mat_slice, n3vec_mat_slice,
         n1vecSQ_mat_slice, n2vecSQ_mat_slice, n3vecSQ_mat_slice]\
            = QCFunctions.get_nvec_data(tbks_entry,
                                        row_slicing, col_slicing)
        Pvec = TWOPI*nP/L
        p1specvec_arr_slice\
            = TWOPI*n1vec_arr_slice/L  # called \vec p in 1408.5933
        p2specvec_arr_slice\
            = TWOPI*n2vec_arr_slice/L  # called \vec k in 1408.5933
        p1specvecSQ_arr_slice = (TWOPI**2)*n1vecSQ_arr_slice/L**2
        p2specvecSQ_arr_slice = (TWOPI**2)*n2vecSQ_arr_slice/L**2
        omegap1spec_arr_slice = np.sqrt(m3**2+p1specvecSQ_arr_slice)
        omegap2spec_arr_slice = np.sqrt(m1**2+p2specvecSQ_arr_slice)

        p1specvec_mat_slice\
            = TWOPI*n1vec_mat_slice/L  # called \vec p in 1408.5933
        p2specvec_mat_slice\
            = TWOPI*n2vec_mat_slice/L  # called \vec k in 1408.5933
        p1specvecSQ_mat_slice = (TWOPI**2)*n1vecSQ_mat_slice/L**2
        p2specvecSQ_mat_slice = (TWOPI**2)*n2vecSQ_mat_slice/L**2
        omegap1spec_mat_slice = np.sqrt(m3**2+p1specvecSQ_mat_slice)
        omegap2spec_mat_slice = np.sqrt(m1**2+p2specvecSQ_mat_slice)

        # Following is called \vec \beta_p in 1408.5933
        beta_for1 = -(Pvec-p1specvec_mat_slice)/np.repeat(
            E-omegap1spec_mat_slice, 3, axis=1
            ).reshape((Pvec-p1specvec_mat_slice).shape)

        # Following is called \vec \beta_k in 1408.5933
        beta_for2 = -(Pvec-p2specvec_mat_slice)/np.repeat(
            E-omegap2spec_mat_slice, 3, axis=1
            ).reshape((Pvec-p2specvec_mat_slice).shape)

        # Following is called k^\mu in 1408.5933
        fourmom_for1 = np.concatenate(
            (omegap2spec_mat_slice.reshape(
                omegap2spec_mat_slice.shape+(1,)), p2specvec_mat_slice),
            axis=2)

        # Following is called \vec k^* in 1408.5933
        vecstar_for1 = BKFunctions.standard_boost_array(beta_for1,
                                                        fourmom_for1)[:, :, 1:]

        # Following is called p^\mu in 1408.5933
        fourmom_for2 = np.concatenate(
            (omegap1spec_mat_slice.reshape(omegap1spec_mat_slice.shape+(1,)),
             p1specvec_mat_slice),
            axis=2)

        # Following is called \vec p^* in 1408.5933
        vecstar_for2 = BKFunctions.standard_boost_array(beta_for2,
                                                        fourmom_for2)[:, :, 1:]

        E2CMSQ_for1 = (E-omegap1spec_arr_slice)**2\
            - ((Pvec-p1specvec_arr_slice)*(Pvec-p1specvec_arr_slice)).sum(1)
        E2CMSQ_for2 = (E-omegap2spec_arr_slice)**2\
            - ((Pvec-p2specvec_arr_slice)*(Pvec-p2specvec_arr_slice)).sum(1)

        if m1 == m2:
            qSQ_for1 = E2CMSQ_for1/4.0-m1**2
        else:
            qSQ_for1 = (E2CMSQ_for1**2-2.0*E2CMSQ_for1*m1**2
                        + m1**4-2.0*E2CMSQ_for1*m2**2-2.0*m1**2*m2**2+m2**4)\
                    / (4.0*E2CMSQ_for1)
        if m2 == m3:
            qSQ_for2 = E2CMSQ_for2/4.0-m3**2
        else:
            qSQ_for2 = (E2CMSQ_for2**2-2.0*E2CMSQ_for2*m3**2
                        + m3**4-2.0*E2CMSQ_for2*m2**2-2.0*m3**2*m2**2+m2**4)\
                    / (4.0*E2CMSQ_for2)

        q_for1 = np.sqrt(qSQ_for1+0.*1j)
        q_for2 = np.sqrt(qSQ_for2+0.*1j)

        q_for1_mat = np.repeat(q_for1.reshape(q_for1.shape+(1,)),
                               vecstar_for1.shape[1], axis=1)
        q_for2_mat = np.repeat(q_for2.reshape((1,)+q_for2.shape),
                               vecstar_for2.shape[0], axis=0)

        return [vecstar_for1, vecstar_for2, E2CMSQ_for1,
                E2CMSQ_for2, q_for1, q_for2, q_for1_mat, q_for2_mat,
                omegap1spec_mat_slice, omegap2spec_mat_slice,
                omegap1spec_arr_slice, omegap2spec_arr_slice,
                n3vecSQ_mat_slice]

    @staticmethod
    def getG_array(E, nP, L, m1, m2, m3,
                   tbks_entry,
                   row_slicing, col_slicing,
                   ell1, ell2,
                   alpha, beta,
                   qc_impl, three_scheme,
                   g_rescale):
        """
        Get G, numpy accelerated.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.

        three_scheme is drawn from the following:
            'original pole'
            'relativistic pole'
        """
        J_slow = False
        [vecstar_for1, vecstar_for2, E2CMSQ_for1,
         E2CMSQ_for2, q_for1, q_for2, q_for1_mat, q_for2_mat,
         omegap1spec_mat_slice, omegap2spec_mat_slice,
         omegap1spec_arr_slice, omegap2spec_arr_slice,
         n3vecSQ_mat_slice]\
            = QCFunctions.__helperG_array(E, nP, L, m1, m2, m3,
                                          tbks_entry,
                                          row_slicing,
                                          col_slicing)

        shape1_tmp = vecstar_for1.shape
        r2_shape = shape1_tmp[:-1]

        calY1mat = [[]]
        calY2conjmat = [[]]
        for mazi1 in np.arange(-ell1, ell1+1):
            calY1row = []
            calY2conjrow = []
            for mazi2 in np.arange(-ell2, ell2+1):
                calY1 = BKFunctions.calY(ell1, mazi1,
                                         vecstar_for1.reshape(
                                             (shape1_tmp[0]*shape1_tmp[1], 3)),
                                         q_for1_mat.reshape(q_for1_mat.size),
                                         qc_impl)
                shape2_tmp = vecstar_for2.shape
                calY2 = BKFunctions.calY(ell2, mazi2,
                                         vecstar_for2.reshape(
                                             (shape2_tmp[0]*shape2_tmp[1], 3)),
                                         q_for2_mat.reshape(q_for2_mat.size),
                                         qc_impl)
                calY2conj = np.conjugate(calY2)
                calY1 = (calY1).reshape(r2_shape)
                calY1row = calY1row+[calY1]
                calY2conj = (calY2conj).reshape(r2_shape)
                calY2conjrow = calY2conjrow+[calY2conj]
            calY1mat = calY1mat+[calY1row]
            calY2conjmat = calY2conjmat+[calY2conjrow]

        calY1mat = np.transpose(np.array(calY1mat[1:]), axes=(2, 0, 3, 1))
        Y1shapetmp = calY1mat.shape
        calY1mat = calY1mat.reshape((Y1shapetmp[0], Y1shapetmp[1],
                                     Y1shapetmp[2]*Y1shapetmp[3]))
        calY1mat = np.transpose(calY1mat, axes=(2, 0, 1))
        Y1shapetmp = calY1mat.shape
        calY1mat = calY1mat.reshape((Y1shapetmp[0],
                                     Y1shapetmp[1]*Y1shapetmp[2]))
        calY1mat = np.transpose(calY1mat)

        calY2conjmat = np.transpose(np.array(calY2conjmat[1:]),
                                    axes=(2, 0, 3, 1))
        Y2conjshapetmp = calY2conjmat.shape
        calY2conjmat = calY2conjmat.reshape((Y2conjshapetmp[0],
                                             Y2conjshapetmp[1],
                                             Y2conjshapetmp[2]
                                             * Y2conjshapetmp[3]))
        calY2conjmat = np.transpose(calY2conjmat, axes=(2, 0, 1))
        Y2conjshapetmp = calY2conjmat.shape
        calY2conjmat = calY2conjmat.reshape((Y2conjshapetmp[0],
                                             Y2conjshapetmp[1]
                                             * Y2conjshapetmp[2]))
        calY2conjmat = np.transpose(calY2conjmat)
        YY = calY1mat*calY2conjmat

        H1 = BKFunctions.H(E2CMSQ_for1.reshape(E2CMSQ_for1.size), m1+m2,
                           alpha, beta, J_slow)
        H2 = BKFunctions.H(E2CMSQ_for2.reshape(E2CMSQ_for2.size), m2+m3,
                           alpha, beta, J_slow)

        omega1_mat = omegap2spec_mat_slice
        omega2_mat = np.sqrt(m2**2+FOURPI2*n3vecSQ_mat_slice/L**2)
        omega3_mat = omegap1spec_mat_slice

        if three_scheme == 'original pole':
            simple_factor_mat = 1.0/(2.0*omega1_mat*omega2_mat*L**3)
        elif three_scheme == 'relativistic pole':
            simple_factor_mat = 1.0/(2.0*omega1_mat*L**3
                                     * (E-omega1_mat-omega3_mat+omega2_mat))
        else:
            raise ValueError('three_scheme not recognized')

        if (('hermitian' not in qc_impl.keys())
           or (qc_impl['hermitian'])):
            simple_factor_mat = simple_factor_mat/(2.0*omega3_mat*L**3)

        pole_factor = 1.0/(E-omega1_mat-omega2_mat-omega3_mat)
        full_mat = simple_factor_mat*pole_factor
        full_mat_big = np.repeat(np.repeat(full_mat, 2*ell1+1, axis=0),
                                 2*ell2+1, axis=1)

        H1_mat = np.repeat((np.repeat(H2.reshape((1,)+H2.shape),
                                      (full_mat_big.shape)[0], axis=0)),
                           2*ell2+1, axis=1)
        H2_mat = np.repeat((np.repeat(H1.reshape(H1.shape+(1,)),
                                      (full_mat_big.shape)[1], axis=1)),
                           2*ell1+1, axis=0)
        return YY*full_mat_big*H1_mat*H2_mat*g_rescale

    @staticmethod
    def summand(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0,
                nvec_arr=np.array([[0, 0, 0]]), alphaKSS=1.0,
                ell1=0, mazi1=0, ell2=0, mazi2=0,
                qc_impl={}):
        """
        Regulated sum entering the F function.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.
        """
        nP2SQ = nP2@nP2
        nP2mag = np.sqrt(nP2SQ)
        q = np.sqrt(np.abs(qSQ))
        sph_harm_value = 1.0
        if nP2SQ == 0.0:
            rSQ_arr = (nvec_arr**2).sum(1)
            if ell1 != 0:
                calY1 = BKFunctions.calY(ell1, mazi1, nvec_arr,
                                         q, qc_impl)
                sph_harm_value = sph_harm_value*calY1

            if ell2 != 0:
                calY2 = BKFunctions.calY(ell2, mazi2, nvec_arr,
                                         q, qc_impl)
                calY2conj = np.conjugate(calY2)
                sph_harm_value = sph_harm_value*calY2conj

            if ((ell1 == ell2) and (mazi1 == mazi2)):
                sph_harm_value = sph_harm_value\
                    - (rSQ_arr**ell1/q**(2*ell1)-1.0)

        else:
            if (ell1 == 0 and ell2 == 0):
                npar_component_arr = ((nvec_arr*nP2).sum(1))/nP2mag
                rparSQ_arr = (npar_component_arr-nP2mag/2.0)**2/gamSQ
                nP2_hat = nP2/nP2mag
                npar_vec_arr = np.dot(np.transpose([npar_component_arr]),
                                      [nP2_hat])
                rperpSQ_arr = ((nvec_arr-npar_vec_arr)**2).sum(1)
                rSQ_arr = rparSQ_arr+rperpSQ_arr
            else:
                npar_component_arrtmp = ((nvec_arr*nP2).sum(1))/nP2mag
                npar_component_arr\
                    = npar_component_arrtmp.reshape(
                        (len(npar_component_arrtmp),
                         1))
                nP2_hat = nP2/nP2mag
                npar_vec_arr = nP2_hat*npar_component_arr
                rpar_vec_arr = (npar_vec_arr - nP2/2.0)/np.sqrt(gamSQ)
                rperp_vec_arr = nvec_arr - npar_vec_arr
                rvec_arr = rpar_vec_arr + rperp_vec_arr
                rSQ_arr = (rvec_arr**2).sum(1)
                if ell1 != 0:
                    calY1 = BKFunctions.calY(ell1, mazi1, rvec_arr,
                                             q, qc_impl)
                    sph_harm_value = sph_harm_value*calY1
                if ell2 != 0:
                    calY2 = BKFunctions.calY(ell2, mazi2, rvec_arr,
                                             q, qc_impl)
                    calY2conj = np.conjugate(calY2)
                    sph_harm_value = sph_harm_value*calY2conj
                if ((ell1 == ell2) and (mazi1 == mazi2)):
                    sph_harm_value = sph_harm_value\
                        - (rSQ_arr**ell1/q**(2*ell1)-1.0)
        Ds = rSQ_arr-qSQ
        return sph_harm_value*np.exp(-alphaKSS*Ds)/Ds

    @staticmethod
    def __T1(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0, C1cut=3,
             alphaKSS=1.0, ell1=0, mazi1=0, ell2=0, mazi2=0,
             qc_impl={}):
        rng = range(-C1cut, C1cut+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        return np.sum(QCFunctions.summand(nP2, qSQ, gamSQ, nvec_arr, alphaKSS,
                                          ell1, mazi1, ell2, mazi2,
                                          qc_impl))/ROOT4PI

    @staticmethod
    def __T2(qSQ=1.5, gamSQ=1.0, alphaKSS=1.0,
             ell1=0, mazi1=0, ell2=0, mazi2=0):
        if ((ell1 == ell2) and (mazi1 == mazi2)):
            gamma = np.sqrt(gamSQ)
            if qSQ >= 0:
                ttmp = 2.0*(np.pi**2)*np.sqrt(qSQ)\
                      * erfi(np.sqrt(alphaKSS*qSQ))\
                      - 2.0*np.exp(alphaKSS*qSQ)\
                      * np.sqrt(np.pi**3)/np.sqrt(alphaKSS)
            else:
                ttmp = -2.0*(np.pi**2)*np.sqrt(-qSQ)\
                      * erf(np.sqrt(-alphaKSS*qSQ))\
                      - 2.0*np.exp(alphaKSS*qSQ)\
                      * np.sqrt(np.pi**3)/np.sqrt(alphaKSS)
            return gamma*ttmp/np.sqrt(2.0*TWOPI)
        else:
            return 0.0

    @staticmethod
    def getZ_single_entry(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0,
                          C1cut=3, alphaKSS=1.0,
                          ell1=0, mazi1=0, ell2=0, mazi2=0,
                          qc_impl={}):
        r"""Evaluate a single entry of \\(Z\\)."""
        return QCFunctions.__T1(nP2, qSQ, gamSQ, C1cut, alphaKSS,
                                ell1, mazi1, ell2, mazi2, qc_impl)\
            + QCFunctions.__T2(qSQ, gamSQ, alphaKSS, ell1, mazi1, ell2, mazi2)

    @staticmethod
    def getFtwo_single_entry(E2=3.0, nP2=np.array([0, 0, 0]), L=5.0,
                             m1=1.0, m2=1.0, C1cut=3, alphaKSS=1.0,
                             ell1=0, mazi1=0, ell2=0, mazi2=0,
                             qc_impl={}):
        r"""Evaluate a single entry of \\(F_2\\)."""
        P2 = TWOPI*nP2/L
        E2SQ = E2**2
        P2SQ = P2@P2
        E2CMSQ = E2SQ-P2SQ
        gamSQ = E2SQ/E2CMSQ
        if m1 == m2:
            qSQ = E2CMSQ/4.0-m1**2
            qSQ_dimless = (L**2)*(qSQ)/FOURPI2
        else:
            qSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
                   + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
                / (4.0*E2CMSQ)
            qSQ_dimless = (L**2)*(qSQ)/FOURPI2
        if E2CMSQ < 0.0:
            return 0.0
        E2CM = np.sqrt(E2CMSQ)
        gamma = np.sqrt(gamSQ)
        pre = -2.0/(L*np.sqrt(PI)*16.0*PI*E2CM*gamma)
        return pre*(QCFunctions.getZ_single_entry(nP2, qSQ_dimless, gamSQ,
                                                  C1cut, alphaKSS,
                                                  ell1, mazi1, ell2, mazi2,
                                                  qc_impl))

    @staticmethod
    def getF_single_entry(E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                          npspec=np.array([0, 0, 0]),
                          m1=1.0, m2=1.0, mspec=1.0,
                          C1cut=3, alphaKSS=1.0, alpha=-1.0, beta=0.0,
                          ell1=0, mazi1=0, ell2=0, mazi2=0,
                          three_scheme='relativistic pole',
                          qc_impl={}):
        """
        Evaluate a single entry of F.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.
        """
        nP2 = nP - npspec
        pspec = TWOPI*npspec/L
        pspecSQ = pspec@pspec
        omspec = np.sqrt(pspecSQ+mspec**2)
        E2 = E-omspec
        P2 = TWOPI*nP2/L
        E2SQ = E2**2
        P2SQ = P2@P2
        E2CMSQ = E2SQ-P2SQ
        if (E2CMSQ < 0.0) or (E2 < 0.0):
            return 0.0
        gamSQ = E2SQ/E2CMSQ
        if m1 == m2:
            qSQ = E2CMSQ/4.0-m1**2
            qSQ_dimless = (L**2)*(qSQ)/FOURPI2
        else:
            qSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
                   + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
                / (4.0*E2CMSQ)
            qSQ_dimless = (L**2)*(qSQ)/FOURPI2
        E2CM = np.sqrt(E2CMSQ)
        gamma = np.sqrt(gamSQ)
        Htmp = BKFunctions.H(E2CMSQ, m1+m2, alpha, beta)
        pre = -Htmp*2.0/(L*np.sqrt(PI)*16.0*PI*E2CM*gamma)
        if ('hermitian' not in qc_impl.keys()) or (qc_impl['hermitian']):
            pre = pre/(2.0*omspec*L**3)
        return pre*(QCFunctions.getZ_single_entry(nP2, qSQ_dimless, gamSQ,
                                                  C1cut, alphaKSS,
                                                  ell1, mazi1,
                                                  ell2, mazi2, qc_impl))

    def getF_array(E, nP, L, m1, m2, m3, tbks_entry, slice_entry,
                   ell1, ell2, alpha, beta, C1cut, alphaKSS, qc_impl, ts):
        """
        Get F, numpy accelerated.

        three_scheme is drawn from the following:
            'original pole'
            'relativistic pole'
        """
        nvec_arr_slice = tbks_entry.nvec_arr[slice_entry[0]:slice_entry[1]]
        f_list = []
        for nvec in nvec_arr_slice:
            f_mat_entry = [[]]
            for mazi1 in range(-ell1, ell1+1):
                f_row = []
                for mazi2 in range(-ell2, ell2+1):
                    # Awkward notation for masses here
                    f_tmp = QCFunctions.getF_single_entry(E=E, nP=nP, L=L,
                                                          npspec=nvec,
                                                          m1=m2, m2=m3,
                                                          mspec=m1,
                                                          C1cut=C1cut,
                                                          alphaKSS=alphaKSS,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          ell1=ell1,
                                                          mazi1=mazi1,
                                                          ell2=ell2,
                                                          mazi2=mazi2,
                                                          three_scheme=ts,
                                                          qc_impl=qc_impl)
                    if np.abs(f_tmp.imag) < EPSILON:
                        f_tmp = f_tmp.real
                    if np.abs(f_tmp) < EPSILON:
                        f_tmp = 0.0
                    f_row = f_row+[f_tmp]
                f_mat_entry = f_mat_entry+[f_row]
            f_mat_entry = np.array(f_mat_entry[1:])
            f_list = f_list+[f_mat_entry]
        return block_diag(*f_list)

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

    def pcotdelta_scattering_length_str():
        """Print behavior for pcotdelta_scattering_length."""
        return "pcotdelta_scattering_length"

    @staticmethod
    @with_str(pcotdelta_scattering_length_str)
    def pcotdelta_scattering_length(pSQ=1.5, a=1.0):
        r"""Scattering function \\(p cot \delta\\), scattering-length only."""
        return -1.0/a

    def pcotdelta_breit_wigner_str():
        """Print behavior for pcotdelta_breit_wigner."""
        return "pcotdelta_breit_wigner"

    @staticmethod
    @with_str(pcotdelta_breit_wigner_str)
    def pcotdelta_breit_wigner(pSQ=1.5, g_value=6.0, mrho_value=3.0):
        """
        Scattering function, Breit-Wigner.

        Warning: This is multiplied by a factor of pSQ to cancel the threshold
        scaling, which is included elsewhere.
        """
        Ecm = 2.0*np.sqrt(1.0+pSQ)
        GammaEcmop = g_value**2/(6.0*np.pi)*((pSQ))/Ecm**2
        tandop = GammaEcmop*Ecm/(mrho_value**2-Ecm**2)
        return pSQ/tandop

    @staticmethod
    def getK_single_entry(pcotdelta_function=None,
                          pcotdelta_parameters=[1.0],
                          E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                          npspec=np.array([0, 0, 0]),
                          m1=1.0, m2=1.0, mspec=1.0,
                          alpha=-1.0, beta=0.0,
                          ell=0,
                          qc_impl={}):
        """
        Evluate a single entry of K.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.
        """
        if pcotdelta_function is None:
            pcotdelta_function = QCFunctions.pcotdelta_scattering_length
        P = TWOPI*nP/L
        pspec = TWOPI*npspec/L
        omspec = np.sqrt(mspec**2+pspec@pspec)
        E2 = E-omspec
        P2 = P-pspec
        E2CMSQ = E2**2-P2@P2
        if E2CMSQ <= 0.0 or E2 < 0.0:
            return np.inf
        ECM = np.sqrt(E2CMSQ)
        if m1 == m2:
            pSQ = E2CMSQ/4.0-m1**2
        else:
            pSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
                   + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
                / (4.0*E2CMSQ)
        pcotdelta = pcotdelta_function(pSQ, *pcotdelta_parameters)
        q_one_minus_H_tmp = BKFunctions.q_one_minus_H(E2CMSQ=E2CMSQ,
                                                      m1=m1, m2=m2,
                                                      alpha=alpha,
                                                      beta=beta)
        pre = 1.0
        if ('hermitian' not in qc_impl.keys()) or (qc_impl['hermitian']):
            pre = pre*(2.0*omspec*L**3)
        pcotdelta = pcotdelta/np.abs(pSQ**(ell))
        return pre*16.0*PI*ECM/(pcotdelta+q_one_minus_H_tmp)

    def getK_array(E, nP, L, m1, m2, m3,
                   tbks_entry,
                   slice_entry,
                   ell,
                   pcotdelta_function,
                   pcotdelta_parameter_list,
                   alpha, beta,
                   qc_impl, ts):
        """
        Get K, numpy accelerated.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.

        three_scheme is drawn from the following:
            'original pole'
            'relativistic pole'
        """
        nvec_arr_slice = tbks_entry.nvec_arr[slice_entry[0]:slice_entry[1]]
        k_list = []
        for nvec in nvec_arr_slice:
            k_tmp = QCFunctions.getK_single_entry(pcotdelta_function=pcotdelta_function,
                                                  pcotdelta_parameters=pcotdelta_parameter_list,
                                                  E=E, nP=nP, L=L,
                                                  npspec=nvec,
                                                  m1=m2, m2=m3, mspec=m1,
                                                  alpha=alpha, beta=beta,
                                                  ell=ell,
                                                  qc_impl=qc_impl)
            if np.abs(k_tmp.imag) < EPSILON:
                k_tmp = k_tmp.real
            if np.abs(k_tmp) < EPSILON:
                k_tmp = 0.0
            k_list = k_list+[k_tmp]*(2*ell+1)
        return block_diag(*k_list)
