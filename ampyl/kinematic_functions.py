#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# kinematic_functions.py
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
from scipy.special import sph_harm
from sympy.physics.quantum.cg import CG

from .constants import EPSILON15
from .constants import PI
from .constants import QC_IMPL_DEFAULTS
from .constants import R4PI


def J_slow(z=0.5):
    r"""Return the slow implementation of the cutoff function ``J(z)``.

    The function smoothly interpolates between ``0`` for ``z < 0`` and
    ``1`` for ``z > 1``.

    Parameters
    ----------
    z : float or numpy.ndarray, optional
        Cutoff-function argument.

    Returns
    -------
    float or numpy.ndarray
        Value of the cutoff function evaluated at ``z``.

    Raises
    ------
    ValueError
        If ``z`` is neither a float nor a NumPy array.
    """
    if isinstance(z, np.ndarray):
        J_array = np.array([])
        for z_val in z:
            if z_val <= 0.0:
                J_array = np.append(J_array, 0.0)
            elif z_val >= 1.0:
                J_array = np.append(J_array, 1.0)
            else:
                J_array = np.append(
                    J_array,
                    np.exp(-1.0/z_val*np.exp(-1.0/(1.0-z_val)))
                    )
        return J_array
    if isinstance(z, float):
        if z <= 0:
            return 0.0
        if z >= 1.0:
            return 1.0
        return np.exp(-1.0/z*np.exp(-1.0/(1.0-z)))
    raise ValueError("z must be a float or np.ndarray")

def J(z=np.array([0.5])):
    r"""Return the vectorized cutoff function ``J(z)``.

    The function smoothly interpolates between ``0`` for ``z < 0`` and
    ``1`` for ``z > 1``.

    Parameters
    ----------
    z : float or numpy.ndarray, optional
        Cutoff-function argument.

    Returns
    -------
    float or numpy.ndarray
        Value of the cutoff function evaluated at ``z``.

    Raises
    ------
    ValueError
        If ``z`` is neither a float nor a NumPy array.
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
    raise ValueError("z must be a float or np.ndarray")

def H(E2CMSQ=9.0, threshold=2.0, alpha=-1.0, beta=0.0, J_slow=False):
    r"""Return the kinematic cutoff function ``H``.

    This function is built from ``J(z)`` and ranges from ``0`` below a
    chosen two-particle CMF energy to ``1`` above threshold.

    Parameters
    ----------
    E2CMSQ : float or numpy.ndarray, optional
        Squared two-particle center-of-mass energy.
    threshold : float, optional
        Two-particle threshold value.
    alpha : float, optional
        Width parameter. The standard choice is ``-1.0``.
    beta : float, optional
        Shift parameter. The standard choice is ``0.0``.
    J_slow : bool, optional
        If ``True``, use :meth:`J_slow` instead of :meth:`J`.

    Returns
    -------
    float or numpy.ndarray
        Value of the cutoff function.
    """
    z = (E2CMSQ-(1.0+alpha)*threshold**2/4.0)\
        / ((3.0-alpha)*threshold**2/4.0)+beta
    if J_slow:
        return globals()['J_slow'](z)
    return J(z)

def phase_space(E2CMSQ=9.0, omk=1.0):
    r"""Return the two-body phase space including ``2\omega_k``.

    Parameters
    ----------
    E2CMSQ : float or numpy.ndarray, optional
        Squared two-particle center-of-mass energy.
    omk : float or numpy.ndarray, optional
        Time component of the four-vector ``k``.

    Returns
    -------
    float or numpy.ndarray
        Two-body phase-space factor.
    """
    return 1.0/(16.0*PI*np.sqrt(E2CMSQ))/(2.0*omk)

def phase_space_alt(omk=1.0, m=1.0):
    r"""Return the alternate definition of the two-body phase space.

    Parameters
    ----------
    omk : float or numpy.ndarray, optional
        Time component of the four-vector ``k``.
    m : float or numpy.ndarray, optional
        Mass entering the alternate normalization.

    Returns
    -------
    float or numpy.ndarray
        Two-body phase-space factor.
    """
    return 1.0/(32.0*PI*m)/(2.0*omk)

def q_one_minus_H(E2CMSQ=9.0, m1=1.0, m2=1.0, alpha=-1.0, beta=0.0,
                  J_slow=False):
    r"""Return the term ``|q| * (1 - H(...))`` relating ``K2`` and ``M2``.

    Parameters
    ----------
    E2CMSQ : float or numpy.ndarray, optional
        Squared two-particle center-of-mass energy.
    m1 : float or numpy.ndarray, optional
        First mass.
    m2 : float or numpy.ndarray, optional
        Second mass.
    alpha : float, optional
        Width parameter. The standard choice is ``-1.0``.
    beta : float, optional
        Shift parameter. The standard choice is ``0.0``.
    J_slow : bool, optional
        If ``True``, use :meth:`J_slow` instead of :meth:`J`.

    Returns
    -------
    float or numpy.ndarray
        Value of ``|q| * (1 - H(...))``.
    """
    threshold = m1+m2
    if m1 == m2:
        qCMSQ = E2CMSQ/4.0-m1**2
    else:
        qCMSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
                 + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*E2CMSQ)
    qCM = np.sqrt(np.abs(qCMSQ))
    return qCM*(1.0-H(E2CMSQ, threshold, alpha, beta, J_slow))

def cart_sph_harm(ell=0, mazi=0,
                  nvec_arr=np.array([[1.0, 2.0, 3.0]])):
    r"""Return Cartesian spherical harmonics.

    The normalization includes a factor of ``\sqrt{4\pi}``, so
    ``Y_{00} == 1``.

    Parameters
    ----------
    ell : int, optional
        Orbital angular momentum.
    mazi : int, optional
        Azimuthal component.
    nvec_arr : numpy.ndarray, optional
        Array of three-vectors. This routine also relies on the global
        constant ``EPSILON15`` when building the angular coordinates.

    Returns
    -------
    numpy.ndarray
        Complex Cartesian spherical harmonics evaluated on ``nvec_arr``.
    """
    nxs = (nvec_arr.T)[0]
    nys = (nvec_arr.T)[1]
    nzs = (nvec_arr.T)[2]
    nmags = np.sqrt((nvec_arr**2).sum(1))
    thetas = np.arccos(nzs/(nmags+EPSILON15))
    phis = np.arctan(nys/(nxs+EPSILON15))\
        + (1.0-np.sign(nxs+EPSILON15))*PI/2.0
    return R4PI*(nmags**ell)*sph_harm(mazi, ell, phis, thetas)

def cart_sph_harm_real(ell=0, mazi=0,
                       nvec_arr=np.array([[1.0, 2.0, 3.0]])):
    r"""Return real Cartesian spherical harmonics.

    The normalization includes a factor of ``\sqrt{4\pi}``, so
    ``Y_{00} == 1``.

    Parameters
    ----------
    ell : int, optional
        Orbital angular momentum.
    mazi : int, optional
        Azimuthal component.
    nvec_arr : numpy.ndarray, optional
        Array of three-vectors. This routine also relies on the global
        constant ``EPSILON15`` when building the angular coordinates.

    Returns
    -------
    numpy.ndarray
        Real Cartesian spherical harmonics evaluated on ``nvec_arr``.
    """
    if mazi == 0:
        return cart_sph_harm(ell, mazi, nvec_arr).real
    if mazi < 0:
        return (np.sqrt(2.0)*(-1.0)**mazi)\
            * cart_sph_harm(ell, np.abs(mazi), nvec_arr).imag
    if mazi > 0:
        return (np.sqrt(2.0)*(-1.0)**mazi)\
            * cart_sph_harm(ell, mazi, nvec_arr).real

def recombine_YY(ell1, mazi1, ell2, mazi2):
    """Recombine a harmonic product into a single-harmonic basis.

    Parameters
    ----------
    ell1 : int
        Angular momentum of the first harmonic.
    mazi1 : int
        Azimuthal component of the first harmonic.
    ell2 : int
        Angular momentum of the second, conjugated harmonic.
    mazi2 : int
        Azimuthal component of the second, conjugated harmonic.

    Returns
    -------
    list[list[float]]
        Entries of the form ``[ell, mazi, coeff]`` describing the
        recombination.
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

def recombine_YY_real(ell1, mazi1, ell2, mazi2):
    """Recombine products of real spherical harmonics.

    Parameters
    ----------
    ell1 : int
        Angular momentum of the first harmonic.
    mazi1 : int
        Azimuthal component of the first harmonic.
    ell2 : int
        Angular momentum of the second harmonic.
    mazi2 : int
        Azimuthal component of the second harmonic.

    Returns
    -------
    list[list[complex]]
        Entries of the form ``[ell, mazi, coeff]`` describing the
        recombination in the real-harmonic basis.

    Raises
    ------
    ValueError
        If the azimuthal inputs cannot be interpreted.
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
        foil_set = [[ell1, mazi1, ell2, mazi2, 1j/np.sqrt(2.)],
                    [ell1, mazi1, ell2, -mazi2,
                     -1j/np.sqrt(2.)*((-1.)**mazi2)]]
    elif mazi2 == 0 and mazi1 < 0:
        foil_set = [[ell1, mazi1, ell2, mazi2, 1j/np.sqrt(2.)],
                    [ell1, -mazi1, ell2, mazi2,
                     -1j/np.sqrt(2.)*((-1.)**mazi1)]]
    elif mazi1 == 0 and mazi2 > 0:
        foil_set = [[ell1, mazi1, ell2, mazi2,
                     1./np.sqrt(2.)*((-1.)**mazi2)],
                    [ell1, mazi1, ell2, -mazi2, 1./np.sqrt(2.)]]
    elif mazi2 == 0 and mazi1 > 0:
        foil_set = [[ell1, mazi1, ell2, mazi2,
                     1./np.sqrt(2.)*((-1.)**mazi1)],
                    [ell1, -mazi1, ell2, mazi2, 1./np.sqrt(2.)]]
    elif mazi1 == 0 and mazi2 == 0:
        foil_set = [[ell1, mazi1, ell2, mazi2, 1.]]
    else:
        raise ValueError("Values for (mazi1, mazi2) not understood")

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
            raise ValueError("Values for mazi not understood")

    final_reco_list = [[]]
    for key in reco_dict:
        if np.abs(reco_dict[key]) > 1.e-10:
            final_reco_list = final_reco_list+[[key[0], key[1],
                                                reco_dict[key]]]
    final_reco_list = final_reco_list[1:]
    return final_reco_list

def calY(ell=0, mazi=0, nvec_arr=np.array([[1.0, 2.0, 3.0]]),
         q=1.0, qc_impl={}):
    r"""Return the caligraphic spherical harmonics.

    Parameters
    ----------
    ell : int, optional
        Orbital angular momentum.
    mazi : int, optional
        Azimuthal component.
    nvec_arr : numpy.ndarray, optional
        Array of three-vectors.
    q : float or numpy.ndarray, optional
        On-shell back-to-back momentum magnitude.
    qc_impl : dict, optional
        Organization of the quantization-condition implementation.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The caligraphic spherical harmonics and their complex conjugates.

    Notes
    -----
    This routine also relies on the global constant ``EPSILON15`` through
    the spherical-harmonic helpers. See ``FiniteVolumeSetup`` for the
    supported entries in ``qc_impl``.
    """
    real_harmonics = QC_IMPL_DEFAULTS['real_harmonics']
    if 'real_harmonics' in qc_impl:
        real_harmonics = qc_impl['real_harmonics']
    if real_harmonics:
        if ell == 0:
            Y = np.ones(len(nvec_arr))
        elif ell == 1 and mazi == 0:
            Y = np.sqrt(3.)*(nvec_arr.T)[2]
        elif ell == 1 and mazi == -1:
            Y = np.sqrt(3.)*(nvec_arr.T)[1]
        elif ell == 1 and mazi == 1:
            Y = np.sqrt(3.)*(nvec_arr.T)[0]
        else:
            Y = cart_sph_harm_real(ell, mazi, nvec_arr)
    else:
        Y = cart_sph_harm(ell, mazi, nvec_arr)
    Yconj = np.conjugate(Y)
    smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
    if 'smarter_q_rescale' in qc_impl:
        smarter_q_rescale = qc_impl['smarter_q_rescale']
    if smarter_q_rescale:
        calY = Y
        calYconj = Yconj
    else:
        calY = Y/q**ell
        calYconj = Yconj/q**ell
    return calY, calYconj

def standard_boost(beta_vec=np.array([0.0, 0.0, 0.0]),
                   four_momentum=np.array([1.0, 0.0, 0.0, 0.0])):
    r"""Return the Lorentz boost of a single four-momentum.

    Parameters
    ----------
    beta_vec : numpy.ndarray, optional
        Boost velocity vector.
    four_momentum : numpy.ndarray, optional
        Four-momentum to be boosted.

    Returns
    -------
    numpy.ndarray
        Boosted four-momentum. If the boost is unphysical, a zero
        four-vector is returned.
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

def standard_boost_array(beta_vec=np.array([[[0.0, 0.0, 0.0]]]),
                         four_momentum=np.array([[[1.0, 0.0, 0.0, 0.0]]])):
    r"""Return Lorentz boosts applied elementwise to an array of vectors.

    Parameters
    ----------
    beta_vec : numpy.ndarray, optional
        Array of boost velocity vectors. Each entry is treated
        independently; magnitudes need not agree across entries.
    four_momentum : numpy.ndarray, optional
        Array of four-momenta to be boosted.

    Returns
    -------
    numpy.ndarray
        Boosted four-momenta. Entries with a vanishing boost velocity
        are returned unchanged, and entries with an unphysical boost
        (``beta**2 >= 1``) are returned as zero four-vectors, matching
        ``standard_boost``.
    """
    betaSQ = (beta_vec*beta_vec).sum(-1)
    physical = betaSQ < 1.0
    safe_betaSQ = np.where(physical, betaSQ, 0.0)
    beta = np.sqrt(safe_betaSQ)
    # Safe as a divisor for building beta_hat only: for vanishing
    # entries beta_hat becomes the zero vector, so the boost reduces to
    # the identity; for unphysical entries beta_hat is not a unit
    # vector, but those entries are zeroed in the final return.
    beta_hat_safe_divisor = np.where(beta > 0.0, beta, 1.0)
    beta_hat = beta_vec/np.repeat(beta_hat_safe_divisor, 3, axis=1
                                  ).reshape(beta_vec.shape)
    gamma = np.sqrt(1.0/(1.0-safe_betaSQ))
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
    return np.where(physical.reshape(physical.shape+(1,)),
                    four_momentum_boosted,
                    np.zeros(four_momentum.shape))
