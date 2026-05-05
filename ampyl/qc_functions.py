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
from scipy.special import erfi
from scipy.special import erf
from scipy.linalg import block_diag
from .kinematic_functions import H
from .kinematic_functions import calY
from .kinematic_functions import q_one_minus_H
from .kinematic_functions import standard_boost
from .kinematic_functions import standard_boost_array
from .constants import PI
from .constants import TWOPI
from .constants import FOURPI2
from .constants import R4PI
from .constants import EPSILON15
from .constants import QC_IMPL_DEFAULTS
import functools
import warnings
warnings.simplefilter("once")


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
    vecstar_for1 = standard_boost(beta_for1, fourmom_for1)[1:]
    # Following is called p^\mu in 1408.5933
    fourmom_for2 = np.concatenate(([omega_p1spec], p1spec_vec))
    # Following is called \vec p^* in 1408.5933
    vecstar_for2 = standard_boost(beta_for2, fourmom_for2)[1:]

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
    if np.abs(q_for1.imag) < EPSILON15:
        q_for1 = q_for1.real
    if np.abs(q_for2.imag) < EPSILON15:
        q_for2 = q_for2.real
    return [vecstar_for1, vecstar_for2, E2CMSQ_for1,
            E2CMSQ_for2, q_for1, q_for2]

def getG_single_entry(E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                      np1spec=np.array([0, 0, 0]),
                      np2spec=np.array([0, 0, 0]),
                      ell1=0, mazi1=0, ell2=0, mazi2=0,
                      m1=1.0, m2=1.0, m3=1.0,
                      alpha=-1.0, beta=0.0,
                      J_slow=False,
                      three_scheme='relativistic pole',
                      qc_impl={},
                      g_rescale=1.0):
    """Evaluate a single entry of the finite-volume ``G`` matrix.

    Parameters
    ----------
    E : float, optional
        Total energy.
    nP : numpy.ndarray, optional
        Dimensionless total momentum.
    L : float, optional
        Spatial box length.
    np1spec : numpy.ndarray, optional
        First spectator momentum index.
    np2spec : numpy.ndarray, optional
        Second spectator momentum index.
    ell1, mazi1, ell2, mazi2 : int, optional
        Angular-momentum labels for the row and column indices.
    m1, m2, m3 : float, optional
        Channel masses.
    alpha, beta : float, optional
        Cutoff parameters.
    J_slow : bool, optional
        Whether to use the slower cutoff implementation.
    three_scheme : str, optional
        Three-body interaction scheme.
    qc_impl : dict, optional
        Quantization-condition implementation options.
    g_rescale : float, optional
        Overall rescaling applied to the result.

    Returns
    -------
    complex or float
        Requested matrix element.

    Notes
    -----
    See ``FiniteVolumeSetup`` for the supported entries in ``qc_impl``.
    """
    [vecstar_for1, vecstar_for2, E2CMSQ_for1, E2CMSQ_for2, q_for1, q_for2]\
        = __helperG_single_entry(E, nP, L,
                                             np1spec, np2spec, m1, m2, m3)

    calY1, _ = calY(ell1, mazi1,
                                vecstar_for1.reshape((1, 3)),
                                q_for1, qc_impl)[0]
    _, calY2conj = calY(ell2, mazi2,
                                    vecstar_for2.reshape((1, 3)),
                                    q_for2, qc_impl)[0]

    HH = H(E2CMSQ_for1, m1+m2, alpha, beta, J_slow)\
        * H(E2CMSQ_for2, m2+m3, alpha, beta, J_slow)

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
        raise ValueError("three_scheme not recognized")
    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        simple_factor = simple_factor/(2.0*omega3)

    pole_factor = 1.0/(E-omega1-omega2-omega3)
    return calY1*calY2conj*HH*simple_factor*pole_factor*g_rescale

def get_nvec_data(tbks_entry, row_shell, col_shell):
    """Get the data for the nvecs."""
    n1vec_arr_shell = tbks_entry.nvec_arr[row_shell[0]:row_shell[1]]
    n1vecSQ_arr_shell = tbks_entry.nvecSQ_arr[
        row_shell[0]:row_shell[1]]

    n2vec_arr_shell = tbks_entry.nvec_arr[col_shell[0]:col_shell[1]]
    n2vecSQ_arr_shell = tbks_entry.nvecSQ_arr[
        col_shell[0]:col_shell[1]]

    # Awkward swap here
    n1vec_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n2vec_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    n2vec_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n1vec_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    n3vec_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n3vec_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    n1vecSQ_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n2vecSQ_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    n2vecSQ_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n1vecSQ_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    n3vecSQ_mat_shell = np.swapaxes(
        np.swapaxes(
            ((tbks_entry.n3vecSQ_mat)[
                row_shell[0]:row_shell[1]]),
            0, 1
            )[col_shell[0]:col_shell[1]],
        0, 1
        )

    return [n1vec_arr_shell, n1vecSQ_arr_shell,
            n2vec_arr_shell, n2vecSQ_arr_shell,
            n1vec_mat_shell, n2vec_mat_shell, n3vec_mat_shell,
            n1vecSQ_mat_shell, n2vecSQ_mat_shell, n3vecSQ_mat_shell]

def __helperG_array(E, nP, L, m1, m2, m3,
                    tbks_entry,
                    row_shell,
                    col_shell):
    [n1vec_arr_shell, n1vecSQ_arr_shell,
     n2vec_arr_shell, n2vecSQ_arr_shell,
     n1vec_mat_shell, n2vec_mat_shell, n3vec_mat_shell,
     n1vecSQ_mat_shell, n2vecSQ_mat_shell, n3vecSQ_mat_shell]\
        = get_nvec_data(tbks_entry,
                                    row_shell, col_shell)
    Pvec = TWOPI*nP/L
    p1specvec_arr_slice\
        = TWOPI*n1vec_arr_shell/L  # called \vec p in 1408.5933
    p2specvec_arr_slice\
        = TWOPI*n2vec_arr_shell/L  # called \vec k in 1408.5933
    p1specvecSQ_arr_slice = (TWOPI**2)*n1vecSQ_arr_shell/L**2
    p2specvecSQ_arr_slice = (TWOPI**2)*n2vecSQ_arr_shell/L**2
    omegap1spec_arr_slice = np.sqrt(m3**2+p1specvecSQ_arr_slice)
    omegap2spec_arr_slice = np.sqrt(m1**2+p2specvecSQ_arr_slice)

    p1specvec_mat_shell\
        = TWOPI*n1vec_mat_shell/L  # called \vec p in 1408.5933
    p2specvec_mat_shell\
        = TWOPI*n2vec_mat_shell/L  # called \vec k in 1408.5933
    p1specvecSQ_mat_shell = (TWOPI**2)*n1vecSQ_mat_shell/L**2
    p2specvecSQ_mat_shell = (TWOPI**2)*n2vecSQ_mat_shell/L**2
    omegap1spec_mat_shell = np.sqrt(m3**2+p1specvecSQ_mat_shell)
    omegap2spec_mat_shell = np.sqrt(m1**2+p2specvecSQ_mat_shell)

    # Following is called \vec \beta_p in 1408.5933
    beta_for1 = -(Pvec-p1specvec_mat_shell)/np.repeat(
        E-omegap1spec_mat_shell, 3, axis=1
        ).reshape((Pvec-p1specvec_mat_shell).shape)

    # Following is called \vec \beta_k in 1408.5933
    beta_for2 = -(Pvec-p2specvec_mat_shell)/np.repeat(
        E-omegap2spec_mat_shell, 3, axis=1
        ).reshape((Pvec-p2specvec_mat_shell).shape)

    # Following is called k^\mu in 1408.5933
    fourmom_for1 = np.concatenate(
        (omegap2spec_mat_shell.reshape(
            omegap2spec_mat_shell.shape+(1,)), p2specvec_mat_shell),
        axis=2)

    # Following is called \vec k^* in 1408.5933
    vecstar_for1 = standard_boost_array(beta_for1,
                                                    fourmom_for1)[:, :, 1:]

    # Following is called p^\mu in 1408.5933
    fourmom_for2 = np.concatenate(
        (omegap1spec_mat_shell.reshape(omegap1spec_mat_shell.shape+(1,)),
         p1specvec_mat_shell),
        axis=2)

    # Following is called \vec p^* in 1408.5933
    vecstar_for2 = standard_boost_array(beta_for2,
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
            omegap1spec_mat_shell, omegap2spec_mat_shell,
            omegap1spec_arr_slice, omegap2spec_arr_slice,
            n3vecSQ_mat_shell]

def get_nvecSQ_mat_shells(tbks_entry,
                          row_shell,
                          col_shell):
    """Get n1vecSQ_mat_shell, n2vecSQ_mat_shell, n3vecSQ_mat_shell."""
    return get_nvec_data(tbks_entry,
                                     row_shell, col_shell)[-3:]

def __helperG_array_prep_mat(E, nP, L, m1, m2, m3,
                             tbks_entry,
                             row_shell_index,
                             col_shell_index):
    n1vec_arr_shell\
        = tbks_entry.n1vec_arr_all_shells[row_shell_index][
            col_shell_index]
    n1vecSQ_arr_shell\
        = tbks_entry.n1vecSQ_arr_all_shells[row_shell_index][
            col_shell_index]
    n2vec_arr_shell\
        = tbks_entry.n2vec_arr_all_shells[row_shell_index][
            col_shell_index]
    n2vecSQ_arr_shell\
        = tbks_entry.n2vecSQ_arr_all_shells[row_shell_index][
            col_shell_index]
    n1vec_mat_shell\
        = tbks_entry.n1vec_mat_all_shells[row_shell_index][
            col_shell_index]
    n2vec_mat_shell\
        = tbks_entry.n2vec_mat_all_shells[row_shell_index][
            col_shell_index]
    # n3vec_mat_shell\
    #     = tbks_entry.n3vec_mat_all_shells[row_shell_index][
    #         col_shell_index]
    n1vecSQ_mat_shell\
        = tbks_entry.n1vecSQ_mat_all_shells[row_shell_index][
            col_shell_index]
    n2vecSQ_mat_shell\
        = tbks_entry.n2vecSQ_mat_all_shells[row_shell_index][
            col_shell_index]
    n3vecSQ_mat_shell\
        = tbks_entry.n3vecSQ_mat_all_shells[row_shell_index][
            col_shell_index]

    Pvec = TWOPI*nP/L
    p1specvec_arr_slice\
        = TWOPI*n1vec_arr_shell/L  # called \vec p in 1408.5933
    p2specvec_arr_slice\
        = TWOPI*n2vec_arr_shell/L  # called \vec k in 1408.5933
    p1specvecSQ_arr_slice = (TWOPI**2)*n1vecSQ_arr_shell/L**2
    p2specvecSQ_arr_slice = (TWOPI**2)*n2vecSQ_arr_shell/L**2
    omegap1spec_arr_slice = np.sqrt(m3**2+p1specvecSQ_arr_slice)
    omegap2spec_arr_slice = np.sqrt(m1**2+p2specvecSQ_arr_slice)

    p1specvec_mat_shell\
        = TWOPI*n1vec_mat_shell/L  # called \vec p in 1408.5933
    p2specvec_mat_shell\
        = TWOPI*n2vec_mat_shell/L  # called \vec k in 1408.5933
    p1specvecSQ_mat_shell = (TWOPI**2)*n1vecSQ_mat_shell/L**2
    p2specvecSQ_mat_shell = (TWOPI**2)*n2vecSQ_mat_shell/L**2
    omegap1spec_mat_shell = np.sqrt(m3**2+p1specvecSQ_mat_shell)
    omegap2spec_mat_shell = np.sqrt(m1**2+p2specvecSQ_mat_shell)

    # Following is called \vec \beta_p in 1408.5933
    beta_for1 = -(Pvec-p1specvec_mat_shell)/np.repeat(
        E-omegap1spec_mat_shell, 3, axis=1
        ).reshape((Pvec-p1specvec_mat_shell).shape)

    # Following is called \vec \beta_k in 1408.5933
    beta_for2 = -(Pvec-p2specvec_mat_shell)/np.repeat(
        E-omegap2spec_mat_shell, 3, axis=1
        ).reshape((Pvec-p2specvec_mat_shell).shape)

    # Following is called k^\mu in 1408.5933
    fourmom_for1 = np.concatenate(
        (omegap2spec_mat_shell.reshape(
            omegap2spec_mat_shell.shape+(1,)), p2specvec_mat_shell),
        axis=2)

    # Following is called \vec k^* in 1408.5933
    vecstar_for1 = standard_boost_array(beta_for1,
                                                    fourmom_for1)[:, :, 1:]

    # Following is called p^\mu in 1408.5933
    fourmom_for2 = np.concatenate(
        (omegap1spec_mat_shell.reshape(omegap1spec_mat_shell.shape+(1,)),
         p1specvec_mat_shell),
        axis=2)

    # Following is called \vec p^* in 1408.5933
    vecstar_for2 = standard_boost_array(beta_for2,
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
            omegap1spec_mat_shell, omegap2spec_mat_shell,
            omegap1spec_arr_slice, omegap2spec_arr_slice,
            n3vecSQ_mat_shell]

def getG_array(E, nP, L, m1, m2, m3,
               tbks_entry,
               row_shell, col_shell,
               ell1, ell2,
               alpha, beta,
               qc_impl, three_scheme,
               g_rescale):
    """Return a NumPy-accelerated block of the ``G`` matrix.

    Parameters
    ----------
    E : float
        Total energy.
    nP : numpy.ndarray
        Dimensionless total momentum.
    L : float
        Spatial box length.
    m1, m2, m3 : float
        Channel masses.
    tbks_entry : object
        TBKS entry providing shell data.
    row_shell, col_shell : tuple[int, int]
        Row and column shell slices.
    ell1, ell2 : int
        Row and column angular momenta.
    alpha, beta : float
        Cutoff parameters.
    qc_impl : dict
        Quantization-condition implementation options.
    three_scheme : str
        Three-body interaction scheme.
    g_rescale : float
        Overall rescaling applied to the result.

    Returns
    -------
    numpy.ndarray
        Matrix block for the requested shell pair.
    """
    J_slow = False
    [vecstar_for1, vecstar_for2, E2CMSQ_for1,
     E2CMSQ_for2, q_for1, q_for2, q_for1_mat, q_for2_mat,
     omegap1spec_mat_shell, omegap2spec_mat_shell,
     omegap1spec_arr_slice, omegap2spec_arr_slice,
     n3vecSQ_mat_shell]\
        = __helperG_array(E, nP, L, m1, m2, m3,
                                      tbks_entry,
                                      row_shell,
                                      col_shell)

    shape1_tmp = vecstar_for1.shape
    r2_shape = shape1_tmp[:-1]

    calY1mat = [[]]
    calY2conjmat = [[]]
    for mazi1 in np.arange(-ell1, ell1+1):
        calY1row = []
        calY2conjrow = []
        for mazi2 in np.arange(-ell2, ell2+1):
            calY1, _ = calY(
                ell1, mazi1, vecstar_for1.reshape(
                    (shape1_tmp[0]*shape1_tmp[1], 3)),
                q_for1_mat.reshape(q_for1_mat.size), qc_impl)
            shape2_tmp = vecstar_for2.shape
            _, calY2conj = calY(
                ell2, mazi2, vecstar_for2.reshape(
                    (shape2_tmp[0]*shape2_tmp[1], 3)),
                q_for2_mat.reshape(q_for2_mat.size), qc_impl)
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

    H1 = H(E2CMSQ_for1.reshape(E2CMSQ_for1.size), m1+m2,
                       alpha, beta, J_slow)
    H2 = H(E2CMSQ_for2.reshape(E2CMSQ_for2.size), m2+m3,
                       alpha, beta, J_slow)

    omega1_mat = omegap2spec_mat_shell
    omega2_mat = np.sqrt(m2**2+FOURPI2*n3vecSQ_mat_shell/L**2)
    omega3_mat = omegap1spec_mat_shell

    if three_scheme == 'original pole':
        simple_factor_mat = 1.0/(2.0*omega1_mat*omega2_mat*L**3)
    elif three_scheme == 'relativistic pole':
        simple_factor_mat = 1.0/(2.0*omega1_mat*L**3
                                 * (E-omega1_mat-omega3_mat+omega2_mat))
    else:
        raise ValueError("three_scheme not recognized")

    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        simple_factor_mat = simple_factor_mat/(2.0*omega3_mat)

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
    if np.sum(H1_mat**2) < 1.e-20 or np.sum(H2_mat**2) < 1.e-20:
        return np.zeros_like(full_mat_big)
    return YY*full_mat_big*H1_mat*H2_mat*g_rescale


def getG_array_two_tbks(E, nP, L, m1, m2, m3,
                        row_tbks_entry, col_tbks_entry,
                        row_shell, col_shell,
                        ell1, ell2,
                        alpha, beta,
                        qc_impl, three_scheme,
                        g_rescale):
    """Return a G block whose row and column use different TBKS entries."""
    row_nvec_arr = row_tbks_entry.nvec_arr[row_shell[0]:row_shell[1]]
    row_nvecSQ_arr = row_tbks_entry.nvecSQ_arr[row_shell[0]:row_shell[1]]
    col_nvec_arr = col_tbks_entry.nvec_arr[col_shell[0]:col_shell[1]]
    col_nvecSQ_arr = col_tbks_entry.nvecSQ_arr[col_shell[0]:col_shell[1]]

    nrow = len(row_nvec_arr)
    ncol = len(col_nvec_arr)
    ntotal = nrow+ncol
    row_slice = [0, nrow]
    col_slice = [nrow, ntotal]

    class _MixedTBKSEntry:
        pass

    mixed_tbks_entry = _MixedTBKSEntry()
    mixed_tbks_entry.nvec_arr = np.concatenate(
        (row_nvec_arr, col_nvec_arr), axis=0)
    mixed_tbks_entry.nvecSQ_arr = np.concatenate(
        (row_nvecSQ_arr, col_nvecSQ_arr), axis=0)
    shape = (ntotal, ntotal, 3)
    mixed_tbks_entry.n1vec_mat = np.zeros(shape, dtype=int)
    mixed_tbks_entry.n2vec_mat = np.zeros(shape, dtype=int)
    mixed_tbks_entry.n3vec_mat = np.zeros(shape, dtype=int)
    mixed_tbks_entry.n1vecSQ_mat = np.zeros((ntotal, ntotal), dtype=int)
    mixed_tbks_entry.n2vecSQ_mat = np.zeros((ntotal, ntotal), dtype=int)
    mixed_tbks_entry.n3vecSQ_mat = np.zeros((ntotal, ntotal), dtype=int)

    row_mat = np.repeat(row_nvec_arr[:, np.newaxis, :], ncol, axis=1)
    col_mat = np.repeat(col_nvec_arr[np.newaxis, :, :], nrow, axis=0)
    exchange_mat = nP-row_mat-col_mat

    row_idx = slice(row_slice[0], row_slice[1])
    col_idx = slice(col_slice[0], col_slice[1])
    mixed_tbks_entry.n2vec_mat[row_idx, col_idx] = row_mat
    mixed_tbks_entry.n1vec_mat[row_idx, col_idx] = col_mat
    mixed_tbks_entry.n3vec_mat[row_idx, col_idx] = exchange_mat
    mixed_tbks_entry.n2vecSQ_mat[row_idx, col_idx] = (
        row_mat*row_mat).sum(axis=2)
    mixed_tbks_entry.n1vecSQ_mat[row_idx, col_idx] = (
        col_mat*col_mat).sum(axis=2)
    mixed_tbks_entry.n3vecSQ_mat[row_idx, col_idx] = (
        exchange_mat*exchange_mat).sum(axis=2)

    return getG_array(E, nP, L, m1, m2, m3,
                      mixed_tbks_entry, row_slice, col_slice,
                      ell1, ell2, alpha, beta, qc_impl, three_scheme,
                      g_rescale)


def getG_array_prep_mat(E, nP, L, m1, m2, m3,
                        tbks_entry,
                        row_shell_index, col_shell_index,
                        ell1, ell2,
                        alpha, beta,
                        qc_impl, three_scheme,
                        g_rescale):
    """Return a prepared-shell, NumPy-accelerated block of ``G``.

    Parameters
    ----------
    E : float
        Total energy.
    nP : numpy.ndarray
        Dimensionless total momentum.
    L : float
        Spatial box length.
    m1, m2, m3 : float
        Channel masses.
    tbks_entry : object
        TBKS entry providing shell data.
    row_shell_index, col_shell_index : int
        Prepared row and column shell indices.
    ell1, ell2 : int
        Row and column angular momenta.
    alpha, beta : float
        Cutoff parameters.
    qc_impl : dict
        Quantization-condition implementation options.
    three_scheme : str
        Three-body interaction scheme.
    g_rescale : float
        Overall rescaling applied to the result.

    Returns
    -------
    numpy.ndarray
        Matrix block for the requested prepared shell pair.
    """
    J_slow = False
    [vecstar_for1, vecstar_for2, E2CMSQ_for1,
     E2CMSQ_for2, q_for1, q_for2, q_for1_mat, q_for2_mat,
     omegap1spec_mat_shell, omegap2spec_mat_shell,
     omegap1spec_arr_slice, omegap2spec_arr_slice,
     n3vecSQ_mat_shell]\
        = __helperG_array_prep_mat(E, nP, L, m1, m2, m3,
                                               tbks_entry,
                                               row_shell_index,
                                               col_shell_index)

    shape1_tmp = vecstar_for1.shape
    r2_shape = shape1_tmp[:-1]

    calY1mat = [[]]
    calY2conjmat = [[]]
    for mazi1 in np.arange(-ell1, ell1+1):
        calY1row = []
        calY2conjrow = []
        for mazi2 in np.arange(-ell2, ell2+1):
            calY1, _ = calY(
                ell1, mazi1, vecstar_for1.reshape(
                    (shape1_tmp[0]*shape1_tmp[1], 3)),
                q_for1_mat.reshape(q_for1_mat.size), qc_impl)
            shape2_tmp = vecstar_for2.shape
            _, calY2conj = calY(
                ell2, mazi2, vecstar_for2.reshape(
                    (shape2_tmp[0]*shape2_tmp[1], 3)),
                q_for2_mat.reshape(q_for2_mat.size), qc_impl)
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

    H1 = H(E2CMSQ_for1.reshape(E2CMSQ_for1.size), m1+m2,
                       alpha, beta, J_slow)
    H2 = H(E2CMSQ_for2.reshape(E2CMSQ_for2.size), m2+m3,
                       alpha, beta, J_slow)

    omega1_mat = omegap2spec_mat_shell
    omega2_mat = np.sqrt(m2**2+FOURPI2*n3vecSQ_mat_shell/L**2)
    omega3_mat = omegap1spec_mat_shell

    if three_scheme == 'original pole':
        simple_factor_mat = 1.0/(2.0*omega1_mat*omega2_mat*L**3)
    elif three_scheme == 'relativistic pole':
        simple_factor_mat = 1.0/(2.0*omega1_mat*L**3
                                 * (E-omega1_mat-omega3_mat+omega2_mat))
    else:
        raise ValueError("three_scheme not recognized")

    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        simple_factor_mat = simple_factor_mat/(2.0*omega3_mat)

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

def summand(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0, alpha_mass=0.5,
            nvec_arr=np.array([[0, 0, 0]]), alphaKSS=1.0,
            ell1=0, mazi1=0, ell2=0, mazi2=0,
            qc_impl={}):
    """Return the regulated summand entering the ``F`` function.

    Parameters
    ----------
    nP2 : numpy.ndarray, optional
        Dimensionless two-particle momentum.
    qSQ : float, optional
        Squared on-shell momentum.
    gamSQ : float, optional
        Squared Lorentz factor.
    alpha_mass : float, optional
        Mass-dependent boost parameter.
    nvec_arr : numpy.ndarray, optional
        Integer vectors included in the regulated sum.
    alphaKSS : float, optional
        Exponential damping parameter.
    ell1, mazi1, ell2, mazi2 : int, optional
        Angular-momentum labels.
    qc_impl : dict, optional
        Quantization-condition implementation options.

    Returns
    -------
    numpy.ndarray
        Value of the regulated summand on ``nvec_arr``.
    """
    nP2SQ = nP2@nP2
    nP2mag = np.sqrt(nP2SQ)
    q = np.sqrt(qSQ+0j)
    sph_harm_value = 1.0
    if nP2SQ == 0.0:
        rSQ_arr = (nvec_arr**2).sum(1)
        if ell1 != 0:
            calY1, _ = calY(ell1, mazi1, nvec_arr,
                                        q, qc_impl)
            sph_harm_value = sph_harm_value*calY1

        if ell2 != 0:
            _, calY2conj = calY(ell2, mazi2, nvec_arr,
                                            q, qc_impl)
            sph_harm_value = sph_harm_value*calY2conj

        if ((ell1 == ell2) and (mazi1 == mazi2)):
            smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
            if 'smarter_q_rescale' in qc_impl:
                smarter_q_rescale = qc_impl['smarter_q_rescale']
            if smarter_q_rescale:
                sph_harm_value = sph_harm_value\
                    - (rSQ_arr**ell1-qSQ**(ell1))
            else:
                sph_harm_value = sph_harm_value\
                    - (rSQ_arr**ell1-qSQ**(ell1))/(qSQ**(ell1))
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
            rpar_vec_arr = (npar_vec_arr-nP2*alpha_mass)/np.sqrt(gamSQ)
            rperp_vec_arr = nvec_arr - npar_vec_arr
            rvec_arr = rpar_vec_arr + rperp_vec_arr
            rSQ_arr = (rvec_arr**2).sum(1)
            if ell1 != 0:
                calY1, _ = calY(ell1, mazi1, rvec_arr,
                                            q, qc_impl)
                sph_harm_value = sph_harm_value*calY1
            if ell2 != 0:
                _, calY2conj = calY(ell2, mazi2, rvec_arr,
                                                q, qc_impl)
                sph_harm_value = sph_harm_value*calY2conj
            if ((ell1 == ell2) and (mazi1 == mazi2)):
                smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
                if 'smarter_q_rescale' in qc_impl:
                    smarter_q_rescale = qc_impl['smarter_q_rescale']

                if smarter_q_rescale:
                    sph_harm_value = sph_harm_value\
                        - (rSQ_arr**ell1-qSQ**(ell1))
                else:
                    sph_harm_value = sph_harm_value\
                        - (rSQ_arr**ell1-qSQ**(ell1))/(qSQ**(ell1))
    Ds = rSQ_arr-qSQ
    return sph_harm_value*np.exp(-alphaKSS*Ds)/Ds

def __T1(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0, alpha_mass=0.5,
         C1cut=3, alphaKSS=1.0, ell1=0, mazi1=0, ell2=0, mazi2=0,
         qc_impl={}):
    rng = range(-C1cut, C1cut+1)
    mesh = np.meshgrid(*([rng]*3))
    nvec_arr = np.vstack([y.flat for y in mesh]).T
    return np.sum(summand(nP2, qSQ, gamSQ, alpha_mass,
                                      nvec_arr, alphaKSS,
                                      ell1, mazi1, ell2, mazi2,
                                      qc_impl))/R4PI

def __T2(qSQ=1.5, gamSQ=1.0, alphaKSS=1.0,
         ell1=0, mazi1=0, ell2=0, mazi2=0, qc_impl={}):
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
        smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
        if 'smarter_q_rescale' in qc_impl:
            smarter_q_rescale = qc_impl['smarter_q_rescale']
        if smarter_q_rescale:
            ttmp = ttmp*(qSQ)**ell1
        return gamma*ttmp/np.sqrt(2.0*TWOPI)
    else:
        return 0.0

def getZ_single_entry(nP2=np.array([0, 0, 0]), qSQ=1.5, gamSQ=1.0,
                      alpha_mass=0.5, C1cut=3, alphaKSS=1.0,
                      ell1=0, mazi1=0, ell2=0, mazi2=0,
                      qc_impl={}):
    r"""Evaluate a single entry of ``Z``."""
    return __T1(nP2, qSQ, gamSQ, alpha_mass, C1cut, alphaKSS,
                            ell1, mazi1, ell2, mazi2, qc_impl)\
        + __T2(qSQ, gamSQ, alphaKSS, ell1, mazi1, ell2, mazi2,
                           qc_impl)

def getFtwo_single_entry(E2=3.0, nP2=np.array([0, 0, 0]), L=5.0,
                         m1=1.0, m2=1.0, C1cut=3, alphaKSS=1.0,
                         ell1=0, mazi1=0, ell2=0, mazi2=0,
                         qc_impl={}):
    r"""Evaluate a single entry of ``F_2``."""
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
    alpha_mass = 0.5*(1.+(m1**2-m2**2)/E2CMSQ)
    pre = -2.0/(L*np.sqrt(PI)*16.0*PI*E2CM*gamma)
    return pre*(getZ_single_entry(nP2, qSQ_dimless, gamSQ,
                                              alpha_mass, C1cut, alphaKSS,
                                              ell1, mazi1, ell2, mazi2,
                                              qc_impl))

def getF_single_entry(E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                      npspec=np.array([0, 0, 0]),
                      m1=1.0, m2=1.0, mspec=1.0,
                      C1cut=3, alphaKSS=1.0, alpha=-1.0, beta=0.0,
                      ell1=0, mazi1=0, ell2=0, mazi2=0,
                      three_scheme='relativistic pole',
                      qc_impl={}):
    """Evaluate a single entry of the finite-volume ``F`` matrix.

    Parameters
    ----------
    E : float, optional
        Total energy.
    nP : numpy.ndarray, optional
        Dimensionless total momentum.
    L : float, optional
        Spatial box length.
    npspec : numpy.ndarray, optional
        Spectator momentum index.
    m1, m2, mspec : float, optional
        Channel masses.
    C1cut : int, optional
        Cutoff on the regulated sum.
    alphaKSS : float, optional
        Exponential damping parameter.
    alpha, beta : float, optional
        Cutoff parameters.
    ell1, mazi1, ell2, mazi2 : int, optional
        Angular-momentum labels.
    three_scheme : str, optional
        Three-body interaction scheme.
    qc_impl : dict, optional
        Quantization-condition implementation options.

    Returns
    -------
    complex or float
        Requested matrix element.
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
    alpha_mass = 0.5*(1.+(m1**2-m2**2)/E2CMSQ)

    Htmp = H(E2CMSQ, m1+m2, alpha, beta)
    pre = -Htmp*2.0/(L*np.sqrt(PI)*16.0*PI*E2CM*gamma)
    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        pre = pre/(2.0*omspec)
    smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
    if 'smarter_q_rescale' in qc_impl:
        smarter_q_rescale = qc_impl['smarter_q_rescale']
    if smarter_q_rescale:
        pre = pre*(FOURPI2/L**2)**ell1
    return pre*(getZ_single_entry(nP2, qSQ_dimless, gamSQ,
                                              alpha_mass, C1cut, alphaKSS,
                                              ell1, mazi1,
                                              ell2, mazi2, qc_impl))

def getF_single_entry_IPV(IPV_function=None, IPV_parameters=[1.0],
                          E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                          npspec=np.array([0, 0, 0]),
                          m1=1.0, m2=1.0, mspec=1.0,
                          C1cut=3, alphaKSS=1.0, alpha=-1.0, beta=0.0,
                          ell1=0, mazi1=0, ell2=0, mazi2=0,
                          three_scheme='relativistic pole', qc_impl={}):
    """Evaluate a single ``F`` entry including the PV-shift prescription.

    Parameters
    ----------
    IPV_function : callable, optional
        Principal-value shift function.
    IPV_parameters : list[float], optional
        Parameters passed to ``IPV_function``.
    E : float, optional
        Total energy.
    nP : numpy.ndarray, optional
        Dimensionless total momentum.
    L : float, optional
        Spatial box length.
    npspec : numpy.ndarray, optional
        Spectator momentum index.
    m1, m2, mspec : float, optional
        Channel masses.
    C1cut : int, optional
        Cutoff on the regulated sum.
    alphaKSS : float, optional
        Exponential damping parameter.
    alpha, beta : float, optional
        Cutoff parameters.
    ell1, mazi1, ell2, mazi2 : int, optional
        Angular-momentum labels.
    three_scheme : str, optional
        Three-body interaction scheme.
    qc_impl : dict, optional
        Quantization-condition implementation options.

    Returns
    -------
    complex or float
        Requested matrix element including the PV-shift term.
    """
    if IPV_function is None:
        IPV_function = IPV_constant
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
    alpha_mass = 0.5*(1.+(m1**2-m2**2)/E2CMSQ)
    Htmp = H(E2CMSQ, m1+m2, alpha, beta)
    pre = -Htmp*2.0/(L*np.sqrt(PI)*16.0*PI*E2CM*gamma)
    pv_shift_value = 0.0
    if ell1 == ell2 and mazi1 == mazi2:
        ell = ell1
        pSQ = qSQ
        IPV = IPV_function(qSQ, *IPV_parameters)
        include_H_in_IPV = QC_IMPL_DEFAULTS['include_H_in_IPV']
        if 'include_H_in_IPV' in qc_impl:
            include_H_in_IPV = qc_impl['include_H_in_IPV']
        if include_H_in_IPV:
            partial_shift = IPV/pSQ**(ell)*np.sqrt(pSQ+1.0)
        else:
            if np.abs(Htmp) < EPSILON15:
                partial_shift = 0.0
            else:
                partial_shift = IPV/pSQ**(ell)*np.sqrt(pSQ+1.0)/Htmp
        smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
        if 'smarter_q_rescale' in qc_impl:
            smarter_q_rescale = qc_impl['smarter_q_rescale']
        if smarter_q_rescale:
            pv_shift_value = 0.5*np.sqrt(PI)*L*gamma*partial_shift\
                * qSQ_dimless**(ell)
        else:
            pv_shift_value = 0.5*np.sqrt(PI)*L*gamma*partial_shift
    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        pre = pre/(2.0*omspec)
    smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
    if 'smarter_q_rescale' in qc_impl:
        smarter_q_rescale = qc_impl['smarter_q_rescale']
    if smarter_q_rescale:
        pre = pre*(FOURPI2/L**2)**ell1
    return pre*(getZ_single_entry(nP2, qSQ_dimless, gamSQ,
                                              alpha_mass, C1cut, alphaKSS,
                                              ell1, mazi1,
                                              ell2, mazi2, qc_impl)
                - pv_shift_value)

def getF_array(E, nP, L, m1, m2, m3, tbks_entry, slice_entry,
               ell1, ell2, alpha, beta, C1cut, alphaKSS, qc_impl,
               three_scheme, use_pv_shift_prescription=False,
               IPV_function=None, pv_shift_parameters=[0.0]):
    """Return the block-diagonal finite-volume ``F`` matrix.

    Parameters
    ----------
    E : float
        Total energy.
    nP : numpy.ndarray
        Dimensionless total momentum.
    L : float
        Spatial box length.
    m1, m2, m3 : float
        Channel masses.
    tbks_entry : object
        TBKS entry providing shell data.
    slice_entry : tuple[int, int]
        Slice selecting the spectator shell.
    ell1, ell2 : int
        Row and column angular momenta.
    alpha, beta : float
        Cutoff parameters.
    C1cut : int
        Cutoff on the regulated sum.
    alphaKSS : float
        Exponential damping parameter.
    qc_impl : dict
        Quantization-condition implementation options.
    three_scheme : str
        Three-body interaction scheme.
    use_pv_shift_prescription : bool, optional
        Whether to include the PV-shift prescription.
    IPV_function : callable, optional
        Principal-value shift function.
    pv_shift_parameters : list[float], optional
        Parameters passed to ``IPV_function``.

    Returns
    -------
    numpy.ndarray
        Block-diagonal ``F`` matrix.
    """
    nvec_arr_slice = tbks_entry.nvec_arr[slice_entry[0]:slice_entry[1]]
    f_list = []
    for nvec in nvec_arr_slice:
        f_mat_entry = [[]]
        for mazi1 in range(-ell1, ell1+1):
            f_row = []
            for mazi2 in range(-ell2, ell2+1):
                # Awkward notation for masses here
                if use_pv_shift_prescription:
                    f_entry = getF_single_entry_IPV(
                        IPV_function=IPV_function,
                        IPV_parameters=pv_shift_parameters,
                        E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3,
                        mspec=m1, C1cut=C1cut, alphaKSS=alphaKSS,
                        alpha=alpha, beta=beta, ell1=ell1, mazi1=mazi1,
                        ell2=ell2, mazi2=mazi2, three_scheme=three_scheme,
                        qc_impl=qc_impl)
                else:
                    f_entry = getF_single_entry(
                        E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3,
                        mspec=m1, C1cut=C1cut, alphaKSS=alphaKSS,
                        alpha=alpha, beta=beta, ell1=ell1, mazi1=mazi1,
                        ell2=ell2, mazi2=mazi2, three_scheme=three_scheme,
                        qc_impl=qc_impl)
                if np.abs(f_entry.imag) < EPSILON15:
                    f_entry = f_entry.real
                if np.abs(f_entry) < EPSILON15:
                    f_entry = 0.0
                f_row = f_row+[f_entry]
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

@with_str(pcotdelta_scattering_length_str)
def pcotdelta_scattering_length(pSQ=1.5, a=1.0):
    r"""Evaluate ``p cot(delta)`` in the scattering-length approximation."""
    return -1.0/a

def IPV_constant(pSQ=1.5, c=1.0):
    """Return a constant principal-value shift."""
    return c

def IPV_poly(pSQ=1.5, c=1.0, d=1.0):
    """Return a linear polynomial principal-value shift."""
    return c+pSQ*d

def IPV_poly_root_removal(pSQ=1.5, c=1.0, d=1.0):
    """Return a polynomial PV shift with threshold-root removal."""
    return (c+pSQ*d)/np.sqrt(pSQ+1.)

def pcotdelta_breit_wigner_str():
    """Print behavior for pcotdelta_breit_wigner."""
    return "pcotdelta_breit_wigner"

@with_str(pcotdelta_breit_wigner_str)
def pcotdelta_breit_wigner(pSQ=1.5, g_value=6.0, mrho_value=3.0):
    """Evaluate the Breit-Wigner ``p cot(delta)`` parametrization.

    Parameters
    ----------
    pSQ : float, optional
        Squared two-particle momentum.
    g_value : float, optional
        Coupling parameter.
    mrho_value : float, optional
        Resonance mass parameter.

    Returns
    -------
    float or complex
        Value of ``p cot(delta)``.

    Notes
    -----
    The result includes a factor of ``pSQ`` to cancel threshold scaling that
    is handled elsewhere in the formalism.
    """
    # print Ecm with label
    # print("Ecm: ", Ecm)
    # print pcotdelta with label
    # print("pcotdelta in original function: ", 1/tandop)
    Ecm = 2.0*np.sqrt(1.0+pSQ)
    GammaEcmop = g_value**2/(6.0*np.pi)*((pSQ))/Ecm**2
    tandop = GammaEcmop*Ecm/(mrho_value**2-Ecm**2)
    return pSQ/tandop

def pcotdelta_ere_breit_wigner(pSQ=1.5, g_value=6.0, mrho_value=3.0):
    """Evaluate the ERE-style Breit-Wigner ``p cot(delta)`` parametrization."""
    Ecm = 2.0*np.sqrt(1.0+pSQ)
    GammaEcmop = g_value**2/(6.0*np.pi)*((pSQ))/mrho_value**2
    tandop = GammaEcmop*Ecm/(mrho_value**2-Ecm**2)
    return pSQ/tandop

def getK_single_entry(pcotdelta_function=None,
                      pcotdelta_parameter_list=[1.0],
                      E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                      npspec=np.array([0, 0, 0]),
                      m1=1.0, m2=1.0, mspec=1.0,
                      alpha=-1.0, beta=0.0,
                      ell=0,
                      qc_impl={}):
    """Evaluate a single entry of the two-body ``K`` matrix.

    Parameters
    ----------
    pcotdelta_function : callable, optional
        Function used to evaluate ``p cot(delta)``.
    pcotdelta_parameter_list : list[float], optional
        Parameters passed to ``pcotdelta_function``.
    E : float, optional
        Total energy.
    nP : numpy.ndarray, optional
        Dimensionless total momentum.
    L : float, optional
        Spatial box length.
    npspec : numpy.ndarray, optional
        Spectator momentum index.
    m1, m2, mspec : float, optional
        Channel masses.
    alpha, beta : float, optional
        Cutoff parameters.
    ell : int, optional
        Angular momentum.
    qc_impl : dict, optional
        Quantization-condition implementation options.

    Returns
    -------
    complex or float
        Requested matrix element.
    """
    if pcotdelta_function is None:
        pcotdelta_function = pcotdelta_scattering_length
    P = TWOPI*nP/L
    pspec = TWOPI*npspec/L
    omspec = np.sqrt(mspec**2+pspec@pspec)
    E2 = E-omspec
    P2 = P-pspec
    E2CMSQ = E2**2-P2@P2
    if E2CMSQ <= 0.0 or E2 < 0.0:
        return np.nan
    ECM = np.sqrt(E2CMSQ)
    if m1 == m2:
        pSQ = E2CMSQ/4.0-m1**2
    else:
        pSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
               + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*E2CMSQ)
    pcotdelta = pcotdelta_function(pSQ, *pcotdelta_parameter_list)
    # print pcotdelta with label
    # print("pcotdelta: ", pcotdelta)
    q_one_minus_H_tmp = q_one_minus_H(E2CMSQ=E2CMSQ,
                                                  m1=m1, m2=m2,
                                                  alpha=alpha,
                                                  beta=beta)
    pre = 1.0
    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        pre = pre*(2.0*omspec)

    smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
    if 'smarter_q_rescale' in qc_impl:
        smarter_q_rescale = qc_impl['smarter_q_rescale']

    if smarter_q_rescale:
        pcotdelta = pcotdelta/pSQ**(ell)
        return pre*16.0*PI*ECM/(pcotdelta+q_one_minus_H_tmp)\
            / pSQ**(ell)
    else:
        pcotdelta = pcotdelta/pSQ**(ell)
        return pre*16.0*PI*ECM/(pcotdelta+q_one_minus_H_tmp)

def getK_single_entry_IPV(pcotdelta_function=None,
                          IPV_function=None,
                          pcotdelta_parameter_list=[1.0],
                          pv_shift_parameters=[1.0],
                          E=4.0, nP=np.array([0, 0, 0]), L=5.0,
                          npspec=np.array([0, 0, 0]),
                          m1=1.0, m2=1.0, mspec=1.0,
                          alpha=-1.0, beta=0.0,
                          ell=0,
                          qc_impl={}):
    """Evaluate a single ``K`` entry including the PV-shift prescription.

    Parameters
    ----------
    pcotdelta_function : callable, optional
        Function used to evaluate ``p cot(delta)``.
    IPV_function : callable, optional
        Principal-value shift function.
    pcotdelta_parameter_list : list[float], optional
        Parameters passed to ``pcotdelta_function``.
    pv_shift_parameters : list[float], optional
        Parameters passed to ``IPV_function``.
    E : float, optional
        Total energy.
    nP : numpy.ndarray, optional
        Dimensionless total momentum.
    L : float, optional
        Spatial box length.
    npspec : numpy.ndarray, optional
        Spectator momentum index.
    m1, m2, mspec : float, optional
        Channel masses.
    alpha, beta : float, optional
        Cutoff parameters.
    ell : int, optional
        Angular momentum.
    qc_impl : dict, optional
        Quantization-condition implementation options.

    Returns
    -------
    complex or float
        Requested matrix element including the PV-shift term.
    """
    if pcotdelta_function is None:
        pcotdelta_function = pcotdelta_scattering_length
    if IPV_function is None:
        IPV_function = IPV_constant
    P = TWOPI*nP/L
    pspec = TWOPI*npspec/L
    omspec = np.sqrt(mspec**2+pspec@pspec)
    E2 = E-omspec
    P2 = P-pspec
    E2CMSQ = E2**2-P2@P2
    if E2CMSQ <= 0.0 or E2 < 0.0:
        return np.nan
    ECM = np.sqrt(E2CMSQ)
    if m1 == m2:
        pSQ = E2CMSQ/4.0-m1**2
    else:
        pSQ = (E2CMSQ**2-2.0*E2CMSQ*m1**2
               + m1**4-2.0*E2CMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*E2CMSQ)
    pcotdelta = pcotdelta_function(pSQ, *pcotdelta_parameter_list)
    q_one_minus_H_value = q_one_minus_H(E2CMSQ=E2CMSQ,
                                        m1=m1, m2=m2,
                                        alpha=alpha,
                                        beta=beta)
    IPV = IPV_function(pSQ, *pv_shift_parameters)
    include_H_in_IPV = QC_IMPL_DEFAULTS['include_H_in_IPV']
    if 'include_H_in_IPV' in qc_impl:
        include_H_in_IPV = qc_impl['include_H_in_IPV']
    if include_H_in_IPV:
        Htmp = H(E2CMSQ, m1+m2, alpha, beta)
        pcot_shift = IPV/pSQ**(ell)*np.sqrt(pSQ+1.0)*Htmp
    else:
        pcot_shift = IPV/pSQ**(ell)*np.sqrt(pSQ+1.0)
    qH_IPV = q_one_minus_H_value-pcot_shift

    pre = 1.0
    hermitian = QC_IMPL_DEFAULTS['hermitian']
    if 'hermitian' in qc_impl:
        hermitian = qc_impl['hermitian']
    if hermitian:
        pre = pre*(2.0*omspec)

    smarter_q_rescale = QC_IMPL_DEFAULTS['smarter_q_rescale']
    if 'smarter_q_rescale' in qc_impl:
        smarter_q_rescale = qc_impl['smarter_q_rescale']

    if smarter_q_rescale:
        pcotdelta = pcotdelta/pSQ**(ell)
        return pre*16.0*PI*ECM/(pcotdelta+qH_IPV)\
            / pSQ**(ell)
    else:
        pcotdelta = pcotdelta/pSQ**(ell)
        return pre*16.0*PI*ECM/(pcotdelta+qH_IPV)

def getK_array(E, nP, L, m1, m2, m3, tbks_entry, slice_entry, ell,
               pcotdelta_function, pcotdelta_parameter_list, alpha, beta,
               qc_impl, three_scheme, use_pv_shift_prescription=False,
               IPV_function=None,
               pv_shift_parameters=[0.]):
    """Return the block-diagonal two-body ``K`` matrix.

    Parameters
    ----------
    E : float
        Total energy.
    nP : numpy.ndarray
        Dimensionless total momentum.
    L : float
        Spatial box length.
    m1, m2, m3 : float
        Channel masses.
    tbks_entry : object
        TBKS entry providing shell data.
    slice_entry : tuple[int, int]
        Slice selecting the spectator shell.
    ell : int
        Angular momentum.
    pcotdelta_function : callable
        Function used to evaluate ``p cot(delta)``.
    pcotdelta_parameter_list : list[float]
        Parameters passed to ``pcotdelta_function``.
    alpha, beta : float
        Cutoff parameters.
    qc_impl : dict
        Quantization-condition implementation options.
    three_scheme : str
        Three-body interaction scheme.
    use_pv_shift_prescription : bool, optional
        Whether to include the PV-shift prescription.
    IPV_function : callable, optional
        Principal-value shift function.
    pv_shift_parameters : list[float], optional
        Parameters passed to ``IPV_function``.

    Returns
    -------
    numpy.ndarray
        Block-diagonal ``K`` matrix.
    """
    nvec_arr_slice = tbks_entry.nvec_arr[slice_entry[0]:slice_entry[1]]
    k_list = []
    for nvec in nvec_arr_slice:
        if use_pv_shift_prescription:
            k_entry = getK_single_entry_IPV(
                pcotdelta_function=pcotdelta_function,
                IPV_function=IPV_function,
                pcotdelta_parameter_list=pcotdelta_parameter_list,
                pv_shift_parameters=pv_shift_parameters,
                E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3, mspec=m1,
                alpha=alpha, beta=beta, ell=ell, qc_impl=qc_impl)
        else:
            k_entry = getK_single_entry(
                pcotdelta_function=pcotdelta_function,
                pcotdelta_parameter_list=pcotdelta_parameter_list,
                E=E, nP=nP, L=L, npspec=nvec, m1=m2, m2=m3, mspec=m1,
                alpha=alpha, beta=beta, ell=ell, qc_impl=qc_impl)
        if np.abs(k_entry.imag) < EPSILON15:
            k_entry = k_entry.real
        if np.abs(k_entry) < EPSILON15:
            k_entry = 0.0
        k_list = k_list+[k_entry]*(2*ell+1)
    return block_diag(*k_list)

def get_kdf_array(E, nP, L, m1, m2, m3,
                  tbks_entry, slice_entry, ell, k3_params):
    """Return the block-diagonal ``Kdf`` contribution for a shell slice."""
    nvec_arr_slice = tbks_entry.nvec_arr[slice_entry[0]:slice_entry[1]]
    len_slice = len(nvec_arr_slice)
    if ell == 1:
        k_block = np.ones((len_slice*(2*ell+1), len_slice*(2*ell+1)))
    else:
        k_block = np.zeros((len_slice*(2*ell+1), len_slice*(2*ell+1)))
    return k_block*k3_params[0]

def getKdf_array(E, nP, L, m1, m2, m3,
                 tbks_entry,
                 row_shell, col_shell,
                 ell1, ell2,
                 k3_params,
                 alpha, beta,
                 qc_impl, three_scheme,
                 g_rescale):
    """Return the shell-resolved ``Kdf`` block for the requested channel."""
    nvec_arr_slice = tbks_entry.nvec_arr
    nvec_arr_row_slice = nvec_arr_slice[row_shell[0]:row_shell[1]]
    nvec_arr_col_slice = nvec_arr_slice[col_shell[0]:col_shell[1]]
    space_size_row = len(nvec_arr_row_slice)
    space_size_col = len(nvec_arr_col_slice)

    if ell1 == 0:
        kdf_value = np.zeros(((2*ell1+1)*space_size_row,
                              (2*ell2+1)*space_size_col))

    elif ell2 == 0:
        kdf_value = np.zeros(((2*ell1+1)*space_size_row,
                              (2*ell2+1)*space_size_col))
    else:
        single_entry = k3_params*np.identity(2*ell1+1)
        kdf_value = np.tile(single_entry, (space_size_row, space_size_col))
    return kdf_value
