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

