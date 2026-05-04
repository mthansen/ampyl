#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# fv_spectrum_utils.py
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

import warnings

import numpy as np
from scipy.optimize import root_scalar

from .constants import DEFAULT_CUTS
from .constants import DEFAULT_EMIN
from .constants import DEFAULT_LMIN
from .constants import EPSILON4
from .constants import EPSILON6
from .constants import EPSILON10
from .constants import MINMAXOFFSET
from .constants import NONINT_DIST_CUT
from .constants import QC_IMPL_DEFAULTS
from .constants import bcolors

warnings.simplefilter("once")


def _initialize_energy_scan(spectrum, version, irrep, qc_dict, dL):
    E_range, L, L_vals, Lmin, Lmax, Emax, Emin = \
        _extract_EL_set(spectrum, version, irrep, dL)
    ni_functions = _get_ni_functions(spectrum, irrep)
    all_E_vals = _get_roots_for_Erange_and_LdL(
        spectrum, E_range, L, dL, ni_functions, qc_dict)
    interp_E_vals, interp_L_vals = _build_interpolated_E_vals(
        all_E_vals, L_vals)
    return {
        'Emax': Emax,
        'Emin': Emin,
        'L': L,
        'Lmax': Lmax,
        'Lmin': Lmin,
        'interp_E_vals': interp_E_vals,
        'interp_L_vals': interp_L_vals,
        'ni_functions': ni_functions
    }


def _refine_interpolated_energies(spectrum, interp_E_vals, interp_L_vals,
                                  qc_dict, ni_functions):
    for band_index, _ in enumerate(interp_E_vals):
        for point_index, _ in enumerate(interp_E_vals[band_index]):
            _refine_interpolated_energy(
                spectrum, band_index, point_index, interp_E_vals,
                interp_L_vals, qc_dict, ni_functions)


def _refine_interpolated_energy(spectrum, band_index, point_index,
                                interp_E_vals, interp_L_vals, qc_dict,
                                ni_functions):
    Ltmp = interp_L_vals[band_index][point_index]
    Etmp = _get_interpolated_energy_guess(
        band_index, point_index, interp_E_vals, interp_L_vals)
    interp_E_vals[band_index][point_index] = Etmp
    Eupdate = _find_updated_energy(spectrum, Etmp, Ltmp, qc_dict, ni_functions)
    interp_E_vals[band_index][point_index] = Eupdate
    spectrum.qc.qcis.fvs.qc_impl['fplusg_smart_interpolate'] = True


def _get_interpolated_energy_guess(band_index, point_index,
                                   interp_E_vals, interp_L_vals):
    E_vals_tmp = np.array(interp_E_vals[band_index])
    L_vals_tmp = np.array(interp_L_vals[band_index])
    if point_index != 0 and point_index != len(E_vals_tmp)-1:
        E_vals_tmp = np.delete(E_vals_tmp, point_index)
        L_vals_tmp = np.delete(L_vals_tmp, point_index)
    sorted_indices = np.argsort(L_vals_tmp)
    Ltmp = interp_L_vals[band_index][point_index]
    return np.interp(Ltmp, L_vals_tmp[sorted_indices],
                     E_vals_tmp[sorted_indices])


def _find_updated_energy(spectrum, Etmp, Ltmp, qc_dict, ni_functions):
    Eupdate = _find_root_near_interpolated_energy(
        spectrum, Etmp, Ltmp, qc_dict, ni_functions)
    if np.isnan(Eupdate):
        return _retry_root_near_interpolated_energy(
            spectrum, Etmp, Ltmp, qc_dict, ni_functions)
    return Eupdate


def _find_root_near_interpolated_energy(spectrum, Etmp, Ltmp, qc_dict,
                                        ni_functions):
    Eupdate = np.nan
    bracket_shift = EPSILON10
    while np.isnan(Eupdate) and bracket_shift < 1.e-1:
        E_range = [Etmp-bracket_shift, Etmp+bracket_shift]
        cuts = _get_refinement_cuts()
        E_set = spectrum.get_roots_from_range(
            E_range, Ltmp, qc_dict, ni_functions, cuts=cuts)
        if len(E_set) == 1:
            Eupdate = E_set[0]
        elif len(E_set) > 1:
            index = np.abs(E_set - Etmp).argmin()
            Eupdate = E_set[index]
            warnings.warn(f"\n{bcolors.WARNING}"
                          f"multiple solutions found for L = {Ltmp},"
                          f"differences are {np.abs(E_set - Etmp)}"
                          f"{bcolors.ENDC}")
        bracket_shift = bracket_shift*10.
    return Eupdate


def _retry_root_near_interpolated_energy(spectrum, Etmp, Ltmp, qc_dict,
                                         ni_functions):
    Eupdate = np.nan
    bracket_shift = 1.e-10
    nonint_energies = np.array([ni_function(Ltmp)
                                for ni_function in ni_functions])
    while np.isnan(Eupdate) and bracket_shift < 3.e-1:
        E_bracket = [Etmp-bracket_shift, Etmp+bracket_shift]
        Eupdate = _simple_try_at_fixed_L(
            spectrum, E_bracket, Ltmp, qc_dict, nonint_energies)
        bracket_shift = bracket_shift*5.
        if np.isnan(Eupdate):
            warnings.warn(f"\n{bcolors.WARNING}"
                          "failed to find solution for "
                          f"L = {Ltmp}"
                          f"{bcolors.ENDC}")
    return Eupdate


def _get_refinement_cuts():
    cuts_a = np.logspace(-8, -2, 4)
    cuts_b = np.linspace(0.011, 0.989, 10)
    cuts_c = 1.-np.logspace(-8, -2, 4)
    cuts = np.concatenate((cuts_a, cuts_b, cuts_c))
    return np.sort(cuts)


def _extend_energy_levels(spectrum, L, Lmin, Lmax, Emin, Emax, dL, qc_dict,
                          ni_functions, interp_E_vals, interp_L_vals):
    while Lmin+np.abs(dL) <= L <= Lmax-np.abs(dL):
        L = L+dL
        for band_index, _ in enumerate(interp_E_vals):
            _append_energy_level_at_volume(
                spectrum, band_index, L, Emin, Emax, qc_dict, ni_functions,
                interp_E_vals, interp_L_vals)


def _append_energy_level_at_volume(spectrum, band_index, L, Emin, Emax,
                                   qc_dict, ni_functions, interp_E_vals,
                                   interp_L_vals):
    E_guess = _fit_energy_guess(
        interp_L_vals[band_index], interp_E_vals[band_index], L)
    E_val, dE = _find_roots_near_energy_guess(
        spectrum, E_guess, L, Emin, Emax, qc_dict, ni_functions)
    if len(E_val) == 1:
        print(f'Unique solution found with dE = {dE}')
        print(f'L = {L}, E = {E_val[0]}')
        interp_E_vals[band_index].append(E_val[0])
        interp_L_vals[band_index].append(L)
    elif len(E_val) > 1:
        index = np.abs(E_val - E_guess).argmin()
        Eupdate = E_val[index]
        interp_E_vals[band_index].append(Eupdate)
        interp_L_vals[band_index].append(L)
        warnings.warn(f'Multiple solutions found for L = {L}.\n'
                      f'Differences are {np.abs(E_val - E_guess)}')
        spectrum.qc.qcis.fvs.qc_impl['fplusg_smart_interpolate'] = True


def _fit_energy_guess(L_vals, E_vals, L):
    degree = min(len(L_vals)-1, 3)
    if len(L_vals) > 9:
        fit = np.polyfit(L_vals[-9:], E_vals[-9:], degree)
    else:
        fit = np.polyfit(L_vals, E_vals, degree)
    line = np.poly1d(fit)
    return line(L)


def _find_roots_near_energy_guess(spectrum, E_guess, L, Emin, Emax, qc_dict,
                                  ni_functions):
    dE = 1.e-6
    E_val = []
    while len(E_val) == 0 and dE < 1.e-1:
        print(f'E_guess = {E_guess}, dE = {dE}')
        E_range = [E_guess-dE, E_guess+dE]
        if _energy_guess_is_out_of_bounds(E_guess, dE, Emin, Emax):
            warnings.warn('E_guess+-dE out of bounds')
            dE = 1.0
            continue
        cuts = np.linspace(0.1, 0.9, 3)
        E_val = spectrum.get_roots_from_range(
            E_range, L, qc_dict, ni_functions, cuts=cuts)
        print(f'E_val = {E_val}')
        dE = dE*10.
    return E_val, dE


def _energy_guess_is_out_of_bounds(E_guess, dE, Emin, Emax):
    return (E_guess+dE > Emax or E_guess-dE > Emax or
            E_guess-dE < Emin or E_guess+dE < Emin)


def _get_version_and_irrep(qc_dict):
    project = qc_dict['project']
    if not project:
        raise ValueError("project must be True")
    irrep = qc_dict['irrep']
    if 'policy' in qc_dict:
        policy = qc_dict['policy']
        if hasattr(policy, 'elements'):
            version = policy.elements[-1]['version']
        elif isinstance(policy, dict):
            version = policy['version']
        else:
            version = policy[-1]['version']
    else:
        version = qc_dict['version']
    return version, irrep


def _extract_EL_set(spectrum, version, irrep, dL):
    if (version in ['kdf_zero_1+_fgcombo',
                    'kdf_zero_detf3inv_asym_fgcombo',
                    'kdf+f3inv_asym_fgcombo']
       and spectrum.qc.qcis.fvs.qc_impl['fplusg_smart_interpolate']):

        Emin_interp, Emax_interp, Lmin_interp, Lmax_interp = \
            spectrum.qc.fplusg.interp_data_lists[irrep][0][0][0]

        Emin = Emin_interp + MINMAXOFFSET
        Emax = Emax_interp - MINMAXOFFSET
        Lmin = Lmin_interp + MINMAXOFFSET
        Lmax = Lmax_interp - MINMAXOFFSET
    else:
        Emin = DEFAULT_EMIN
        Emax = spectrum.qc.qcis.Emax
        Lmin = DEFAULT_LMIN
        Lmax = spectrum.qc.qcis.Lmax
    if dL > 0.:
        L = Lmin+dL+EPSILON4
    else:
        L = Lmax+dL-EPSILON4
    E_range = [Emin, Emax]
    L_vals = [L-dL, L]
    return E_range, L, L_vals, Lmin, Lmax, Emax, Emin


def _get_ni_functions(spectrum, irrep):
    ni_functions = []
    for ni_function_channel in spectrum.qc.qcis.nonint_functions:
        ni_functions.extend(ni_function_channel[irrep])
    return ni_functions


def _get_roots_for_Erange_and_LdL(spectrum, E_range, L, dL, ni_functions,
                                  qc_dict, cuts=DEFAULT_CUTS):
    L_values = [L-dL, L]
    E_sets = []
    for Ltmp in L_values:
        E_set = spectrum.get_roots_from_range(E_range, Ltmp, qc_dict,
                                              ni_functions, cuts=cuts)
        E_sets.append(E_set)
    return E_sets


def _get_roots_from_range(spectrum, E_range, L, qc_dict, ni_functions,
                          cuts=DEFAULT_CUTS):
    if not isinstance(E_range, list) or len(E_range) != 2 or \
            not all(isinstance(E, float) for E in E_range):
        raise TypeError("E_range must be a list of two floats")
    if not isinstance(L, float):
        raise TypeError("L must be a float")
    spectrum.qc.validate_qc_dict(qc_dict)
    nonint_energies = []
    for ni_function in ni_functions:
        nonint_energies.append(ni_function(L))
    nonint_energies = np.array(nonint_energies)
    nonint_in_range = nonint_energies[E_range[0] < nonint_energies]
    nonint_in_range = nonint_in_range[nonint_in_range < E_range[1]]
    breakpoints = np.concatenate(([E_range[0]], nonint_in_range,
                                  [E_range[1]]))
    breakpoints = np.sort(breakpoints)
    differences = np.diff(breakpoints)
    cuts = np.sort(cuts)
    all_breakpoints = []
    for i in range(len(differences)):
        for cut in cuts:
            all_breakpoints.append(breakpoints[i]+cut*differences[i])
    all_breakpoints.append(breakpoints[-1])
    all_breakpoints = np.array(all_breakpoints)
    all_ranges = []
    for i in range(len(all_breakpoints)-1):
        candidate_range = [all_breakpoints[i], all_breakpoints[i+1]]
        no_nonint_in_candidate = True
        for nonint_energy in nonint_in_range:
            if candidate_range[0] < nonint_energy < candidate_range[1]:
                no_nonint_in_candidate = False
        if no_nonint_in_candidate:
            all_ranges.append([all_breakpoints[i], all_breakpoints[i+1]])
    all_roots = []
    for E_bracket in all_ranges:
        root = _simple_try_at_fixed_L(spectrum, E_bracket, L, qc_dict)
        if root is not np.nan:
            all_roots.append(root)
    return all_roots


def _simple_try_at_fixed_L(spectrum, E_bracket, L, qc_dict):
    try:
        root = root_scalar(spectrum.qc.get_value,
                           args=(L, qc_dict),
                           bracket=E_bracket).root
        abs_qc_value_at_root = np.abs(spectrum.qc.get_value(root, L, qc_dict))
        abs_qc_value_at_root_plus = np.abs(
            spectrum.qc.get_value(root+EPSILON6, L, qc_dict)
        )
        qc_ratio = abs_qc_value_at_root / abs_qc_value_at_root_plus
        if qc_ratio < EPSILON6:
            refine_roots = QC_IMPL_DEFAULTS['refine_roots']
            if 'refine_roots' in spectrum.qc.qcis.fvs.qc_impl:
                refine_roots = spectrum.qc.qcis.fvs.qc_impl['refine_roots']
            if not refine_roots:
                return root
            return _refine_root_without_interpolation(spectrum, root, L,
                                                      qc_dict)
        warnings.warn("Root was found but it failed the QC consistency "
                      "checks, returning NaN.")
        return np.nan
    except ValueError:
        warnings.warn("Root not found and ValueError was raised either by "
                      "root_scalar or by get_value. Returning NaN.")
        return np.nan
    warnings.warn("Root not found, not sure why. Returning NaN.")
    return np.nan


def _refine_root_without_interpolation(spectrum, root, L, qc_dict):
    qc_impl = spectrum.qc.qcis.fvs.qc_impl
    default_interpolation_keys = (
        'zeta_interp',
        'g_interpolate',
        'g_smart_interpolate',
        'f_interpolate',
        'f_smart_interpolate',
        'fplusg_interpolate',
        'fplusg_smart_interpolate',
        'populate_interp_zeros',
    )
    interpolation_keys = sorted(
        key for key in set(default_interpolation_keys) | set(qc_impl)
        if ('interp' in key or 'interpolate' in key)
    )
    previous_settings = {
        key: qc_impl[key] for key in interpolation_keys if key in qc_impl
    }
    try:
        for key in interpolation_keys:
            qc_impl[key] = False
        for bracket_shift in np.logspace(-9, -3, 7):
            E_bracket = [root-bracket_shift, root+bracket_shift]
            try:
                true_root = root_scalar(spectrum.qc.get_value,
                                        args=(L, qc_dict),
                                        bracket=E_bracket).root
            except ValueError:
                continue
            abs_qc_value_at_root = np.abs(
                spectrum.qc.get_value(true_root, L, qc_dict)
            )
            abs_qc_value_at_root_plus = np.abs(
                spectrum.qc.get_value(true_root+EPSILON6, L, qc_dict)
            )
            qc_ratio = abs_qc_value_at_root / abs_qc_value_at_root_plus
            if qc_ratio < EPSILON6:
                return true_root
        return np.nan
    finally:
        for key in interpolation_keys:
            if key in previous_settings:
                qc_impl[key] = previous_settings[key]
            else:
                del qc_impl[key]


def _build_interpolated_E_vals(all_E_vals, L_vals, n_interp_points=4):
    cleaned = []
    for E_vals in all_E_vals:
        unique_vals = []
        for val in E_vals:
            if not any(np.isclose(val, seen) for seen in unique_vals):
                unique_vals.append(val)
        cleaned.append(sorted(unique_vals))
    n = min(len(vals) for vals in cleaned)
    trimmed = [vals[:n] for vals in cleaned]
    for vals in trimmed:
        assert np.all(np.diff(vals) >= 0)
    grouped_E_vals = np.array(trimmed).T
    target_L_vals = np.linspace(*L_vals, n_interp_points)
    source_L_vals = np.array([target_L_vals[0], target_L_vals[-1]])
    interp_E_vals = [
        np.interp(target_L_vals, source_L_vals, E_pair).tolist()
        for E_pair in grouped_E_vals
    ]
    interp_L_vals = [target_L_vals.copy().tolist()
                     for _ in range(len(interp_E_vals))]
    return interp_E_vals, interp_L_vals
