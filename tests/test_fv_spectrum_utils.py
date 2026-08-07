#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_fv_spectrum_utils.py
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

import unittest
import warnings
import numpy as np
from ampyl import fv_spectrum_utils
from ampyl.ampyl import FVSpectrum
from ampyl.constants import DEFAULT_EMIN
from ampyl.constants import DEFAULT_LMIN
from ampyl.constants import EPSILON4
from ampyl.constants import MINMAXOFFSET


class _FakeFVS:

    def __init__(self, qc_impl=None):
        self.qc_impl = {} if qc_impl is None else qc_impl


class _FakeNIS:

    def __init__(self, nonint_functions=None):
        self.nonint_functions = ([] if nonint_functions is None
                                 else nonint_functions)


class _FakeQCIS:

    def __init__(self, Emax=4.5, Lmax=5.0, qc_impl=None,
                 nonint_functions=None):
        self.Emax = Emax
        self.Lmax = Lmax
        self.fvs = _FakeFVS(qc_impl)
        self.nis = _FakeNIS(nonint_functions)


class _FakeQC:
    """Analytic stand-in for QC whose zeros sit at known root functions."""

    def __init__(self, root_functions, qcis):
        self.root_functions = root_functions
        self.qcis = qcis

    def validate_qc_dict(self, qc_dict):
        return None

    def get_value(self, E, L, qc_dict):
        value = 1.0
        for root_function in self.root_functions:
            value = value*(E-root_function(L))
        return value


class _StubSpectrum:
    """Spectrum whose get_roots_from_range returns canned root sets."""

    def __init__(self, roots_for_call, qc_impl=None):
        self.roots_for_call = roots_for_call
        self.calls = []
        self.qc = _FakeQC([], _FakeQCIS(qc_impl=qc_impl))

    def get_roots_from_range(self, E_range, L, qc_dict, ni_functions,
                             cuts=None):
        self.calls.append((E_range[0], E_range[1], L))
        return self.roots_for_call(E_range, L)


def _analytic_spectrum(root_functions, **qcis_kwargs):
    qcis = _FakeQCIS(**qcis_kwargs)
    return FVSpectrum(_FakeQC(root_functions, qcis))


class _StubPolicy:

    def __init__(self, elements):
        self.elements = elements


class TestPureHelpers(unittest.TestCase):
    """Tests for the helpers that need no spectrum object."""

    def test_get_refinement_cuts(self):
        cuts = fv_spectrum_utils._get_refinement_cuts()
        self.assertEqual(len(cuts), 18)
        self.assertTrue(np.all(np.diff(cuts) > 0.))
        self.assertTrue(np.all(cuts > 0.))
        self.assertTrue(np.all(cuts < 1.))

    def test_fit_energy_guess_is_exact_on_linear_data(self):
        L_vals = [3.0, 3.2, 3.4, 3.6]
        E_vals = [2.0+0.5*L for L in L_vals]
        guess = fv_spectrum_utils._fit_energy_guess(L_vals, E_vals, 3.8)
        self.assertAlmostEqual(guess, 2.0+0.5*3.8, places=10)

    def test_fit_energy_guess_uses_last_nine_points(self):
        L_vals = list(np.linspace(3.0, 4.1, 12))
        E_vals = [100.0]*3+[2.0+0.5*L for L in L_vals[3:]]
        guess = fv_spectrum_utils._fit_energy_guess(L_vals, E_vals, 4.2)
        self.assertAlmostEqual(guess, 2.0+0.5*4.2, places=8)

    def test_fit_energy_guess_with_two_points(self):
        guess = fv_spectrum_utils._fit_energy_guess([3.0, 3.5], [2.0, 2.5],
                                                    4.0)
        self.assertAlmostEqual(guess, 3.0, places=10)

    def test_energy_guess_is_out_of_bounds(self):
        out = fv_spectrum_utils._energy_guess_is_out_of_bounds
        self.assertFalse(out(3.0, 0.1, 2.0, 4.0))
        self.assertTrue(out(4.05, 0.1, 2.0, 4.0))
        self.assertTrue(out(1.95, 0.1, 2.0, 4.0))
        self.assertTrue(out(5.0, 0.1, 2.0, 4.0))

    def test_get_version_and_irrep_requires_project(self):
        with self.assertRaises(ValueError):
            fv_spectrum_utils._get_version_and_irrep(
                {'project': False, 'irrep': ('A1PLUS', 0)})

    def test_get_version_and_irrep_from_version_entry(self):
        version, irrep = fv_spectrum_utils._get_version_and_irrep(
            {'project': True, 'irrep': ('A1PLUS', 0),
             'version': 'kdf_zero_1+_fgcombo'})
        self.assertEqual(version, 'kdf_zero_1+_fgcombo')
        self.assertEqual(irrep, ('A1PLUS', 0))

    def test_get_version_and_irrep_from_policy_elements(self):
        policy = _StubPolicy([{'version': 'early'}, {'version': 'late'}])
        version, irrep = fv_spectrum_utils._get_version_and_irrep(
            {'project': True, 'irrep': ('T1MINUS', 1), 'policy': policy})
        self.assertEqual(version, 'late')
        self.assertEqual(irrep, ('T1MINUS', 1))

    def test_get_version_and_irrep_from_policy_dict(self):
        version, _ = fv_spectrum_utils._get_version_and_irrep(
            {'project': True, 'irrep': ('A1PLUS', 0),
             'policy': {'version': 'dict_version'}})
        self.assertEqual(version, 'dict_version')

    def test_get_version_and_irrep_from_policy_list(self):
        version, _ = fv_spectrum_utils._get_version_and_irrep(
            {'project': True, 'irrep': ('A1PLUS', 0),
             'policy': [{'version': 'first'}, {'version': 'list_version'}]})
        self.assertEqual(version, 'list_version')

    def test_build_interpolated_E_vals(self):
        all_E_vals = [[3.0, 2.0, 3.0+1.0e-12], [3.4, 2.2]]
        interp_E_vals, interp_L_vals = \
            fv_spectrum_utils._build_interpolated_E_vals(
                all_E_vals, [3.0, 3.6])
        self.assertEqual(len(interp_E_vals), 2)
        self.assertEqual(len(interp_L_vals), 2)
        for band in interp_E_vals:
            self.assertEqual(len(band), 4)
        for L_band in interp_L_vals:
            self.assertTrue(np.allclose(L_band,
                                        np.linspace(3.0, 3.6, 4)))
        self.assertAlmostEqual(interp_E_vals[0][0], 2.0)
        self.assertAlmostEqual(interp_E_vals[0][-1], 2.2)
        self.assertAlmostEqual(interp_E_vals[1][0], 3.0)
        self.assertAlmostEqual(interp_E_vals[1][-1], 3.4)
        self.assertAlmostEqual(interp_E_vals[0][1], 2.0+0.2/3.)

    def test_build_interpolated_E_vals_trims_to_shortest_band(self):
        all_E_vals = [[2.0, 3.0, 3.5], [2.2, 3.4]]
        interp_E_vals, _ = fv_spectrum_utils._build_interpolated_E_vals(
            all_E_vals, [3.0, 3.6], n_interp_points=3)
        self.assertEqual(len(interp_E_vals), 2)
        for band in interp_E_vals:
            self.assertEqual(len(band), 3)

    def test_get_interpolated_energy_guess_interior_point(self):
        interp_E_vals = [[2.0, 2.1, 2.6, 2.3]]
        interp_L_vals = [[3.0, 3.1, 3.2, 3.3]]
        guess = fv_spectrum_utils._get_interpolated_energy_guess(
            0, 2, interp_E_vals, interp_L_vals)
        self.assertAlmostEqual(guess, 2.2, places=10)

    def test_get_interpolated_energy_guess_endpoint(self):
        interp_E_vals = [[2.0, 2.1, 2.2, 2.3]]
        interp_L_vals = [[3.0, 3.1, 3.2, 3.3]]
        guess = fv_spectrum_utils._get_interpolated_energy_guess(
            0, 0, interp_E_vals, interp_L_vals)
        self.assertAlmostEqual(guess, 2.0, places=10)

    def test_get_ni_functions_flattens_channels(self):
        irrep = ('A1PLUS', 0)

        def first(L):
            return 2.0

        def second(L):
            return 3.0

        def third(L):
            return 4.0

        spectrum = _analytic_spectrum(
            [], nonint_functions=[{irrep: [first, second]},
                                  {irrep: [third]}])
        ni_functions = fv_spectrum_utils._get_ni_functions(spectrum, irrep)
        self.assertEqual(ni_functions, [first, second, third])


class TestExtractELSet(unittest.TestCase):
    """Tests for the (E, L) window construction."""

    def test_default_window_with_positive_dL(self):
        spectrum = _analytic_spectrum([], Emax=4.5, Lmax=5.0)
        E_range, L, L_vals, Lmin, Lmax, Emax, Emin = \
            fv_spectrum_utils._extract_EL_set(
                spectrum, 'kdf_zero_1+', ('A1PLUS', 0), 0.1)
        self.assertEqual(E_range, [DEFAULT_EMIN, 4.5])
        self.assertEqual((Emin, Emax), (DEFAULT_EMIN, 4.5))
        self.assertEqual((Lmin, Lmax), (DEFAULT_LMIN, 5.0))
        self.assertAlmostEqual(L, DEFAULT_LMIN+0.1+EPSILON4, places=12)
        self.assertAlmostEqual(L_vals[0], L-0.1, places=12)
        self.assertAlmostEqual(L_vals[1], L, places=12)

    def test_default_window_with_negative_dL(self):
        spectrum = _analytic_spectrum([], Emax=4.5, Lmax=5.0)
        _, L, L_vals, _, _, _, _ = fv_spectrum_utils._extract_EL_set(
            spectrum, 'kdf_zero_1+', ('A1PLUS', 0), -0.1)
        self.assertAlmostEqual(L, 5.0-0.1-EPSILON4, places=12)
        self.assertAlmostEqual(L_vals[0], L+0.1, places=12)

    def test_interpolated_window_shrinks_by_offset(self):
        irrep = ('A1PLUS', 0)
        spectrum = _analytic_spectrum(
            [], Emax=4.5, Lmax=5.0, qc_impl={'fplusg_interpolate': True})

        class _FakeFplusG:
            interp_data_lists = {irrep: [[[[1.9, 4.4, 3.05, 4.95]]]]}

        spectrum.qc.fplusg = _FakeFplusG()
        E_range, L, _, Lmin, Lmax, Emax, Emin = \
            fv_spectrum_utils._extract_EL_set(
                spectrum, 'kdf_zero_1+_fgcombo', irrep, 0.1)
        self.assertAlmostEqual(Emin, 1.9+MINMAXOFFSET, places=12)
        self.assertAlmostEqual(Emax, 4.4-MINMAXOFFSET, places=12)
        self.assertAlmostEqual(Lmin, 3.05+MINMAXOFFSET, places=12)
        self.assertAlmostEqual(Lmax, 4.95-MINMAXOFFSET, places=12)
        self.assertEqual(E_range, [Emin, Emax])
        self.assertAlmostEqual(L, Lmin+0.1+EPSILON4, places=12)


class TestGetRootsFromRange(unittest.TestCase):
    """Tests for root finding at fixed volume with an analytic QC."""

    def test_E_range_type_validation(self):
        spectrum = _analytic_spectrum([])
        qc_dict = {}
        with self.assertRaises(TypeError):
            spectrum.get_roots_from_range((1.9, 4.4), 4.0, qc_dict, [])
        with self.assertRaises(TypeError):
            spectrum.get_roots_from_range([1.9, 3.0, 4.4], 4.0, qc_dict, [])
        with self.assertRaises(TypeError):
            spectrum.get_roots_from_range([2, 4], 4.0, qc_dict, [])
        with self.assertRaises(TypeError):
            spectrum.get_roots_from_range([1.9, 4.4], 4, qc_dict, [])

    def test_finds_single_root(self):
        spectrum = _analytic_spectrum([lambda L: 2.5+1./L])
        roots = spectrum.get_roots_from_range(
            [1.9, 4.4], 4.0, {}, [lambda L: 4.2])
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 2.75, places=10)

    def test_finds_two_roots_in_order(self):
        spectrum = _analytic_spectrum(
            [lambda L: 2.5+1./L, lambda L: 3.3+2./L])
        roots = spectrum.get_roots_from_range(
            [1.9, 4.4], 4.0, {}, [lambda L: 4.2])
        self.assertEqual(len(roots), 2)
        self.assertAlmostEqual(roots[0], 2.75, places=10)
        self.assertAlmostEqual(roots[1], 3.8, places=10)

    def test_discards_root_near_nonint_energy(self):
        spectrum = _analytic_spectrum([lambda L: 2.5+1./L])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            roots = spectrum.get_roots_from_range(
                [1.9, 4.4], 4.0, {}, [lambda L: 2.7503])
        self.assertEqual(len(roots), 0)

    def test_keeps_root_near_nonint_energy_when_discard_disabled(self):
        spectrum = _analytic_spectrum(
            [lambda L: 2.5+1./L],
            qc_impl={'discard_non_interacting': False})
        roots = spectrum.get_roots_from_range(
            [1.9, 4.4], 4.0, {}, [lambda L: 2.7503])
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 2.75, places=8)

    def test_refine_roots_restores_interpolation_settings(self):
        qc_impl = {'refine_roots': True, 'f_interpolate': True}
        spectrum = _analytic_spectrum([lambda L: 2.5+1./L], qc_impl=qc_impl)
        roots = spectrum.get_roots_from_range(
            [1.9, 4.4], 4.0, {}, [lambda L: 4.2])
        self.assertEqual(len(roots), 1)
        self.assertAlmostEqual(roots[0], 2.75, places=8)
        self.assertEqual(qc_impl,
                         {'refine_roots': True, 'f_interpolate': True})


class TestSimpleTryAtFixedL(unittest.TestCase):
    """Tests for the single-bracket root attempt."""

    def test_returns_root_for_bracket_with_sign_change(self):
        spectrum = _analytic_spectrum([lambda L: 2.75])
        root = fv_spectrum_utils._simple_try_at_fixed_L(
            spectrum, [2.7, 2.8], 4.0, {})
        self.assertAlmostEqual(root, 2.75, places=10)

    def test_returns_nan_without_sign_change(self):
        spectrum = _analytic_spectrum([lambda L: 2.75])
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            root = fv_spectrum_utils._simple_try_at_fixed_L(
                spectrum, [3.0, 3.1], 4.0, {})
        self.assertTrue(np.isnan(root))

    def test_discards_root_near_nonint_energy(self):
        spectrum = _analytic_spectrum([lambda L: 2.75])
        root = fv_spectrum_utils._simple_try_at_fixed_L(
            spectrum, [2.7, 2.8], 4.0, {},
            nonint_energies=np.array([2.7502]))
        self.assertTrue(np.isnan(root))

    def test_sign_change_without_zero_fails_consistency_check(self):
        spectrum = _analytic_spectrum([])
        spectrum.qc.get_value = \
            lambda E, L, qc_dict: -1.0 if E < 2.75 else 1.0
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            root = fv_spectrum_utils._simple_try_at_fixed_L(
                spectrum, [2.7, 2.8], 4.0, {})
        self.assertTrue(np.isnan(root))


class TestRefineRootWithoutInterpolation(unittest.TestCase):
    """Tests for the interpolation-free refinement pass."""

    def test_refines_root_and_restores_settings(self):
        qc_impl = {'f_interpolate': True, 'fplusg_interpolate': True}
        spectrum = _analytic_spectrum([lambda L: 2.75], qc_impl=qc_impl)
        root = fv_spectrum_utils._refine_root_without_interpolation(
            spectrum, 2.75+1.0e-8, 4.0, {})
        self.assertAlmostEqual(root, 2.75, places=10)
        self.assertEqual(
            qc_impl, {'f_interpolate': True, 'fplusg_interpolate': True})

    def test_returns_nan_and_restores_settings_on_failure(self):
        qc_impl = {'f_interpolate': True}
        spectrum = _analytic_spectrum([], qc_impl=qc_impl)
        root = fv_spectrum_utils._refine_root_without_interpolation(
            spectrum, 2.75, 4.0, {})
        self.assertTrue(np.isnan(root))
        self.assertEqual(qc_impl, {'f_interpolate': True})

    def test_discards_refined_root_near_nonint_energy(self):
        qc_impl = {}
        spectrum = _analytic_spectrum([lambda L: 2.75], qc_impl=qc_impl)
        root = fv_spectrum_utils._refine_root_without_interpolation(
            spectrum, 2.75+1.0e-8, 4.0, {},
            nonint_energies=np.array([2.7502]))
        self.assertTrue(np.isnan(root))
        self.assertEqual(qc_impl, {})

    def test_explicit_discard_override_keeps_nearby_root(self):
        qc_impl = {'discard_non_interacting': False}
        spectrum = _analytic_spectrum([lambda L: 2.75], qc_impl=qc_impl)
        root = fv_spectrum_utils._refine_root_without_interpolation(
            spectrum, 2.75+1.0e-8, 4.0, {},
            nonint_energies=np.array([2.7502]))
        self.assertAlmostEqual(root, 2.75, places=10)
        self.assertEqual(qc_impl, {'discard_non_interacting': False})


class TestInterpolatedEnergyRefinement(unittest.TestCase):
    """Tests for the interpolation-based refinement drivers."""

    def test_find_updated_energy_uses_direct_root(self):
        spectrum = _StubSpectrum(lambda E_range, L: np.array([2.31]))
        Eupdate = fv_spectrum_utils._find_updated_energy(
            spectrum, 2.3, 4.0, {}, [])
        self.assertAlmostEqual(Eupdate, 2.31, places=12)

    def test_find_root_near_interpolated_energy_picks_nearest(self):
        spectrum = _StubSpectrum(
            lambda E_range, L: np.array([2.31, 2.9]))
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            Eupdate = fv_spectrum_utils._find_root_near_interpolated_energy(
                spectrum, 2.3, 4.0, {}, [])
        self.assertAlmostEqual(Eupdate, 2.31, places=12)

    def test_find_updated_energy_falls_back_to_retry(self):
        spectrum = _StubSpectrum(lambda E_range, L: np.array([]))
        spectrum.qc = _FakeQC([lambda L: 2.31], _FakeQCIS())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Eupdate = fv_spectrum_utils._find_updated_energy(
                spectrum, 2.3, 4.0, {}, [lambda L: 4.2])
        self.assertAlmostEqual(Eupdate, 2.31, places=8)

    def test_refine_interpolated_energies_updates_all_points(self):
        spectrum = _StubSpectrum(lambda E_range, L: np.array([2.22]))
        interp_E_vals = [[2.0, 2.1, 2.2, 2.3]]
        interp_L_vals = [[3.0, 3.1, 3.2, 3.3]]
        fv_spectrum_utils._refine_interpolated_energies(
            spectrum, interp_E_vals, interp_L_vals, {}, [])
        self.assertTrue(np.allclose(interp_E_vals[0], [2.22]*4))
        self.assertTrue(
            spectrum.qc.qcis.fvs.qc_impl['fplusg_interpolate'])


class TestExtendEnergyLevels(unittest.TestCase):
    """Tests for extending bands to new volumes."""

    def test_append_energy_level_unique_solution(self):
        spectrum = _StubSpectrum(lambda E_range, L: np.array([2.55]))
        interp_E_vals = [[2.4, 2.45, 2.5]]
        interp_L_vals = [[3.0, 3.1, 3.2]]
        fv_spectrum_utils._append_energy_level_at_volume(
            spectrum, 0, 3.3, 1.801, 4.5, {}, [],
            interp_E_vals, interp_L_vals)
        self.assertEqual(len(interp_E_vals[0]), 4)
        self.assertAlmostEqual(interp_E_vals[0][-1], 2.55, places=12)
        self.assertAlmostEqual(interp_L_vals[0][-1], 3.3, places=12)

    def test_append_energy_level_multiple_solutions_picks_nearest(self):
        spectrum = _StubSpectrum(
            lambda E_range, L: np.array([2.56, 3.2]))
        interp_E_vals = [[2.4, 2.45, 2.5]]
        interp_L_vals = [[3.0, 3.1, 3.2]]
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            fv_spectrum_utils._append_energy_level_at_volume(
                spectrum, 0, 3.3, 1.801, 4.5, {}, [],
                interp_E_vals, interp_L_vals)
        self.assertEqual(len(interp_E_vals[0]), 4)
        self.assertAlmostEqual(interp_E_vals[0][-1], 2.56, places=12)
        self.assertTrue(
            spectrum.qc.qcis.fvs.qc_impl['fplusg_interpolate'])

    def test_append_energy_level_out_of_bounds_appends_nothing(self):
        spectrum = _StubSpectrum(lambda E_range, L: np.array([4.6]))
        interp_E_vals = [[4.4, 4.45, 4.5]]
        interp_L_vals = [[3.0, 3.1, 3.2]]
        with self.assertWarns(UserWarning):
            warnings.simplefilter("always")
            fv_spectrum_utils._append_energy_level_at_volume(
                spectrum, 0, 3.3, 1.801, 4.5, {}, [],
                interp_E_vals, interp_L_vals)
        self.assertEqual(len(interp_E_vals[0]), 3)
        self.assertEqual(len(spectrum.calls), 0)

    def test_extend_energy_levels_tracks_band_across_volumes(self):
        def root_function(L):
            return 2.0+2./L

        spectrum = _analytic_spectrum([root_function], Emax=4.5, Lmax=4.0)
        L_seed = [3.0501, 3.1501, 3.2501]
        interp_E_vals = [[root_function(L) for L in L_seed]]
        interp_L_vals = [list(L_seed)]
        fv_spectrum_utils._extend_energy_levels(
            spectrum, 3.2501, 3.0, 4.0, 1.801, 4.5, 0.25, {},
            [lambda L: 4.3], interp_E_vals, interp_L_vals)
        self.assertEqual(len(interp_L_vals[0]), 5)
        self.assertAlmostEqual(interp_L_vals[0][-2], 3.5001, places=10)
        self.assertAlmostEqual(interp_L_vals[0][-1], 3.7501, places=10)
        for L, E in zip(interp_L_vals[0], interp_E_vals[0]):
            self.assertAlmostEqual(E, root_function(L), places=6)


class TestInitializeEnergyScan(unittest.TestCase):
    """End-to-end test of the scan initialization on an analytic QC."""

    def test_initialize_energy_scan_with_two_bands(self):
        irrep = ('A1PLUS', 0)

        def band0(L):
            return 2.2+1./L

        def band1(L):
            return 3.3+2./L

        spectrum = _analytic_spectrum(
            [band0, band1], Emax=4.5, Lmax=5.0,
            nonint_functions=[{irrep: [lambda L: 4.3]}])
        state = fv_spectrum_utils._initialize_energy_scan(
            spectrum, 'kdf_zero_1+', irrep, {}, 0.1)
        self.assertEqual(
            sorted(state.keys()),
            ['Emax', 'Emin', 'L', 'Lmax', 'Lmin', 'interp_E_vals',
             'interp_L_vals', 'ni_functions'])
        self.assertEqual(state['Emax'], 4.5)
        self.assertEqual(state['Emin'], DEFAULT_EMIN)
        self.assertEqual(state['Lmin'], DEFAULT_LMIN)
        self.assertEqual(state['Lmax'], 5.0)
        self.assertEqual(len(state['ni_functions']), 1)
        self.assertEqual(len(state['interp_E_vals']), 2)
        L_start = state['L']-0.1
        L_end = state['L']
        self.assertTrue(np.allclose(
            state['interp_L_vals'][0], np.linspace(L_start, L_end, 4)))
        self.assertAlmostEqual(
            state['interp_E_vals'][0][0], band0(L_start), places=8)
        self.assertAlmostEqual(
            state['interp_E_vals'][0][-1], band0(L_end), places=8)
        self.assertAlmostEqual(
            state['interp_E_vals'][1][0], band1(L_start), places=8)
        self.assertAlmostEqual(
            state['interp_E_vals'][1][-1], band1(L_end), places=8)


class Template(unittest.TestCase):
    """Test."""

    def setUp(self):
        """Exectue set-up."""
        pass

    def tearDown(self):
        """Execute tear-down."""
        pass

    def __example(self, x):
        return x

    def test(self):
        """Example test."""
        self.assertEqual(10.0, self.__example(10.0))


if __name__ == '__main__':
    unittest.main()
