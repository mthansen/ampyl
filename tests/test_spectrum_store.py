#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# test_spectrum_store.py
#
# MIT License
# Copyright (c) 2026 Maxwell T. Hansen
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

import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

from ampyl import spectrum_store_utils
from ampyl.ampyl import EvaluationPolicy
from ampyl.ampyl import FVSpectrum
from ampyl.constants import QC_DICT_DEFAULTS
from ampyl.spectrum_store_utils import SpectrumValueStore

IRREP = ('A1PLUS', 0)


class _FakeFVS:

    def __init__(self, qc_impl=None):
        self.qc_impl = {} if qc_impl is None else qc_impl


class _FakeNIS:

    def __init__(self, nonint_functions=None):
        self.nonint_functions = ([] if nonint_functions is None
                                 else nonint_functions)


class _FakeQCIS:

    def __init__(self, Emax=4.5, Lmax=4.0, qc_impl=None,
                 nonint_functions=None):
        self.Emax = Emax
        self.Lmax = Lmax
        self.fvs = _FakeFVS(qc_impl)
        self.nis = _FakeNIS(nonint_functions)


class _FakeQC:
    """Analytic stand-in for QC that counts its evaluations."""

    def __init__(self, root_functions, qcis):
        self.root_functions = root_functions
        self.qcis = qcis
        self.calls = 0

    def validate_qc_dict(self, qc_dict):
        return None

    def get_value(self, E, L, qc_dict):
        self.calls = self.calls+1
        value = 1.0
        for root_function in self.root_functions:
            value = value*(E-root_function(L))
        return value


def _band(L):
    return 2.2+1./L


def _spectrum(root_functions=None, value_store=None, store_context=None,
              **qcis_kwargs):
    if root_functions is None:
        root_functions = [_band]
    qcis_kwargs.setdefault('nonint_functions',
                           [{IRREP: [lambda L: 4.3]}])
    qcis = _FakeQCIS(**qcis_kwargs)
    return FVSpectrum(_FakeQC(root_functions, qcis),
                      value_store=value_store,
                      store_context=store_context)


def _qc_dict():
    return {'project': True, 'irrep': IRREP, 'version': 'kdf_zero_1+',
            'k_params': [[[0.1]], [0.0]]}


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class TestCanonicalValue(unittest.TestCase):
    """Tests for the canonicalization used to build context keys."""

    def test_dictionaries_are_sorted_by_string_key(self):
        canonical = spectrum_store_utils.canonical_value(
            {2: 'b', 'a': 1, 1: 'c'})
        self.assertEqual(list(canonical), ['1', '2', 'a'])

    def test_arrays_and_tuples_become_lists(self):
        canonical = spectrum_store_utils.canonical_value(
            {'a': np.array([1.5, 2.5]), 'b': (1, 2)})
        self.assertEqual(canonical, {'a': [1.5, 2.5], 'b': [1, 2]})

    def test_booleans_stay_booleans(self):
        canonical = spectrum_store_utils.canonical_value(
            {'a': True, 'b': 1, 'c': np.bool_(False)})
        self.assertIs(canonical['a'], True)
        self.assertIsInstance(canonical['b'], int)
        self.assertNotIsInstance(canonical['b'], bool)
        self.assertIs(canonical['c'], False)

    def test_floats_are_rounded(self):
        canonical = spectrum_store_utils.canonical_value(
            {'a': 0.1+1.0e-15})
        self.assertEqual(canonical['a'], 0.1)

    def test_unsupported_objects_fall_back_to_their_string(self):
        canonical = spectrum_store_utils.canonical_value({'a': object})
        self.assertIsInstance(canonical['a'], str)

    def test_coordinate_key_is_round_trip_stable(self):
        key = spectrum_store_utils.coordinate_key(3.51)
        self.assertEqual(key, '3.5100000000')
        self.assertEqual(spectrum_store_utils.coordinate_key(float(key)), key)

    def test_coordinate_key_absorbs_round_off(self):
        self.assertEqual(spectrum_store_utils.coordinate_key(3.51),
                         spectrum_store_utils.coordinate_key(3.51+1.0e-13))


class TestContextKey(unittest.TestCase):
    """Tests for context construction and hashing."""

    def test_key_is_insensitive_to_dictionary_order(self):
        first = spectrum_store_utils.context_key({'a': 1, 'b': 2})
        second = spectrum_store_utils.context_key({'b': 2, 'a': 1})
        self.assertEqual(first, second)

    def test_key_changes_with_k_params(self):
        spectrum = _spectrum()
        qc_dict = _qc_dict()
        first = spectrum_store_utils.build_context(spectrum.qc, qc_dict)
        qc_dict['k_params'] = [[[0.2]], [0.0]]
        second = spectrum_store_utils.build_context(spectrum.qc, qc_dict)
        self.assertNotEqual(spectrum_store_utils.context_key(first),
                            spectrum_store_utils.context_key(second))

    def test_key_changes_with_extra_entries(self):
        spectrum = _spectrum()
        qc_dict = _qc_dict()
        first = spectrum_store_utils.build_context(spectrum.qc, qc_dict)
        second = spectrum_store_utils.build_context(
            spectrum.qc, qc_dict, extra={'dE': 0.005})
        self.assertNotEqual(spectrum_store_utils.context_key(first),
                            spectrum_store_utils.context_key(second))

    def test_key_survives_in_place_qc_dict_validation(self):
        """A validated qc_dict must not land in a different context.

        ``QC.validate_qc_dict`` fills defaults and replaces the policy
        with an ``EvaluationPolicy`` in place, which happens on the first
        evaluation of a run.
        """
        spectrum = _spectrum()
        qc_dict = _qc_dict()
        before = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        for key, default in QC_DICT_DEFAULTS.items():
            qc_dict.setdefault(key, default)
        qc_dict['policy'] = EvaluationPolicy.from_qc_dict(qc_dict)
        after = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        self.assertEqual(before, after)

    def test_key_ignores_interpolation_flags(self):
        """Toggling an interpolator must not orphan a stored spectrum."""
        spectrum = _spectrum(qc_impl={'refine_roots': True})
        qc_dict = _qc_dict()
        before = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        spectrum.qc.qcis.fvs.qc_impl['fplusg_interpolate'] = True
        after = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        self.assertEqual(before, after)

    def test_key_changes_with_refine_roots(self):
        spectrum = _spectrum(qc_impl={'refine_roots': False})
        qc_dict = _qc_dict()
        before = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        spectrum.qc.qcis.fvs.qc_impl['refine_roots'] = True
        after = spectrum_store_utils.context_key(
            spectrum_store_utils.build_context(spectrum.qc, qc_dict))
        self.assertNotEqual(before, after)

    def test_policy_summary_matches_evaluation_policy_form(self):
        elements = [{'version': 'first', 'Lmin': 3.0, 'Lmax': 4.0,
                     'Emin': 2.0, 'Emax': 5.0},
                    {'version': 'default'}]
        from_list = spectrum_store_utils.policy_summary(
            {'policy': elements})
        from_policy = spectrum_store_utils.policy_summary(
            {'policy': EvaluationPolicy(elements)})
        self.assertEqual(from_list, from_policy)
        self.assertEqual(from_list[0]['id'], 0)
        self.assertIsNone(from_list[1]['Lmin'])

    def test_qc_impl_summary_fills_defaults_and_extras(self):
        summary = spectrum_store_utils.qc_impl_summary({'custom_flag': 7})
        self.assertEqual(summary['custom_flag'], 7)
        self.assertTrue(summary['smarter_q_rescale'])
        self.assertNotIn('fplusg_interpolate', summary)

    def test_qc_impl_summary_can_keep_interpolation_flags(self):
        summary = spectrum_store_utils.qc_impl_summary(
            {'fplusg_interpolate': True}, include_interpolation=True)
        self.assertTrue(summary['fplusg_interpolate'])


class TestSpectrumValueStore(unittest.TestCase):
    """Tests for the in-memory behavior of the store."""

    def setUp(self):
        """Build a store and a context to record against."""
        self.store = SpectrumValueStore()
        self.context = {'label': 'test'}

    def test_records_and_returns_one_energy(self):
        self.store.put_root(self.context, 0, 3.51, 3.8)
        self.assertAlmostEqual(
            self.store.get_root(self.context, 0, 3.51), 3.8, places=12)

    def test_missing_energy_returns_none(self):
        self.assertIsNone(self.store.get_root(self.context, 0, 3.51))
        self.assertIsNone(self.store.get_curves(self.context))

    def test_non_finite_energies_are_ignored(self):
        self.store.put_root(self.context, 0, 3.51, np.nan)
        self.store.put_root(self.context, 0, 3.61, np.inf)
        self.assertIsNone(self.store.get_curves(self.context))

    def test_curves_are_sorted_by_volume_and_grouped_by_band(self):
        self.store.put_root(self.context, 1, 3.6, 4.1)
        self.store.put_root(self.context, 0, 3.6, 3.9)
        self.store.put_root(self.context, 0, 3.5, 3.8)
        L_vals, E_vals = self.store.get_curves(self.context)
        self.assertEqual(self.store.band_count(self.context), 2)
        self.assertTrue(np.allclose(L_vals[0], [3.5, 3.6]))
        self.assertTrue(np.allclose(E_vals[0], [3.8, 3.9]))
        self.assertTrue(np.allclose(L_vals[1], [3.6]))
        self.assertTrue(np.allclose(E_vals[1], [4.1]))

    def test_repeated_volume_overwrites_the_energy(self):
        self.store.put_root(self.context, 0, 3.5, 3.8)
        self.store.put_root(self.context, 0, 3.5+1.0e-13, 3.9)
        _, E_vals = self.store.get_curves(self.context)
        self.assertEqual(E_vals[0], [3.9])

    def test_metadata_round_trip(self):
        self.store.set_meta(self.context, complete=True, dL=0.25)
        meta = self.store.get_meta(self.context)
        self.assertIs(meta['complete'], True)
        self.assertEqual(meta['dL'], 0.25)

    def test_values_are_not_memoized_by_default(self):
        self.store.put_value(self.context, 3.9, 4.0, 1.25)
        self.assertIsNone(self.store.get_value(self.context, 3.9, 4.0))

    def test_values_are_memoized_when_enabled(self):
        store = SpectrumValueStore(memoize_values=True)
        store.put_value(self.context, 3.9, 4.0, 1.25)
        self.assertAlmostEqual(store.get_value(self.context, 3.9, 4.0),
                               1.25, places=12)
        self.assertIsNone(store.get_value(self.context, 3.9, 4.5))

    def test_nearest_context_picks_the_closest_populated_entry(self):
        near = {'a': 0.2}
        far = {'a': 0.9}
        empty = {'a': 0.21}
        self.store.put_root(near, 0, 3.5, 3.8)
        self.store.put_root(far, 0, 3.5, 3.9)
        self.store.register(empty)
        reference = {'a': 0.3}
        self.store.register(reference)
        found = self.store.nearest_context(
            reference, lambda other: abs(other['a']-0.3))
        self.assertEqual(found, near)

    def test_nearest_context_ignores_incomparable_entries(self):
        self.store.put_root({'a': 0.2}, 0, 3.5, 3.8)
        found = self.store.nearest_context({'a': 0.3}, lambda other: None)
        self.assertIsNone(found)

    def test_clear_discards_recorded_energies(self):
        self.store.put_root(self.context, 0, 3.5, 3.8)
        self.store.set_meta(self.context, complete=True)
        self.store.clear(self.context)
        self.assertIsNone(self.store.get_curves(self.context))
        self.assertEqual(self.store.get_meta(self.context), {})

    def test_memoized_values_are_keyed_at_full_precision(self):
        """A root finder samples far closer together than the rounding."""
        store = SpectrumValueStore(memoize_values=True)
        store.put_value(self.context, 3.9, 4.0, 1.25)
        self.assertIsNone(store.get_value(self.context, 3.9+1.0e-12, 4.0))

    def test_merge_combines_bands_and_values(self):
        other = SpectrumValueStore()
        self.store.put_root(self.context, 0, 3.5, 3.8)
        other.put_root(self.context, 0, 3.6, 3.9)
        other.put_root(self.context, 1, 3.6, 4.1)
        self.store.merge(other)
        L_vals, E_vals = self.store.get_curves(self.context)
        self.assertTrue(np.allclose(L_vals[0], [3.5, 3.6]))
        self.assertTrue(np.allclose(E_vals[0], [3.8, 3.9]))
        self.assertEqual(self.store.band_count(self.context), 2)


class TestSpectrumValueStoreOnDisk(unittest.TestCase):
    """Tests for persistence and lookup of stored energies."""

    def setUp(self):
        """Create a scratch directory for the store."""
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.context = {'label': 'test'}

    def tearDown(self):
        """Remove the scratch directory."""
        self._tmp.cleanup()

    def test_save_and_reload_a_context(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0)
        store.put_root(self.context, 0, 3.5, 3.8)
        store.set_meta(self.context, complete=True, dL=0.25)
        store.save()
        reloaded = spectrum_store_utils.load_store(self.directory)
        L_vals, E_vals = reloaded.get_curves(self.context)
        self.assertTrue(np.allclose(L_vals[0], [3.5]))
        self.assertTrue(np.allclose(E_vals[0], [3.8]))
        self.assertIs(reloaded.get_meta(self.context)['complete'], True)

    def test_a_fresh_store_finds_the_context_by_key(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0)
        store.put_root(self.context, 0, 3.5, 3.8)
        store.save()
        fresh = SpectrumValueStore(directory=self.directory)
        self.assertIsNotNone(fresh.get_curves(self.context))

    def test_autosave_writes_without_an_explicit_save(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=2)
        store.put_root(self.context, 0, 3.5, 3.8)
        key = spectrum_store_utils.context_key(self.context)
        path = self.directory/f"roots_{key}.json"
        self.assertFalse(path.is_file())
        store.put_root(self.context, 0, 3.6, 3.9)
        self.assertTrue(path.is_file())

    def test_stored_file_is_plain_json(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0)
        store.put_root(self.context, 0, 3.5, 3.8)
        store.save()
        key = spectrum_store_utils.context_key(self.context)
        with open(self.directory/f"roots_{key}.json", 'r') as file_handle:
            payload = json.load(file_handle)
        self.assertEqual(payload['context'], self.context)
        self.assertEqual(payload['bands']['0']['3.5000000000'], 3.8)
        self.assertIn('ampyl_version', payload)

    def test_no_temporary_files_are_left_behind(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0)
        store.put_root(self.context, 0, 3.5, 3.8)
        store.save()
        self.assertEqual(list(self.directory.glob('*.tmp*')), [])

    def test_memoized_values_are_persisted_separately(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0, memoize_values=True)
        store.put_value(self.context, 3.9, 4.0, 1.25)
        store.save()
        reloaded = SpectrumValueStore(directory=self.directory,
                                      memoize_values=True)
        self.assertAlmostEqual(reloaded.get_value(self.context, 3.9, 4.0),
                               1.25, places=12)

    def test_disk_entries_do_not_overwrite_newer_memory_entries(self):
        store = SpectrumValueStore(directory=self.directory,
                                   autosave_every=0)
        store.put_root(self.context, 0, 3.5, 3.8)
        store.save()
        fresh = SpectrumValueStore(directory=self.directory)
        fresh.put_root(self.context, 0, 3.5, 4.4)
        _, E_vals = fresh.get_curves(self.context)
        self.assertEqual(E_vals[0], [4.4])


class TestFVSpectrumWithoutStore(unittest.TestCase):
    """Tests that the solver is unchanged when no store is attached."""

    def test_record_root_is_a_no_op(self):
        spectrum = _spectrum()
        spectrum.record_root(0, 3.5, 3.8)
        self.assertIsNone(spectrum.value_store)

    def test_get_value_delegates_to_the_qc(self):
        spectrum = _spectrum()
        value = spectrum.get_value(3.0, 4.0, _qc_dict())
        self.assertAlmostEqual(value, 3.0-_band(4.0), places=12)
        self.assertEqual(spectrum.qc.calls, 1)


class TestFVSpectrumValueMemoization(unittest.TestCase):
    """Tests for the optional QC-value cache."""

    def setUp(self):
        """Attach a memoizing store to an analytic spectrum."""
        self.spectrum = _spectrum(
            value_store=SpectrumValueStore(memoize_values=True))
        self.qc_dict = _qc_dict()

    def test_repeated_evaluation_is_served_from_the_store(self):
        first = self.spectrum.get_value(3.0, 4.0, self.qc_dict)
        calls = self.spectrum.qc.calls
        second = self.spectrum.get_value(3.0, 4.0, self.qc_dict)
        self.assertEqual(self.spectrum.qc.calls, calls)
        self.assertAlmostEqual(first, second, places=12)

    def test_a_new_point_is_evaluated(self):
        self.spectrum.get_value(3.0, 4.0, self.qc_dict)
        calls = self.spectrum.qc.calls
        self.spectrum.get_value(3.1, 4.0, self.qc_dict)
        self.assertEqual(self.spectrum.qc.calls, calls+1)

    def test_toggling_interpolation_forces_a_new_evaluation(self):
        """The interpolation-free refinement pass must not be short-cut."""
        self.spectrum.get_value(3.0, 4.0, self.qc_dict)
        calls = self.spectrum.qc.calls
        self.spectrum.qc.qcis.fvs.qc_impl['fplusg_interpolate'] = True
        self.spectrum.get_value(3.0, 4.0, self.qc_dict)
        self.assertEqual(self.spectrum.qc.calls, calls+1)


class TestFVSpectrumStoredSpectra(unittest.TestCase):
    """End-to-end tests of the stored spectrum on an analytic QC."""

    def test_get_all_energies_is_served_from_the_store_on_re_entry(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        qc_dict = _qc_dict()
        with _quiet():
            L_first, E_first = spectrum.get_all_energies(qc_dict, dL=0.25)
        calls = spectrum.qc.calls
        self.assertGreater(calls, 0)
        with _quiet():
            L_second, E_second = spectrum.get_all_energies(qc_dict, dL=0.25)
        self.assertEqual(spectrum.qc.calls, calls)
        self.assertTrue(np.allclose(L_first[0], L_second[0]))
        self.assertTrue(np.allclose(E_first[0], E_second[0]))

    def test_stored_energies_solve_the_quantization_condition(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        with _quiet():
            L_vals, E_vals = spectrum.get_all_energies(_qc_dict(), dL=0.25)
        for L, E in zip(L_vals[0], E_vals[0]):
            self.assertAlmostEqual(E, _band(L), places=6)

    def test_a_different_step_size_is_not_served_from_the_store(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        qc_dict = _qc_dict()
        with _quiet():
            spectrum.get_all_energies(qc_dict, dL=0.25)
        calls = spectrum.qc.calls
        with _quiet():
            spectrum.get_all_energies(qc_dict, dL=0.5)
        self.assertGreater(spectrum.qc.calls, calls)

    def test_a_different_parameter_set_is_not_served_from_the_store(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        with _quiet():
            spectrum.get_all_energies(_qc_dict(), dL=0.25)
        calls = spectrum.qc.calls
        other_qc_dict = _qc_dict()
        other_qc_dict['k_params'] = [[[0.2]], [0.0]]
        with _quiet():
            spectrum.get_all_energies(other_qc_dict, dL=0.25)
        self.assertGreater(spectrum.qc.calls, calls)

    def test_energies_survive_a_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            spectrum = _spectrum()
            spectrum.enable_value_store(directory=directory)
            qc_dict = _qc_dict()
            with _quiet():
                L_first, E_first = spectrum.get_all_energies(qc_dict, dL=0.25)
            resumed = _spectrum()
            resumed.enable_value_store(directory=directory)
            with _quiet():
                L_second, E_second = resumed.get_all_energies(qc_dict,
                                                              dL=0.25)
            self.assertEqual(resumed.qc.calls, 0)
            self.assertTrue(np.allclose(L_first[0], L_second[0]))
            self.assertTrue(np.allclose(E_first[0], E_second[0]))

    def test_an_interrupted_run_is_continued_from_the_store(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        qc_dict = _qc_dict()
        with _quiet():
            L_vals, E_vals = spectrum.get_all_energies(qc_dict, dL=0.25)
        context = spectrum.store_context_for(qc_dict)
        store.set_meta(context, complete=False)
        resumed = _spectrum(value_store=store)
        with _quiet():
            L_resumed, E_resumed = resumed.get_all_energies(qc_dict, dL=0.25)
        self.assertTrue(np.allclose(L_vals[0], L_resumed[0]))
        self.assertTrue(np.allclose(E_vals[0], E_resumed[0]))


class TestFVSpectrumStoreFallback(unittest.TestCase):
    """Tests that unusable stored energies are discarded, not returned."""

    def _poison(self, spectrum, store, qc_dict):
        """Record energies above the non-interacting pole, where no root is."""
        context = spectrum.store_context_for(qc_dict, refresh=True)
        for L in [3.05, 3.3, 3.55]:
            store.put_root(context, 0, L, 4.4)
        return context

    def test_unrefinable_stored_energies_trigger_a_full_scan(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        qc_dict = _qc_dict()
        context = self._poison(spectrum, store, qc_dict)
        with _quiet():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                L_vals, E_vals = spectrum.get_all_energies(qc_dict, dL=0.25)
        for L, E in zip(L_vals[0], E_vals[0]):
            self.assertAlmostEqual(E, _band(L), places=6)
        _, stored_E_vals = store.get_curves(context)
        self.assertTrue(np.allclose(stored_E_vals[0], E_vals[0]))

    def test_caller_supplied_guesses_are_not_second_guessed(self):
        spectrum = _spectrum()
        with _quiet():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                _, E_vals = spectrum.get_all_energies(
                    _qc_dict(), dL=0.25,
                    initial_curves=([[3.05, 3.3, 3.55]], [[4.4, 4.4, 4.4]]))
        self.assertTrue(np.all(np.isnan(E_vals[0][:3])))


class TestFVSpectrumEnergiesFromGuesses(unittest.TestCase):
    """Tests for the continuation entry point."""

    def test_wrong_guesses_are_refined_onto_the_band(self):
        spectrum = _spectrum()
        L_seed = [3.05, 3.3, 3.55]
        E_seed = [_band(L)+0.01 for L in L_seed]
        with _quiet():
            L_vals, E_vals = spectrum.get_energies_from_guesses(
                _qc_dict(), [L_seed], [E_seed], dL=0.25)
        for L, E in zip(L_vals[0], E_vals[0]):
            self.assertAlmostEqual(E, _band(L), places=6)

    def test_bands_are_extended_to_the_edge_of_the_window(self):
        spectrum = _spectrum()
        L_seed = [3.05, 3.3, 3.55]
        E_seed = [_band(L) for L in L_seed]
        with _quiet():
            L_vals, _ = spectrum.get_energies_from_guesses(
                _qc_dict(), [L_seed], [E_seed], dL=0.25)
        self.assertEqual(len(L_vals[0]), 4)
        self.assertAlmostEqual(L_vals[0][-1], 3.8, places=10)

    def test_extension_can_be_disabled(self):
        spectrum = _spectrum()
        L_seed = [3.05, 3.3, 3.55]
        E_seed = [_band(L) for L in L_seed]
        with _quiet():
            L_vals, _ = spectrum.get_energies_from_guesses(
                _qc_dict(), [L_seed], [E_seed], dL=0.25, extend=False)
        self.assertEqual(len(L_vals[0]), 3)

    def test_guesses_from_a_neighbouring_parameter_set_seed_a_new_run(self):
        store = SpectrumValueStore()
        spectrum = _spectrum(value_store=store)
        qc_dict = _qc_dict()
        with _quiet():
            L_vals, E_vals = spectrum.get_all_energies(qc_dict, dL=0.25)
        shifted = _spectrum([lambda L: 2.25+1./L], value_store=store)
        shifted_qc_dict = _qc_dict()
        shifted_qc_dict['k_params'] = [[[0.2]], [0.0]]
        with _quiet():
            L_shifted, E_shifted = shifted.get_all_energies(
                shifted_qc_dict, dL=0.25, initial_curves=(L_vals, E_vals))
        for L, E in zip(L_shifted[0], E_shifted[0]):
            self.assertAlmostEqual(E, 2.25+1./L, places=6)

    def test_empty_band_lists_are_rejected(self):
        spectrum = _spectrum()
        with self.assertRaises(ValueError):
            spectrum.get_energies_from_guesses(_qc_dict(), [], [])

    def test_mismatched_band_lists_are_rejected(self):
        spectrum = _spectrum()
        with self.assertRaises(ValueError):
            spectrum.get_energies_from_guesses(
                _qc_dict(), [[3.0, 3.1]], [[3.5]])


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
