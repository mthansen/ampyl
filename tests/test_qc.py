#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_qc.py
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
from scipy.optimize import root_scalar
import importlib.util
from pathlib import Path
import ampyl
from ampyl import fv_spectrum_utils
from ampyl.ampyl import QCMatrixBuilder


_KKPI_DETF3_HELPER_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "KKpi_detF3inverse"
    / "KKpi_detF3inverse_helper.py"
)
_KKPI_DETF3_HELPER_SPEC = importlib.util.spec_from_file_location(
    "kkpi_detf3inverse_helper",
    _KKPI_DETF3_HELPER_PATH,
)
kkpi_detf3inverse_helper = importlib.util.module_from_spec(
    _KKPI_DETF3_HELPER_SPEC
)
_KKPI_DETF3_HELPER_SPEC.loader.exec_module(kkpi_detf3inverse_helper)


class TestQC(unittest.TestCase):
    def build_qc(self):
        mrho = 2.197791
        pion = ampyl.flavor.Particle(mass=1., spin=0., flavor='pi',
                                     isospin_multiplet=True, isospin=1.)
        rho = ampyl.flavor.Particle(mass=mrho, spin=1., flavor='rho',
                                    isospin_multiplet=True, isospin=1.)
        fc_three_pi = ampyl.flavor.FlavorChannel(3,
                                                 particles=[pion, pion, pion],
                                                 isospin=2.)
        fc_rho_pi = ampyl.flavor.FlavorChannel(2, particles=[rho, pion],
                                               isospin=2.)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_three_pi],
                                              ni_list=[fc_three_pi, fc_rho_pi])
        fcs.sc_list[0].p_cot_deltas[0]\
            = ampyl.qc_functions.pcotdelta_breit_wigner
        fvs = ampyl.spaces.FiniteVolumeSetup()
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(
            fcs=fcs, ESQmins=[0.3]*len(fcs.sc_list_sorted),
            scheme_data=[[-0.7, 0.0] for _ in fcs.sc_list_sorted],
            use_pv_shift_prescription=[True, False],
            pv_shift_parameters=[[-20.], [0.]])
        qcis = ampyl.spaces.QCIndexSpace(fcs=fcs, fvs=fvs, tbis=tbis,
                                         Emax=5.5, Lmax=4.0)
        qcis.populate()
        return ampyl.QC(qcis=qcis)

    def build_qc_case(self):
        L = 16*0.06906*3.444
        k_params = [[[5.80, 2.184], [0.296]], [-9.0]]
        project = True
        irrep = ('T1MINUS', 1)
        qc_dict = {'k_params': k_params, 'project': project, 'irrep': irrep,
                   'version': 'kdf+f3inv_asym_fgcombo'}
        return L, qc_dict

    def test_qc(self):
        qc = self.build_qc()
        L, qc_dict = self.build_qc_case()
        brackets = [[4.6, 4.7], [4.7, 4.9]]
        roots = []
        for bracket in brackets:
            root = root_scalar(qc.get_value, args=(L, qc_dict),
                               bracket=bracket).root
            roots.append(root)
        roots = np.array(roots)

        roots_expected = np.array([4.63304377, 4.84871987])
        diffSQ = np.sum((roots - roots_expected)**2)
        self.assertTrue(diffSQ < 1.e-15)

    def test_qcis_populates_kkpi_spectator_slices(self):
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi',
                                     isospin_multiplet=True, isospin=1.0)
        kaon = ampyl.flavor.Particle(mass=2.5, spin=0.0, flavor='K',
                                     isospin_multiplet=True, isospin=0.5)
        fc_kkpi = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion], isospin=2.0)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_kkpi], ni_list=[])
        fvs = ampyl.spaces.FiniteVolumeSetup(
            qc_impl={'discard_non_interacting': False})
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, fvs=fvs, tbis=tbis, Emax=7.0, Lmax=4.0)

        qcis.populate()

        self.assertEqual(qcis.sc_to_three_slice, [0, 1])
        self.assertTrue(all(len(tbks_set) > 0 for tbks_set in qcis.tbks_list))

    def test_multislice_interpolator_builds_cob_matrices(self):
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi',
                                     isospin_multiplet=True, isospin=1.0)
        kaon = ampyl.flavor.Particle(mass=2.5, spin=0.0, flavor='K',
                                     isospin_multiplet=True, isospin=0.5)
        fc_kkpi = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion], isospin=2.0)
        fcs = ampyl.flavor.FlavorChannelSpace(fc_list=[fc_kkpi], ni_list=[])
        fvs = ampyl.spaces.FiniteVolumeSetup(
            qc_impl={'discard_non_interacting': False,
                     'populate_interp_zeros': True})
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, fvs=fvs, tbis=tbis, Emax=7.0, Lmax=4.0)
        qcis.populate()
        f = ampyl.F(qcis=qcis)
        irrep = ('A1PLUS', 0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f.build_interpolator(6.1, 6.2, 0.1, 3.0, 3.1, 0.1,
                                 True, irrep)

        cob_warning_found = any(
            'Change-of-basis interpolation matrices are not supported'
            in str(warning.message) for warning in caught)
        expected_cob_count = 1
        for tbks_set in qcis.tbks_list:
            expected_cob_count *= len(tbks_set)
        self.assertFalse(cob_warning_found)
        self.assertEqual(len(f.cob_matrix_lists[irrep]), expected_cob_count)
        self.assertEqual(len(f.cob_matrix_key_lists[irrep]),
                         expected_cob_count)
        max_dim = f.cob_matrix_lists[irrep][0].shape[1]
        self.assertTrue(
            all(cob_matrix.shape[1] == max_dim
                for cob_matrix in f.cob_matrix_lists[irrep]))
        self.assertTrue(
            any(cob_matrix.shape[0] < cob_matrix.shape[1]
                for cob_matrix in f.cob_matrix_lists[irrep]))

    def test_multislice_fplusg_interpolator_skips_zero_g_blocks(self):
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi',
                                     isospin_multiplet=True, isospin=1.0)
        kaon = ampyl.flavor.Particle(mass=0.09698/0.06906, spin=0.0,
                                     flavor='K',
                                     isospin_multiplet=True, isospin=0.5)
        fc_kkpi = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion], isospin=2.0)
        fcs = ampyl.flavor.FlavorChannelSpace(
            fc_list=[fc_kkpi], ni_list=[fc_kkpi])
        fvs = ampyl.spaces.FiniteVolumeSetup(
            qc_impl={'discard_non_interacting': False,
                     'populate_interp_zeros': True})
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, fvs=fvs, tbis=tbis, Emax=5.7, Lmax=6.0)
        qcis.populate()
        fplusg = ampyl.FplusG(qcis=qcis)
        irrep = ('A1PLUS', 0)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            fplusg.build_interpolator(4.6, 4.7, 0.1, 5.5, 5.6, 0.1,
                                      True, irrep)

        self.assertIn(irrep, fplusg.pole_mass_lists)

    def test_qcis_populates_kkpi_aab_nonint_functions(self):
        pion = ampyl.flavor.Particle(mass=1.0, spin=0.0, flavor='pi',
                                     isospin_multiplet=True, isospin=1.0)
        kaon = ampyl.flavor.Particle(mass=2.5, spin=0.0, flavor='K',
                                     isospin_multiplet=True, isospin=0.5)
        fc_kkpi = ampyl.flavor.FlavorChannel(
            3, particles=[kaon, kaon, pion], isospin=2.0)
        fcs = ampyl.flavor.FlavorChannelSpace(
            fc_list=[fc_kkpi], ni_list=[fc_kkpi])
        tbis = ampyl.spaces.ThreeBodyInteractionScheme(fcs=fcs)
        qcis = ampyl.spaces.QCIndexSpace(
            fcs=fcs, tbis=tbis, Emax=7.0, Lmax=4.0)

        qcis.populate()

        irrep = ('A1PLUS', 0)
        self.assertEqual(qcis._nonint_channel_particle_label(0), 'aab')
        self.assertEqual(len(qcis.nonint_functions), 1)
        self.assertGreater(len(qcis.nonint_functions[0][irrep]), 0)
        self.assertAlmostEqual(
            qcis.nonint_functions[0][irrep][0](4.0),
            2.0*kaon.mass+pion.mass,
        )

    def test_qc_energy_solver_is_explicit(self):
        qc = self.build_qc()
        L, qc_dict = self.build_qc_case()
        spectrum = ampyl.FVSpectrum(qc)
        irrep = qc_dict['irrep']
        ni_functions = []
        for ni_function_channel in qc.qcis.nonint_functions:
            ni_functions.extend(ni_function_channel[irrep])

        self.assertFalse(hasattr(qc, 'energy_solver'))
        self.assertFalse(hasattr(qc, 'get_all_energies'))
        self.assertFalse(hasattr(spectrum, 'simple_try_at_fixed_L'))

        roots = np.array(
            spectrum.get_roots_from_range([4.6, 4.9], L, qc_dict, ni_functions)
        )

        roots_expected = np.array([4.63304377, 4.84871987])
        diffSQ = np.sum((roots - roots_expected)**2)
        self.assertTrue(diffSQ < 1.e-15)

    def test_evaluation_policy_uses_first_matching_element(self):
        policy = ampyl.EvaluationPolicy([
            {'Lmin': 3.0, 'Lmax': 5.0, 'Emin': 4.0, 'Emax': 6.0,
             'version': 'matched-high-id'},
            {'Lmin': 3.0, 'Lmax': 5.0, 'Emin': 4.0, 'Emax': 6.0,
             'version': 'matched-lowest-after-order'},
            {'Lmin': None, 'Lmax': None, 'Emin': None, 'Emax': None,
             'version': 'default'},
        ])

        self.assertEqual(policy.select(4.0, 3.0)['version'],
                         'matched-high-id')
        self.assertEqual(policy.select(7.0, 3.0)['version'], 'default')

    def test_qc_component_registries_have_ids_and_names(self):
        qc = self.build_qc()
        fplusg = qc.add_fplusg(qcis_id=0, name='coarse')

        self.assertEqual(fplusg.id, 1)
        self.assertEqual(fplusg.name, 'coarse')
        self.assertIs(qc.fplusg_list[1], fplusg)
        self.assertIs(qc.fplusg_list['coarse'], fplusg)

    def test_matrix_builder_uses_policy_selected_f_and_fplusg_interpolators(self):
        class FakeComponent:
            def __init__(self, value):
                self.value = np.array([[value]], dtype=float)
                self.calls = []

            def get_value(self, *args, **kwargs):
                self.calls.append(kwargs)
                return self.value

        class FakeRegistry:
            def __init__(self, component):
                self.component = component
                self.identifiers = []

            def get(self, identifier):
                self.identifiers.append(identifier)
                return self.component

        class FakeOwner:
            def __init__(self):
                self.f = FakeComponent(3.0)
                self.fplusg = FakeComponent(4.0)
                self.k = FakeComponent(5.0)
                self.kdf = FakeComponent(6.0)
                self.g = FakeComponent(7.0)
                self.f_list = FakeRegistry(self.f)
                self.fplusg_list = FakeRegistry(self.fplusg)
                self.k_list = FakeRegistry(self.k)
                self.kdf_list = FakeRegistry(self.kdf)
                self.g_list = FakeRegistry(self.g)

        owner = FakeOwner()
        builder = QCMatrixBuilder(owner=owner)
        qc_dict = {
            'k_params': [[[]], []],
            'project': True,
            'irrep': ('A1PLUS', 0),
            'version': 'detF3inverse',
            'rescale': 1.0,
            'shift': 0.0,
        }
        policy_element = {
            'version': 'detF3inverse',
            'f_id': 'f-segment',
            'f_interpolator': True,
            'f_interpolator_id': 12,
            'fplusg_id': 'fg-segment',
            'fplusg_interpolator': True,
            'fplusg_interpolator_name': 'L20',
            'k_id': 'k-segment',
        }

        matrices = builder.build(4.5, 20.0, qc_dict, policy_element)

        self.assertEqual(owner.f_list.identifiers, ['f-segment'])
        self.assertEqual(owner.fplusg_list.identifiers, ['fg-segment'])
        self.assertEqual(owner.k_list.identifiers, ['k-segment'])
        self.assertEqual(
            owner.f.calls[0],
            {'short_string': 'f', 'interpolate': True, 'interpolator_id': 12},
        )
        self.assertEqual(
            owner.fplusg.calls[0],
            {
                'short_string': 'fplusg',
                'interpolate': True,
                'interpolator_name': 'L20',
            },
        )
        self.assertIn('F', matrices)
        self.assertIn('FplusG', matrices)

    def test_kkpi_detf3inverse_segment_defaults_to_single_l20_volume(self):
        segment = kkpi_detf3inverse_helper.make_detf3inverse_segment(0.25)

        self.assertAlmostEqual(segment['Lmin'], 20.0 * 0.25)
        self.assertAlmostEqual(segment['Lmax'], 20.0 * 0.25)

    def test_kkpi_detf3inverse_policy_enables_f_and_fplusg_interpolators(self):
        segment = kkpi_detf3inverse_helper.make_detf3inverse_segment(0.25)
        element = kkpi_detf3inverse_helper.policy_element_for_segment(
            segment,
            3,
        )

        self.assertEqual(element['f_id'], 3)
        self.assertTrue(element['f_interpolator'])
        self.assertEqual(element['f_interpolator_id'], 0)
        self.assertEqual(element['fplusg_id'], 3)
        self.assertTrue(element['fplusg_interpolator'])
        self.assertEqual(element['fplusg_interpolator_id'], 0)

    def test_kkpi_detf3inverse_disable_interpolator_fallback_turns_off_f_and_fplusg(self):
        segment = kkpi_detf3inverse_helper.make_detf3inverse_segment(0.25)
        qc_dict = {
            'policy': kkpi_detf3inverse_helper.make_policy([segment]),
        }

        qc_dict_no_interp = kkpi_detf3inverse_helper._qc_dict_without_interpolator(
            qc_dict
        )
        policy_elements = qc_dict_no_interp['policy']

        self.assertFalse(policy_elements[0]['f_interpolator'])
        self.assertFalse(policy_elements[0]['fplusg_interpolator'])
        self.assertFalse(policy_elements[-1]['f_interpolator'])
        self.assertFalse(policy_elements[-1]['fplusg_interpolator'])

    def test_root_finder_refinement_defaults_to_false(self):
        class FakeFVS:
            def __init__(self):
                self.qc_impl = {'fplusg_interpolate': True}

        class FakeQCIS:
            def __init__(self):
                self.fvs = FakeFVS()

        class FakeQC:
            def __init__(self):
                self.qcis = FakeQCIS()

            def get_value(self, E, L, qc_dict):
                interpolation_on = any(
                    value
                    for key, value in self.qcis.fvs.qc_impl.items()
                    if 'interp' in key or 'interpolate' in key
                )
                root = 1.0 if interpolation_on else 1.00002
                return E - root

        class FakeSpectrum:
            def __init__(self):
                self.qc = FakeQC()

        root = fv_spectrum_utils._simple_try_at_fixed_L(
            FakeSpectrum(), [0.99, 1.01], 5.0, {}
        )

        self.assertAlmostEqual(root, 1.0)

    def test_root_finder_discards_near_noninteracting_by_default(self):
        class FakeFVS:
            def __init__(self):
                self.qc_impl = {'fplusg_interpolate': True}

        class FakeQCIS:
            def __init__(self):
                self.fvs = FakeFVS()

        class FakeQC:
            def __init__(self):
                self.qcis = FakeQCIS()

            def get_value(self, E, L, qc_dict):
                return E - 1.0

        class FakeSpectrum:
            def __init__(self):
                self.qc = FakeQC()

        root = fv_spectrum_utils._simple_try_at_fixed_L(
            FakeSpectrum(), [0.99, 1.01], 5.0, {}, np.array([1.00005])
        )

        self.assertTrue(np.isnan(root))

    def test_root_finder_can_keep_near_noninteracting_roots(self):
        class FakeFVS:
            def __init__(self):
                self.qc_impl = {
                    'fplusg_interpolate': True,
                    'discard_non_interacting': False,
                }

        class FakeQCIS:
            def __init__(self):
                self.fvs = FakeFVS()

        class FakeQC:
            def __init__(self):
                self.qcis = FakeQCIS()

            def get_value(self, E, L, qc_dict):
                return E - 1.0

        class FakeSpectrum:
            def __init__(self):
                self.qc = FakeQC()

        root = fv_spectrum_utils._simple_try_at_fixed_L(
            FakeSpectrum(), [0.99, 1.01], 5.0, {}, np.array([1.00005])
        )

        self.assertAlmostEqual(root, 1.0)

    def test_root_finder_refines_without_interpolation(self):
        class FakeFVS:
            def __init__(self):
                self.qc_impl = {
                    'fplusg_interpolate': True,
                    'zeta_interp': True,
                    'refine_roots': True,
                }

        class FakeQCIS:
            def __init__(self):
                self.fvs = FakeFVS()

        class FakeQC:
            def __init__(self):
                self.qcis = FakeQCIS()

            def get_value(self, E, L, qc_dict):
                interpolation_on = any(
                    value
                    for key, value in self.qcis.fvs.qc_impl.items()
                    if 'interp' in key or 'interpolate' in key
                )
                root = 1.0 if interpolation_on else 1.00002
                return E - root

        class FakeSpectrum:
            def __init__(self):
                self.qc = FakeQC()

        spectrum = FakeSpectrum()
        qc_impl_before = dict(spectrum.qc.qcis.fvs.qc_impl)

        root = fv_spectrum_utils._simple_try_at_fixed_L(
            spectrum, [0.99, 1.01], 5.0, {}
        )

        self.assertAlmostEqual(root, 1.00002)
        self.assertEqual(spectrum.qc.qcis.fvs.qc_impl, qc_impl_before)

    def test_root_finder_returns_nan_if_uninterpolated_root_is_absent(self):
        class FakeFVS:
            def __init__(self):
                self.qc_impl = {
                    'fplusg_interpolate': True,
                    'refine_roots': True,
                }

        class FakeQCIS:
            def __init__(self):
                self.fvs = FakeFVS()

        class FakeQC:
            def __init__(self):
                self.qcis = FakeQCIS()

            def get_value(self, E, L, qc_dict):
                interpolation_on = any(
                    value
                    for key, value in self.qcis.fvs.qc_impl.items()
                    if 'interp' in key or 'interpolate' in key
                )
                root = 1.0 if interpolation_on else 1.01
                return E - root

        class FakeSpectrum:
            def __init__(self):
                self.qc = FakeQC()

        spectrum = FakeSpectrum()
        qc_impl_before = dict(spectrum.qc.qcis.fvs.qc_impl)

        root = fv_spectrum_utils._simple_try_at_fixed_L(
            spectrum, [0.99, 1.01], 5.0, {}
        )

        self.assertTrue(np.isnan(root))
        self.assertEqual(spectrum.qc.qcis.fvs.qc_impl, qc_impl_before)


if __name__ == '__main__':
    unittest.main()
