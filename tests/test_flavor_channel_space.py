#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# test_flavor_channel_space.py
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
import numpy as np
from ampyl import Particle
from ampyl import FlavorChannel
from ampyl import SpectatorChannel
from ampyl import FlavorChannelSpace


class TestFlavorChannelSpace(unittest.TestCase):

    def test_init(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[kaon, kaon, kaon])
        fcb = FlavorChannel(3, particles=[pion, kaon, kaon])
        fcs = FlavorChannelSpace(fc_list=[fca, fcb])

        self.assertEqual(fcs.fc_list[0], fca)
        self.assertEqual(fcs.ni_list[0], fca)
        self.assertEqual(len(fcs.sc_list), 3)
        self.assertEqual(fcs.sc_list[0].fc, fca)
        self.assertEqual(fcs.sc_list[1].fc, fcb)
        self.assertEqual(fcs.sc_list[2].fc, fcb)
        self.assertEqual(fcs.sc_list_sorted[0].fc, fcb)
        self.assertEqual(fcs.sc_list_sorted[1].fc, fcb)
        self.assertEqual(fcs.sc_list_sorted[2].fc, fca)
        self.assertEqual(fcs.n_particles_max, 3)
        self.assertEqual(fcs.possible_numbers_of_particles, [3])
        self.assertEqual(fcs.n_particle_numbers, 1)
        self.assertEqual(fcs.n_channels_by_particle_number, [3])
        self.assertEqual(fcs.slices_by_particle_number, [[0, 3]])
        self.assertEqual(fcs.slices_by_three_masses, [[0, 1], [1, 2], [2, 3]])
        self.assertEqual(fcs.n_three_slices, 3)
        self.assertEqual(fcs.n_three_slices,
                         len(fcs.slices_by_three_masses))
        g_templates_expected = [[np.array([[0.]]),
                                 np.array([[1.]]),
                                 np.array([[0.]])],
                                [np.array([[1.]]),
                                 np.array([[1.]]),
                                 np.array([[0.]])],
                                [np.array([[0.]]),
                                 np.array([[0.]]),
                                 np.array([[1.]])]]
        self.assertTrue((fcs.g_templates == g_templates_expected))
        g_templates_ell_specific_expected = {
            (0, 0, 0, 0):
                np.array([np.array([[0.]]), list([0]), list([0]),
                          list([0]), list([0])], dtype=object),
            (0, 1, 0, 0):
                np.array([np.array([[1.]]), list([0]), list([0]),
                          list([0]), list([1])], dtype=object),
            (0, 2, 0, 0):
                np.array([np.array([[0.]]), list([0]), list([0]),
                          list([0]), list([2])], dtype=object),
            (1, 0, 0, 0):
                np.array([np.array([[1.]]), list([0]), list([0]),
                          list([1]), list([0])], dtype=object),
            (1, 1, 0, 0):
                np.array([np.array([[1.]]), list([0]), list([0]),
                          list([1]), list([1])], dtype=object),
            (1, 2, 0, 0):
                np.array([np.array([[0.]]), list([0]), list([0]),
                          list([1]), list([2])], dtype=object),
            (2, 0, 0, 0):
                np.array([np.array([[0.]]), list([0]), list([0]),
                          list([2]), list([0])], dtype=object),
            (2, 1, 0, 0):
                np.array([np.array([[0.]]), list([0]), list([0]),
                          list([2]), list([1])], dtype=object),
            (2, 2, 0, 0):
                np.array([np.array([[1.]]), list([0]), list([0]),
                          list([2]), list([2])], dtype=object)
            }

        self.assertEqual(fcs.g_templates_ell_specific.keys(),
                         g_templates_ell_specific_expected.keys())
        for key in g_templates_ell_specific_expected.keys():
            for entry_index in range(len(g_templates_ell_specific_expected[
                    key])):
                self.assertTrue(
                    (fcs.g_templates_ell_specific[key][entry_index] ==
                     g_templates_ell_specific_expected[key][entry_index])
                    .all())

    def test_add_spectator_channel(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[pion, kaon, kaon])
        fcb = FlavorChannel(3, particles=[pion, pion, pion])
        fcs = FlavorChannelSpace(fc_list=[fca])
        previous_sc_list = []
        for sc in fcs.sc_list:
            previous_sc_list.append(sc)
        sc = SpectatorChannel(fc=fcb)
        fcs._add_spectator_channel(sc)
        expected_sc_list = previous_sc_list+[sc]
        self.assertEqual(len(fcs.sc_list), len(expected_sc_list))
        for i in range(len(fcs.sc_list)):
            self.assertEqual(fcs.sc_list[i], expected_sc_list[i])

    def test_add_flavor_channel(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[pion, kaon, kaon])
        fcb = FlavorChannel(3, particles=[pion, pion, pion])
        fcs = FlavorChannelSpace(fc_list=[fcb])
        sca1 = SpectatorChannel(fc=fca, indexing=[0, 1, 2])
        sca2 = SpectatorChannel(fc=fca, indexing=[1, 2, 0])
        previous_sc_list = []
        for sc in fcs.sc_list:
            previous_sc_list.append(sc)
        fcs._add_flavor_channel(fca)
        expected_sc_list = previous_sc_list+[sca1]
        expected_sc_list = expected_sc_list+[sca2]
        self.assertEqual(len(fcs.sc_list), len(expected_sc_list))
        for i in range(len(fcs.sc_list)):
            self.assertEqual(fcs.sc_list[i], expected_sc_list[i])

    def test_build_sorted_sc_list(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[kaon, kaon, kaon])
        fcs = FlavorChannelSpace(fc_list=[fca])
        fcb = FlavorChannel(3, particles=[pion, pion, pion])
        scb = SpectatorChannel(fc=fcb)
        fcs._add_spectator_channel(scb)
        self.assertEqual(len(fcs.sc_list_sorted), 1)
        fcs._build_sorted_sc_list()
        self.assertEqual(len(fcs.sc_list_sorted), 2)
        self.assertEqual(fcs.sc_list_sorted[0], scb)

    def test_add_three_particle_compact(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[kaon, kaon, kaon])
        fcs = FlavorChannelSpace(fc_list=[fca])
        fcb = FlavorChannel(3, particles=[pion, pion, pion])
        scb = SpectatorChannel(fc=fcb)
        sc_compact_single = fcs._add_three_particle_compact(scb, 0, [3])
        sc_compact_single_expected = [3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                                      'pi', 'pi', 'pi', False, None, None,
                                      None, None, None, 0]
        self.assertEqual(sc_compact_single, sc_compact_single_expected)

    def test_add_two_particle_compact(self):
        pass

    def test_build_g_templates(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        kaon = Particle(mass=2.0, spin=0.0, flavor='k')
        fca = FlavorChannel(3, particles=[pion, kaon, kaon])
        fcs = FlavorChannelSpace(fc_list=[fca])
        expected_g_templates = [[np.array([[0.]]),
                                 np.array([[1.]])],
                                [np.array([[1.]]),
                                 np.array([[1.]])]]
        self.assertEqual(fcs.g_templates, expected_g_templates)
        sigma = Particle(mass=3.0, spin=0.0, flavor='sigma')
        fcb = FlavorChannel(3, particles=[sigma, sigma, sigma])
        fcs._add_flavor_channel(fcb)
        self.assertEqual(fcs.g_templates, expected_g_templates)
        fcs._build_sorted_sc_list()
        fcs._build_g_templates()
        expected_g_templates = [[np.array([[0.]]),
                                 np.array([[1.]]),
                                 np.array([[0.]])],
                                [np.array([[1.]]),
                                 np.array([[1.]]),
                                 np.array([[0.]])],
                                [np.array([[0.]]),
                                 np.array([[0.]]),
                                 np.array([[1.]])]]
        self.assertEqual(fcs.g_templates, expected_g_templates)

    def test_get_g_isospin_ij(self):
        pass

    def test_build_g_templates_ell_specific(self):
        pass

    def test_populate_g_clustered(self):
        pass

    def test_sort_db(self):
        pass

    def test_populate_g_templates_db(self):
        pass

    def test_flavor_channel_space_str(self):
        pion = Particle(mass=1.0, spin=0.0, flavor='pi')
        fc = FlavorChannel(3, particles=[pion, pion, pion])
        fcs = FlavorChannelSpace(fc_list=[fc])
        print(fcs.__str__())
        self.assertEqual(fcs.__str__(),
                         "FlavorChannelSpace with the following "
                         "SpectatorChannels:\n"
                         "    SpectatorChannel with the following details:\n"
                         "        3 particles,\n"
                         "        masses: [1.0, 1.0, 1.0],\n"
                         "        spins: [0.0, 0.0, 0.0],\n"
                         "        flavors: ['pi', 'pi', 'pi'],\n"
                         "        isospin_channel: False,\n"
                         "        indexing: [0, 1, 2],\n"
                         "        ell_set: [0],\n"
                         "        p_cot_delta_0: "
                         "pcotdelta_scattering_length,\n"
                         "        n_params_set: [1].")


if __name__ == '__main__':
    unittest.main()
