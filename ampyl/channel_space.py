#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# channel_space.py
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
from copy import deepcopy
from .constants import G_TEMPLATE_DICT
from .constants import bcolors
from .channel import SpectatorChannel
import warnings
warnings.simplefilter("once")


class FlavorChannelSpace:
    """
    Class used to represent a flavor-channel space.

    :param fc_list: list of FlavorChannel objects
    :type fc_list: list
    :param ni_list: list of FlavorChannel objects (corresponding to
    non-interacting channels)
    :type ni_list: list
    :param sc_list: list of SpectatorChannel objects built autmatically from
    fc_list
    :type sc_list: list
    :param sc_list_sorted: list of SpectatorChannel objects sorted first by
    particle number, then mass, then other properties
    :type sc_list_sorted: list
    :param n_particles_max: maximum number of particles in the space
    :type n_particles_max: int
    :param possible_numbers_of_particles: possible numbers of particles in the
    space, typically either ``[2]``, ``[3]`` or ``[2, 3]``
    :type possible_numbers_of_particles: list
    :param n_particle_numbers: number of distinct counts in the space (e.g.
    for ``possible_numbers_of_particles == [2, 3]`` one has
    ``n_particle_numbers == 2``)
    :type n_particle_numbers: int
    :param n_channels_by_particle_number: number of spectator channels for a
    fixed number of particles. For example, for ``possible_numbers_of_particles
    == [2, 3]`` and ``n_channels_by_particle_number == [2, 3]`` one has two
    two-particle and three three-particle channels. The ordering matches
    ``possible_numbers_of_particles``.
    :type n_channels_by_particle_number: list
    :param slices_by_particle_number: slices of the channel space by particle
    number (e.g. for three two- and one three-particle channel one has
    ``slices_by_particle_number = [[0, 3], [3, 4]]``)
    :type slices_by_particle_number: list
    :param slices_by_three_masses: mass-dependent slicing of the three-particle
    channel space (e.g. for one two- and two three-particle channels with
    distinct masses one has ``slices_by_three_masses = [[1, 2], [2, 3]]``)
    :type slices_by_three_masses: list
    :param n_three_slices: length of ``slices_by_three_masses``
    :type n_three_slices: int
    :param g_templates: templates for the g matrices
    :type g_templates: list
    :param g_templates_ell_specific: templates for the g matrices, ell-specific
    :type g_templates_ell_specific: dict
    """

    def __init__(self, fc_list=[], ni_list=None, verbosity=0):
        self.fc_list = fc_list
        if ni_list is None:
            self.ni_list = fc_list
        else:
            self.ni_list = ni_list
        self.sc_list = []
        for fc in fc_list:
            self._add_flavor_channel(fc)
        self._verbosity = verbosity
        self.verbosity = self._verbosity
        self._build_sorted_sc_list()
        self._build_g_templates()
        self._build_g_templates_ell_specific()

        if self.verbosity >= 2:
            self.print_summary()

    def print_summary(self):
        print(f"{bcolors.OKGREEN}FlavorChannelSpace initialized with the "
              "following properties:\n"
              f"    fc_list: {self.fc_list}\n"
              f"    ni_list: {self.ni_list}\n"
              f"    sc_list: {self.sc_list}\n"
              f"    sc_list_sorted: {self.sc_list_sorted}\n"
              f"    n_particles_max: {self.n_particles_max}\n"
              f"    possible_numbers_of_particles: "
              f"{self.possible_numbers_of_particles}\n"
              f"    n_particle_numbers: {self.n_particle_numbers}\n"
              f"    n_channels_by_particle_number: "
              f"{self.n_channels_by_particle_number}\n"
              f"    slices_by_particle_number: "
              f"{self.slices_by_particle_number}\n"
              f"    slices_by_three_masses: "
              f"{self.slices_by_three_masses}\n"
              f"    n_three_slices: {self.n_three_slices}\n"
              f"    g_templates:\n"
              f"        {self.g_templates}\n"
              f"    g_templates_ell_specific:\n"
              "        Key is built from four entries:\n"
              "        [slice_index_i,  slice_index_j, ell_i, ell_j]\n"
              "        Entry is built from five entries:\n"
              "        [np.array([[g_template_ij[sc_index_i][sc_index_j]]], "
              "sc_index_i, sc_index_j, collective_index_i, "
              "collective_index_j]")
        for g_temp_key in self.g_templates_ell_specific:
            print(f"        key = {g_temp_key}:\n"
                  f"        {self.g_templates_ell_specific[g_temp_key]}")
        print(f"{bcolors.ENDC}")

    def update_g_templates(self):
        """Update the g templates."""
        self._build_g_templates()
        self._build_g_templates_ell_specific()
        if self.verbosity >= 2:
            self.print_summary()

    @property
    def verbosity(self):
        """Verbosity of the channel space."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the channel space."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

    def _add_spectator_channel(self, sc):
        self.sc_list.append(sc)

    def _add_flavor_channel(self, fc):
        """
        Add a flavor channel to the flavor channel space.

        This method hard codes some choices for the ell_set and p_cot_deltas.
        This should be changed in the future.
        """
        if fc.n_particles == 2:
            sc1 = SpectatorChannel(fc, indexing=None)
            self._add_spectator_channel(sc1)
        elif fc.isospin_channel:
            for entry in fc.summary_reduced:
                flavors = entry[[2, 4, 5]]
                sub_isospin = entry[1]
                indexing = []
                for flavor in flavors:
                    tmp_locations = np.where(np.array(fc.flavors)
                                             == flavor)[0]
                    added = False
                    for tmp_location in tmp_locations:
                        if (tmp_location not in indexing) and not added:
                            indexing.append(tmp_location)
                            added = True
                if (sub_isospin == 0.0) and (flavors[1] == flavors[2]):
                    ell_set = [0]
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Assuming ell_set = [0] for spectator with "
                                  f"sub_isospin = {sub_isospin} and "
                                  f"flavors = {flavors}"
                                  f"{bcolors.ENDC}", stacklevel=2)
                elif (sub_isospin == 1.0) and (flavors[1] == flavors[2]):
                    ell_set = [1]
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Assuming ell_set = [1] for spectator with "
                                  f"sub_isospin = {sub_isospin} and "
                                  f"flavors = {flavors}"
                                  f"{bcolors.ENDC}", stacklevel=2)
                elif (sub_isospin == 2.0) and (flavors[1] == flavors[2]):
                    ell_set = [0]
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Assuming ell_set = [0] for spectator with "
                                  f"sub_isospin = {sub_isospin} and "
                                  f"flavors = {flavors}"
                                  f"{bcolors.ENDC}", stacklevel=2)
                else:
                    ell_set = [0]
                    warnings.warn(f"\n{bcolors.WARNING}"
                                  "Assuming ell_set = [0] for spectator with "
                                  f"sub_isospin = {sub_isospin} and "
                                  f"flavors = {flavors}"
                                  f"{bcolors.ENDC}", stacklevel=2)
                sc_tmp = SpectatorChannel(fc, indexing=indexing,
                                          sub_isospin=sub_isospin,
                                          ell_set=ell_set)
                self._add_spectator_channel(sc_tmp)
        else:
            if fc.flavors[0] == fc.flavors[1]\
               == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                self._add_spectator_channel(sc1)
            elif fc.flavors[0] == fc.flavors[1]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[2, 0, 1])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            elif fc.flavors[0] == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[1, 2, 0])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            elif fc.flavors[1] == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[1, 2, 0])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            else:
                sc1 = SpectatorChannel(fc=fc)
                sc2 = SpectatorChannel(fc=fc, indexing=[1, 2, 0])
                sc3 = SpectatorChannel(fc=fc, indexing=[2, 0, 1])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
                self._add_spectator_channel(sc3)

    def _build_sorted_sc_list(self):
        n_particles_max = 0
        possible_numbers_of_particles = []
        for fc in self.fc_list:
            if fc.n_particles > n_particles_max:
                n_particles_max = fc.n_particles
            if fc.n_particles not in possible_numbers_of_particles:
                possible_numbers_of_particles.append(fc.n_particles)
        possible_numbers_of_particles.sort()
        n_particle_numbers = len(possible_numbers_of_particles)
        n_channels_by_particle_number = [0 for _ in range(n_particle_numbers)]
        for sc in self.sc_list:
            n_channels_by_particle_number[possible_numbers_of_particles.index(
                sc.fc.n_particles)] += 1
        slices_by_particle_number = []
        n_channels_prev = 0
        for n_channels in n_channels_by_particle_number:
            slices_by_particle_number.append([n_channels_prev,
                                              n_channels
                                              + n_channels_prev])
            n_channels_prev = n_channels
        self.n_particles_max = n_particles_max
        self.possible_numbers_of_particles = possible_numbers_of_particles
        self.n_particle_numbers = n_particle_numbers
        self.n_channels_by_particle_number = n_channels_by_particle_number
        self.slices_by_particle_number = slices_by_particle_number

        sc_compact = [[] for _ in range(n_particle_numbers)]
        sc_index = -1
        for sc in self.sc_list:
            sc_index += 1
            sc_compact_single = [sc.fc.n_particles]
            if sc.fc.n_particles == 2:
                sc_compact_single = self.\
                    _add_two_particle_compact(sc, sc_index, sc_compact_single)
            elif sc.fc.n_particles == 3:
                sc_compact_single = self.\
                    _add_three_particle_compact(sc, sc_index,
                                                sc_compact_single)
            else:
                return ValueError("n_particles > 3 not implemented yet")
            sc_compact[possible_numbers_of_particles.index(sc.fc.n_particles)]\
                .append(sc_compact_single)

        for j in range(len(sc_compact)):
            sc_compact[j] = np.array(sc_compact[j], dtype=object)
            len_tmp = len(sc_compact[j].T)
            for i in range(len_tmp):
                try:
                    sc_compact[j] = sc_compact[j][
                        sc_compact[j][:, len_tmp-i-1].argsort(
                            kind='mergesort')]
                except TypeError:
                    pass

        three_particle_channel_included\
            = (3 in self.possible_numbers_of_particles)
        if three_particle_channel_included:
            slices_by_three_masses = []

            if 2 in self.possible_numbers_of_particles:
                three_offset = self.n_channels_by_particle_number[
                    possible_numbers_of_particles.index(2)]
            else:
                three_offset = 0

            sc_compact_three_subspace = sc_compact[
                possible_numbers_of_particles.index(3)]
            first_mass_index = 1
            last_mass_index = 4
            sc_three_previous_masses = sc_compact_three_subspace[0][
                first_mass_index:last_mass_index]
            slice_min = three_offset
            slice_max = three_offset
            for sc_compact_entry in sc_compact_three_subspace:
                sc_three_masses_current = sc_compact_entry[first_mass_index:
                                                           last_mass_index]
                if (sc_three_previous_masses == sc_three_masses_current).all():
                    slice_max = slice_max+1
                else:
                    slices_by_three_masses.append([slice_min, slice_max])
                    slice_min = slice_max
                    slice_max = slice_max+1
                    sc_three_previous_masses = sc_three_masses_current
            slices_by_three_masses.append([slice_min, slice_max])
            self.slices_by_three_masses = slices_by_three_masses
            self.n_three_slices = len(slices_by_three_masses)
        else:
            self.slices_by_three_masses = []
            self.n_three_slices = 0

        sc_list_sorted = []
        for sc_group in sc_compact:
            for sc_entry in sc_group:
                sc_list_sorted.append(self.sc_list[sc_entry[-1]])
        self.sc_list_sorted = sc_list_sorted

    def _add_three_particle_compact(self, sc, sc_index, sc_compact_single):
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.masses)[sc.indexing])
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.spins)[sc.indexing])
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.flavors)[sc.indexing])
        sc_compact_single = sc_compact_single+[sc.fc.isospin_channel]
        if sc.fc.isospin_channel:
            sc_compact_single = sc_compact_single\
                        + list(np.array(sc.fc.isospins)[sc.indexing])
            sc_compact_single = sc_compact_single+[sc.fc.isospin]
            sc_compact_single = sc_compact_single+[sc.sub_isospin]
        else:
            sc_compact_single = sc_compact_single\
                        + [None, None, None, None, None]
        sc_compact_single = sc_compact_single+[sc_index]
        return sc_compact_single

    def _add_two_particle_compact(self, sc, sc_index, sc_compact_single):
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.masses))
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.spins))
        sc_compact_single = sc_compact_single\
            + list(np.array(sc.fc.flavors))
        sc_compact_single = sc_compact_single+[sc.fc.isospin_channel]
        if sc.fc.isospin_channel:
            sc_compact_single = sc_compact_single\
                        + list(np.array(sc.fc.isospins))
            sc_compact_single = sc_compact_single+[sc.fc.isospin]
        else:
            sc_compact_single = sc_compact_single+[None, None, None]
        sc_compact_single = sc_compact_single+[sc_index]
        return sc_compact_single

    def _build_g_templates(self):
        g_templates = []
        for slice_i in self.slices_by_three_masses:
            slice_i_len = slice_i[1]-slice_i[0]
            g_templates_row = []
            for slice_j in self.slices_by_three_masses:
                slice_j_len = slice_j[1]-slice_j[0]
                g_template = np.zeros((slice_i_len, slice_j_len))
                for i in range(slice_i_len):
                    for j in range(slice_j_len):
                        flavors_i = self.sc_list_sorted[slice_i[0]+i].\
                            flavors_indexed
                        flavors_j = self.sc_list_sorted[slice_j[0]+j].\
                            flavors_indexed
                        g_is_nonzero = (
                                ((flavors_i[0] == flavors_j[2])
                                 and (np.sort(flavors_i[1:])
                                      == np.sort(flavors_j[:-1])).all())
                                or
                                ((flavors_i[0] == flavors_j[1])
                                 and (np.sort(flavors_i[1:])
                                      == np.sort([flavors_j[0]]
                                                 + [flavors_j[2]])).all())
                                )
                        if g_is_nonzero:
                            isospin_channel_i = self.sc_list_sorted[
                                slice_i[0]+i].fc.isospin_channel
                            isospin_channel_j = self.sc_list_sorted[
                                slice_j[0]+j].fc.isospin_channel
                            neither_are_isospin_channels\
                                = ((not isospin_channel_i)
                                   and (not isospin_channel_j))
                            both_are_isospin_channels\
                                = isospin_channel_i and isospin_channel_j
                            if neither_are_isospin_channels:
                                g_template[i][j] = 1.0
                            elif both_are_isospin_channels:
                                g_isospin_ij\
                                    = self._get_g_isospin_ij(slice_i, slice_j,
                                                             i, j)
                                g_template[i][j] = g_isospin_ij
                            else:
                                raise NotImplementedError(
                                    "Mixing of isospin and non-isospin "
                                    "channels is not implemented.")
                g_templates_row.append(g_template)
            g_templates.append(g_templates_row)
        self.g_templates = g_templates

    def _get_g_isospin_ij(self, slice_i, slice_j, i, j):
        isospin_i = self.sc_list_sorted[slice_i[0]+i].fc.isospin
        isospin_j = self.sc_list_sorted[slice_j[0]+j].fc.isospin
        flavors_indexed_i = self.sc_list_sorted[slice_i[0]+i].flavors_indexed
        flavors_indexed_j = self.sc_list_sorted[slice_j[0]+j].flavors_indexed
        flavors_sorted_i = deepcopy(flavors_indexed_i)
        flavors_sorted_j = deepcopy(flavors_indexed_j)
        flavors_sorted_i.sort()
        flavors_sorted_j.sort()
        isospins_indexed_i = self.sc_list_sorted[slice_i[0]+i].isospins_indexed

        if not flavors_sorted_i == flavors_sorted_j:
            return 0.
        if isospin_i != isospin_j:
            return 0.
        if not ((flavors_indexed_i[0] == flavors_indexed_j[1])
                or (flavors_indexed_i[0] == flavors_indexed_j[2])):
            return 0.
        if ((flavors_indexed_i[0] == flavors_indexed_i[1]
             and flavors_indexed_i[0] == flavors_indexed_i[2])
            and (isospins_indexed_i[0] == 1.)
            and (isospin_i == 0. or isospin_i == 1.
                 or isospin_i == 2. or isospin_i == 3.)):
            g_template_isospin = G_TEMPLATE_DICT[int(isospin_i)]
            sub_isospin_i = self.sc_list_sorted[
                slice_i[0]+i].sub_isospin
            sub_isospin_j = self.sc_list_sorted[
                slice_j[0]+j].sub_isospin
            if isospin_i == 3.0:
                ind_i = int(sub_isospin_i-2.0)
                ind_j = int(sub_isospin_j-2.0)
            elif isospin_i == 2.0:
                ind_i = int(sub_isospin_i-1.0)
                ind_j = int(sub_isospin_j-1.0)
            elif isospin_i == 1.0:
                ind_i = int(sub_isospin_i)
                ind_j = int(sub_isospin_j)
            elif isospin_i == 0.0:
                ind_i = int(sub_isospin_i-1.0)
                ind_j = int(sub_isospin_j-1.0)
            return g_template_isospin[ind_i][ind_j]
        warnings.warn(f"\n{bcolors.WARNING}"
                      f"Unknown value within _get_g_isospin_ij; "
                      f"assuming 100"
                      f"{bcolors.ENDC}", stacklevel=2)
        return 100.

    def _build_g_templates_ell_specific(self):
        if 3 in self.possible_numbers_of_particles:
            g_templates_ell_specific_db = self._populate_g_templates_db()
            g_templates_ell_specific_db = self._sort_db(
                g_templates_ell_specific_db)
            g_templates_clustered = self._populate_g_clustered(
                g_templates_ell_specific_db)
            g_templates_ell_specific = {}
            for g_key in g_templates_clustered:
                g_key_list = list(g_templates_clustered[g_key][:4])
                g_key_tuple = tuple(g_key_list)
                g_templates_ell_specific[g_key_tuple] \
                    = g_templates_clustered[g_key][4:]
            self.g_templates_ell_specific = g_templates_ell_specific
        else:
            self.g_templates_ell_specific = {}

    def _populate_g_clustered(self, g_templates_ell_specific_db):
        g_templates_clustered = {}
        for g_template_entry in g_templates_ell_specific_db:
            g_key = str(g_template_entry[:4])
            if g_key not in g_templates_clustered:
                g_templates_clustered[g_key]\
                        = np.array(list(g_template_entry[:4])
                                   + [g_template_entry[4]]
                                   + [[g_template_entry[5]]]
                                   + [[g_template_entry[6]]]
                                   + [[g_template_entry[7]]]
                                   + [[g_template_entry[8]]],
                                   dtype=object)
            else:
                g_template_entry_prev = g_templates_clustered[g_key]
                g_template_matrix_prev = g_template_entry_prev[4]
                sc_indexset_i_prev = g_template_entry_prev[5]
                sc_indexset_j_prev = g_template_entry_prev[6]
                collective_set_i_prev = g_template_entry_prev[7]
                collective_set_j_prev = g_template_entry_prev[8]
                sc_index_i = g_template_entry[5]
                sc_index_j = g_template_entry[6]
                collective_index_i = g_template_entry[7]
                collective_index_j = g_template_entry[8]
                if sc_index_i not in sc_indexset_i_prev:
                    sc_indexset_i_prev.append(sc_index_i)
                    collective_set_i_prev.append(collective_index_i)
                if sc_index_j not in sc_indexset_j_prev:
                    sc_indexset_j_prev.append(sc_index_j)
                    collective_set_j_prev.append(collective_index_j)
                if (len(sc_indexset_i_prev) != len(g_template_matrix_prev)) or\
                    (len(sc_indexset_j_prev) !=
                     len(g_template_matrix_prev.T)):
                    g_template_matrix_new = np.zeros((len(sc_indexset_i_prev),
                                                      len(sc_indexset_j_prev)))
                    g_template_matrix_new[:len(sc_indexset_i_prev),
                                          :len(sc_indexset_j_prev)]\
                        = g_template_matrix_prev
                else:
                    g_template_matrix_new = g_template_matrix_prev
                i_ind = np.where(np.array(sc_indexset_i_prev) ==
                                 sc_index_i)[0][0]
                j_ind = np.where(np.array(sc_indexset_j_prev) ==
                                 sc_index_j)[0][0]
                g_template_matrix_new[i_ind, j_ind] = g_template_entry[4][0][0]
                g_templates_clustered[g_key]\
                    = np.array(list(g_template_entry[:4])
                               + [g_template_matrix_new]
                               + [sc_indexset_i_prev]
                               + [sc_indexset_j_prev]
                               + [collective_set_i_prev]
                               + [collective_set_j_prev], dtype=object)
        return g_templates_clustered

    def _sort_db(self, g_templates_ell_specific_db):
        len_dbT = len(g_templates_ell_specific_db.T)
        for slice_index_i in range(len_dbT):
            try:
                g_templates_ell_specific_db = g_templates_ell_specific_db[
                        g_templates_ell_specific_db[:, len_dbT-slice_index_i-1]
                        .argsort(kind='mergesort')]
            except TypeError:
                pass
        return g_templates_ell_specific_db

    def _populate_g_templates_db(self):
        g_templates_ell_specific_db = []
        collective_index_i = 0
        for slice_index_i in range(len(self.slices_by_three_masses)):
            slice_i = self.slices_by_three_masses[slice_index_i]
            three_mass_slice_i = self.sc_list_sorted[slice_i[0]:slice_i[1]]
            for sc_index_i in range(len(three_mass_slice_i)):
                sc_i = three_mass_slice_i[sc_index_i]
                for ell_i in sc_i.ell_set:
                    collective_index_j = 0
                    for slice_index_j in range(len(
                                self.slices_by_three_masses)):
                        slice_j = self.slices_by_three_masses[
                                slice_index_j]
                        three_mass_slice_j = self.sc_list_sorted[
                                slice_j[0]:slice_j[1]]
                        g_template_ij = self.g_templates[slice_index_i][
                                slice_index_j]
                        for sc_index_j in range(len(three_mass_slice_j)):
                            sc_j = three_mass_slice_j[sc_index_j]
                            for ell_j in sc_j.ell_set:
                                g_templates_ell_specific_db.append(
                                        np.array(
                                            [slice_index_i,
                                             slice_index_j,
                                             ell_i, ell_j,
                                             np.array([[g_template_ij[
                                                 sc_index_i][sc_index_j]]]),
                                             sc_index_i, sc_index_j,
                                             collective_index_i,
                                             collective_index_j],
                                            dtype=object))
                                collective_index_j += 1
                    collective_index_i += 1
        g_templates_ell_specific_db = np.array(g_templates_ell_specific_db)
        return g_templates_ell_specific_db

    def __str__(self):
        """Return a string representation of the FlavorChannelSpace object."""
        flavor_channel_space_str = "FlavorChannelSpace with the following "\
            + "SpectatorChannels:\n"
        for sc in self.sc_list_sorted:
            flavor_channel_space_str += "    "
            flavor_channel_space_str += sc.__str__().replace("\n    ",
                                                             "\n        ")[:-1]
            flavor_channel_space_str += ",\n"
        return flavor_channel_space_str[:-2]+"."
