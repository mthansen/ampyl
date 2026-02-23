#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# flavor_utils.py
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
from .constants import EPSILON4
from .constants import G_TEMPLATE_DICT
from .constants import bcolors
import warnings
warnings.simplefilter("once")


def check_type(variable, variable_string, variable_type,
               variable_type_str):
    """Check that a variable is of the correct type."""
    if variable is not None:
        if not isinstance(variable, variable_type):
            raise TypeError(f"{variable_string} must be of type "
                            f"{variable_type_str}")


def particle_to_string(particle):
    """Convert a particle to a string."""
    particle_str = "Particle with the following properties:\n"
    particle_str += f"    mass: {particle.mass},\n"
    particle_str += f"    spin: {particle.spin},\n"
    particle_str += f"    flavor: {particle.flavor},\n"
    particle_str += f"    isospin_multiplet: {particle.isospin_multiplet},\n"
    if particle.isospin_multiplet:
        particle_str += f"    isospin: {particle.isospin},\n"
    return particle_str[:-2]+"."


def particles_equal(particle1, particle2):
    """Check if two particles are equivalent."""
    return (particle1.mass == particle2.mass and
            particle1.spin == particle2.spin and
            particle1.flavor == particle2.flavor and
            particle1.isospin_multiplet == particle2.isospin_multiplet and
            particle1.isospin == particle2.isospin)


def get_allowed_total_isospins(channel, isospins=None):
    if not channel._isospin_channel:
        return None
    if isospins is None:
        none_was_passed = True
        isospins = channel.isospins
    else:
        none_was_passed = False
    n_isospins = len(isospins)
    if n_isospins == 1:
        return isospins
    if n_isospins == 2:
        min_isospin = abs(isospins[0]-isospins[1])
        max_isospin = abs(isospins[0]+isospins[1])
        return list(np.arange(min_isospin, max_isospin+0.5+EPSILON4))
    if n_isospins == 3 and none_was_passed:
        unique_flavors = np.unique(channel.flavors)
        redundant_list = []
        counting_list = []
        for j in range(len(unique_flavors)):
            spectator_flavor = unique_flavors[j]
            i = np.where(np.array(channel.flavors) == spectator_flavor)[0][0]
            spectator_isospin = isospins[i]
            pair_isospins = isospins[:i] + isospins[i+1:]
            pair_flavors = channel.flavors[:i] + channel.flavors[i+1:]
            combined_pair_isospins =\
                get_allowed_total_isospins(channel, isospins=pair_isospins)
            for combined_pair_isospin in combined_pair_isospins:
                combined_three_isospins =\
                    get_allowed_total_isospins(
                        channel, isospins=[spectator_isospin,
                                           combined_pair_isospin]
                        )
                for combined_three_isospin in combined_three_isospins:
                    redundant_list.append(combined_three_isospin)
                    candidate = (combined_three_isospin,
                                 combined_pair_isospin,
                                 spectator_flavor,
                                 spectator_isospin,
                                 *pair_flavors, *pair_isospins)
                    if candidate not in counting_list:
                        counting_list.append(candidate)
        allowed_total_isospins\
            = list(np.sort(np.unique(redundant_list)))
        channel.summary = np.array([entry for entry in counting_list],
                                   dtype=object)
        if channel._isospin is not None:
            if channel._isospin not in allowed_total_isospins:
                raise ValueError(f"total isospin {channel._isospin} not "
                                 f"allowed with these particles")
            channel.summary_reduced\
                = np.array([entry for entry in channel.summary
                            if entry[0] == channel.isospin],
                           dtype=object)
        return allowed_total_isospins
    raise NotImplementedError("more than three particles not implemented yet")


def get_allowed_sub_isospins(channel):
    if not channel.isospin_channel:
        return None
    if channel.n_particles == 2:
        return None
    if channel.n_particles == 3:
        if channel.isospin == 0.:
            return [1.]
        if channel.isospin == 1.:
            return [0., 1., 2.]
        if channel.isospin == 2.:
            return [1., 2.]
        if channel.isospin == 3.:
            return [2.]
    raise NotImplementedError(f"total isospin {channel.isospin} not "
                              f"implemented yet for {channel._n_particles} "
                              f"particles")


def flavor_channel_to_string(channel):
    """Convert a FlavorChannel to a string."""
    channel_str = "FlavorChannel with the following details:\n"
    channel_str += f"    {channel.n_particles} particles,\n"
    channel_str += f"    masses: {channel.masses},\n"
    channel_str += f"    spins: {channel.spins},\n"
    channel_str += f"    flavors: {channel.flavors},\n"
    channel_str += f"    isospin_channel: {channel.isospin_channel},\n"
    if channel.isospin_channel:
        channel_str += f"    isospins: {channel.isospins},\n"
        channel_str += "    allowed_total_isospins: "
        for i, isospin in enumerate(channel.allowed_total_isospins):
            if i < len(channel.allowed_total_isospins) - 1:
                channel_str += f"{isospin}, "
            else:
                channel_str += f"{isospin},\n"
        channel_str += f"    isospin: {channel.isospin},\n"
    return channel_str[:-2]+"."


def spectator_channel_to_string(channel):
    """Convert a spectator channel to a string."""
    channel_str = channel.fc.__str__().replace("Flavor", "Spectator")
    channel_str = channel_str[:-1]+",\n"
    channel_str += f"    indexing: {channel.indexing},\n"
    if channel.fc.isospin_channel:
        if channel.sub_isospin is not None:
            channel_str += f"    sub_isospin: "\
                f"{channel.sub_isospin},\n"
        if channel.allowed_sub_isospins is not None:
            channel_str += f"    allowed sub_isospins: "\
                f"{channel.allowed_sub_isospins},\n"
    channel_str += f"    ell_set: {channel.ell_set},\n"
    for i, ell in enumerate(channel.ell_set):
        channel_str += f"    p_cot_delta_{ell}: "\
            f"{channel.p_cot_deltas[i]},\n"
    channel_str += f"    n_params_set: {channel.n_params_set},\n"
    return channel_str[:-2]+"."


def spectator_channel_equality(channel1, channel2):
    """Check if two spectator channels are equivalent."""
    if not (channel1.fc == channel2.fc):
        return False
    if not (channel1.indexing == channel2.indexing):
        return False
    if not (channel1.sub_isospin == channel2.sub_isospin):
        return False
    if not (channel1.ell_set == channel2.ell_set):
        return False
    if not (channel1.p_cot_deltas == channel2.p_cot_deltas):
        return False
    if not (channel1.n_params_set == channel2.n_params_set):
        return False
    return True


def parse_three_iso_fc_entry(entry, fc):
    """
    Parse an entry of the summary attribute of a three-particle
    isospin channel.
    """
    flavors = entry[[2, 4, 5]]
    sub_isospin = entry[1]
    indexing = []
    for flavor in flavors:
        tmp_locations = np.where(np.array(fc.flavors) == flavor)[0]
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
    return indexing, sub_isospin, ell_set


def build_sorted_sc_list(fcs):
    """Build the sc_list_sorted attribute of the FlavorChannelSpace."""
    n_particles_max = 0
    possible_numbers_of_particles = []
    for fc in fcs:
        if fc.n_particles > n_particles_max:
            n_particles_max = fc.n_particles
        if fc.n_particles not in possible_numbers_of_particles:
            possible_numbers_of_particles.append(fc.n_particles)
    possible_numbers_of_particles.sort()
    n_particle_numbers = len(possible_numbers_of_particles)
    n_channels_by_particle_number = [0 for _ in range(n_particle_numbers)]
    for sc in fcs.sc_list:
        n_channels_by_particle_number[possible_numbers_of_particles.index(
            sc.fc.n_particles)] += 1
    slices_by_particle_number = []
    n_channels_prev = 0
    for n_channels in n_channels_by_particle_number:
        slices_by_particle_number.append([n_channels_prev,
                                          n_channels + n_channels_prev])
        n_channels_prev = n_channels
    fcs.n_particles_max = n_particles_max
    fcs.possible_numbers_of_particles = possible_numbers_of_particles
    fcs.n_particle_numbers = n_particle_numbers
    fcs.n_channels_by_particle_number = n_channels_by_particle_number
    fcs.slices_by_particle_number = slices_by_particle_number

    sc_compact = [[] for _ in range(n_particle_numbers)]
    sc_index = -1
    for sc in fcs.sc_list:
        sc_index += 1
        sc_compact_single = [sc.fc.n_particles]
        if sc.fc.n_particles == 2:
            sc_compact_single = fcs.\
                _add_two_particle_compact(sc, sc_index, sc_compact_single)
        elif sc.fc.n_particles == 3:
            sc_compact_single = fcs.\
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
        = (3 in fcs.possible_numbers_of_particles)
    if three_particle_channel_included:
        slices_by_three_masses = []

        if 2 in fcs.possible_numbers_of_particles:
            three_offset = fcs.n_channels_by_particle_number[
                fcs.possible_numbers_of_particles.index(2)]
        else:
            three_offset = 0

        sc_compact_three_subspace = sc_compact[
            fcs.possible_numbers_of_particles.index(3)]
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
        fcs.slices_by_three_masses = slices_by_three_masses
        fcs.n_three_slices = len(slices_by_three_masses)
    else:
        fcs.slices_by_three_masses = []
        fcs.n_three_slices = 0

    sc_list_sorted = []
    for sc_group in sc_compact:
        for sc_entry in sc_group:
            sc_list_sorted.append(fcs.sc_list[sc_entry[-1]])
    fcs.sc_list_sorted = sc_list_sorted


def generate_flavor_channel_space_summary(fcs):
    fcs_summary = ("FlavorChannelSpace initialized with the "
                   "following properties:\n"
                   "    fc_list (Flavor Channel list) with the following "
                   "channels:\n")
    for i, fc in enumerate(fcs.fc_list):
        fc_str = "        "+(fc.__str__().replace("    ", "            "))
        fcs_summary += f"    fc_list[{i}]:\n{fc_str}\n"
    fcs_summary += ("    ni_list (Non-Interacting list) with the following "
                    "channels:\n")
    for i, ni in enumerate(fcs.ni_list):
        ni_str = "        "+(ni.__str__().replace("    ", "            "))
        fcs_summary += f"    ni_list[{i}]:\n{ni_str}\n"
    fcs_summary += ("    sc_list (Spectator Channel list) with the following "
                    "channels:\n")
    for i, sc in enumerate(fcs.sc_list):
        sc_str = "        "+(sc.__str__().replace("    ", "            "))
        fcs_summary += f"    sc_list[{i}]:\n{sc_str}\n"
    indices = [fcs.sc_list_sorted.index(sc) for sc in fcs.sc_list]
    if indices != list(range(len(fcs.sc_list))):
        fcs_summary += ("    sc_list_sorted rearranges sc_list according to"
                        f" the following indexing: {indices}\n")
    else:
        fcs_summary += ("    sc_list_sorted is the same as sc_list.\n")
    fcs_summary += f"    n_particles_max: {fcs.n_particles_max}\n"
    fcs_summary += f"    possible_numbers_of_particles: "\
        f"{fcs.possible_numbers_of_particles}\n"
    fcs_summary += f"    n_particle_numbers: {fcs.n_particle_numbers}\n"
    fcs_summary += f"    n_channels_by_particle_number: "\
        f"{fcs.n_channels_by_particle_number}\n"
    fcs_summary += f"    slices_by_particle_number: "\
        f"{fcs.slices_by_particle_number}\n"
    fcs_summary += f"    slices_by_three_masses: "\
        f"{fcs.slices_by_three_masses}\n"
    fcs_summary += f"    n_three_slices: {fcs.n_three_slices}\n"
    fcs_summary += f"    g_templates:\n        {fcs.g_templates}\n"
    fcs_summary +=\
        ("    g_templates_ell_specific:\n"
         "        Key is built from four entries:\n"
         "        [slice_index_i,  slice_index_j, ell_i, ell_j]\n"
         "        Entry is built from five entries:\n"
         "        [np.array([[g_template_ij[sc_index_i][sc_index_j]]]),\n"
         "         sc_index_i, sc_index_j,\n"
         "         collective_index_i, collective_index_j]\n")
    for g_temp_key in fcs.g_templates_ell_specific:
        fcs_summary += f"        key = {g_temp_key}:\n"
        fcs_summary += f"        {fcs.g_templates_ell_specific[g_temp_key]}\n"
    return fcs_summary


def flavor_channel_space_to_string(fcs):
    """Convert a FlavorChannelSpace to a string."""
    flavor_channel_space_str = ("FlavorChannelSpace with the following "
                                "SpectatorChannels:\n")
    for sc in fcs.sc_list_sorted:
        flavor_channel_space_str += "    "
        flavor_channel_space_str += sc.__str__().replace("\n    ",
                                                         "\n        ")[:-1]
        flavor_channel_space_str += ",\n"
    return flavor_channel_space_str[:-2]+"."


def build_g_templates_ell_specific(fcs):
    """Build the ell-specific g-templates of the FlavorChannelSpace."""
    if 3 in fcs.possible_numbers_of_particles:
        g_templates_ell_specific_db = populate_g_templates_db(fcs)
        g_templates_ell_specific_db = sort_db(fcs, g_templates_ell_specific_db)
        g_templates_clustered =\
            populate_g_clustered(fcs, g_templates_ell_specific_db)
        g_templates_ell_specific = {}
        for g_key in g_templates_clustered:
            g_key_list = list(g_templates_clustered[g_key][:4])
            g_key_tuple = tuple(g_key_list)
            g_templates_ell_specific[g_key_tuple] \
                = g_templates_clustered[g_key][4:]
        fcs.g_templates_ell_specific = g_templates_ell_specific
    else:
        fcs.g_templates_ell_specific = {}


def sort_db(fcs, g_templates_ell_specific_db):
    len_dbT = len(g_templates_ell_specific_db.T)
    for slice_index_i in range(len_dbT):
        try:
            g_templates_ell_specific_db = g_templates_ell_specific_db[
                    g_templates_ell_specific_db[:, len_dbT-slice_index_i-1]
                    .argsort(kind='mergesort')]
        except TypeError:
            pass
    return g_templates_ell_specific_db


def populate_g_templates_db(fcs):
    """Populate the g_templates_ell_specific_db attribute."""
    g_templates_ell_specific_db = []
    collective_index_i = 0
    for slice_index_i in range(len(fcs.slices_by_three_masses)):
        slice_i = fcs.slices_by_three_masses[slice_index_i]
        three_mass_slice_i = fcs.sc_list_sorted[slice_i[0]:slice_i[1]]
        for sc_index_i in range(len(three_mass_slice_i)):
            sc_i = three_mass_slice_i[sc_index_i]
            for ell_i in sc_i.ell_set:
                collective_index_j = 0
                for slice_index_j in range(len(
                            fcs.slices_by_three_masses)):
                    slice_j = fcs.slices_by_three_masses[
                            slice_index_j]
                    three_mass_slice_j = fcs.sc_list_sorted[
                            slice_j[0]:slice_j[1]]
                    g_template_ij = fcs.g_templates[slice_index_i][
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


def get_g_isospin_ij(fcs, slice_i, slice_j, i, j):
    """Get the g_ij contribution from the isospin of the i,j entry."""
    isospin_i = fcs.sc_list_sorted[slice_i[0]+i].fc.isospin
    isospin_j = fcs.sc_list_sorted[slice_j[0]+j].fc.isospin
    flavors_indexed_i = fcs.sc_list_sorted[slice_i[0]+i].flavors_indexed
    flavors_indexed_j = fcs.sc_list_sorted[slice_j[0]+j].flavors_indexed
    flavors_sorted_i = deepcopy(flavors_indexed_i)
    flavors_sorted_j = deepcopy(flavors_indexed_j)
    flavors_sorted_i.sort()
    flavors_sorted_j.sort()
    isospins_indexed_i = fcs.sc_list_sorted[slice_i[0]+i].isospins_indexed

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
        sub_isospin_i = fcs.sc_list_sorted[
            slice_i[0]+i].sub_isospin
        sub_isospin_j = fcs.sc_list_sorted[
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
                  f"Unknown value within get_g_isospin_ij; assuming 100"
                  f"{bcolors.ENDC}", stacklevel=2)
    return 100.


def add_to_g_template(fcs, slice_i, i, slice_j, j, g_template):
    """Add the contribution from the i,j entry to the g_template."""
    flavors_i = fcs.sc_list_sorted[slice_i[0]+i].flavors_indexed
    flavors_j = fcs.sc_list_sorted[slice_j[0]+j].flavors_indexed

    i0_is_j2 = (flavors_i[0] == flavors_j[2])
    flav_i0_is_j2 = (np.sort(flavors_i[1:]) == np.sort(flavors_j[:-1])).all()
    i0_is_j1 = (flavors_i[0] == flavors_j[1])
    flav_i0_is_j1 = (np.sort(flavors_i[1:]) == np.sort([flavors_j[0]]
                                                       + [flavors_j[2]])).all()

    g_is_nonzero = ((i0_is_j2 and flav_i0_is_j2)
                    or (i0_is_j1 and flav_i0_is_j1))

    if g_is_nonzero:
        isospin_channel_i = fcs.sc_list_sorted[slice_i[0]+i].fc.isospin_channel
        isospin_channel_j = fcs.sc_list_sorted[slice_j[0]+j].fc.isospin_channel
        neither_are_isospin_channels = ((not isospin_channel_i)
                                        and (not isospin_channel_j))
        both_are_isospin_channels = isospin_channel_i and isospin_channel_j
        if neither_are_isospin_channels:
            g_template[i][j] = 1.0
        elif both_are_isospin_channels:
            g_isospin_ij = get_g_isospin_ij(fcs, slice_i, slice_j, i, j)
            g_template[i][j] = g_isospin_ij
        else:
            raise NotImplementedError("Mixing of isospin and non-isospin "
                                      "channels is not implemented.")
    return g_template


def build_g_templates(fcs):
    """Build the g_templates attribute of the FlavorChannelSpace."""
    g_templates = []
    for slice_i in fcs.slices_by_three_masses:
        slice_i_len = slice_i[1]-slice_i[0]
        g_templates_row = []
        for slice_j in fcs.slices_by_three_masses:
            slice_j_len = slice_j[1]-slice_j[0]
            g_template = np.zeros((slice_i_len, slice_j_len))
            for i in range(slice_i_len):
                for j in range(slice_j_len):
                    g_template = add_to_g_template(
                        fcs, slice_i, i, slice_j, j, g_template)
            g_templates_row.append(g_template)
        g_templates.append(g_templates_row)
    fcs.g_templates = g_templates
