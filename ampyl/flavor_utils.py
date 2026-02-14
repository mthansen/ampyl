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
from .constants import EPSILON4


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


def flavor_channel_to_str(channel):
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


def generate_flavor_channel_space_summary(fcs):
    fcs_summary = ("FlavorChannelSpace initialized with the "
                   "following properties:\n")
    fcs_summary += f"    fc_list: {fcs.fc_list}\n"
    fcs_summary += f"    ni_list: {fcs.ni_list}\n"
    fcs_summary += f"    sc_list: {fcs.sc_list}\n"
    fcs_summary += f"    sc_list_sorted: {fcs.sc_list_sorted}\n"
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
    for g_temp_key in fcs.g_templates:
        fcs_summary += f"        key = {g_temp_key}:\n"
        fcs_summary += f"        {fcs.g_templates[g_temp_key]}\n"
    return fcs_summary
