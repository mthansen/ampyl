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


