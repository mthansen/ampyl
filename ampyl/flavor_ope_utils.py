#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# flavor_ope_utils.py
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
import warnings
warnings.simplefilter("once")


def _build_g_templates_ell_specific(fcs):
    """Build the ell-specific g-templates of the FlavorChannelSpace."""
    if 3 in fcs.possible_numbers_of_particles:
        g_templates_ell_specific_db = _populate_g_templates_db(fcs)
        g_templates_ell_specific_db = _sort_db(
            fcs, g_templates_ell_specific_db)
        g_templates_clustered =\
            _populate_g_clustered(fcs, g_templates_ell_specific_db)
        g_templates_ell_specific = {}
        for g_key in g_templates_clustered:
            g_key_list = list(g_templates_clustered[g_key][:4])
            g_key_tuple = tuple(g_key_list)
            g_templates_ell_specific[g_key_tuple] \
                = g_templates_clustered[g_key][4:]
        fcs.g_templates_ell_specific = g_templates_ell_specific
    else:
        fcs.g_templates_ell_specific = {}


def _sort_db(fcs, g_templates_ell_specific_db):
    len_dbT = len(g_templates_ell_specific_db.T)
    for slice_index_i in range(len_dbT):
        try:
            g_templates_ell_specific_db = g_templates_ell_specific_db[
                    g_templates_ell_specific_db[:, len_dbT-slice_index_i-1]
                    .argsort(kind='mergesort')]
        except TypeError:
            pass
    return g_templates_ell_specific_db


def _populate_g_templates_db(fcs):
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


def _populate_g_clustered(fcs, g_templates_ell_specific_db):
    """Cluster the g_templates_ell_specific_db by the first four entries."""
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


def _get_g_isospin_ij(fcs, slice_i, slice_j, i, j):
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
    isospins_indexed_j = fcs.sc_list_sorted[slice_j[0]+j].isospins_indexed
    isospins_sorted_i = deepcopy(isospins_indexed_i)
    isospins_sorted_j = deepcopy(isospins_indexed_j)
    isospins_sorted_i.sort()
    isospins_sorted_j.sort()

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
    if (np.all(isospins_sorted_i == np.array([0.5, 0.5, 1.0]))
       and np.all(isospins_sorted_j == np.array([0.5, 0.5, 1.0]))
       and isospin_i == 2.0
       and isospin_j == 2.0):
        return 1.0
    warnings.warn(f"\n{bcolors.WARNING}"
                  "Unknown value within get_g_isospin_ij; assuming 100"
                  f"{bcolors.ENDC}", stacklevel=2)
    return 100.


def _add_to_g_template(fcs, slice_i, i, slice_j, j, g_template):
    """Add the contribution from the i,j entry to the g_template."""
    flavors_i = fcs.sc_list_sorted[slice_i[0]+i].flavors_indexed
    flavors_j = fcs.sc_list_sorted[slice_j[0]+j].flavors_indexed

    spec_i0_is_j2 = (flavors_i[0] == flavors_j[2])
    dim_i0_is_j2 = (np.sort(flavors_i[1:]) == np.sort(flavors_j[:-1])).all()
    spec_i0_is_j1 = (flavors_i[0] == flavors_j[1])
    dim_i0_is_j1 = (np.sort(flavors_i[1:]) == np.sort([flavors_j[0]]
                                                      + [flavors_j[2]])).all()

    g_is_nonzero = ((spec_i0_is_j2 and dim_i0_is_j2)
                    or (spec_i0_is_j1 and dim_i0_is_j1))

    if g_is_nonzero:
        isospin_channel_i = fcs.sc_list_sorted[slice_i[0]+i].fc.isospin_channel
        isospin_channel_j = fcs.sc_list_sorted[slice_j[0]+j].fc.isospin_channel
        neither_are_isospin_channels = ((not isospin_channel_i)
                                        and (not isospin_channel_j))
        both_are_isospin_channels = isospin_channel_i and isospin_channel_j
        if neither_are_isospin_channels:
            g_template[i][j] = 1.0
        elif both_are_isospin_channels:
            g_isospin_ij = _get_g_isospin_ij(fcs, slice_i, slice_j, i, j)
            g_template[i][j] = g_isospin_ij
        else:
            raise NotImplementedError("Mixing of isospin and non-isospin "
                                      "channels is not implemented.")
    return g_template


def _build_g_templates(fcs):
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
                    g_template = _add_to_g_template(
                        fcs, slice_i, i, slice_j, j, g_template)
            g_templates_row.append(g_template)
        g_templates.append(g_templates_row)
    fcs.g_templates = g_templates
