#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# ampyl.py
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
from scipy.linalg import block_diag
from scipy.optimize import root_scalar
import functools
from copy import deepcopy
from .group_theory import Groups
from .group_theory import Irreps
from .qc_functions import QCFunctions
from inspect import signature
from .qc_functions import BKFunctions
from scipy.interpolate import RegularGridInterpolator
from .global_constants import QC_IMPL_DEFAULTS
from .global_constants import TWOPI
from .global_constants import FOURPI2
from .global_constants import EPSILON4
from .global_constants import EPSILON10
from .global_constants import EPSILON20
from .global_constants import BAD_MIN_GUESS
from .global_constants import BAD_MAX_GUESS
from .global_constants import DELTA_L_FOR_GRID
from .global_constants import DELTA_E_FOR_GRID
from .global_constants import L_GRID_SHIFT
from .global_constants import E_GRID_SHIFT
from .global_constants import G_TEMPLATE_DICT
from .global_constants import ISO_PROJECTORS
from .global_constants import CAL_C_ISO
from .global_constants import SPARSE_CUT
from .global_constants import POLE_CUT
import warnings
warnings.simplefilter("once")

PRINT_THRESHOLD_DEFAULT = np.get_printoptions()['threshold']


class bcolors:
    """Class collecting text colors."""

    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Particle:
    """
    Class used to represent a particle.

    :param mass: mass of the particle (default is ``1.``)
    :type mass: float
    :param spin: spin of the particle (default is ``0.``)
    :type spin: float
    :param flavor: flavor of the particle (default is ``'pi'``)
    :type flavor: str
    :param isospin_multiplet: specifies whether this is an isospin multiplet
    :type isospin_multiplet: bool
    :param isospin: isospin of the particle (default is ``None``)
    :type isospin: float
    """
    def __init__(self, mass=1., spin=0., flavor='pi',
                 isospin_multiplet=False, isospin=None):
        if not isospin_multiplet and isospin is not None:
            isospin_multiplet = True
        if isospin_multiplet and isospin is None:
            isospin = 1.

        self._mass = mass
        self._spin = spin
        self._flavor = flavor
        self._isospin_multiplet = isospin_multiplet
        self._isospin = isospin

        self.mass = self._mass
        self.spin = self._spin
        self.flavor = self._flavor
        self.isospin_multiplet = self._isospin_multiplet
        self.isospin = self._isospin

    def _check_type(self, variable, variable_string, variable_type,
                    variable_type_str):
        """Check that a variable is of the correct type."""
        if variable is not None:
            if not isinstance(variable, variable_type):
                raise TypeError(f"{variable_string} must be of type "
                                f"{variable_type_str}")

    @property
    def mass(self):
        """Mass of the particle."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Set the mass of the particle."""
        self._check_type(mass, 'mass', float, 'float')
        self._mass = mass

    @property
    def spin(self):
        """Spin of the particle."""
        return self._spin

    @spin.setter
    def spin(self, spin):
        """Set the spin of the particle."""
        self._check_type(spin, 'spin', float, 'float')
        self._spin = spin

    @property
    def flavor(self):
        """Flavor of the particle."""
        return self._flavor

    @flavor.setter
    def flavor(self, flavor):
        """Set the flavor of the particle."""
        self._check_type(flavor, 'flavor', str, 'str')
        self._flavor = flavor

    @property
    def isospin_multiplet(self):
        """Whether the particle is an isospin multiplet."""
        return self._isospin_multiplet

    @isospin_multiplet.setter
    def isospin_multiplet(self, isospin_multiplet):
        """Set whether the particle is an isospin multiplet."""
        self._check_type(isospin_multiplet, 'isospin_multiplet', bool, 'bool')
        self._isospin_multiplet = isospin_multiplet

    @property
    def isospin(self):
        """Isospin of the particle."""
        return self._isospin

    @isospin.setter
    def isospin(self, isospin):
        """Set the isospin of the particle."""
        self._check_type(isospin, 'isospin', float, 'float')
        self._isospin = isospin

    def __eq__(self, other):
        """Check if two particles are equal."""
        if not isinstance(other, Particle):
            return False
        return (self.mass == other.mass and
                self.spin == other.spin and
                self.flavor == other.flavor and
                self.isospin_multiplet == other.isospin_multiplet and
                self.isospin == other.isospin)

    def __str__(self):
        """Return a string representation of the particle object."""
        particle_str = "Particle with the following properties:\n"
        particle_str += f"    mass: {self.mass},\n"
        particle_str += f"    spin: {self.spin},\n"
        particle_str += f"    flavor: {self.flavor},\n"
        particle_str += f"    isospin_multiplet: {self.isospin_multiplet},\n"
        if self.isospin_multiplet:
            particle_str += f"    isospin: {self.isospin},\n"
        return particle_str[:-2]+"."


class FlavorChannel:
    """
    Class used to represent a flavor channel.

    :param n_particles: number of particles in the channel
    :type n_particles: int
    :param particles: list of particles in the channel
    :type particles: list
    :param masses: list of masses of the particles in the channel
    :type masses: list
    :param spins: list of spins of the particles in the channel
    :type spins: list
    :param flavors: list of flavors of the particles in the channel
    :type flavors: list
    :param isospins: list of isospins of the particles in the channel
    :type isospins: list
    :param isospin_channel: specifies whether this is an isospin channel
    :type isospin_channel: bool
    :param isospin: isospin of the flavor channel (default is ``None``)
    :type isospin: float
    :param allowed_total_isospins: list of allowed total isospins for the
    :type allowed_total_isospins: list
    :param summary: summary of the flavor channel
    :type summary: list
    :param summary_reduced: reduced summary of the flavor channel
    :type summary_reduced: list
    """
    def __init__(self, n_particles, particles=[], isospin_channel=False,
                 isospin=None):
        if not isinstance(n_particles, int):
            raise ValueError("n_particles must be an int")
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        self._n_particles = n_particles

        self.summary = None
        self.summary_reduced = None

        if isinstance(particles, list) and len(particles) == 0:
            particles = [Particle() for _ in range(n_particles)]
        if not isospin_channel and isospin is not None:
            isospin_channel = True
        if isospin_channel and isospin is None:
            isospin_channel = False
            warnings.warn(f"\n{bcolors.WARNING}"
                          "isospin_channel is True but isospin is None; "
                          "setting isospin_channel to False"
                          f"{bcolors.ENDC}", stacklevel=2)

        self._particles = particles
        self._isospin_channel = isospin_channel
        self._isospin = isospin

        self.particles = self._particles

        self.masses = self._get_masses()
        self.spins = self._get_spins()
        self.flavors = self._get_flavors()
        self.isospins = self._get_isospins()

        self.allowed_total_isospins = self._get_allowed_total_isospins()

        self.isospin_channel = self._isospin_channel
        self.isospin = self._isospin
        self.n_particles = self._n_particles

    def _get_masses(self):
        return [particle.mass for particle in self.particles]

    def _get_spins(self):
        return [particle.spin for particle in self.particles]

    def _get_flavors(self):
        return [particle.flavor for particle in self.particles]

    def _get_isospins(self):
        return [particle.isospin for particle in self.particles]

    def _get_allowed_total_isospins(self, isospins=None):
        if not self._isospin_channel:
            return None
        if isospins is None:
            none_was_passed = True
            isospins = self.isospins
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
            unique_flavors = np.unique(self.flavors)
            redundant_list = []
            counting_list = []
            for j in range(len(unique_flavors)):
                spectator_flavor = unique_flavors[j]
                i = np.where(np.array(self.flavors) == spectator_flavor)[0][0]
                spectator_isospin = isospins[i]
                pair_isospins = isospins[:i] + isospins[i+1:]
                pair_flavors = self.flavors[:i] + self.flavors[i+1:]
                combined_pair_isospins = self.\
                    _get_allowed_total_isospins(isospins=pair_isospins)
                for combined_pair_isospin in combined_pair_isospins:
                    combined_three_isospins = self\
                        ._get_allowed_total_isospins(
                            isospins=[spectator_isospin, combined_pair_isospin]
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
            self.summary = np.array([entry for entry in counting_list],
                                    dtype=object)
            if self._isospin is not None:
                if self._isospin not in allowed_total_isospins:
                    raise ValueError(f"total isospin {self._isospin} not "
                                     f"allowed with these particles")
                self.summary_reduced\
                    = np.array([entry for entry in self.summary
                                if entry[0] == self.isospin],
                               dtype=object)
            return allowed_total_isospins
        raise NotImplementedError("more than three particles not implemented "
                                  "yet")

    @property
    def particles(self):
        """Particles in the channel."""
        return self._particles

    @particles.setter
    def particles(self, particles):
        """Set the particles in the channel."""
        if not isinstance(particles, list):
            raise ValueError("particles must be a list")
        if len(particles) != self.n_particles:
            raise ValueError("len(particles) must be equal to n_particles")
        for particle in particles:
            if not isinstance(particle, Particle):
                raise ValueError("particles must be a list of Particle "
                                 "objects")
        for particle_a in particles:
            for particle_b in particles:
                if (particle_a.flavor == particle_b.flavor)\
                   and (particle_a != particle_b):
                    raise ValueError("particles with the same flavors must be "
                                     + "identical")
        if self._isospin_channel:
            for particle in particles:
                if not particle.isospin_multiplet:
                    raise ValueError("all particles must be in an isospin "
                                     "multiplet if the channel is an "
                                     "isospin channel")
        self._particles = particles

    @property
    def isospin_channel(self):
        """Whether the channel is an isospin channel."""
        return self._isospin_channel

    @isospin_channel.setter
    def isospin_channel(self, isospin_channel):
        """Set whether the channel is an isospin channel."""
        if not isinstance(isospin_channel, bool):
            raise ValueError("isospin_channel must be a bool")
        self._isospin_channel = isospin_channel

    @property
    def isospin(self):
        """Isospin value of the channel."""
        return self._isospin

    @isospin.setter
    def isospin(self, isospin):
        """Set the isospin value of the channel."""
        if isospin is not None and not isinstance(isospin, float):
            raise ValueError("isospin must be an float")
        if isospin is not None and isospin not in self.allowed_total_isospins:
            raise ValueError("isospin must be in allowed_total_isospins")
        self._isospin = isospin

    @property
    def n_particles(self):
        """Number of particles in the channel."""
        return self._n_particles

    @n_particles.setter
    def n_particles(self, n_particles):
        """Set the number of particles in the channel."""
        if not isinstance(n_particles, int):
            raise ValueError("n_particles must be an int")
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")

    def __str__(self):
        """Return a string representation of the FlavorChannel object."""
        flavor_channel_str = "FlavorChannel with the following details:\n"
        flavor_channel_str += f"    {self.n_particles} particles,\n"
        flavor_channel_str += f"    masses: {self.masses},\n"
        flavor_channel_str += f"    spins: {self.spins},\n"
        flavor_channel_str += f"    flavors: {self.flavors},\n"
        flavor_channel_str += f"    isospin_channel: {self.isospin_channel},\n"
        if self.isospin_channel:
            flavor_channel_str += f"    isospins: {self.isospins},\n"
            flavor_channel_str += f"    allowed_total_isospins: "\
                f"{self.allowed_total_isospins},\n"
            flavor_channel_str += f"    isospin: {self.isospin},\n"
        return flavor_channel_str[:-2]+"."


class SpectatorChannel:
    """
    Class used to represent a spectator channel.

    :param fc: FlavorChannel object required to define the spectator channel
    :type fc: FlavorChannel
    :param indexing: list of indices of the particles in the FlavorChannel. The
    first entry corresponds to the spectator particle.
    :type indexing: list
    :param masses_indexed: masses of the particles in the channel with the
    spectator first
    :type masses_indexed: list
    :param spins_indexed: spins of the particles in the channel with the
    spectator first
    :type spins_indexed: list
    :param flavors_indexed: flavors of the particles in the channel with the
    spectator first
    :type flavors_indexed: list
    :param isospins_indexed: isospins of the particles in the channel with the
    spectator first
    :type isospins_indexed: list
    :param sub_isospin: sub-isospin value of the channel
    :type sub_isospin: float
    :param ell_set: list of angular momentum values of the channel
    :type ell_set: list
    :param p_cot_deltas: list of p_cot_delta values of the channel
    :type p_cot_deltas: list
    :param n_params_set: list of parameter counts for the channel. Set
    automatically.
    :type n_params_set: list
    :param allowed_sub_isospins: list of allowed sub-channel isospins
    :type allowed_sub_isospins: list
    """

    def __init__(self, fc=FlavorChannel(3), indexing=[0, 1, 2],
                 sub_isospin=None, ell_set=[0], p_cot_deltas=None,
                 n_params_set=[1]):

        self.allowed_sub_isospins = None

        self._fc = fc
        self._indexing = indexing
        self._sub_isospin = sub_isospin
        self._ell_set = ell_set
        self._p_cot_deltas = p_cot_deltas
        self._n_params_set = n_params_set

        if fc.isospin_channel and fc.n_particles > 2:
            allowed_sub_isospins = []
            for entry in fc.summary_reduced:
                if fc.flavors[indexing[0]] == entry[2]:
                    allowed_sub_isospins.append(entry[1])
            self.allowed_sub_isospins = allowed_sub_isospins

        self.fc = fc
        self.indexing = indexing
        self.sub_isospin = sub_isospin
        self.ell_set = ell_set

        if self.fc.n_particles == 2:
            self.masses_indexed = self.fc.masses
            self.spins_indexed = self.fc.spins
            self.flavors_indexed = self.fc.flavors
            self.isospins_indexed = self.fc.isospins
        elif self.fc.n_particles == 3:
            self.masses_indexed = list(np.array(self.fc.masses)[indexing])
            self.spins_indexed = list(np.array(self.fc.spins)[indexing])
            self.flavors_indexed = list(np.array(self.fc.flavors)[indexing])
            self.isospins_indexed = list(np.array(self.fc.isospins)[indexing])
        else:
            raise NotImplementedError("only 2- and 3-body channels supported")

        if p_cot_deltas is None:
            p_cot_deltas = []
            for _ in range(len(ell_set)):
                p_cot_deltas.append(QCFunctions.pcotdelta_scattering_length)
                self._p_cot_deltas = p_cot_deltas
                self.p_cot_deltas = p_cot_deltas
        else:
            self._p_cot_deltas = p_cot_deltas
            self.p_cot_deltas = p_cot_deltas

        self._n_params_set = n_params_set

    @property
    def fc(self):
        """FlavorChannel object of the spectator channel."""
        return self._fc

    @fc.setter
    def fc(self, fc):
        """Set the FlavorChannel object of the spectator channel."""
        self._fc = fc
        self.indexing = self._indexing
        self.sub_isospin = self._sub_isospin
        self.ell_set = self._ell_set
        self.p_cot_deltas = self._p_cot_deltas

    @property
    def indexing(self):
        """Indexing of the spectator channel."""
        return self._indexing

    @indexing.setter
    def indexing(self, indexing):
        """Set the indexing of the spectator channel."""
        if (self.fc.n_particles == 2) and (indexing is not None):
            warnings.warn(f"\n{bcolors.WARNING}"
                          f"n_particles == 2 and indexing is not None; "
                          f"setting it to None"
                          f"{bcolors.ENDC}", stacklevel=2)
            self._indexing = None
        elif (self.fc.n_particles == 2) and (indexing is None):
            self._indexing = None
        elif self.fc.n_particles >= 3:
            if not isinstance(indexing, list):
                raise ValueError("for n_particles > 2, indexing must be a "
                                 "list")
            if len(indexing) != self.fc.n_particles:
                raise ValueError("indexing must have length n_particles")
            if (np.sort(indexing) != np.arange(self.fc.n_particles)).any():
                raise ValueError("indexing must be a permuatation of "
                                 "ascending integers")
            self._indexing = indexing
        else:
            raise ValueError("unknown problem with indexing")

    @property
    def sub_isospin(self):
        """Sub-channel isospin of the spectator channel."""
        return self._sub_isospin

    @sub_isospin.setter
    def sub_isospin(self, sub_isospin):
        """Set the sub-channel isospin of the spectator channel."""
        if ((sub_isospin is not None)
           and (self.fc.n_particles == 2)):
            raise ValueError("sub_isospin must be None "
                             "for n_particles == 2")
        if ((sub_isospin is not None)
           and (not isinstance(sub_isospin, float))):
            raise ValueError("sub_isospin must be an float")
        if ((sub_isospin is not None)
           and (self.allowed_sub_isospins is not None)
           and (sub_isospin not in self.allowed_sub_isospins)):
            raise ValueError("sub-isospin is not in allowed set")
        if (not self.fc.isospin_channel) and (sub_isospin is not None):
            raise ValueError("sub_isospin cannot be set because "
                             "isospin_channel is False")
        if (self.fc.isospin_channel and (sub_isospin is None)
           and (self.fc.n_particles != 2)):
            raise ValueError("sub_isospin cannot be set to None because "
                             "isospin_channel is True")
        self._sub_isospin = sub_isospin

    @property
    def ell_set(self):
        """Angular-momentum set of the spectator channel."""
        return self._ell_set

    @ell_set.setter
    def ell_set(self, ell_set):
        """Set the angular-momentum set of the spectator channel."""
        if ell_set is None:
            self._ell_set = None
            self._p_cot_deltas = None
            self._n_params_set = None
        else:
            if self._p_cot_deltas is None:
                self._p_cot_deltas = []
            if len(self._p_cot_deltas) > len(ell_set):
                for _ in range(len(self._p_cot_deltas)-len(self._ell_set)):
                    self._p_cot_deltas.pop()
            elif (len(self._p_cot_deltas) < len(ell_set)
                  and len(self._p_cot_deltas) != 0):
                for _ in range(len(self._ell_set)-len(self.p_cot_deltas)):
                    self._p_cot_deltas.append(self._p_cot_deltas[-1])
            elif len(self._p_cot_deltas) < len(ell_set):
                for _ in range(len(self._ell_set)-len(self.p_cot_deltas)):
                    self._p_cot_deltas.append(
                        QCFunctions.pcotdelta_scattering_length)
            self._ell_set = ell_set
            self._n_params_set = []
            for p_cot_delta in self._p_cot_deltas:
                self._n_params_set.append(
                    len(signature(p_cot_delta).parameters)-1)

    @property
    def p_cot_deltas(self):
        """p-cot-delta functions of the spectator channel."""
        return self._p_cot_deltas

    @p_cot_deltas.setter
    def p_cot_deltas(self, p_cot_deltas):
        """Set the p-cot-delta functions of the spectator channel."""
        if p_cot_deltas is None:
            self._ell_set = None
            self._p_cot_deltas = None
            self._n_params_set = None
        else:
            self._n_params_set = []
            for p_cot_delta in p_cot_deltas:
                self._n_params_set.append(
                    len(signature(p_cot_delta).parameters)-1)
            if self._ell_set is None:
                self._ell_set = []
            elif len(p_cot_deltas) < len(self._ell_set):
                for _ in range(len(self._ell_set)-len(self.p_cot_deltas)):
                    self._ell_set.pop()
            elif len(p_cot_deltas) > len(self._ell_set):
                for _ in range(len(self.p_cot_deltas)-len(self._ell_set)):
                    if len(self._ell_set) == 0:
                        self._ell_set.append(0)
                    else:
                        self._ell_set.append(self._ell_set[-1]+1)
            self._p_cot_deltas = p_cot_deltas

    @property
    def n_params_set(self):
        """Parameter counts of the spectator channel p-cot-deltas."""
        return self._n_params_set

    def __eq__(self, other):
        """Return True if the two SpectatorChannel objects are equal."""
        if not isinstance(other, SpectatorChannel):
            return False
        if not (self.fc == other.fc):
            return False
        if not (self.indexing == other.indexing):
            return False
        if not (self.sub_isospin == other.sub_isospin):
            return False
        if not (self.ell_set == other.ell_set):
            return False
        if not (self.p_cot_deltas == other.p_cot_deltas):
            return False
        if not (self.n_params_set == other.n_params_set):
            return False
        return True

    def __str__(self):
        """Return a string representation of the SpectatorChannel object."""
        spectator_channel_str = self.fc.__str__().replace("Flavor",
                                                          "Spectator")
        spectator_channel_str = spectator_channel_str[:-1]+",\n"
        spectator_channel_str += f"    indexing: {self.indexing},\n"
        if self.fc.isospin_channel:
            if self.sub_isospin is not None:
                spectator_channel_str += f"    sub_isospin: "\
                    f"{self.sub_isospin},\n"
            if self.allowed_sub_isospins is not None:
                spectator_channel_str += f"    allowed sub_isospins: "\
                    f"{self.allowed_sub_isospins},\n"
        spectator_channel_str += f"    ell_set: {self.ell_set},\n"
        for i, ell in enumerate(self.ell_set):
            spectator_channel_str += f"    p_cot_delta_{ell}: "\
                f"{self.p_cot_deltas[i]},\n"
        spectator_channel_str += f"    n_params_set: {self.n_params_set},\n"
        return spectator_channel_str[:-2]+"."


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

    def __init__(self, fc_list=[], ni_list=None):
        self.fc_list = fc_list
        if ni_list is None:
            self.ni_list = fc_list
        else:
            self.ni_list = ni_list
        self.sc_list = []
        for fc in fc_list:
            self._add_flavor_channel(fc)
        self._build_sorted_sc_list()
        self._build_g_templates()
        self._build_g_templates_ell_specific()

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
                                  "Assuming ell_set = [0] for spectator with "
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
        if isospin_i == isospin_j:
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


class FiniteVolumeSetup:
    """
    Class used to represent a finite volume setup.

    :param formalism: formalism used (currently only ``'RFT'`` supported)
    :type formalism: str
    :param nP: total momentum in the finite-volume frame
    :type nP: numpy.ndarray
    :param qc_impl: implementation details of the finite-volume setup
    :type qc_impl: dict
    :param irreps: irreducible representations of the finite-volume
    symmetry group
    :type irreps: Irreps
    """

    def __init__(self, formalism='RFT', nP=np.array([0, 0, 0]), qc_impl={}):
        self.formalism = formalism
        self.qc_impl = qc_impl
        self.nP = nP
        self.irreps = Irreps(nP=self.nP)

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def qc_impl(self):
        """
        Implementation of the quantization condition.

        See FiniteVolumeSetup for documentation of possible keys included in
        qc_impl.
        """
        return self._qc_impl

    @qc_impl.setter
    def qc_impl(self, qc_impl):
        """Set the implementation of the quantization condition."""
        if not isinstance(qc_impl, dict):
            raise ValueError("qc_impl must be a dict")
        for key in qc_impl.keys():
            if key not in QC_IMPL_DEFAULTS.keys():
                raise ValueError("key", key, "not recognized")
        for key in QC_IMPL_DEFAULTS.keys():
            if (key in qc_impl.keys()
               and (not isinstance(qc_impl[key],
                                   type(QC_IMPL_DEFAULTS[key])))):
                raise ValueError(f"qc_impl entry {key} mest be a "
                                 f"{type(QC_IMPL_DEFAULTS[key])}")
        self._qc_impl = qc_impl

    def __str__(self):
        """Return a string representation of the FiniteVolumeSetup object."""
        finite_volume_setup_str =\
            f"FiniteVolumeSetup using the {self.formalism}:\n"
        finite_volume_setup_str += f"    nP = {self._nP},\n"
        finite_volume_setup_str += f"    qc_impl = {self.qc_impl},\n"
        return finite_volume_setup_str[:-2]+"."


class ThreeBodyInteractionScheme:
    """
    Class used to represent all details of the three-body interaction.

    :param fcs: FlavorChannelSpace object, needed to define the space for Kdf
    :type fcs: FlavorChannelSpace
    :param Emin: minimum energy defining the subthreshold region
    :type Emin: float
    :param three_scheme: three-body interaction scheme
    :type three_scheme: str
    :param scheme_data: data needed to define the three-body interaction,
    typically ``[alpha, beta]`` where alpha dictates the width of the smooth
    cutoff function and beta dictates the position
    :type scheme_data: list
    :param kdf_functions: list of functions used to define the three-body
    interaction for each pair of FlavorChannels
    :type kdf_functions: list
    """

    def __init__(self, fcs=None, Emin=0.0, three_scheme='relativistic pole',
                 scheme_data=[-1.0, 0.0], kdf_functions=None):
        self.Emin = Emin
        if fcs is None:
            self.fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        else:
            self.fcs = fcs
        self.three_scheme = three_scheme
        self.scheme_data = scheme_data
        if kdf_functions is None:
            self.kdf_functions = []
            for fc in self.fcs.fc_list:
                self.kdf_functions = self.kdf_functions+[self.kdf_iso_constant]
        else:
            self.kdf_functions = kdf_functions

    def with_str(str_func):
        """Change print behavior of a function."""
        def wrapper(f):
            class FuncType:
                def __call__(self, *args, **kwargs):
                    return f(*args, **kwargs)

                def __str__(self):
                    return str_func()
            return functools.wraps(f)(FuncType())
        return wrapper

    def kdf_iso_constant_str():
        """Print behavior for kdf_iso_constant."""
        return "kdf_iso_constant"

    @with_str(kdf_iso_constant_str)
    def kdf_iso_constant(beta_0):
        """Kdf: constant form."""
        return beta_0

    def __str__(self):
        """Return a string representation of the ThreeBodyInteractionScheme."""
        three_body_interaction_scheme_str =\
            "ThreeBodyInteractionScheme with the following data:\n"
        three_body_interaction_scheme_str += f"    Emin = {self.Emin},\n"
        three_body_interaction_scheme_str +=\
            f"    three_scheme = {self.three_scheme},\n"
        three_body_interaction_scheme_str +=\
            f"    [alpha, beta] = {self.scheme_data},\n"
        three_body_interaction_scheme_str +=\
            "    kdf_functions as follows:\n"
        for i in range(len(self.fcs.fc_list)):
            three_body_interaction_scheme_str +=\
                f"        {self.kdf_functions[i]} for\n"
            three_body_interaction_scheme_str +=\
                "        "+str(self.fcs.fc_list[i]).replace(
                    "   ", "            ")[:-1]+",\n"
        return three_body_interaction_scheme_str[:-2]+"."


class ThreeBodyKinematicSpace:
    """
    Class encoding the spectator-momentum kinematics.

    :param nP: total momentum in the finite-volume frame
    :type nP: numpy.ndarray
    :param nvec_arr: array of nvecs
    :type nvec_arr: numpy.ndarray
    :param build_shell_acc: whether to build the data to accelerate the
    evaluations by shell
    :type build_shell_acc: bool
    :param verbosity: verbosity level
    :type verbosity: int
    :param nPSQ: total momentum squared in the finite-volume frame
    :type nPSQ: int
    :param nPmag: magnitude of the total momentum in the finite-volume frame
    :type nPmag: float
    :param shells: list of shells
    :type shells: list
    :param nvecSQ_arr: array of nvec squared
    :type nvecSQ_arr: numpy.ndarray
    :param n1vec_mat: matrix of n1vecs
    :type n1vec_mat: numpy.ndarray
    :param n2vec_mat: matrix of n2vecs
    :type n2vec_mat: numpy.ndarray
    :param n3vec_mat: matrix of n3vecs
    :type n3vec_mat: numpy.ndarray
    :param nP_minus_nvec_arr: array of nP - nvecs
    :type nP_minus_nvec_arr: numpy.ndarray
    :param nP_minus_nvec_SQ_arr: array of (nP - nvecs)^2
    :type nP_minus_nvec_SQ_arr: numpy.ndarray
    :param nvecmag_arr: array of nvec magnitudes
    :type nvecmag_arr: numpy.ndarray
    :param nP_minus_nvec_mag_arr: array of |nP - nvec|
    :type nP_minus_nvec_mag_arr: numpy.ndarray
    :param n1vecSQ_mat: matrix of n1vec squared
    :type n1vecSQ_mat: numpy.ndarray
    :param n2vecSQ_mat: matrix of n2vec squared
    :type n2vecSQ_mat: numpy.ndarray
    :param n3vecSQ_mat: matrix of n3vec squared
    :type n3vecSQ_mat: numpy.ndarray
    :param nP_minus_n1vec_mat: matrix of nP - n1vecs
    :type nP_minus_n1vec_mat: numpy.ndarray
    :param nP_minus_n2vec_mat: matrix of nP - n2vecs
    :type nP_minus_n2vec_mat: numpy.ndarray
    :param n1vec_stacked: stacked n1vecs
    :type n1vec_stacked: numpy.ndarray
    :param n2vec_stacked: stacked n2vecs
    :type n2vec_stacked: numpy.ndarray
    :param n3vec_stacked: stacked n3vecs
    :type n3vec_stacked: numpy.ndarray
    :param stack_multiplicities: multiplicities of the stacked nvecs
    :type stack_multiplicities: numpy.ndarray
    :param n1vecSQ_stacked: stacked n1vec^2
    :type n1vecSQ_stacked: numpy.ndarray
    :param n2vecSQ_stacked: stacked n2vec^2
    :type n2vecSQ_stacked: numpy.ndarray
    :param n3vecSQ_stacked: stacked n3vec^2
    :type n3vecSQ_stacked: numpy.ndarray
    :param n1vec_arr_all_shells: array of n1vecs for all shells
    :type n1vec_arr_all_shells: numpy.ndarray
    :param n2vec_arr_all_shells: array of n2vecs for all shells
    :type n2vec_arr_all_shells: numpy.ndarray
    :param n1vecSQ_arr_all_shells: array of n1vec^2 for all shells
    :type n1vecSQ_arr_all_shells: numpy.ndarray
    :param n2vecSQ_arr_all_shells: array of n2vec^2 for all shells
    :type n2vecSQ_arr_all_shells: numpy.ndarray
    :param n3vecSQ_arr_all_shells: array of n3vec^2 for all shells
    :type n1vec_mat_all_shells: numpy.ndarray
    :param n2vec_mat_all_shells: matrix of n2vecs for all shells
    :type n2vec_mat_all_shells: numpy.ndarray
    :param n3vec_mat_all_shells: matrix of n3vecs for all shells
    :type n3vec_mat_all_shells: numpy.ndarray
    :param n1vecSQ_mat_all_shells: matrix of n1vec^2 for all shells
    :type n1vecSQ_mat_all_shells: numpy.ndarray
    :param n2vecSQ_mat_all_shells: matrix of n2vec^2 for all shells
    :type n2vecSQ_mat_all_shells: numpy.ndarray
    :param n3vecSQ_mat_all_shells: matrix of n3vec^2 for all shells
    :type n3vecSQ_mat_all_shells: numpy.ndarray
    """

    def __init__(self, nP=np.array([0, 0, 0]), nvec_arr=np.array([]),
                 build_shell_acc=True, verbosity=0):
        self.build_shell_acc = build_shell_acc
        self.nP = nP
        self.nvec_arr = nvec_arr
        self.verbosity = verbosity

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def nvec_arr(self):
        """Array of spectator-momentum vectors in the finite-volume frame."""
        return self._nvec_arr

    def _get_first_sort(self, nvec_arr):
        nvecSQ_arr = (nvec_arr*nvec_arr).sum(1)
        nP_minus_nvec_arr = self.nP - nvec_arr
        nP_minus_nvec_SQ_arr = (nP_minus_nvec_arr
                                * nP_minus_nvec_arr).sum(1)
        arrtmp = np.concatenate(
            (
                nvecSQ_arr.reshape((1, len(nvec_arr))),
                nP_minus_nvec_SQ_arr.reshape((1, len(nvec_arr))),
                nvec_arr.T
             )
            ).T
        for i in range(5):
            arrtmp = arrtmp[arrtmp[:, 4-i].argsort(kind='mergesort')]
        return (arrtmp.T)[2:].T

    def _get_shell_sort(self, nvec_arr_first_sort):
        group = Groups(ell_max=0)
        little_group = group.get_little_group(self._nP)
        nvec_arr_copy = np.copy(nvec_arr_first_sort)
        shell_dict_nvec_arr = {}
        shell_dict_index = 0
        while len(nvec_arr_copy) > 0:
            nvec_tmp = nvec_arr_copy[0].reshape((3, 1))
            nvec_rotations = (little_group*nvec_tmp).sum(1)
            nvec_rotations_unique = np.unique(nvec_rotations, axis=0)
            shell_dict_nvec_arr[shell_dict_index] = nvec_rotations_unique
            shell_dict_index = shell_dict_index+1
            mins = np.minimum(nvec_rotations_unique.min(0),
                              nvec_arr_copy.min(0))
            nvec_rotations_shifted = nvec_rotations_unique-mins
            nvec_arr_shifted = nvec_arr_copy-mins
            dims = np.maximum(nvec_rotations_shifted.max(0),
                              nvec_arr_shifted.max(0))+1
            nvec_arr_shifted_purged = nvec_arr_shifted[~np.in1d(
                np.ravel_multi_index(
                    nvec_arr_shifted.T, dims
                    ),
                np.ravel_multi_index(
                    nvec_rotations_shifted.T, dims
                    )
                )]
            nvec_arr_copy = nvec_arr_shifted_purged+mins
        nvec_arr_shell_sort = None
        shells = [[]]
        shells_counter = 0
        for i in range(len(shell_dict_nvec_arr)):
            if nvec_arr_shell_sort is None:
                nvec_arr_shell_sort = shell_dict_nvec_arr[i]
            else:
                nvec_arr_shell_sort = np.concatenate((nvec_arr_shell_sort,
                                                     shell_dict_nvec_arr[i]))
            shells = shells+[[shells_counter,
                              shells_counter+len(shell_dict_nvec_arr[i])]]
            shells_counter = shells_counter+len(shell_dict_nvec_arr[i])
        return nvec_arr_shell_sort, shells[1:]

    def _get_stacks(self):
        n1vec_stackdict = {}
        n2vec_stackdict = {}
        n3vec_stackdict = {}
        stackdict_multiplicities = {}
        for i1 in self.shells:
            for j1 in self.shells:
                stri = str(i1[0])+'_'+str(i1[1])
                strj = str(j1[0])+'_'+str(j1[1])
                n1vec_stackdict[stri+'_'+strj] = 0.0
                n2vec_stackdict[stri+'_'+strj] = 0.0
                n3vec_stackdict[stri+'_'+strj] = 0.0
                stackdict_multiplicities[stri+'_'+strj]\
                    = (i1[1]-i1[0])*(j1[1]-j1[0])

        for i1 in range(len(self.nvecSQ_arr)):
            for j1 in range(len(self.nvecSQ_arr)):
                shell_index_i = 0
                shell_index_j = 0
                for i2 in range(len(self.shells)):
                    shell_tmp = self.shells[i2]
                    if shell_tmp[0] <= i1 < shell_tmp[1]:
                        shell_index_i = i2
                for j2 in range(len(self.shells)):
                    shell_tmp = self.shells[j2]
                    if shell_tmp[0] <= j1 < shell_tmp[1]:
                        shell_index_j = j2
                stri = str(self.shells[shell_index_i][0])+'_'\
                    + str(self.shells[shell_index_i][1])
                strj = str(self.shells[shell_index_j][0])+'_'\
                    + str(self.shells[shell_index_j][1])
                sizei = self.shells[shell_index_i][1]\
                    - self.shells[shell_index_i][0]
                sizej = self.shells[shell_index_j][1]\
                    - self.shells[shell_index_j][0]
                if (sizei >= sizej and i1 == self.shells[shell_index_i][0])\
                   or (sizej > sizei and j1 == self.shells[shell_index_j][0]):
                    if n1vec_stackdict[stri+'_'+strj] == 0.0:
                        n1vec_stackdict[stri+'_'+strj]\
                            = [self.n1vec_mat[i1][j1]]
                    else:
                        n1vec_stackdict[stri+'_'+strj] =\
                            n1vec_stackdict[stri+'_'+strj]\
                            + [self.n1vec_mat[i1][j1]]
                    if n2vec_stackdict[stri+'_'+strj] == 0.0:
                        n2vec_stackdict[stri+'_'+strj]\
                            = [self.n2vec_mat[i1][j1]]
                    else:
                        n2vec_stackdict[stri+'_'+strj]\
                            = n2vec_stackdict[stri+'_'+strj]\
                            + [self.n2vec_mat[i1][j1]]
                    if n3vec_stackdict[stri+'_'+strj] == 0.0:
                        n3vec_stackdict[stri+'_'+strj]\
                            = [self.n3vec_mat[i1][j1]]
                    else:
                        n3vec_stackdict[stri+'_'+strj] =\
                            n3vec_stackdict[stri+'_'+strj]\
                            + [self.n3vec_mat[i1][j1]]
        n1vec_stacked = [[]]
        n2vec_stacked = [[]]
        n3vec_stacked = [[]]
        stack_multiplicities = [[]]
        for i1 in self.shells:
            n1row = []
            n2row = []
            n3row = []
            mrow = []
            for j1 in self.shells:
                stri = str(i1[0])+'_'+str(i1[1])
                strj = str(j1[0])+'_'+str(j1[1])
                n1row = n1row + [np.array(n1vec_stackdict[stri+'_'+strj])]
                n2row = n2row + [np.array(n2vec_stackdict[stri+'_'+strj])]
                n3row = n3row + [np.array(n3vec_stackdict[stri+'_'+strj])]
                mrow = mrow\
                    + [stackdict_multiplicities[stri+'_'+strj]
                       / len(np.array(n1vec_stackdict[stri+'_'+strj]))]
            n1vec_stacked = n1vec_stacked+[n1row]
            n2vec_stacked = n2vec_stacked+[n2row]
            n3vec_stacked = n3vec_stacked+[n3row]
            stack_multiplicities = stack_multiplicities + [mrow]
        return n1vec_stacked[1:], n2vec_stacked[1:], n3vec_stacked[1:],\
            np.array(stack_multiplicities[1:])

    @nvec_arr.setter
    def nvec_arr(self, nvec_arr):
        """Set the nvec_arr attribute."""
        if self.build_shell_acc:
            if len(nvec_arr) == 0:
                self._nvec_arr = nvec_arr
            else:
                nvec_arr_first_sort = self._get_first_sort(nvec_arr)
                self._nvec_arr, self.shells\
                    = self._get_shell_sort(nvec_arr_first_sort)
                self.nvecSQ_arr = (self._nvec_arr*self._nvec_arr).sum(1)
                self.nP_minus_nvec_arr = self.nP - self._nvec_arr
                self.nP_minus_nvec_SQ_arr = (self.nP_minus_nvec_arr
                                             * self.nP_minus_nvec_arr).sum(1)
                self.nvecmag_arr = np.sqrt(self.nvecSQ_arr)
                self.nP_minus_nvec_mag_arr = np.sqrt(self.nP_minus_nvec_SQ_arr)
                self.n1vec_mat = (np.tile(self._nvec_arr,
                                          (len(self._nvec_arr), 1))).reshape(
                    (len(self._nvec_arr), len(self._nvec_arr), 3))
                self.n2vec_mat = np.transpose(self.n1vec_mat, (1, 0, 2))
                self.n3vec_mat = self.nP-self.n1vec_mat-self.n2vec_mat
                self.n1vecSQ_mat = (self.n1vec_mat*self.n1vec_mat).sum(2)
                self.n2vecSQ_mat = (self.n2vec_mat*self.n2vec_mat).sum(2)
                self.n3vecSQ_mat = (self.n3vec_mat*self.n3vec_mat).sum(2)
                self.nP_minus_n1vec_mat = self.nP - self.n1vec_mat
                self.nP_minus_n2vec_mat = self.nP - self.n2vec_mat
                self.n1vec_stacked, self.n2vec_stacked, self.n3vec_stacked,\
                    self.stack_multiplicities = self._get_stacks()
                n1vecSQ_stacked = [[]]
                for tmprow in self.n1vec_stacked:
                    n1vecSQ_row = []
                    for ent in tmprow:
                        n1vecSQ_row = n1vecSQ_row+[(ent*ent).sum(1)]
                    n1vecSQ_stacked = n1vecSQ_stacked+[n1vecSQ_row]
                self.n1vecSQ_stacked = n1vecSQ_stacked[1:]
                n2vecSQ_stacked = [[]]
                for tmprow in self.n2vec_stacked:
                    n2vecSQ_row = []
                    for ent in tmprow:
                        n2vecSQ_row = n2vecSQ_row+[(ent*ent).sum(1)]
                    n2vecSQ_stacked = n2vecSQ_stacked+[n2vecSQ_row]
                self.n2vecSQ_stacked = n2vecSQ_stacked[1:]
                n3vecSQ_stacked = [[]]
                for tmprow in self.n3vec_stacked:
                    n3vecSQ_row = []
                    for ent in tmprow:
                        n3vecSQ_row = n3vecSQ_row+[(ent*ent).sum(1)]
                    n3vecSQ_stacked = n3vecSQ_stacked+[n3vecSQ_row]
                self.n3vecSQ_stacked = n3vecSQ_stacked[1:]

                n1vec_arr_all_shells = [[]]
                n1vecSQ_arr_all_shells = [[]]
                n2vec_arr_all_shells = [[]]
                n2vecSQ_arr_all_shells = [[]]
                n1vec_mat_all_shells = [[]]
                n2vec_mat_all_shells = [[]]
                n3vec_mat_all_shells = [[]]
                n1vecSQ_mat_all_shells = [[]]
                n2vecSQ_mat_all_shells = [[]]
                n3vecSQ_mat_all_shells = [[]]
                for row_shell in self.shells:
                    n1vec_arr_row_shells = []
                    n1vecSQ_arr_row_shells = []
                    n2vec_arr_row_shells = []
                    n2vecSQ_arr_row_shells = []
                    n1vec_mat_row_shells = []
                    n2vec_mat_row_shells = []
                    n3vec_mat_row_shells = []
                    n1vecSQ_mat_row_shells = []
                    n2vecSQ_mat_row_shells = []
                    n3vecSQ_mat_row_shells = []
                    for col_shell in self.shells:
                        n1vec_arr_shell = self.nvec_arr[
                            row_shell[0]:row_shell[1]]
                        n1vecSQ_arr_shell = self.nvecSQ_arr[
                            row_shell[0]:row_shell[1]]
                        n2vec_arr_shell = self.nvec_arr[
                            col_shell[0]:col_shell[1]]
                        n2vecSQ_arr_shell = self.nvecSQ_arr[
                            col_shell[0]:col_shell[1]]

                        # Awkward swap here
                        n1vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n2vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

                        n2vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n1vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

                        n3vec_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n3vec_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

                        n1vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n2vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

                        n2vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n1vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )

                        n3vecSQ_mat_shell = np.swapaxes(
                            np.swapaxes(
                                ((self.n3vecSQ_mat)[
                                    row_shell[0]:row_shell[1]]),
                                0, 1
                                )[col_shell[0]:col_shell[1]],
                            0, 1
                            )
                        n1vec_arr_row_shells\
                            = n1vec_arr_row_shells+[n1vec_arr_shell]
                        n1vecSQ_arr_row_shells\
                            = n1vecSQ_arr_row_shells+[n1vecSQ_arr_shell]
                        n2vec_arr_row_shells\
                            = n2vec_arr_row_shells+[n2vec_arr_shell]
                        n2vecSQ_arr_row_shells\
                            = n2vecSQ_arr_row_shells+[n2vecSQ_arr_shell]
                        n1vec_mat_row_shells\
                            = n1vec_mat_row_shells+[n1vec_mat_shell]
                        n2vec_mat_row_shells\
                            = n2vec_mat_row_shells+[n2vec_mat_shell]
                        n3vec_mat_row_shells\
                            = n3vec_mat_row_shells+[n3vec_mat_shell]
                        n1vecSQ_mat_row_shells\
                            = n1vecSQ_mat_row_shells+[n1vecSQ_mat_shell]
                        n2vecSQ_mat_row_shells\
                            = n2vecSQ_mat_row_shells+[n2vecSQ_mat_shell]
                        n3vecSQ_mat_row_shells\
                            = n3vecSQ_mat_row_shells+[n3vecSQ_mat_shell]
                    n1vec_arr_all_shells\
                        = n1vec_arr_all_shells+[n1vec_arr_row_shells]
                    n1vecSQ_arr_all_shells\
                        = n1vecSQ_arr_all_shells+[n1vecSQ_arr_row_shells]
                    n2vec_arr_all_shells\
                        = n2vec_arr_all_shells+[n2vec_arr_row_shells]
                    n2vecSQ_arr_all_shells\
                        = n2vecSQ_arr_all_shells+[n2vecSQ_arr_row_shells]
                    n1vec_mat_all_shells\
                        = n1vec_mat_all_shells+[n1vec_mat_row_shells]
                    n2vec_mat_all_shells\
                        = n2vec_mat_all_shells+[n2vec_mat_row_shells]
                    n3vec_mat_all_shells\
                        = n3vec_mat_all_shells+[n3vec_mat_row_shells]
                    n1vecSQ_mat_all_shells\
                        = n1vecSQ_mat_all_shells+[n1vecSQ_mat_row_shells]
                    n2vecSQ_mat_all_shells\
                        = n2vecSQ_mat_all_shells+[n2vecSQ_mat_row_shells]
                    n3vecSQ_mat_all_shells\
                        = n3vecSQ_mat_all_shells+[n3vecSQ_mat_row_shells]
                    self.n1vec_arr_all_shells = n1vec_arr_all_shells[1:]
                    self.n1vecSQ_arr_all_shells = n1vecSQ_arr_all_shells[1:]
                    self.n2vec_arr_all_shells = n2vec_arr_all_shells[1:]
                    self.n2vecSQ_arr_all_shells = n2vecSQ_arr_all_shells[1:]
                    self.n1vec_mat_all_shells = n1vec_mat_all_shells[1:]
                    self.n2vec_mat_all_shells = n2vec_mat_all_shells[1:]
                    self.n3vec_mat_all_shells = n3vec_mat_all_shells[1:]
                    self.n1vecSQ_mat_all_shells = n1vecSQ_mat_all_shells[1:]
                    self.n2vecSQ_mat_all_shells = n2vecSQ_mat_all_shells[1:]
                    self.n3vecSQ_mat_all_shells = n3vecSQ_mat_all_shells[1:]
        else:
            self._nvec_arr = nvec_arr

    def __str__(self):
        """Return a string representation of the ThreeBodyKinematicSpace."""
        np.set_printoptions(threshold=10)
        three_body_kinematic_space_str =\
            "ThreeBodyKinematicSpace with the following data:\n"
        three_body_kinematic_space_str += "    nvec_arr="\
            + str(self.nvec_arr).replace("\n", "\n             ")+",\n"
        np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)
        return three_body_kinematic_space_str[:-2]+"."


class QCIndexSpace:
    """
    Class representing the quantization condition index space.

    :param fcs: flavor-channel space
    :type fcs: FlavorChannelSpace
    :param fvs: finite-volume setup
    :type fvs: FiniteVolumeSetup
    :param tbis: three-body interaction scheme
    :type tbis: ThreeBodyInteractionScheme
    :param Emax: maximum energy
    :type Emax: float
    :param Lmax: maximum volume
    :type Lmax: float
    :param verbosity: verbosity level
    :type verbosity: int
    :param nP: total momentum in the finite-volume frame
    :type nP: numpy.ndarray
    :param nPSQ: squared total momentum in the finite-volume frame
    :type nPSQ: int
    :param nPmag: magnitude of the total momentum in the finite-volume frame
    :type nPmag: float
    :param Evals: energy values used to build the grid for non-zero nP
    :type Evals: numpy.ndarray
    :param Lvals: volume values used to build the grid for non-zero nP
    :type Lvals: numpy.ndarray
    :param param_structure: structure of the quantization condition parameters
    :type param_structure: list
    :param ell_sets: list of sets of angular momenta
    :type ell_sets: list
    :param ellm_sets: list of sets of angular momenta and their azimuthal
    values
    :type ellm_sets: list
    :param proj_dict: dictionary of projection matrices
    :type proj_dict: dict
    :param non_int_proj_dict: dictionary of projection matrices for the
    non-interacting states
    :type non_int_proj_dict: dict
    :param group: relevant symmetry group
    :type group: Group
    :param n_channels: number of channels
    :type n_channels: int
    :param n_two_channels: number of two-particle channels
    :type n_two_channels: int
    :param n_three_channels: number of three-particle channels
    :type n_three_channels: int
    :param tbks_list: list of three-body kinematic spaces
    :type tbks_list: list of ThreeBodyKinematicSpace
    :param kellm_spaces: list of spectator + angular-momentum spaces
    :type kellm_spaces: list
    :param kellm_shells: list of spectator + angular-momentum spaces,
    organized by shell
    :type kellm_shells: list
    :param sc_proj_dicts: list of projection dictionaries by spectator channel
    :type sc_proj_dicts: list
    :param sc_proj_dicts_by_shell: list of projection dictionaries by spectator
    channel, organized by momentum shell
    :type sc_proj_dicts_by_shell: list

    :param nvecset_arr: nvecs
    :type nvecset_arr: numpy.ndarray
    :param nvecset_SQs: nvecs^2
    :type nvecset_SQs: numpy.ndarray
    :param nvecset_reps: nvec representatives
    :type nvecset_reps: numpy.ndarray
    :param nvecset_SQreps: nvec^2 representatives
    :type nvecset_SQreps: numpy.ndarray
    :param nvecset_inds: indices for the nvecs
    :type nvecset_inds: numpy.ndarray
    :param nvecset_counts: counts for the nvecs
    :type nvecset_counts: numpy.ndarray
    :param nvecset_batched: nvecs organized by batch
    :type nvecset_batched: numpy.ndarray

    :param nvecset_ident: nvecs for identical particles
    :type nvecset_ident: numpy.ndarray
    :param nvecset_ident_SQs: nvecs^2 for identical particles
    :type nvecset_ident_SQs: numpy.ndarray
    :param nvecset_ident_reps: nvec representatives for identical particles
    :type nvecset_ident_reps: numpy.ndarray
    :param nvecset_ident_SQreps: nvec^2 representatives for identical particles
    :type nvecset_ident_SQreps: numpy.ndarray
    :param nvecset_ident_inds: indices for the nvecs for identical particles
    :type nvecset_ident_inds: numpy.ndarray
    :param nvecset_ident_counts: counts for the nvecs for identical particles
    :type nvecset_ident_counts: numpy.ndarray
    :param nvecset_ident_batched: nvecs organized by batch for identical
    particles
    :type nvecset_ident_batched: numpy.ndarray
    """

    def __init__(self, fcs=None, fvs=None, tbis=None,
                 Emax=5.0, Lmax=5.0, verbosity=0):
        self.verbosity = verbosity
        self.Emax = Emax
        self.Lmax = Lmax

        if fcs is None:
            if verbosity == 2:
                print('setting the flavor-channel space, None was passed')
            self._fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        else:
            if verbosity == 2:
                print('setting the flavor-channel space')
            self._fcs = fcs

        if fvs is None:
            if verbosity == 2:
                print('setting the finite-volume setup, None was passed')
            self.fvs = FiniteVolumeSetup()
        else:
            if verbosity == 2:
                print('setting the finite-volume setup')
            self.fvs = fvs

        if tbis is None:
            if verbosity == 2:
                print('setting the three-body interaction scheme, '
                      + 'None was passed')
            self.tbis = ThreeBodyInteractionScheme()
        else:
            if verbosity == 2:
                print('setting the three-body interaction scheme')
            self.tbis = tbis

        self._nP = self.fvs.nP
        if verbosity == 2:
            print('setting nP to '+str(self.fvs.nP))
        self.nP = self.fvs.nP
        self.fcs = self._fcs

    def populate(self):
        """Populate the index space."""
        ell_max = 4
        for sc in self.fcs.sc_list_sorted:
            if np.max(sc.ell_set) > ell_max:
                ell_max = np.max(sc.ell_set)
        for nic in self.fcs.ni_list:
            spins = nic.spins
            for spin in spins:
                spin_int = int(spin)
                if np.abs(spin-spin_int) > EPSILON10:
                    raise ValueError("only integer spin currently supported")
            maxspin = int(np.max(nic.spins))
            if maxspin > ell_max:
                ell_max = maxspin
        self.group = Groups(ell_max=ell_max)

        if self.nPSQ != 0:
            if self.verbosity == 2:
                print('nPSQ is nonzero, will use grid')
            [self.Evals, self.Lvals] = self._get_grid_nonzero_nP(self.Emax,
                                                                 self.Lmax)
            if self.verbosity == 2:
                print('Lvals =', self.Lvals)
                print('Evals =', self.Evals)
        else:
            self.Lvals = None
            self.Evals = None

        parametrization_structure = []
        two_param_struc_tmp = []
        for sc in self.fcs.sc_list_sorted:
            tmp_entry = []
            for n_params_tmp in sc.n_params_set:
                tmp_entry.append([0.0]*n_params_tmp)
            two_param_struc_tmp.append(tmp_entry)
        parametrization_structure.append(two_param_struc_tmp)
        three_param_struc_tmp = []
        for _ in self.tbis.kdf_functions:
            three_param_struc_tmp = three_param_struc_tmp+[0.0]
        parametrization_structure.append(three_param_struc_tmp)
        self.param_structure = parametrization_structure

        self.populate_all_nvec_arr()
        self.ell_sets = self._get_ell_sets()
        self.populate_all_kellm_spaces()
        self.populate_all_proj_dicts()
        self.proj_dict = self.group.get_full_proj_dict(qcis=self)
        self.populate_all_nonint_data()
        self.populate_nonint_proj_dict()

    def populate_nonint_proj_dict(self):
        """Populate the non-interacting projection dictionary."""
        nonint_proj_dict = []
        for nic_index in range(len(self.fcs.ni_list)):
            if self.fcs.ni_list[nic_index].n_particles == 2:
                isospin_channel = self.fcs.ni_list[nic_index].isospin_channel
                nonint_proj_dict\
                    .append(self.group.get_noninttwo_proj_dict(
                        qcis=self, nic_index=nic_index,
                        isospin_channel=isospin_channel))
            elif self.fcs.ni_list[nic_index].n_particles == 3:
                nonint_proj_dict.append(
                    self.group.get_nonint_proj_dict(qcis=self,
                                                    nic_index=nic_index))
            else:
                raise ValueError("only two and three particles supported "
                                 + "by nonint_proj_dict")
        self.nonint_proj_dict = nonint_proj_dict

    def _get_grid_nonzero_nP(self, Emax, Lmax):
        deltaL = DELTA_L_FOR_GRID
        deltaE = DELTA_E_FOR_GRID
        Lmin = np.mod(Lmax-L_GRID_SHIFT, deltaL)+L_GRID_SHIFT
        Emin = np.mod(Emax-E_GRID_SHIFT, deltaE)+E_GRID_SHIFT
        Lvals = np.arange(Lmin, Lmax, deltaL)
        Evals = np.arange(Emin, Emax, deltaE)
        if np.abs(Lvals[-1] - Lmax) > EPSILON20:
            Lvals = np.append(Lvals, Lmax)
        if np.abs(Evals[-1] - Emax) > EPSILON20:
            Evals = np.append(Evals, Emax)
        Lvals = Lvals[::-1]
        Evals = Evals[::-1]
        return [Evals, Lvals]

    @property
    def nP(self):
        """Total momentum in the finite-volume frame."""
        return self._nP

    @nP.setter
    def nP(self, nP):
        """Set the total momentum in the finite-volume frame."""
        if not isinstance(nP, np.ndarray):
            raise ValueError("nP must be a numpy array")
        elif not np.array(nP).shape == (3,):
            raise ValueError("nP must have shape (3,)")
        elif not ((isinstance(nP[0], np.int64))
                  and (isinstance(nP[1], np.int64))
                  and (isinstance(nP[2], np.int64))):
            raise ValueError("nP must be populated with ints")
        else:
            self._nP = nP
            self.nPSQ = (nP)@(nP)
            self.nPmag = np.sqrt(self.nPSQ)

    @property
    def fcs(self):
        """FlavorChannelSpace object."""
        return self._fcs

    @fcs.setter
    def fcs(self, fcs):
        """Set the FlavorChannelSpace object."""
        self.n_channels = len(fcs.sc_list_sorted)
        n_two_channels = 0
        n_three_channels = 0
        for sc in fcs.sc_list_sorted:
            if np.sum(sc.masses_indexed) > self.Emax:
                raise ValueError("QCIndexSpace includes channel with "
                                 + "threshold exceeding Emax")
            if sc.fc.n_particles == 2:
                n_two_channels += 1
            elif sc.fc.n_particles == 3:
                n_three_channels += 1
            else:
                raise ValueError("QCIndexSpace currently only supports "
                                 + "two- and three-particle channels")
        self.n_two_channels = n_two_channels
        self.n_three_channels = n_three_channels

        tbks_list = []
        if self.n_two_channels > 0:
            tbks_list.append(ThreeBodyKinematicSpace(nP=self.nP))
        for _ in range(self.fcs.n_three_slices):
            tbks_list.append(ThreeBodyKinematicSpace(nP=self.nP))
        self.tbks_list = tbks_list
        self._fcs = fcs

    def _get_nPspecmax(self, three_slice_index):
        sc = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]
        m_spec = sc.masses_indexed[0]
        Emax = self.Emax
        EmaxSQ = Emax**2
        nPSQ = self.nPSQ
        Lmax = self.Lmax
        EminSQ = self.tbis.Emin**2
        if (EminSQ != 0.0):
            if nPSQ == 0:
                nPspecmax = (Lmax*np.sqrt(
                    Emax**4+(EminSQ-m_spec**2)**2-2.*Emax**2*(EminSQ+m_spec**2)
                    ))/(2.*Emax*TWOPI)
                return nPspecmax
            else:
                raise ValueError("simultaneous nonzero nP and Emin not"
                                 + " supported")
        else:
            if nPSQ == 0:
                nPspecmax = Lmax*(EmaxSQ-m_spec**2)/(2.0*TWOPI*Emax)
                return nPspecmax
            else:
                nPmag = np.sqrt(nPSQ)
                nPspecmax = (FOURPI2*nPmag*(
                    Lmax**2*(EmaxSQ+m_spec**2)-FOURPI2*nPSQ
                    )+np.sqrt(EmaxSQ*FOURPI2*Lmax**2*(
                        Lmax**2*(-EmaxSQ+m_spec**2)+FOURPI2*nPSQ
                        )**2))/(2.*FOURPI2*(EmaxSQ*Lmax**2-FOURPI2*nPSQ))
                return nPspecmax

    def populate_nvec_arr_slot(self, slot_index, three_particle_channel=True):
        """Populate a given nvec_arr slot."""
        if three_particle_channel:
            if self.n_two_channels > 0:
                three_slice_index = slot_index-1
            else:
                three_slice_index = slot_index
            if self.nPSQ == 0:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                if self.verbosity >= 2:
                    print(f"populating nvec array, three_slice_index = "
                          f"{three_slice_index}")
                self._populate_slot_zero_momentum(slot_index, nPspecmax)
            else:
                nPspecmax = self._get_nPspecmax(three_slice_index)
                self._populate_slot_nonzero_momentum(slot_index,
                                                     three_slice_index,
                                                     nPspecmax)
        else:
            nPspecmax = EPSILON4
            self._populate_slot_zero_momentum(slot_index, nPspecmax)

    def _populate_slot_nonzero_momentum(self, slot_index, three_slice_index,
                                        nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is a list, "
                      "taking first entry")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is not a list")
        rng = range(-int(nPspecmax), int(nPspecmax)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        Lmax = self.Lmax
        Emax = self.Emax
        masses = self.fcs.sc_list_sorted[
            self.fcs.slices_by_three_masses[three_slice_index][0]]\
            .masses_indexed
        m_spec = masses[0]
        nP = self.nP
        [Evals, Lvals] = self._get_grid_nonzero_nP(Emax, Lmax)
        for Ltmp in Lvals:
            for Etmp in Evals:
                self._populate_EL_iteration(slot_index, tbks_tmp, nvec_arr,
                                            m_spec, nP, Ltmp, Etmp)
        self.tbks_list[slot_index] = self.tbks_list[slot_index][:-1]

    def _populate_EL_iteration(self, slot_index, tbks_tmp, nvec_arr, m_spec,
                               nP, Ltmp, Etmp):
        E2CMSQ = (Etmp-np.sqrt(m_spec**2+FOURPI2/Ltmp**2
                               * ((nvec_arr**2)
                                  .sum(axis=1))))**2\
            - FOURPI2/Ltmp**2*((nP-nvec_arr)**2).sum(axis=1)
        carr = E2CMSQ < 0.0
        E2CMSQ = E2CMSQ.reshape((len(E2CMSQ), 1))
        E2nvec_arr = np.concatenate((E2CMSQ, nvec_arr), axis=1)
        E2nvec_arr = np.delete(E2nvec_arr, np.where(carr), axis=0)
        nvec_arr_tmp = ((E2nvec_arr.T)[1:]).T
        nvec_arr_tmp = nvec_arr_tmp.astype(np.int64)
        if self.verbosity >= 2:
            print(f"L = {np.round(Ltmp, 10)}, "
                  f"E = {np.round(Etmp, 10)}")
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr_tmp
        if self.verbosity >= 2:
            print(self.tbks_list[slot_index][-1])
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = self.tbks_list[slot_index]\
            + [tbks_copy]

    def _populate_slot_zero_momentum(self, slot_index, nPspecmax):
        if isinstance(self.tbks_list[slot_index], list):
            tbks_tmp = self.tbks_list[slot_index][0]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is a list, "
                      "taking first entry")
        else:
            tbks_tmp = self.tbks_list[slot_index]
            if self.verbosity >= 2:
                print("self.tbks_list[slot_index] is not a list")
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] = [tbks_copy]
        nPspec = nPspecmax
        if self.verbosity >= 2:
            print("populating up to nPspecmax = "+str(nPspecmax))
        while nPspec > 0:
            nPspec = self._populate_nP_iteration(slot_index, tbks_tmp, nPspec)
        self.tbks_list[slot_index] = self.tbks_list[slot_index][:-1]

    def _populate_nP_iteration(self, slot_index, tbks_tmp, nPspec):
        if self.verbosity >= 2:
            print("nPspec**2 = ", int(nPspec**2))
        rng = range(-int(nPspec), int(nPspec)+1)
        mesh = np.meshgrid(*([rng]*3))
        nvec_arr = np.vstack([y.flat for y in mesh]).T
        carr = (nvec_arr*nvec_arr).sum(1) > nPspec**2
        nvec_arr = np.delete(nvec_arr, np.where(carr), axis=0)
        self.tbks_list[slot_index][-1].nvec_arr = nvec_arr
        if self.verbosity >= 2:
            print(self.tbks_list[slot_index][-1])
        tbks_copy = deepcopy(tbks_tmp)
        tbks_copy.verbosity = self.verbosity
        self.tbks_list[slot_index] =\
            self.tbks_list[slot_index]+[tbks_copy]
        nPspecSQ = nPspec**2-1.0
        if nPspecSQ >= 0.0:
            nPspec = np.sqrt(nPspecSQ)
        else:
            nPspec = -1.0
        return nPspec

    def populate_all_nvec_arr(self):
        """Populate all nvec_arr slots."""
        if self.n_two_channels > 0:
            slot_index = 0
            self.populate_nvec_arr_slot(slot_index,
                                        three_particle_channel=False)
        for three_slice_index in range(self.fcs.n_three_slices):
            if self.n_two_channels > 0:
                slot_index = three_slice_index+1
            else:
                slot_index = three_slice_index
            self.populate_nvec_arr_slot(slot_index)

    def _get_ell_sets(self):
        ell_sets = [[]]
        for sc_index in range(self.n_channels):
            ell_set = self.fcs.sc_list_sorted[sc_index].ell_set
            ell_sets = ell_sets+[ell_set]
        return ell_sets[1:]

    @property
    def ell_sets(self):
        """Angular-momentum value sets."""
        return self._ell_sets

    @ell_sets.setter
    def ell_sets(self, ell_sets):
        """Set angular-momentum value sets."""
        self._ell_sets = ell_sets
        ellm_sets = []
        for ell_set in ell_sets:
            ellm_set = []
            for ell in ell_set:
                for mazi in range(-ell, ell+1):
                    ellm_set.append((ell, mazi))
            ellm_sets.append(ellm_set)
        self.ellm_sets = ellm_sets

    def _get_three_slice_index(self, sc_index):
        three_channel_max = self.fcs.slices_by_three_masses[-1][-1]-1
        if ((self.n_two_channels == 0) and (sc_index > three_channel_max)):
            raise ValueError(f"using cindex = {sc_index} with three_slices = "
                             f"{self.fcs.slices_by_three_masses} "
                             f"and (no two-particle channels) "
                             f"is not allowed")
        slice_index = 0
        for three_slice in self.fcs.slices_by_three_masses:
            if sc_index > three_slice[1]:
                slice_index = slice_index+1
        if self.n_two_channels > 0:
            slice_index = slice_index+1
        return slice_index

    def populate_all_kellm_spaces(self):
        """Populate all kellm spaces."""
        if self.verbosity >= 2:
            print("populating kellm spaces")
            print(self.n_channels, "channels to populate")
        kellm_shells = [[]]
        kellm_spaces = [[]]
        for cindex in range(self.n_channels):
            if cindex < self.n_two_channels:
                slot_index = 0
            else:
                cindex_shift = cindex-self.n_two_channels
                slot_index = -1
                for k in range(len(self.fcs.slices_by_three_masses)):
                    three_slice = self.fcs.slices_by_three_masses[k]
                    if three_slice[0] <= cindex_shift < three_slice[1]:
                        slot_index = k
                if self.n_two_channels > 0:
                    slot_index = slot_index+1
            tbks_list_tmp = self.tbks_list[slot_index]
            ellm_set = self.ellm_sets[cindex]
            kellm_shells_single = [[]]
            kellm_spaces_single = [[]]
            for tbks_tmp in tbks_list_tmp:
                nvec_arr = tbks_tmp.nvec_arr
                kellm_shell = (len(ellm_set)*np.array(tbks_tmp.shells
                                                      )).tolist()
                kellm_shells_single = kellm_shells_single+[kellm_shell]
                ellm_set_extended = np.tile(ellm_set, (len(nvec_arr), 1))
                nvec_arr_extended = np.repeat(nvec_arr, len(ellm_set),
                                              axis=0)
                kellm_space = np.concatenate((nvec_arr_extended,
                                              ellm_set_extended),
                                             axis=1)
                kellm_spaces_single = kellm_spaces_single+[kellm_space]
            kellm_shells = kellm_shells+[kellm_shells_single[1:]]
            kellm_spaces = kellm_spaces+[kellm_spaces_single[1:]]
        self.kellm_spaces = kellm_spaces[1:]
        self.kellm_shells = kellm_shells[1:]
        if self.verbosity >= 2:
            print("result for kellm spaces")
            print("location: channel index, nvec-space index")
            np.set_printoptions(threshold=20)
            for i in range(len(self.kellm_spaces)):
                for j in range(len(self.kellm_spaces[i])):
                    print('location:', i, j)
                    print(self.kellm_spaces[i][j])
            np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)

    def populate_all_proj_dicts(self):
        """Populate all projector dictionaries."""
        group = self.group
        sc_proj_dicts = []
        sc_proj_dicts_by_shell = [[]]
        if self.verbosity >= 2:
            print("getting the dict for following qcis:")
            print(self)
        for sc_index in range(self.n_channels):
            proj_dict = group.get_channel_proj_dict(qcis=self,
                                                    sc_index=sc_index)
            sc_proj_dicts = sc_proj_dicts+[proj_dict]
            sc_proj_dict_channel_by_shell = [[]]
            for kellm_shell_index in range(len(self.kellm_shells[sc_index])):
                kellm_shell_set = self.kellm_shells[sc_index][
                    kellm_shell_index]
                sc_proj_dict_shell_set = []
                for kellm_shell in kellm_shell_set:
                    sc_proj_dict_shell_set = sc_proj_dict_shell_set\
                        + [group.get_shell_proj_dict(
                            qcis=self,
                            cindex=sc_index,
                            kellm_shell=kellm_shell,
                            shell_index=kellm_shell_index)]
                sc_proj_dict_channel_by_shell = sc_proj_dict_channel_by_shell\
                    + [sc_proj_dict_shell_set]
            sc_proj_dict_channel_by_shell = sc_proj_dict_channel_by_shell[1:]
            sc_proj_dicts_by_shell = sc_proj_dicts_by_shell\
                + [sc_proj_dict_channel_by_shell]
        self.sc_proj_dicts = sc_proj_dicts
        self.sc_proj_dicts_by_shell = sc_proj_dicts_by_shell[1:]

    def get_tbks_sub_indices(self, E, L):
        """Get the indices of the relevant three-body kinematics spaces."""
        if E > self.Emax:
            raise ValueError("get_tbks_sub_indices called with E > Emax")
        if L > self.Lmax:
            raise ValueError("get_tbks_sub_indices called with L > Lmax")
        if self.nPSQ != 0:
            tbks_sub_indices =\
                self._get_tbks_sub_indices_nonzero_mom(E, L)
            return tbks_sub_indices
        tbks_sub_indices = self._get_tbks_sub_indices_zero_mom(E, L)
        return tbks_sub_indices

    def _get_tbks_sub_indices_zero_mom(self, E, L):
        tbks_sub_indices = [0]*len(self.tbks_list)
        for slice_index in range(self.fcs.n_three_slices):
            sc_index = self.fcs.slices_by_three_masses[slice_index][0]
            nPspecmax = self._get_nPspecmax(sc_index)
            sc = self.fcs.sc_list_sorted[sc_index]
            m_spec = sc.fc.masses[sc.indexing[0]]
            ESQ = E**2
            nPSQ = self.nPSQ
            EminSQ = self.tbis.Emin**2
            if (EminSQ != 0.0):
                if nPSQ == 0:
                    nPspecnew = (L*np.sqrt(
                        E**4+(EminSQ-m_spec**2)**2-2.*E**2*(EminSQ+m_spec**2)
                        ))/(2.*E*TWOPI)
                else:
                    raise ValueError("nonzero nP and Emin not supported")
            else:
                if nPSQ == 0:
                    nPspecnew = L*(ESQ-m_spec**2)/(2.0*TWOPI*E)
                else:
                    nPmag = np.sqrt(nPSQ)
                    nPspecnew = (FOURPI2*nPmag*(
                        L**2*(ESQ+m_spec**2)-FOURPI2*nPSQ
                        )+np.sqrt(ESQ*FOURPI2*L**2*(
                            L**2*(-ESQ+m_spec**2)+FOURPI2*nPSQ
                            )**2))/(2.*FOURPI2*(ESQ*L**2-FOURPI2*nPSQ))
            nPmaxintSQ = int(nPspecmax**2)
            nPnewintSQ = int(nPspecnew**2)
            tbks_sub_indices[sc_index] = nPmaxintSQ - nPnewintSQ
        return tbks_sub_indices

    def _get_tbks_sub_indices_nonzero_mom(self, E, L, tbks_sub_indices):
        tbks_sub_indices = [0]*len(self.tbks_list)
        for slice_index in range(self.fcs.n_three_slices):
            sc_index = self.fcs.slices_by_three_masses[slice_index][0]
            sc = self.fcs.sc_list_sorted[sc_index]
            m_spec = sc.masses_indexed[0]
            nP = self.nP
            tbkstmp_set = self.tbks_list[sc_index]
            still_searching = True
            i = 0
            while still_searching:
                tbkstmp = tbkstmp_set[i]
                nvec_arr = tbkstmp.nvec_arr
                E2CMSQfull = (E-np.sqrt(m_spec**2
                                        + FOURPI2/L**2
                                        * ((nvec_arr**2).sum(axis=1))))**2\
                    - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
                still_searching = not (np.sort(E2CMSQfull) > 0.0).all()
                i += 1
            i -= 1
            tbkstmp = tbkstmp_set[i]
            nvec_arr = tbkstmp.nvec_arr
            E2CMSQfull = (E-np.sqrt(m_spec**2
                                    + FOURPI2/L**2
                                    * ((nvec_arr**2).sum(axis=1))))**2\
                - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
            tbks_sub_indices[sc_index] = i
        warnings.warn(f"\n{bcolors.WARNING}"
                      f"get_tbks_sub_indices is being called with "
                      f"non_zero nP; this can lead to shells being "
                      f"missed! result is = {str(tbks_sub_indices)}"
                      f"{bcolors.ENDC}", stacklevel=1)
        return tbks_sub_indices

    def _load_ni_data_three(self, fc):
        [m1, m2, m3] = fc.masses
        Emax = self.Emax
        nP = self.nP
        Lmax = self.Lmax
        pSQ = 0.
        for m in [m1, m2, m3]:
            pSQ_tmp = ((Emax-m)**2-(nP@nP)*(TWOPI/Lmax)**2)/4.-m**2
            if pSQ_tmp > pSQ:
                pSQ = pSQ_tmp
                omp = np.sqrt(pSQ+m**2)
        beta = np.sqrt(nP@nP)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = int(p_cutoff*Lmax/TWOPI)+1
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return [m1, m2, m3, Emax, nP, Lmax, nvec_cutoff, nvecs]

    def _get_nvecset_arr_three(self, nvecset_arr, nmin, nmax,
                               m1, m2, m3, Emax, nP, Lmax, n1, n2):
        n3 = nP-n1-n2
        n1SQ = n1@n1
        n2SQ = n2@n2
        n3SQ = n3@n3
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2)), *(list(n3))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_arr = nvecset_arr+[[n1, n2, n3]]
        return [nvecset_arr, nmin, nmax]

    def _square_and_sort_three(self, nvecset_arr, nmin, nmax,
                               m1, m2, m3, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            n1 = nvecset_arr[i][0]
            n2 = nvecset_arr[i][1]
            n3 = nvecset_arr[i][2]
            n1SQ = n1@n1
            n2SQ = n2@n2
            n3SQ = n3@n3
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m3**2+n3SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            n3_as_num = (n3[2]-nmin)\
                + (n3[1]-nmin)*numsys+(n3[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num,
                                                    n3_as_num]]
            nvecset_SQs = nvecset_SQs+[[n1SQ, n2SQ, n3SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_SQs = np.array(nvecset_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(4):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 3-i].argsort(kind='mergesort')]
        nvecset_arr = nvecset_arr[re_indexing]
        nvecset_SQs = nvecset_SQs[re_indexing]
        return [nvecset_arr, nvecset_SQs]

    def _get_nvecset_ident_three(self, nvecset_arr, nvecset_SQs):
        nvecset_ident = []
        nvecset_ident_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            [n1, n2, n3] = nvecset_arr[i]
            candidates = [np.array([n1, n2, n3]),
                          np.array([n2, n3, n1]),
                          np.array([n3, n1, n2]),
                          np.array([n3, n2, n1]),
                          np.array([n2, n1, n3]),
                          np.array([n1, n3, n2])]

            include_entry = True

            for candidate in candidates:
                for nvecset_tmp_entry in nvecset_ident:
                    nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                    include_entry = include_entry\
                        and (not ((candidate == nvecset_tmp_entry)
                                  .all()))
            if include_entry:
                nvecset_ident = nvecset_ident+[[n1, n2, n3]]
                nvecset_ident_SQs = nvecset_ident_SQs+[nvecset_SQs[i]]
        nvecset_ident = np.array(nvecset_ident)
        nvecset_ident_SQs = np.array(nvecset_ident_SQs)
        return [nvecset_ident, nvecset_ident_SQs]

    def _reps_and_batches_three(self, nvecset_arr, nvecset_SQs,
                                nvecset_ident, nvecset_ident_SQs,
                                nP):
        nvecset_reps = [nvecset_arr[0]]
        nvecset_ident_reps = deepcopy([nvecset_ident[0]])
        nvecset_SQreps = [nvecset_SQs[0]]
        nvecset_ident_SQreps = deepcopy([nvecset_ident_SQs[0]])
        nvecset_inds = [0]
        nvecset_ident_inds = deepcopy([0])
        nvecset_counts = deepcopy([0])
        nvecset_ident_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_arr)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_reps)):
                        n_included = nvecset_reps[k]
                        if (nvecset_arr[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_counts[k] = nvecset_counts[k]+1
            if not already_included:
                nvecset_reps = nvecset_reps+[nvecset_arr[j]]
                nvecset_SQreps = nvecset_SQreps+[nvecset_SQs[j]]
                nvecset_inds = nvecset_inds+[j]
                nvecset_counts = nvecset_counts+[1]

        for j in range(len(nvecset_ident)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_ident_reps)):
                        n_included = nvecset_ident_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2, n3] = nvecset_ident[j]@g_elem
                        candidates = [np.array([n1, n2, n3]),
                                      np.array([n2, n3, n1]),
                                      np.array([n3, n1, n2]),
                                      np.array([n3, n2, n1]),
                                      np.array([n2, n1, n3]),
                                      np.array([n1, n3, n2])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_ident_counts[k]\
                                = nvecset_ident_counts[k]+1
            if not already_included:
                nvecset_ident_reps = nvecset_ident_reps\
                    + [nvecset_ident[j]]
                nvecset_ident_SQreps = nvecset_ident_SQreps\
                    + [nvecset_ident_SQs[j]]
                nvecset_ident_inds = nvecset_ident_inds+[j]
                nvecset_ident_counts = nvecset_ident_counts+[1]

        nvecset_batched = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_arr)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_arr[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_batched[k], np.int64):
                        nvecset_batched[k] = [nvecset_arr[j]]
                    else:
                        nvecset_batched[k] = nvecset_batched[k]\
                            + [nvecset_arr[j]]

        nvecset_ident_batched\
            = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_ident)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2, n3] = nvecset_ident[j]@g_elem
                    candidates = [np.array([n1, n2, n3]),
                                  np.array([n2, n3, n1]),
                                  np.array([n3, n1, n2]),
                                  np.array([n3, n2, n1]),
                                  np.array([n2, n1, n3]),
                                  np.array([n1, n3, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_ident_batched[k], np.int64):
                        nvecset_ident_batched[k] = [nvecset_ident[j]]
                    else:
                        nvecset_ident_batched[k]\
                            = nvecset_ident_batched[k]\
                            + [nvecset_ident[j]]

        for j in range(len(nvecset_batched)):
            nvecset_batched[j] = np.array(nvecset_batched[j])

        for j in range(len(nvecset_ident_batched)):
            nvecset_ident_batched[j]\
                = np.array(nvecset_ident_batched[j])
        return [nvecset_reps, nvecset_ident_reps,
                nvecset_SQreps, nvecset_ident_SQreps,
                nvecset_inds, nvecset_ident_inds,
                nvecset_counts, nvecset_ident_counts,
                nvecset_batched, nvecset_ident_batched]

    def _load_ni_data_two(self, fc):
        Emax = self.Emax
        nP = self.nP
        nPSQ = nP@nP
        Lmax = self.Lmax

        [m1, m2] = fc.masses
        ECMSQ = Emax**2-FOURPI2*nPSQ/Lmax**2
        pSQ = (ECMSQ**2-2.0*ECMSQ*m1**2
               + m1**4-2.0*ECMSQ*m2**2-2.0*m1**2*m2**2+m2**4)\
            / (4.0*ECMSQ)
        mmax = np.max([m1, m2])
        omp = np.sqrt(pSQ+mmax**2)
        beta = np.sqrt(nPSQ)*TWOPI/Lmax/Emax
        gamma = 1./np.sqrt(1.-beta**2)
        p_cutoff = beta*gamma*omp+gamma*np.sqrt(pSQ)
        nvec_cutoff = int(p_cutoff*Lmax/TWOPI)
        rng = range(-nvec_cutoff, nvec_cutoff+1)
        mesh = np.meshgrid(*([rng]*3))
        nvecs = np.vstack([y.flat for y in mesh]).T
        return [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]

    def _get_nvecset_arr_two(self, nvecset_arr, nmin, nmax, m1, m2,
                             Emax, nP, Lmax, n1):
        n2 = nP-n1
        n1SQ = n1@n1
        n2SQ = n2@n2
        E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
            + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
        if E <= Emax:
            comp_set = [*(list(n1)), *(list(n2))]
            min_candidate = np.min(comp_set)
            if min_candidate < nmin:
                nmin = min_candidate
            max_candidate = np.max(comp_set)
            if max_candidate > nmax:
                nmax = max_candidate
            nvecset_arr = nvecset_arr+[[n1, n2]]
        return [nvecset_arr, nmin, nmax]

    def _square_and_sort_two(self, nvecset_arr, nmin, nmax,
                             m1, m2, Lmax):
        numsys = nmax-nmin+1
        E_nvecset_compact = []
        nvecset_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            n1 = nvecset_arr[i][0]
            n2 = nvecset_arr[i][1]
            n1SQ = n1@n1
            n2SQ = n2@n2
            E = np.sqrt(m1**2+n1SQ*(TWOPI/Lmax)**2)\
                + np.sqrt(m2**2+n2SQ*(TWOPI/Lmax)**2)
            n1_as_num = (n1[2]-nmin)\
                + (n1[1]-nmin)*numsys+(n1[0]-nmin)*numsys**2
            n2_as_num = (n2[2]-nmin)\
                + (n2[1]-nmin)*numsys+(n2[0]-nmin)*numsys**2
            E_nvecset_compact = E_nvecset_compact+[[E, n1_as_num,
                                                    n2_as_num]]
            nvecset_SQs = nvecset_SQs+[[n1SQ, n2SQ]]
        E_nvecset_compact = np.array(E_nvecset_compact)
        nvecset_SQs = np.array(nvecset_SQs)

        re_indexing = np.arange(len(E_nvecset_compact))
        for i in range(3):
            re_indexing = re_indexing[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
            E_nvecset_compact = E_nvecset_compact[
                E_nvecset_compact[:, 2-i].argsort(kind='mergesort')]
        nvecset_arr = nvecset_arr[re_indexing]
        nvecset_SQs = nvecset_SQs[re_indexing]
        return [nvecset_arr, nvecset_SQs]

    def _get_nvecset_ident_two(self, nvecset_arr, nvecset_SQs):
        nvecset_ident = []
        nvecset_ident_SQs = deepcopy([])
        for i in range(len(nvecset_arr)):
            [n1, n2] = nvecset_arr[i]
            candidates = [np.array([n1, n2]),
                          np.array([n2, n1])]
            include_entry = True
            for candidate in candidates:
                for nvecset_tmp_entry in nvecset_ident:
                    nvecset_tmp_entry = np.array(nvecset_tmp_entry)
                    include_entry = include_entry\
                        and (not ((candidate == nvecset_tmp_entry)
                                  .all()))
            if include_entry:
                nvecset_ident = nvecset_ident+[[n1, n2]]
                nvecset_ident_SQs = nvecset_ident_SQs+[nvecset_SQs[i]]
        nvecset_ident = np.array(nvecset_ident)
        nvecset_ident_SQs = np.array(nvecset_ident_SQs)
        return [nvecset_ident, nvecset_ident_SQs]

    def _reps_and_batches_two(self, nvecset_arr, nvecset_SQs, nvecset_ident,
                              nvecset_ident_SQs, nP):
        nvecset_reps = [nvecset_arr[0]]
        nvecset_ident_reps = deepcopy([nvecset_ident[0]])
        nvecset_SQreps = [nvecset_SQs[0]]
        nvecset_ident_SQreps = deepcopy([nvecset_ident_SQs[0]])
        nvecset_inds = [0]
        nvecset_ident_inds = deepcopy([0])
        nvecset_counts = deepcopy([0])
        nvecset_ident_counts = deepcopy([0])

        G = self.group.get_little_group(nP)
        for j in range(len(nvecset_arr)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_reps)):
                        n_included = nvecset_reps[k]
                        if (nvecset_arr[j]@g_elem == n_included).all():
                            already_included = True
                            nvecset_counts[k] = nvecset_counts[k]+1
            if not already_included:
                nvecset_reps = nvecset_reps+[nvecset_arr[j]]
                nvecset_SQreps = nvecset_SQreps+[nvecset_SQs[j]]
                nvecset_inds = nvecset_inds+[j]
                nvecset_counts = nvecset_counts+[1]

        for j in range(len(nvecset_ident)):
            already_included = False
            for g_elem in G:
                if not already_included:
                    for k in range(len(nvecset_ident_reps)):
                        n_included = nvecset_ident_reps[k]
                        n_included = np.array(n_included)
                        [n1, n2] = nvecset_ident[j]@g_elem
                        candidates = [np.array([n1, n2]),
                                      np.array([n2, n1])]
                        include_entry = True
                        for candidate in candidates:
                            include_entry = include_entry\
                                and (not ((candidate == n_included)
                                          .all()))
                        if not include_entry:
                            already_included = True
                            nvecset_ident_counts[k]\
                                = nvecset_ident_counts[k]+1
            if not already_included:
                nvecset_ident_reps = nvecset_ident_reps\
                    + [nvecset_ident[j]]
                nvecset_ident_SQreps = nvecset_ident_SQreps\
                    + [nvecset_ident_SQs[j]]
                nvecset_ident_inds = nvecset_ident_inds+[j]
                nvecset_ident_counts = nvecset_ident_counts+[1]

        nvecset_batched = list(np.arange(len(nvecset_reps)))
        for j in range(len(nvecset_arr)):
            for k in range(len(nvecset_reps)):
                include_entry = False
                n_rep = nvecset_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_arr[j]@g_elem
                    candidates = [np.array([n1, n2])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_batched[k], np.int64):
                        nvecset_batched[k] = [nvecset_arr[j]]
                    else:
                        nvecset_batched[k] = nvecset_batched[k]\
                            + [nvecset_arr[j]]

        nvecset_ident_batched\
            = list(np.arange(len(nvecset_ident_reps)))
        for j in range(len(nvecset_ident)):
            for k in range(len(nvecset_ident_reps)):
                include_entry = False
                n_rep = nvecset_ident_reps[k]
                n_rep = np.array(n_rep)
                for g_elem in G:
                    [n1, n2] = nvecset_ident[j]@g_elem
                    candidates = [np.array([n1, n2]),
                                  np.array([n2, n1])]
                    for candidate in candidates:
                        include_entry = include_entry\
                            or (((candidate == n_rep).all()))
                if include_entry:
                    if isinstance(nvecset_ident_batched[k], np.int64):
                        nvecset_ident_batched[k] = [nvecset_ident[j]]
                    else:
                        nvecset_ident_batched[k]\
                            = nvecset_ident_batched[k]\
                            + [nvecset_ident[j]]

        for j in range(len(nvecset_batched)):
            nvecset_batched[j] = np.array(nvecset_batched[j])

        for j in range(len(nvecset_ident_batched)):
            nvecset_ident_batched[j]\
                = np.array(nvecset_ident_batched[j])
        return [nvecset_reps, nvecset_ident_reps,
                nvecset_SQreps, nvecset_ident_SQreps,
                nvecset_inds, nvecset_ident_inds,
                nvecset_counts, nvecset_ident_counts,
                nvecset_batched, nvecset_ident_batched]

    def populate_all_nonint_data(self):
        """Populate all non-interacting data."""
        nvecset_arr_all = []
        nvecset_SQs_all = []
        nvecset_reps_all = []
        nvecset_SQreps_all = []
        nvecset_inds_all = []
        nvecset_counts_all = []
        nvecset_batched_all = []
        nvecset_ident_all = []
        nvecset_ident_SQs_all = []
        nvecset_ident_reps_all = []
        nvecset_ident_SQreps_all = []
        nvecset_ident_inds_all = []
        nvecset_ident_counts_all = []
        nvecset_ident_batched_all = []
        ni_list = self.fcs.ni_list
        for fc in ni_list:
            if fc.n_particles == 3:
                [m1, m2, m3, Emax, nP, Lmax, nvec_cutoff, nvecs]\
                    = self._load_ni_data_three(fc)
                nvecset_arr = []
                nmin = nvec_cutoff
                nmax = nvec_cutoff
                for n1 in nvecs:
                    for n2 in nvecs:
                        [nvecset_arr, nmin, nmax]\
                            = self._get_nvecset_arr_three(nvecset_arr,
                                                          nmin, nmax,
                                                          m1, m2, m3,
                                                          Emax, nP, Lmax,
                                                          n1, n2)
                nvecset_arr = np.array(nvecset_arr)
                [nvecset_arr, nvecset_SQs]\
                    = self._square_and_sort_three(nvecset_arr, nmin, nmax,
                                                  m1, m2, m3, Lmax)
                [nvecset_ident, nvecset_ident_SQs]\
                    = self._get_nvecset_ident_three(nvecset_arr, nvecset_SQs)
                [nvecset_reps, nvecset_ident_reps,
                 nvecset_SQreps, nvecset_ident_SQreps,
                 nvecset_inds, nvecset_ident_inds,
                 nvecset_counts, nvecset_ident_counts,
                 nvecset_batched, nvecset_ident_batched]\
                    = self._reps_and_batches_three(nvecset_arr, nvecset_SQs,
                                                   nvecset_ident,
                                                   nvecset_ident_SQs,
                                                   nP)
            else:
                [m1, m2, Emax, nP, Lmax, nvec_cutoff, nvecs]\
                    = self._load_ni_data_two(fc)
                nvecset_arr = []
                nmin = nvec_cutoff
                nmax = nvec_cutoff
                for n1 in nvecs:
                    [nvecset_arr, nmin, nmax]\
                        = self._get_nvecset_arr_two(nvecset_arr, nmin, nmax,
                                                    m1, m2, Emax, nP, Lmax, n1)
                nvecset_arr = np.array(nvecset_arr)
                [nvecset_arr, nvecset_SQs]\
                    = self._square_and_sort_two(nvecset_arr, nmin, nmax,
                                                m1, m2, Lmax)
                [nvecset_ident, nvecset_ident_SQs]\
                    = self._get_nvecset_ident_two(nvecset_arr, nvecset_SQs)
                [nvecset_reps, nvecset_ident_reps,
                 nvecset_SQreps, nvecset_ident_SQreps,
                 nvecset_inds, nvecset_ident_inds,
                 nvecset_counts, nvecset_ident_counts,
                 nvecset_batched, nvecset_ident_batched]\
                    = self._reps_and_batches_two(nvecset_arr, nvecset_SQs,
                                                 nvecset_ident,
                                                 nvecset_ident_SQs, nP)

            nvecset_arr_all = nvecset_arr_all+[nvecset_arr]
            nvecset_SQs_all = nvecset_SQs_all+[nvecset_SQs]
            nvecset_reps_all = nvecset_reps_all+[nvecset_reps]
            nvecset_SQreps_all = nvecset_SQreps_all+[nvecset_SQreps]
            nvecset_inds_all = nvecset_inds_all+[nvecset_inds]
            nvecset_counts_all = nvecset_counts_all+[nvecset_counts]
            nvecset_batched_all = nvecset_batched_all+[nvecset_batched]

            nvecset_ident_all = nvecset_ident_all\
                + [nvecset_ident]
            nvecset_ident_SQs_all = nvecset_ident_SQs_all+[nvecset_ident_SQs]
            nvecset_ident_reps_all = nvecset_ident_reps_all\
                + [nvecset_ident_reps]
            nvecset_ident_SQreps_all = nvecset_ident_SQreps_all\
                + [nvecset_ident_SQreps]
            nvecset_ident_inds_all = nvecset_ident_inds_all\
                + [nvecset_ident_inds]
            nvecset_ident_counts_all = nvecset_ident_counts_all\
                + [nvecset_ident_counts]
            nvecset_ident_batched_all = nvecset_ident_batched_all\
                + [nvecset_ident_batched]
        self.nvecset_arr = nvecset_arr_all
        self.nvecset_SQs = nvecset_SQs_all
        self.nvecset_reps = nvecset_reps_all
        self.nvecset_SQreps = nvecset_SQreps_all
        self.nvecset_inds = nvecset_inds_all
        self.nvecset_counts = nvecset_counts_all
        self.nvecset_batched = nvecset_batched_all

        self.nvecset_ident = nvecset_ident_all
        self.nvecset_ident_SQs = nvecset_ident_SQs_all
        self.nvecset_ident_reps = nvecset_ident_reps_all
        self.nvecset_ident_SQreps = nvecset_ident_SQreps_all
        self.nvecset_ident_inds = nvecset_ident_inds_all
        self.nvecset_ident_counts = nvecset_ident_counts_all
        self.nvecset_ident_batched = nvecset_ident_batched_all

    @staticmethod
    def count_by_isospin(flavor_basis):
        """Count by isospin."""
        iso_basis = CAL_C_ISO@flavor_basis

        iso_basis_normalized = []
        for entry in iso_basis:
            if entry@entry != 0.:
                entry_norm = entry/np.sqrt(entry@entry)
                iso_basis_normalized = iso_basis_normalized+[entry_norm]
            else:
                iso_basis_normalized = iso_basis_normalized+[entry]
        iso_basis_normalized = np.array(iso_basis_normalized)

        iso_basis_broken = []
        for iso_projector in ISO_PROJECTORS:
            iso_basis_broken_entry = iso_projector@iso_basis_normalized
            iso_basis_broken = iso_basis_broken+[iso_basis_broken_entry]

        iso_basis_broken_collapsed = []
        counts = [0, 0, 0, 0]
        for k in range(len(iso_basis_broken)):
            iso_basis_broken_entry = iso_basis_broken[k]
            reduced_entry = deepcopy(iso_basis_broken_entry)
            for i in range(len(reduced_entry)):
                reduced_entry_line = reduced_entry[i]
                if reduced_entry_line@reduced_entry_line != 0.:
                    for j in range(len(reduced_entry)):
                        if ((j > i) and
                           (reduced_entry_line@reduced_entry[j] != 0.)):
                            reduced_entry[j] = reduced_entry[j]\
                                - reduced_entry_line\
                                * (reduced_entry_line@reduced_entry[j])
            collapsed_entry = []
            for reduced_entry_line in reduced_entry:
                if reduced_entry_line@reduced_entry_line > EPSILON20:
                    collapsed_entry = collapsed_entry+[reduced_entry_line]
                    counts[k] = counts[k]+1
            collapsed_entry = np.array(collapsed_entry)
            iso_basis_broken_collapsed = iso_basis_broken_collapsed\
                + [collapsed_entry]
        return counts, iso_basis_broken_collapsed

    def _get_ibest(self, E, L):
        """Only for non-zero P."""
        Lvals = self.Lvals
        Evals = self.Evals
        i = 0
        ibest = 0
        for Ltmp in Lvals:
            for Etmp in Evals:
                if (Etmp > E) and (Ltmp > L):
                    ibest = i
                i = i+1
        if self.verbosity == 2:
            print('Lvals  =', Lvals)
            print('Evals  =', Evals)
            print('ibest =', ibest, '=', np.mod(ibest,
                                                len(Evals)),
                  '+',
                  int(ibest/len(Evals)), '* len(Evals)')
            print('so Lmaxtmp =',
                  np.round(Lvals[
                      int(ibest/len(Evals))], 10),
                  'and Emaxtmp =',
                  np.round(Evals[
                      np.mod(ibest, len(Evals))], 10))
        return ibest

    def default_k_params(self):
        """Get the default k-matrix parameters."""
        pcotdelta_parameter_list = [[]]
        for sc in self.fcs.sc_list_sorted:
            for n_params in sc.n_params_set:
                pcotdelta_parameter_list = pcotdelta_parameter_list\
                    + [[0.0]*n_params]
        pcotdelta_parameter_list = pcotdelta_parameter_list[1:]
        k3_params = [0.0]
        return [pcotdelta_parameter_list, k3_params]

    def __str__(self):
        """Return a string representation of the QCIndexSpace object."""
        qc_index_space_str = "QCIndexSpace containing:\n"
        qc_index_space_str += "    "\
            + str(self.fcs).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    "\
            + str(self.fvs).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    "\
            + str(self.tbis).replace("\n", "\n    ")+"\n\n"
        qc_index_space_str += "    Parameter input structure:\n"
        qc_index_space_str += "        "\
            + str(self.param_structure)+"\n\n"
        for tbkstmp in self.tbks_list:
            qc_index_space_str += "    "\
                + str(tbkstmp[0]).replace("\n", "\n    ")+"\n"
        return qc_index_space_str[:-1]


class G:
    r"""
    Class for the finite-volume G matrix (responsible for exchanges).

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    """

    def __init__(self, qcis=QCIndexSpace()):
        self.qcis = qcis
        three_scheme = self.qcis.tbis.three_scheme
        if (three_scheme == 'original pole')\
           or (three_scheme == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.all_relevant_nvecSQs = {}
        self.pole_free_interpolator_matrix = {}
        self.interpolator_matrix = {}
        self.cob_matrices = {}
        self.function_set = {}
        self.all_dimensions = {}
        self.total_cobs = {}

    def _get_masks_and_shells(self, E, nP, L, tbks_entry,
                              cindex_row, cindex_col,
                              row_shell_index, col_shell_index):
        three_slice_index_row\
            = self.qcis._get_three_slice_index(cindex_row)
        three_slice_index_col\
            = self.qcis._get_three_slice_index(cindex_col)
        if not (three_slice_index_row == three_slice_index_col == 0):
            raise ValueError("only one mass slice is supported in G")
        three_slice_index = three_slice_index_row
        if nP@nP == 0:
            mask_row_shells, mask_col_shells, row_shell, col_shell\
                = self._mask_and_shell_helper_nPzero(tbks_entry,
                                                     row_shell_index,
                                                     col_shell_index)
        else:
            mask_row_shells, mask_col_shells, row_shell, col_shell = self.\
                _mask_and_shell_helper_nPnonzero(E, nP, L, tbks_entry,
                                                 row_shell_index,
                                                 col_shell_index,
                                                 three_slice_index)
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def _mask_and_shell_helper_nPnonzero(self, E, nP, L, tbks_entry,
                                         row_shell_index, col_shell_index,
                                         three_slice_index):
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[three_slice_index][0]].\
            masses_indexed
        m_spec = masses[0]
        kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
        kvec_arr = TWOPI*tbks_entry.nvec_arr/L
        omk_arr = np.sqrt(m_spec**2+kvecSQ_arr)
        Pvec = TWOPI*nP/L
        PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
        mask_row = (E-omk_arr)**2-PmkSQ_arr > 0.0
        row_shells = tbks_entry.shells
        mask_row_shells = []
        for row_shell in row_shells:
            mask_row_shells = mask_row_shells\
                    + [mask_row[row_shell[0]:row_shell[1]].all()]
        row_shells = list(np.array(row_shells)[mask_row_shells])
        row_shell = list(row_shells[row_shell_index])

        mask_col = mask_row
        col_shells = tbks_entry.shells
        mask_col_shells = []
        for col_shell in col_shells:
            mask_col_shells = mask_col_shells\
                    + [mask_col[col_shell[0]:col_shell[1]].all()]
        col_shells = list(np.array(col_shells)[mask_col_shells])
        col_shell = list(col_shells[col_shell_index])
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def _mask_and_shell_helper_nPzero(self, tbks_entry, row_shell_index,
                                      col_shell_index):
        mask_row_shells = None
        mask_col_shells = None
        row_shell = tbks_entry.shells[row_shell_index]
        col_shell = tbks_entry.shells[col_shell_index]
        return mask_row_shells, mask_col_shells, row_shell, col_shell

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex_row=None, cindex_col=None,  # only for non-zero P
                  sc_index_row=None, sc_index_col=None,
                  ell1=0, ell2=0,
                  g_rescale=1.0, tbks_entry=None,
                  row_shell_index=None,
                  col_shell_index=None,
                  project=False, irrep=None):
        """Build the G matrix on a single shell."""
        three_scheme = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
                                         row_shell_index, col_shell_index)
        if project:
            try:
                if nP@nP != 0:
                    proj_tmp_right, proj_tmp_left = self.\
                        _nP_nonzero_projectors(E, L,
                                               sc_index_row, sc_index_col,
                                               row_shell_index,
                                               col_shell_index,
                                               irrep,
                                               mask_row_shells,
                                               mask_col_shells)
                else:
                    proj_tmp_right, proj_tmp_left = self.\
                        _nPzero_projectors(sc_index_row, sc_index_col,
                                           row_shell_index, col_shell_index,
                                           irrep)
            except KeyError:
                return np.array([])
            proj_support_right, proj_support_left, sparse = self.\
                _get_sparse(proj_tmp_right, proj_tmp_left)
        else:
            sparse = False

        if not sparse:
            g_uses_prep_mat = QC_IMPL_DEFAULTS['g_uses_prep_mat']
            if 'g_uses_prep_mat' in self.qcis.fvs.qc_impl:
                g_uses_prep_mat = self.qcis.fvs.qc_impl['g_uses_prep_mat']
            if g_uses_prep_mat and (nP@nP == 0):
                Gshell = QCFunctions.getG_array_prep_mat(E, nP, L, m1, m2, m3,
                                                         tbks_entry,
                                                         row_shell_index,
                                                         col_shell_index,
                                                         ell1, ell2,
                                                         alpha, beta,
                                                         qc_impl, three_scheme,
                                                         g_rescale)
            else:
                Gshell = QCFunctions.getG_array(E, nP, L, m1, m2, m3,
                                                tbks_entry,
                                                row_shell, col_shell,
                                                ell1, ell2,
                                                alpha, beta,
                                                qc_impl, three_scheme,
                                                g_rescale)
        else:
            Gshell = self.\
                _getG_array_sparse(E, nP, L, m1, m2, m3, ell1, ell2, g_rescale,
                                   tbks_entry, three_scheme, qc_impl,
                                   alpha, beta, row_shell, col_shell,
                                   proj_support_right, proj_support_left)
        if project:
            Gshell = proj_tmp_left@Gshell@proj_tmp_right
        return Gshell

    def _getG_array_sparse(self, E, nP, L, m1, m2, m3, ell1, ell2, g_rescale,
                           tbks_entry, three_scheme, qc_impl, alpha, beta,
                           row_shell, col_shell,
                           proj_support_right, proj_support_left):
        J_slow = False
        n1vec_arr_shell = tbks_entry.nvec_arr[row_shell[0]:row_shell[1]]
        n1vecSQ_arr_shell = tbks_entry.nvecSQ_arr[
                row_shell[0]:row_shell[1]]

        n2vec_arr_shell = tbks_entry.nvec_arr[col_shell[0]:col_shell[1]]
        n2vecSQ_arr_shell = tbks_entry.nvecSQ_arr[
                col_shell[0]:col_shell[1]]
        Gshell = np.zeros((len(n1vecSQ_arr_shell)*(2*ell1+1),
                           len(n2vecSQ_arr_shell)*(2*ell2+1)))
        for i1 in range(len(n1vecSQ_arr_shell)):
            for i2 in range(2*ell1+1):
                ifull = i1*(2*ell1+1)+i2
                np1spec = n1vec_arr_shell[i1]
                mazi1 = i2-ell1
                for j1 in range(len(n2vecSQ_arr_shell)):
                    for j2 in range(2*ell2+1):
                        jfull = j1*(2*ell2+1)+j2
                        np2spec = n2vec_arr_shell[j1]
                        mazi2 = j2-ell2
                        if ((proj_support_left[ifull] != 0.0)
                           and (proj_support_right[jfull] != 0.0)):
                            Gshell[ifull][jfull] = QCFunctions\
                                    .getG_single_entry(E, nP, L,
                                                       np1spec, np2spec,
                                                       ell1, mazi1,
                                                       ell2, mazi2,
                                                       m1, m2, m3,
                                                       alpha, beta,
                                                       J_slow,
                                                       three_scheme,
                                                       qc_impl,
                                                       g_rescale)
        return Gshell

    def _get_sparse(self, proj_tmp_right, proj_tmp_left):
        proj_support_right = np.diag(proj_tmp_right@(proj_tmp_right.T))
        zero_frac_right = float(
            np.count_nonzero(proj_support_right == 0.))\
            / float(len(proj_support_right))
        proj_support_left = np.diag((proj_tmp_left.T)@proj_tmp_left)
        zero_frac_left = float(
            np.count_nonzero(proj_support_left == 0.))\
            / float(len(proj_support_left))
        sparse = ((zero_frac_right > SPARSE_CUT)
                  and (zero_frac_left > SPARSE_CUT))

        return proj_support_right, proj_support_left, sparse

    def _nPzero_projectors(self, sc_index_row, sc_index_col,
                           row_shell_index, col_shell_index, irrep):
        proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_index_col][0][col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
                        self.qcis.sc_proj_dicts_by_shell[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
        return proj_tmp_right, proj_tmp_left

    def _nP_nonzero_projectors(self, E, L, sc_index_row, sc_index_col,
                               row_shell_index, col_shell_index, irrep,
                               mask_row_shells, mask_col_shells):
        ibest = self.qcis._get_ibest(E, L)
        proj_tmp_right = np.array(self.qcis.sc_proj_dicts_by_shell[
            sc_index_col][ibest])[mask_col_shells][
                col_shell_index][irrep]
        proj_tmp_left = np.conjugate((
            np.array(self.qcis.
                     sc_proj_dicts_by_shell[sc_index_row][ibest]
                     )[mask_row_shells][row_shell_index][irrep]).T)
        return proj_tmp_right, proj_tmp_left

    def get_shell_nvecSQs_projs(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                                cindex_row=None, cindex_col=None,
                                # only for non-zero P
                                sc_index_row=None, sc_index_col=None,
                                ell1=0, ell2=0,
                                g_rescale=1.0, tbks_entry=None,
                                row_shell_index=None,
                                col_shell_index=None,
                                project=False, irrep=None):
        """Build the G matrix on a single shell."""
        three_scheme = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta

        mask_row_shells, mask_col_shells, row_shell, col_shell\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
                                         row_shell_index, col_shell_index)
        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(self.qcis
                                              .sc_proj_dicts_by_shell[
                                                  sc_index_col][ibest]
                                              )[mask_col_shells][
                                                  col_shell_index][irrep]
                    proj_tmp_left = np.conjugate((
                        np.array(self.qcis.
                                 sc_proj_dicts_by_shell[sc_index_row][ibest]
                                 )[mask_row_shells][row_shell_index][irrep]
                    ).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_index_col][0][col_shell_index][irrep]
                    proj_tmp_left = np.conjugate((
                        self.qcis.sc_proj_dicts_by_shell[
                            sc_index_row][0][row_shell_index][irrep]
                        ).T)
            except KeyError:
                return np.array([])
            proj_support_right = np.diag(proj_tmp_right@(proj_tmp_right.T))
            zero_frac_right = float(np.count_nonzero(proj_support_right
                                                     == 0.))\
                / float(len(proj_support_right))
            proj_support_left = np.diag((proj_tmp_left.T)@proj_tmp_left)
            zero_frac_left = float(np.count_nonzero(proj_support_left
                                                    == 0.))\
                / float(len(proj_support_left))
            sparse = ((zero_frac_right > SPARSE_CUT)
                      and (zero_frac_left > SPARSE_CUT))
        else:
            sparse = False

        if not sparse:
            g_uses_prep_mat = QC_IMPL_DEFAULTS['g_uses_prep_mat']
            if 'g_uses_prep_mat' in self.qcis.fvs.qc_impl:
                g_uses_prep_mat = self.qcis.fvs.qc_impl['g_uses_prep_mat']
            if g_uses_prep_mat and (nP@nP == 0):
                nvecSQ_mat_shells = QCFunctions\
                    .getG_array_prep_mat(E, nP, L, m1, m2, m3, tbks_entry,
                                         row_shell_index, col_shell_index,
                                         ell1, ell2, alpha, beta, qc_impl,
                                         three_scheme, g_rescale)
            else:
                nvecSQ_mat_shells = QCFunctions\
                    .get_nvecSQ_mat_shells(tbks_entry, row_shell, col_shell)
        return [nvecSQ_mat_shells, proj_tmp_left, proj_tmp_right]

    def _clean_shape(self, g_collection):
        rowsizes = [0]*len(g_collection)
        colsizes = [0]*len(g_collection)
        for i in range(len(g_collection)):
            for j in range(len(g_collection)):
                shtmp = g_collection[i][j].shape
                if shtmp != (0,):
                    if shtmp[0] > rowsizes[i]:
                        rowsizes[i] = shtmp[0]
                    if shtmp[1] > colsizes[j]:
                        colsizes[j] = shtmp[1]
        for i in range(len(g_collection)):
            for j in range(len(g_collection)):
                shtmp = g_collection[i][j].shape
                if shtmp == (0,) or shtmp == (0, 0):
                    g_collection[i][j].shape = (rowsizes[i], colsizes[j])
        return g_collection

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the G matrix in a shell-based way."""
        g_interpolate = QC_IMPL_DEFAULTS['g_interpolate']
        if 'g_interpolate' in self.qcis.fvs.qc_impl:
            g_interpolate = self.qcis.fvs.qc_impl['g_interpolate']
        if g_interpolate:
            g_final_smooth_basis = [[]]
            total_cobs = self.total_cobs[irrep]
            matrix_dimension = self.\
                all_dimensions[irrep][total_cobs
                                      - self.
                                      qcis.get_tbks_sub_indices(E, L)[0]-1]
            m1, m2, m3 = self.extract_masses()
            for i in range(matrix_dimension):
                g_row_tmp = []
                for j in range(matrix_dimension):
                    func_tmp = self.function_set[irrep][i][j]
                    if func_tmp is not None:
                        value_tmp = float(func_tmp((E, L)))
                        for pole_data in (self.
                                          pole_free_interpolator_matrix[
                                              irrep][i][j][2]):
                            factor_tmp = E-self.\
                                get_pole_candidate(L, *pole_data[2],
                                                   m1, m2, m3)
                            value_tmp = value_tmp/factor_tmp
                        g_row_tmp = g_row_tmp+[value_tmp]
                    else:
                        g_row_tmp = g_row_tmp+[0.]
                g_final_smooth_basis = g_final_smooth_basis+[g_row_tmp]
            g_final_smooth_basis = np.array(g_final_smooth_basis[1:])
            cob_matrix = self.\
                cob_matrices[irrep][
                    total_cobs-self.qcis.get_tbks_sub_indices(E, L)[0]-1]
            g_matrix_tmp_rotated\
                = (cob_matrix)@g_final_smooth_basis@(cob_matrix.T)
            return g_matrix_tmp_rotated

        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating G using numpy accelerated version')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

            print(self.qcis.tbis.three_scheme, ',',
                  self.qcis.fvs.qc_impl)
            print('cutoff params:', self.alpha, ',', self.beta)

            if self.qcis.tbis.three_scheme == 'original pole':
                sf = '1./(2.*w1*w2*L**3)'
            elif self.qcis.tbis.three_scheme == 'relativistic pole':
                sf = '1./(2.*w1*L**3)\n    * 1./(E-w1-w3+w2)'
            else:
                raise ValueError("three_scheme not recognized")
            hermitian = QC_IMPL_DEFAULTS['hermitian']
            if 'hermitian' in self.qcis.fvs.qc_impl:
                hermitian = self.qcis.fvs.qc_impl['hermitian']
            if hermitian:
                sf = sf+'\n    * 1./(2.0*w3)'

            print('G = YY*H1*H2\n    * '+sf+'\n    * 1./(E-w1-w2-w3)\n')

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if self.qcis.verbosity >= 2:
            print('representatives of three_slice:')
            print('    cindex_row =', cindex_row,
                  ', cindex_col =', cindex_col)

        if (not ((irrep is None) and (project is False))
           and (not (irrep in self.qcis.proj_dict.keys()))):
            raise ValueError("irrep "+str(irrep)+" not in "
                             + "qcis.proj_dict.keys()")

        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        g_final = [[]]
        if self.qcis.verbosity >= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            g_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                if self.qcis.verbosity >= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     + "supported in G")
                ell2 = col_ell_set[0]
                g_rescale = self.qcis.fcs.g_templates[0][0][
                    sc_row_ind][sc_col_ind]

                g_inner = [[]]
                for row_shell_index in range(len(slices)):
                    g_inner_row = []
                    for col_shell_index in range(len(slices)):
                        g_tmp = self.get_shell(E, L,
                                               m1, m2, m3,
                                               cindex_row, cindex_col,
                                               # only for non-zero P
                                               sc_row_ind, sc_col_ind,
                                               ell1, ell2,
                                               g_rescale,
                                               tbks_entry,
                                               row_shell_index,
                                               col_shell_index,
                                               project, irrep)
                        g_inner_row = g_inner_row+[g_tmp]
                    g_inner = g_inner+[g_inner_row]

                g_inner = g_inner[1:]
                g_inner = self._clean_shape(g_inner)
                g_block_tmp = np.block(g_inner)

                g_outer_row = g_outer_row+[g_block_tmp]
            g_final = g_final+[g_outer_row]

        g_final = g_final[1:]
        g_final = self._clean_shape(g_final)
        g_final = np.block(g_final)
        return g_final

    def get_all_nvecSQs_by_shell(self, E=5.0, L=5.0, project=False,
                                 irrep=None):
        """Build the G matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if (not ((irrep is None) and (project is False))
           and (not (irrep in self.qcis.proj_dict.keys()))):
            raise ValueError("irrep "+str(irrep)+" not in "
                             + "qcis.proj_dict.keys()")

        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        nvecSQs_final = [[]]
        if self.qcis.verbosity >= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            nvecSQs_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                if self.qcis.verbosity >= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list_sorted[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     + "supported in G")
                ell2 = col_ell_set[0]
                g_rescale = self.qcis.fcs.g_templates[0][0][
                    sc_row_ind][sc_col_ind]

                nvecSQs_inner = [[]]
                for row_shell_index in range(len(slices)):
                    nvecSQs_inner_row = []
                    for col_shell_index in range(len(slices)):
                        nvecSQs_tmp = self\
                            .get_shell_nvecSQs_projs(E, L, m1, m2, m3,
                                                     cindex_row, cindex_col,
                                                     # only for non-zero P
                                                     sc_row_ind, sc_col_ind,
                                                     ell1, ell2, g_rescale,
                                                     tbks_entry,
                                                     row_shell_index,
                                                     col_shell_index,
                                                     project, irrep)
                        nvecSQs_inner_row = nvecSQs_inner_row+[nvecSQs_tmp]
                    nvecSQs_inner = nvecSQs_inner+[nvecSQs_inner_row]
                nvecSQs_block_tmp = nvecSQs_inner[1:]
                nvecSQs_outer_row = nvecSQs_outer_row+[nvecSQs_block_tmp]
            nvecSQs_final = nvecSQs_final+[nvecSQs_outer_row]

        nvecSQs_final = nvecSQs_final[1:]
        return nvecSQs_final

    def build_interpolator(self, Emin, Emax, Estep,
                           Lmin, Lmax, Lstep, project, irrep):
        """Build interpolator."""
        assert project
        # Generate grides and matrix structures
        L_grid, E_grid, interp_mat_dim, interpolator_matrix = self\
            .grids_and_matrix(Emin, Emax, Estep, Lmin, Lmax, Lstep,
                              project, irrep)

        # Determine basis where entries are smooth
        dim_with_shell_index_all = self.get_dim_with_shell_index_all(irrep)
        final_set_for_change_of_basis = self\
            .get_final_set_for_change_of_basis(dim_with_shell_index_all)
        cob_matrices = self.get_cob_matrices(final_set_for_change_of_basis)

        # Populate interpolation data
        energy_vol_dat_index = 0
        interp_data_index = 1
        for L in L_grid:
            for E in E_grid:
                g_tmp = self.get_value(E=E, L=L,
                                       project=project, irrep=irrep)
                for cob_matrix in cob_matrices:
                    try:
                        g_tmp = (cob_matrix.T)@g_tmp@cob_matrix
                    except ValueError:
                        pass
                for i in range(len(g_tmp)):
                    for j in range(len(g_tmp)):
                        g_val = g_tmp[i][j]
                        if not np.isnan(g_val) and (g_val != 0.):
                            interpolator_matrix[i][j][interp_data_index]\
                                = interpolator_matrix[i][j][interp_data_index]\
                                + [[E, L, g_val]]
        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                interpolator_matrix[i][j][interp_data_index] =\
                    interpolator_matrix[i][j][interp_data_index][1:]
                if len(interpolator_matrix[i][j][interp_data_index]) == 0:
                    interpolator_matrix[i][j][energy_vol_dat_index] = []
                else:
                    for interpolator_entry in\
                       interpolator_matrix[i][j][interp_data_index]:
                        [E, L, g_val] = interpolator_entry
                        # [Lmin, Lmax, Emin, Emax]
                        if L < interpolator_matrix[i][j][
                                energy_vol_dat_index][0]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][0] = L
                        if L > interpolator_matrix[i][j][
                                energy_vol_dat_index][1]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][1] = L
                        if E < interpolator_matrix[i][j][
                                energy_vol_dat_index][2]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][2] = E
                        if E > interpolator_matrix[i][j][
                                energy_vol_dat_index][3]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][3] = E
                    interpolator_matrix[i][j][energy_vol_dat_index][0]\
                        = interpolator_matrix[i][j][energy_vol_dat_index][0]\
                        - Lstep
                    interpolator_matrix[i][j][energy_vol_dat_index][2]\
                        = interpolator_matrix[i][j][energy_vol_dat_index][2]\
                        - Estep

        # Identify all poles in projected entries
        nvecSQs_by_shell = self.get_all_nvecSQs_by_shell(E=Emax, L=Lmax,
                                                         project=project,
                                                         irrep=irrep)
        all_nvecSQs = self.get_all_nvecSQs(nvecSQs_by_shell)
        m1, m2, m3 = self.extract_masses()
        all_relevant_nvecSQs = self\
            .get_all_relevant_nvecSQs(Emax, project, irrep, interp_mat_dim,
                                      interpolator_matrix, cob_matrices,
                                      all_nvecSQs, m1, m2, m3)

        # Remove poles
        pole_free_interpolator_matrix = self\
            .get_pole_free_interpolator_matrix(interp_mat_dim,
                                               interpolator_matrix,
                                               interp_data_index, m1, m2, m3,
                                               all_relevant_nvecSQs)

        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                interpolator_matrix_complete = [[]]
                pole_free_interpolator_matrix_complete = [[]]
                if len(interpolator_matrix[i][j][energy_vol_dat_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = interpolator_matrix[i][j][energy_vol_dat_index]
                    Lgrid_entry = np.arange(Lmin_entry, Lmax_entry+EPSILON4,
                                            Lstep)
                    Egrid_entry = np.arange(Emin_entry, Emax_entry+EPSILON4,
                                            Estep)
                    for Ltmp in Lgrid_entry:
                        for Etmp in Egrid_entry:
                            not_found = True
                            for interpolator_entry_index in\
                                range(len(interpolator_matrix[i][j][
                                    interp_data_index])):
                                interpolator_entry = interpolator_matrix[i][j][
                                    interp_data_index][
                                        interpolator_entry_index]
                                if ((np.abs(interpolator_entry[0]-Etmp)
                                    < EPSILON10)
                                    and (np.abs(interpolator_entry[1]-Ltmp)
                                         < EPSILON10)):
                                    not_found = False
                                    interpolator_matrix_complete\
                                        = interpolator_matrix_complete\
                                        + [interpolator_entry]
                                    pole_free_interpolator_entry\
                                        = pole_free_interpolator_matrix[i][j][
                                            interp_data_index][
                                                interpolator_entry_index]
                                    pole_free_interpolator_matrix_complete =\
                                        pole_free_interpolator_matrix_complete\
                                        + [pole_free_interpolator_entry]
                            if not_found:
                                interpolator_matrix_complete\
                                    = interpolator_matrix_complete\
                                    + [[Etmp, Ltmp, 0.]]
                                pole_free_interpolator_matrix_complete =\
                                    pole_free_interpolator_matrix_complete\
                                    + [[Etmp, Ltmp, 0.]]
                    interpolator_matrix[i][j][interp_data_index]\
                        = interpolator_matrix_complete[1:]
                    pole_free_interpolator_matrix[i][j][interp_data_index]\
                        = pole_free_interpolator_matrix_complete[1:]

        # Build interpolator functions
        function_set = [[]]
        func_tuple_set = [[]]
        for i in range(interp_mat_dim):
            function_set_row = []
            func_tuple_set_row = []
            for j in range(interp_mat_dim):
                if len(interpolator_matrix[i][j][energy_vol_dat_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = pole_free_interpolator_matrix[i][j][
                            energy_vol_dat_index]
                    L_grid_tmp\
                        = np.arange(Lmin_entry, Lmax_entry+EPSILON4, Lstep)
                    E_grid_tmp\
                        = np.arange(Emin_entry, Emax_entry+EPSILON4, Estep)
                    E_mesh_grid, L_mesh_grid\
                        = np.meshgrid(E_grid_tmp, L_grid_tmp)
                    g_pole_free_mesh_grid\
                        = (np.array(pole_free_interpolator_matrix[i][j][
                            interp_data_index]).T)[2].\
                        reshape(L_mesh_grid.shape).T
                    try:
                        interp =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='cubic')
                    except ValueError:
                        interp =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='linear')
                    function_set_row = function_set_row+[interp]
                    func_tuple_set_row = func_tuple_set_row\
                        + [(E_grid_tmp, L_grid_tmp, g_pole_free_mesh_grid)]
                else:
                    function_set_row = function_set_row+[None]
                    func_tuple_set_row = func_tuple_set_row+[None]
            function_set = function_set+[function_set_row]
            func_tuple_set = func_tuple_set+[func_tuple_set_row]
        function_set = np.array(function_set[1:])
        func_tuple_set = np.array(func_tuple_set[1:])

        # Get unique E and L sets
        E_grid_unique = []
        for i in range(len(func_tuple_set)):
            for j in range(len(func_tuple_set[i])):
                if func_tuple_set[i][j] is not None:
                    E_grid_candidate = func_tuple_set[i][j][0]
                    for E in E_grid_candidate:
                        if E not in E_grid_unique:
                            E_grid_unique.append(E)
        E_grid_unique = np.unique(np.sort(E_grid_unique).round(decimals=10))

        L_grid_unique = []
        for i in range(len(func_tuple_set)):
            for j in range(len(func_tuple_set[i])):
                if func_tuple_set[i][j] is not None:
                    L_grid_candidate = func_tuple_set[i][j][1]
                    for L in L_grid_candidate:
                        if L not in L_grid_unique:
                            L_grid_unique.append(L)
        L_grid_unique = np.unique(np.sort(L_grid_unique).round(decimals=10))

        # Build the rank 4 tensor
        func_set_tensor = []
        for E in E_grid_unique:
            vol_rank = []
            for L in L_grid_unique:
                gi_rank = []
                for i in range(len(func_tuple_set)):
                    gj_rank = []
                    for j in range(len(func_tuple_set[i])):
                        if func_tuple_set[i][j] is None:
                            gj_rank.append(0.0)
                        else:
                            en_bools =\
                                np.abs(func_tuple_set[i][j][0]-E) < 1.e-10
                            vol_bools =\
                                np.abs(func_tuple_set[i][j][1]-L) < 1.e-10
                            if (not en_bools.any()) or (not vol_bools.any()):
                                gj_rank.append(0.0)
                            else:
                                en_loc = np.where(en_bools)[0][0]
                                vol_loc = np.where(vol_bools)[0][0]
                                gj_rank.append(
                                    func_tuple_set[i][j][2][en_loc][vol_loc])
                    gi_rank.append(gj_rank)
                vol_rank.append(gi_rank)
            func_set_tensor.append(vol_rank)
        func_set_tensor = np.array(func_set_tensor)

        warnings.simplefilter('always')

        all_dimensions = []
        for cob_matrix in cob_matrices:
            all_dimensions = all_dimensions+[len(cob_matrix)]

        # Add relevant data to self
        self.all_relevant_nvecSQs[irrep] = all_relevant_nvecSQs
        self.pole_free_interpolator_matrix[irrep]\
            = pole_free_interpolator_matrix
        self.interpolator_matrix[irrep] = interpolator_matrix
        self.cob_matrices[irrep] = cob_matrices
        self.total_cobs[irrep] = len(cob_matrices)
        self.function_set[irrep] = function_set
        self.all_dimensions[irrep] = all_dimensions

    def extract_masses(self):
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses
        return m1, m2, m3

    def get_pole_free_interpolator_matrix(self, interp_mat_dim,
                                          interpolator_matrix,
                                          interp_data_index, m1, m2, m3,
                                          all_relevant_nvecSQs):
        pole_free_interpolator_matrix = [[]]
        for i in range(interp_mat_dim):
            rowtmp = [[]]
            for j in range(interp_mat_dim):
                matrix_entry_interp_data = interpolator_matrix[i][j][
                    interp_data_index]
                relevant_poles = [[]]
                for relevant_candidate in all_relevant_nvecSQs:
                    if ((relevant_candidate[0] == i)
                       and (relevant_candidate[1] == j)):
                        relevant_poles = relevant_poles+[relevant_candidate]
                relevant_poles = relevant_poles[1:]
                for entry_index in range(len(matrix_entry_interp_data)):
                    dim_with_shell_index = matrix_entry_interp_data[
                        entry_index]
                    [E, L, g_val] = dim_with_shell_index
                    for nvecSQs_set in relevant_poles:
                        nvecSQ = nvecSQs_set[2]
                        n1vecSQ = nvecSQ[0]
                        n2vecSQ = nvecSQ[1]
                        n3vecSQ = nvecSQ[2]
                        three_omega = self\
                            .get_pole_candidate(L, n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        pole_removal_factor = E-three_omega
                        g_val = pole_removal_factor*g_val
                    matrix_entry_interp_data[entry_index] = [E, L, g_val]
                final_entry = [interpolator_matrix[i][j][0],
                               matrix_entry_interp_data,
                               relevant_poles]
                rowtmp = rowtmp+[final_entry]
            pole_free_interpolator_matrix = pole_free_interpolator_matrix\
                + [rowtmp[1:]]
        pole_free_interpolator_matrix = pole_free_interpolator_matrix[1:]
        return pole_free_interpolator_matrix

    def get_all_relevant_nvecSQs(self, Emax, project, irrep, interp_mat_dim,
                                 interpolator_matrix, cob_matrices,
                                 all_nvecSQs, m1, m2, m3):
        interp_data_index = 1
        all_relevant_nvecSQs = [[]]
        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                matrix_entry_interp_data = interpolator_matrix[i][j][
                    interp_data_index][1:]
                if len(matrix_entry_interp_data) != 0:
                    Lmin_tmp = BAD_MIN_GUESS
                    Lmax_tmp = BAD_MAX_GUESS
                    Emin_tmp = BAD_MIN_GUESS
                    Emax_tmp = BAD_MAX_GUESS
                    for single_interp_entry in matrix_entry_interp_data:
                        energy_volume_set = single_interp_entry[:-1]
                        [E_candidate, L_candidate] = energy_volume_set
                        if E_candidate < Emin_tmp:
                            Emin_tmp = E_candidate
                        if E_candidate > Emax_tmp:
                            Emax_tmp = E_candidate
                        if L_candidate < Lmin_tmp:
                            Lmin_tmp = L_candidate
                        if L_candidate > Lmax_tmp:
                            Lmax_tmp = L_candidate
                    nvecSQs_all_keeps = [[]]
                    for nvecSQ_entry in all_nvecSQs:
                        n1vecSQ = nvecSQ_entry[0]
                        n2vecSQ = nvecSQ_entry[1]
                        n3vecSQ = nvecSQ_entry[2]
                        removal_at_Lmin = self\
                            .get_pole_candidate(Lmin_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        removal_at_Lmax = self\
                            .get_pole_candidate(Lmax_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        if ((Emin_tmp < removal_at_Lmin < Emax_tmp)
                           or (Emin_tmp < removal_at_Lmax < Emax_tmp)):
                            nvecSQs_all_keeps = nvecSQs_all_keeps\
                                + [[n1vecSQ, n2vecSQ, n3vecSQ]]
                    nvecSQs_all_keeps = nvecSQs_all_keeps[1:]
                    for nvecSQs_keep in nvecSQs_all_keeps:
                        [n1vecSQ, n2vecSQ, n3vecSQ] = nvecSQs_keep
                        Lvals_tmp = [Lmin_tmp+EPSILON4, Lmax_tmp-EPSILON4]
                        for Ltmp in Lvals_tmp:
                            Etmp = self\
                                .get_pole_candidate_eps(Ltmp, n1vecSQ, n2vecSQ,
                                                        n3vecSQ, m1, m2, m3)
                            if Etmp < Emax:
                                try:
                                    g_tmp = self\
                                        .get_value(E=Etmp, L=Ltmp,
                                                   project=project,
                                                   irrep=irrep)
                                    for cob_matrix in cob_matrices:
                                        try:
                                            g_tmp =\
                                                (cob_matrix.T)@g_tmp@cob_matrix
                                        except ValueError:
                                            pass
                                    g_tmp_entry = g_tmp[i][j]
                                    near_pole_mag = np.abs(g_tmp_entry)
                                    pole_found = (near_pole_mag > POLE_CUT)
                                    if (pole_found and
                                        ([i, j, nvecSQs_keep] not in
                                         all_relevant_nvecSQs)):
                                        all_relevant_nvecSQs\
                                            = all_relevant_nvecSQs\
                                            + [[i, j, nvecSQs_keep]]
                                except IndexError:
                                    pass
        all_relevant_nvecSQs = all_relevant_nvecSQs[1:]
        return all_relevant_nvecSQs

    def get_pole_candidate(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                       + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                       + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)
        return pole_candidate

    def get_pole_candidate_eps(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate_eps = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                           + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                           + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)+EPSILON10
        return pole_candidate_eps

    def get_all_nvecSQs(self, nvecSQs_by_shell):
        all_nvecSQs = []
        for outer_nvecSQ_row in nvecSQs_by_shell:
            for outer_nvecSQ_entry in outer_nvecSQ_row:
                for inner_nvecSQ_row in outer_nvecSQ_entry:
                    for inner_nvecSQ_entry in inner_nvecSQ_row:
                        if len(inner_nvecSQ_entry) != 0:
                            n1vecSQs = inner_nvecSQ_entry[0][0]
                            n2vecSQs = inner_nvecSQ_entry[0][1]
                            n3vecSQs = inner_nvecSQ_entry[0][2]
                            for i in range(len(n1vecSQs)):
                                for j in range(len(n1vecSQs[i])):
                                    nvecSQ_sets = [n1vecSQs[i][j],
                                                   n2vecSQs[i][j],
                                                   n3vecSQs[i][j]]
                                    nvecSQ_sets = list(np.sort(nvecSQ_sets))
                                    if nvecSQ_sets not in all_nvecSQs:
                                        all_nvecSQs = all_nvecSQs+[nvecSQ_sets]
        return all_nvecSQs

    def get_cob_matrices(self, final_set_for_change_of_basis):
        all_restacks = [[]]
        for dim_shell_counter_all in final_set_for_change_of_basis:
            restack = [[]]
            for shell_index in range(len(dim_shell_counter_all)):
                for dim_shell_counter in dim_shell_counter_all[shell_index]:
                    restack = restack+[[([dim_shell_counter[0][1],
                                          shell_index]), dim_shell_counter[1]]]
            all_restacks = all_restacks+[sorted(restack[1:])]
        all_restacks = all_restacks[1:]
        all_restacks_second = [[]]
        for restack in all_restacks:
            second_restack = []
            for entry in restack:
                second_restack = second_restack+(entry[1])
            all_restacks_second = all_restacks_second+[second_restack]
        all_restacks_second = all_restacks_second[1:]
        cob_matrices = []
        for restack in all_restacks_second:
            cob_matrices = cob_matrices\
                + [(np.identity(len(restack))[restack]).T]
        return cob_matrices

    def get_final_set_for_change_of_basis(self, dim_with_shell_index_all):
        final_set_for_change_of_basis = [[]]
        for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
            dim_shell_counter_all = [[]]
            dim_counter = 0
            for dim_with_shell_index_for_sc in dim_with_shell_index_all:
                dim_shell_counter = [[]]
                for dim_with_shell_index in dim_with_shell_index_for_sc:
                    if dim_with_shell_index[1] <= shell_index:
                        counter_set = []
                        for _ in range(dim_with_shell_index[0]):
                            counter_set = counter_set+[dim_counter]
                            dim_counter = dim_counter+1
                        dim_shell_counter = dim_shell_counter\
                            + [[dim_with_shell_index, counter_set]]
                dim_shell_counter_all = dim_shell_counter_all\
                    + [dim_shell_counter[1:]]
            final_set_for_change_of_basis = final_set_for_change_of_basis\
                + [dim_shell_counter_all[1:]]
        final_set_for_change_of_basis = final_set_for_change_of_basis[1:]
        return final_set_for_change_of_basis

    def get_dim_with_shell_index_all(self, irrep):
        dim_with_shell_index_all = [[]]
        for spectator_channel_index in range(
             len(self.qcis.fcs.sc_list_sorted)):
            dim_with_shell_index_for_sc = [[]]
            ell_set_tmp = self\
                .qcis.fcs.sc_list_sorted[spectator_channel_index].ell_set
            ang_mom_dim = 0
            for ell_tmp in ell_set_tmp:
                ang_mom_dim = ang_mom_dim+(2*ell_tmp+1)
            for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
                shell = self.qcis.tbks_list[0][0].shells[shell_index]
                try:
                    transposed_proj_dict = self.qcis.sc_proj_dicts[
                        spectator_channel_index][irrep][
                        ang_mom_dim*shell[0]:ang_mom_dim*shell[1]].T
                    support_rows = []
                    for row_index in range(len(transposed_proj_dict)):
                        row = transposed_proj_dict[row_index]
                        if (not (row@row < EPSILON10)):
                            support_rows = support_rows\
                                + [row_index]
                    proj_candidate = transposed_proj_dict[support_rows].T
                    # only purpose of the following is to trigger KeyError
                    self.qcis.sc_proj_dicts_by_shell[
                        spectator_channel_index][0][shell_index][irrep]
                    dim_with_shell_index_for_sc = dim_with_shell_index_for_sc\
                        + [[(proj_candidate.shape)[1], shell_index]]
                except KeyError:
                    pass
            dim_with_shell_index_all = dim_with_shell_index_all\
                + [dim_with_shell_index_for_sc[1:]]
        dim_with_shell_index_all = dim_with_shell_index_all[1:]
        return dim_with_shell_index_all

    def grids_and_matrix(self, Emin, Emax, Estep, Lmin, Lmax, Lstep,
                         project, irrep):
        L_grid = np.arange(Lmin, Lmax+EPSILON4, Lstep)
        E_grid = np.arange(Emin, Emax+EPSILON4, Estep)
        interp_mat_shape = (self.get_value(E=Emax, L=Lmax,
                                           project=project,
                                           irrep=irrep)).shape
        interp_mat_dim = interp_mat_shape[0]
        interpolator_matrix = [[]]
        for _ in range(interp_mat_dim):
            interp_mat_row = []
            for _ in range(interp_mat_dim):
                interp_mat_row = interp_mat_row\
                    + [[[BAD_MIN_GUESS, BAD_MAX_GUESS,
                         BAD_MIN_GUESS, BAD_MAX_GUESS], [[]]]]
            interpolator_matrix = interpolator_matrix+[interp_mat_row]
        interpolator_matrix = interpolator_matrix[1:]
        return L_grid, E_grid, interp_mat_dim, interpolator_matrix


class F:
    """
    Class for the finite-volume F matrix.

    Warning: It is up to the user to select values of alphaKSS and C1cut that
    lead to a sufficient estimate of the F matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    :param alphaKSS: damping factor entering the zeta functions
    :type alphaKSS: float
    :param C1cut: hard cutoff used in the zeta functions
    :type C1cut: int
    """

    def __init__(self, qcis=None, alphaKSS=1.0, C1cut=3):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS

    def _get_masks_and_shells(self, E, nP, L, tbks_entry,
                              cindex, slice_index):
        mask_slices = None
        three_slice_index\
            = self.qcis._get_three_slice_index(cindex)

        if nP@nP == 0:
            slice_entry = tbks_entry.shells[slice_index]
        else:
            masses = self.qcis.fcs.sc_list_sorted[
                self.qcis.fcs.slices_by_three_masses[three_slice_index][0]]\
                .masses_indexed
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            slices = tbks_entry.shells
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        return mask_slices, slice_entry

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell1=0, ell2=0, tbks_entry=None,
                  slice_index=None, project=False, irrep=None,
                  mask=None):
        """Build the F matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta
        C1cut = self.C1cut
        alphaKSS = self.alphaKSS

        mask_slices, slice_entry\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Fshell = QCFunctions.getF_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell1, ell2,
                                        alpha, beta,
                                        C1cut, alphaKSS,
                                        qc_impl, ts)
        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][ibest])[mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][0][slice_index][irrep]
                    proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                return np.array([])
        if project:
            Fshell = proj_tmp_left@Fshell@proj_tmp_right

        return Fshell

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the F matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        three_slice_index = 0
        cindex = 0
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within F assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            mask = None
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within F assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[three_slice_index][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('slices =')
                print(slices)
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])
            if self.qcis.verbosity >= 2:
                print('mask_slices =')
                print(mask_slices)
                print('range for sc_ind =')
                print(range(len(self.qcis.fcs.sc_list_sorted)))
        f_final_list = []
        for sc_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            ell_set = self.qcis.fcs.sc_list_sorted[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in F")
            ell1 = ell_set[0]
            ell2 = ell1
            for slice_index in range(len(slices)):
                if self.qcis.verbosity >= 2:
                    print('get_shell is receiving:')
                    print(E, L,
                          m1, m2, m3)
                    print(f'cindex = {cindex}\n'
                          f'sc_ind = {sc_ind}\n'
                          f'ell1 = {ell1}\n'
                          f'ell2 = {ell2}\n')
                    print(tbks_entry)
                    print('slice_index = '+str(slice_index))
                    print(project, irrep)
                f_tmp = self.get_shell(E, L,
                                       m1, m2, m3,
                                       cindex,  # only for non-zero P
                                       sc_ind,
                                       ell1, ell2,
                                       tbks_entry,
                                       slice_index,
                                       project, irrep,
                                       mask)
                if len(f_tmp) != 0:
                    f_final_list = f_final_list+[f_tmp]
        return block_diag(*f_final_list)


class FplusG:
    r"""
    Class for F+G (typically with interpolation).
    """

    def __init__(self, qcis=QCIndexSpace(), alphaKSS=1.0, C1cut=3):
        self.qcis = qcis
        three_scheme = self.qcis.tbis.three_scheme
        if (three_scheme == 'original pole')\
           or (three_scheme == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.all_relevant_nvecSQs = {}
        self.pole_free_interpolator_matrix = {}
        self.interpolator_matrix = {}
        self.cob_matrices = {}
        self.function_set = {}
        self.all_dimensions = {}
        self.total_cobs = {}

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the F plus G matrix in a shell-based way."""
        g_interpolate = QC_IMPL_DEFAULTS['g_interpolate']
        if 'g_interpolate' in self.qcis.fvs.qc_impl:
            g_interpolate = self.qcis.fvs.qc_impl['g_interpolate']
        if not project:
            raise ValueError("FplusG().get_value() should only be called "
                             + "with project==True")
        if not g_interpolate:
            raise ValueError("FplusG().get_value() should only be called "
                             + "with g_interpolate==True")
        f_plus_g_final_smooth_basis = [[]]
        total_cobs = self.total_cobs[irrep]
        matrix_dimension = self.\
            all_dimensions[irrep][
                total_cobs-self.qcis.get_tbks_sub_indices(E, L)[0]-1]
        m1, m2, m3 = self.extract_masses()
        for i in range(matrix_dimension):
            g_row_tmp = []
            for j in range(matrix_dimension):
                func_tmp = self.function_set[irrep][i][j]
                if func_tmp is not None:
                    value_tmp = float(func_tmp((E, L)))
                    for pole_data in (self.
                                      pole_free_interpolator_matrix[
                                          irrep][i][j][2]):
                        factor_tmp = E-self.\
                            get_pole_candidate(L, *pole_data[2], m1, m2, m3)
                        value_tmp = value_tmp/factor_tmp
                    g_row_tmp = g_row_tmp+[value_tmp]
                else:
                    g_row_tmp = g_row_tmp+[0.]
            f_plus_g_final_smooth_basis = f_plus_g_final_smooth_basis\
                + [g_row_tmp]
        f_plus_g_final_smooth_basis = np.array(f_plus_g_final_smooth_basis[1:])
        cob_matrix = self.\
            cob_matrices[irrep][
                total_cobs-self.qcis.get_tbks_sub_indices(E, L)[0]-1]
        f_plus_g_matrix_tmp_rotated\
            = (cob_matrix)@f_plus_g_final_smooth_basis@(cob_matrix.T)
        return f_plus_g_matrix_tmp_rotated

    def get_all_nvecSQs_by_shell(self, E=5.0, L=5.0, project=False,
                                 irrep=None):
        """Build the G matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        cindex_row = cindex_col = 0
        if (not ((irrep is None) and (project is False))
           and (not (irrep in self.qcis.proj_dict.keys()))):
            raise ValueError("irrep "+str(irrep)+" not in "
                             + "qcis.proj_dict.keys()")

        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within G assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        nvecSQs_final = [[]]
        if self.qcis.verbosity >= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            nvecSQs_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list_sorted[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in G")
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(self.qcis.fcs.sc_list_sorted)):
                if self.qcis.verbosity >= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError("only length-one ell_set currently "
                                     + "supported in G")
                ell2 = col_ell_set[0]
                g_rescale = self.qcis.fcs.g_templates[0][0][
                    sc_row_ind][sc_col_ind]

                nvecSQs_inner = [[]]
                for row_shell_index in range(len(slices)):
                    nvecSQs_inner_row = []
                    for col_shell_index in range(len(slices)):
                        nvecSQs_tmp = self\
                            .get_shell_nvecSQs_projs(E, L, m1, m2, m3,
                                                     cindex_row, cindex_col,
                                                     # only for non-zero P
                                                     sc_row_ind, sc_col_ind,
                                                     ell1, ell2, g_rescale,
                                                     tbks_entry,
                                                     row_shell_index,
                                                     col_shell_index,
                                                     project, irrep)
                        nvecSQs_inner_row = nvecSQs_inner_row+[nvecSQs_tmp]
                    nvecSQs_inner = nvecSQs_inner+[nvecSQs_inner_row]
                nvecSQs_block_tmp = nvecSQs_inner[1:]
                nvecSQs_outer_row = nvecSQs_outer_row+[nvecSQs_block_tmp]
            nvecSQs_final = nvecSQs_final+[nvecSQs_outer_row]

        nvecSQs_final = nvecSQs_final[1:]
        return nvecSQs_final

    def build_interpolator(self, Emin, Emax, Estep,
                           Lmin, Lmax, Lstep, project, irrep):
        """Build interpolator."""
        assert project
        # Generate grides and matrix structures
        L_grid, E_grid, interp_mat_dim, interpolator_matrix = self\
            .grids_and_matrix(Emin, Emax, Estep, Lmin, Lmax, Lstep,
                              project, irrep)

        # Determine basis where entries are smooth
        dim_with_shell_index_all = self.get_dim_with_shell_index_all(irrep)
        final_set_for_change_of_basis = self\
            .get_final_set_for_change_of_basis(dim_with_shell_index_all)
        cob_matrices = self.get_cob_matrices(final_set_for_change_of_basis)

        # Populate interpolation data
        energy_vol_dat_index = 0
        interp_data_index = 1
        for L in L_grid:
            for E in E_grid:
                f_plus_g_tmp = self.g.get_value(E=E, L=L,
                                                project=project, irrep=irrep)\
                             + self.f.get_value(E=E, L=L,
                                                project=project, irrep=irrep)
                for cob_matrix in cob_matrices:
                    try:
                        f_plus_g_tmp = (cob_matrix.T)@f_plus_g_tmp@cob_matrix
                    except ValueError:
                        pass
                for i in range(len(f_plus_g_tmp)):
                    for j in range(len(f_plus_g_tmp)):
                        f_plus_g_val = f_plus_g_tmp[i][j]
                        if not np.isnan(f_plus_g_val) and (f_plus_g_val != 0.):
                            interpolator_matrix[i][j][interp_data_index]\
                                = interpolator_matrix[i][j][interp_data_index]\
                                + [[E, L, f_plus_g_val]]
        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                interpolator_matrix[i][j][interp_data_index] =\
                    interpolator_matrix[i][j][interp_data_index][1:]
                if len(interpolator_matrix[i][j][interp_data_index]) == 0:
                    interpolator_matrix[i][j][energy_vol_dat_index] = []
                else:
                    for interpolator_entry in\
                       interpolator_matrix[i][j][interp_data_index]:
                        [E, L, f_plus_g_val] = interpolator_entry
                        # [Lmin, Lmax, Emin, Emax]
                        if L < interpolator_matrix[i][j][
                                energy_vol_dat_index][0]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][0] = L
                        if L > interpolator_matrix[i][j][
                                energy_vol_dat_index][1]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][1] = L
                        if E < interpolator_matrix[i][j][
                                energy_vol_dat_index][2]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][2] = E
                        if E > interpolator_matrix[i][j][
                                energy_vol_dat_index][3]:
                            interpolator_matrix[i][j][
                                energy_vol_dat_index][3] = E
                    interpolator_matrix[i][j][energy_vol_dat_index][0]\
                        = interpolator_matrix[i][j][energy_vol_dat_index][0]\
                        - Lstep
                    interpolator_matrix[i][j][energy_vol_dat_index][2]\
                        = interpolator_matrix[i][j][energy_vol_dat_index][2]\
                        - Estep

        # Identify all poles in projected entries
        nvecSQs_by_shell = self.get_all_nvecSQs_by_shell(E=Emax, L=Lmax,
                                                         project=project,
                                                         irrep=irrep)
        all_nvecSQs = self.get_all_nvecSQs(nvecSQs_by_shell)
        m1, m2, m3 = self.extract_masses()
        all_relevant_nvecSQs = self\
            .get_all_relevant_nvecSQs(Emax, project, irrep, interp_mat_dim,
                                      interpolator_matrix, cob_matrices,
                                      all_nvecSQs, m1, m2, m3)

        # Remove poles
        pole_free_interpolator_matrix = self\
            .get_pole_free_interpolator_matrix(interp_mat_dim,
                                               interpolator_matrix,
                                               interp_data_index, m1, m2, m3,
                                               all_relevant_nvecSQs)

        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                interpolator_matrix_complete = [[]]
                pole_free_interpolator_matrix_complete = [[]]
                if len(interpolator_matrix[i][j][energy_vol_dat_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = interpolator_matrix[i][j][energy_vol_dat_index]
                    Lgrid_entry = np.arange(Lmin_entry, Lmax_entry+EPSILON4,
                                            Lstep)
                    Egrid_entry = np.arange(Emin_entry, Emax_entry+EPSILON4,
                                            Estep)
                    for Ltmp in Lgrid_entry:
                        for Etmp in Egrid_entry:
                            not_found = True
                            for interpolator_entry_index in\
                                range(len(interpolator_matrix[i][j][
                                    interp_data_index])):
                                interpolator_entry = interpolator_matrix[i][j][
                                    interp_data_index][
                                        interpolator_entry_index]
                                if ((np.abs(interpolator_entry[0]-Etmp)
                                    < EPSILON10)
                                    and (np.abs(interpolator_entry[1]-Ltmp)
                                         < EPSILON10)):
                                    not_found = False
                                    interpolator_matrix_complete\
                                        = interpolator_matrix_complete\
                                        + [interpolator_entry]
                                    pole_free_interpolator_entry\
                                        = pole_free_interpolator_matrix[i][j][
                                            interp_data_index][
                                                interpolator_entry_index]
                                    pole_free_interpolator_matrix_complete =\
                                        pole_free_interpolator_matrix_complete\
                                        + [pole_free_interpolator_entry]
                            if not_found:
                                interpolator_matrix_complete\
                                    = interpolator_matrix_complete\
                                    + [[Etmp, Ltmp, 0.]]
                                pole_free_interpolator_matrix_complete =\
                                    pole_free_interpolator_matrix_complete\
                                    + [[Etmp, Ltmp, 0.]]
                    interpolator_matrix[i][j][interp_data_index]\
                        = interpolator_matrix_complete[1:]
                    pole_free_interpolator_matrix[i][j][interp_data_index]\
                        = pole_free_interpolator_matrix_complete[1:]
        warnings.simplefilter('default')

        # Build interpolator functions
        function_set = [[]]
        for i in range(interp_mat_dim):
            function_set_row = []
            for j in range(interp_mat_dim):
                if len(interpolator_matrix[i][j][energy_vol_dat_index]) == 4:
                    [Lmin_entry, Lmax_entry, Emin_entry, Emax_entry]\
                        = pole_free_interpolator_matrix[i][j][
                            energy_vol_dat_index]
                    L_grid_tmp\
                        = np.arange(Lmin_entry, Lmax_entry+EPSILON4, Lstep)
                    E_grid_tmp\
                        = np.arange(Emin_entry, Emax_entry+EPSILON4, Estep)
                    E_mesh_grid, L_mesh_grid\
                        = np.meshgrid(E_grid_tmp, L_grid_tmp)
                    g_pole_free_mesh_grid\
                        = (np.array(pole_free_interpolator_matrix[i][j][
                            interp_data_index]).T)[2].\
                        reshape(L_mesh_grid.shape).T

                    try:
                        interp =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='cubic')
                    except ValueError:
                        interp =\
                            RegularGridInterpolator((E_grid_tmp, L_grid_tmp),
                                                    g_pole_free_mesh_grid,
                                                    method='linear')
                    function_set_row = function_set_row+[interp]
                else:
                    function_set_row = function_set_row+[None]
            function_set = function_set+[function_set_row]
        function_set = np.array(function_set[1:])
        warnings.simplefilter('always')

        all_dimensions = []
        for cob_matrix in cob_matrices:
            all_dimensions = all_dimensions+[len(cob_matrix)]

        # Add relevant data to self
        self.all_relevant_nvecSQs[irrep] = all_relevant_nvecSQs
        self.pole_free_interpolator_matrix[irrep]\
            = pole_free_interpolator_matrix
        self.interpolator_matrix[irrep] = interpolator_matrix
        self.cob_matrices[irrep] = cob_matrices
        self.total_cobs[irrep] = len(cob_matrices)
        self.function_set[irrep] = function_set
        self.all_dimensions[irrep] = all_dimensions

    def extract_masses(self):
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[0][0]].masses_indexed
        [m1, m2, m3] = masses
        return m1, m2, m3

    def get_pole_free_interpolator_matrix(self, interp_mat_dim,
                                          interpolator_matrix,
                                          interp_data_index, m1, m2, m3,
                                          all_relevant_nvecSQs):
        pole_free_interpolator_matrix = [[]]
        for i in range(interp_mat_dim):
            rowtmp = [[]]
            for j in range(interp_mat_dim):
                matrix_entry_interp_data = interpolator_matrix[i][j][
                    interp_data_index]
                relevant_poles = [[]]
                for relevant_candidate in all_relevant_nvecSQs:
                    if ((relevant_candidate[0] == i)
                       and (relevant_candidate[1] == j)):
                        relevant_poles = relevant_poles+[relevant_candidate]
                relevant_poles = relevant_poles[1:]
                for entry_index in range(len(matrix_entry_interp_data)):
                    dim_with_shell_index = matrix_entry_interp_data[
                        entry_index]
                    [E, L, g_val] = dim_with_shell_index
                    for nvecSQs_set in relevant_poles:
                        nvecSQ = nvecSQs_set[2]
                        n1vecSQ = nvecSQ[0]
                        n2vecSQ = nvecSQ[1]
                        n3vecSQ = nvecSQ[2]
                        three_omega = self\
                            .get_pole_candidate(L, n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        pole_removal_factor = E-three_omega
                        g_val = pole_removal_factor*g_val
                    matrix_entry_interp_data[entry_index] = [E, L, g_val]
                final_entry = [interpolator_matrix[i][j][0],
                               matrix_entry_interp_data,
                               relevant_poles]
                rowtmp = rowtmp+[final_entry]
            pole_free_interpolator_matrix = pole_free_interpolator_matrix\
                + [rowtmp[1:]]
        pole_free_interpolator_matrix = pole_free_interpolator_matrix[1:]
        return pole_free_interpolator_matrix

    def get_all_relevant_nvecSQs(self, Emax, project, irrep, interp_mat_dim,
                                 interpolator_matrix, cob_matrices,
                                 all_nvecSQs, m1, m2, m3):
        interp_data_index = 1
        all_relevant_nvecSQs = [[]]
        for i in range(interp_mat_dim):
            for j in range(interp_mat_dim):
                matrix_entry_interp_data = interpolator_matrix[i][j][
                    interp_data_index][1:]
                if len(matrix_entry_interp_data) != 0:
                    Lmin_tmp = BAD_MIN_GUESS
                    Lmax_tmp = BAD_MAX_GUESS
                    Emin_tmp = BAD_MIN_GUESS
                    Emax_tmp = BAD_MAX_GUESS
                    for single_interp_entry in matrix_entry_interp_data:
                        energy_volume_set = single_interp_entry[:-1]
                        [E_candidate, L_candidate] = energy_volume_set
                        if E_candidate < Emin_tmp:
                            Emin_tmp = E_candidate
                        if E_candidate > Emax_tmp:
                            Emax_tmp = E_candidate
                        if L_candidate < Lmin_tmp:
                            Lmin_tmp = L_candidate
                        if L_candidate > Lmax_tmp:
                            Lmax_tmp = L_candidate
                    nvecSQs_all_keeps = [[]]
                    for nvecSQ_entry in all_nvecSQs:
                        n1vecSQ = nvecSQ_entry[0]
                        n2vecSQ = nvecSQ_entry[1]
                        n3vecSQ = nvecSQ_entry[2]
                        removal_at_Lmin = self\
                            .get_pole_candidate(Lmin_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        removal_at_Lmax = self\
                            .get_pole_candidate(Lmax_tmp,
                                                n1vecSQ, n2vecSQ, n3vecSQ,
                                                m1, m2, m3)
                        if ((Emin_tmp < removal_at_Lmin < Emax_tmp)
                           or (Emin_tmp < removal_at_Lmax < Emax_tmp)):
                            nvecSQs_all_keeps = nvecSQs_all_keeps\
                                + [[n1vecSQ, n2vecSQ, n3vecSQ]]
                    nvecSQs_all_keeps = nvecSQs_all_keeps[1:]
                    for nvecSQs_keep in nvecSQs_all_keeps:
                        [n1vecSQ, n2vecSQ, n3vecSQ] = nvecSQs_keep
                        Lvals_tmp = [Lmin_tmp+EPSILON4, Lmax_tmp-EPSILON4]
                        for Ltmp in Lvals_tmp:
                            Etmp = self\
                                .get_pole_candidate_eps(Ltmp, n1vecSQ, n2vecSQ,
                                                        n3vecSQ, m1, m2, m3)
                            if Etmp < Emax:
                                try:
                                    f_plus_g_tmp = self.\
                                        g.get_value(E=Etmp, L=Ltmp,
                                                    project=project,
                                                    irrep=irrep)\
                                        + self.\
                                        f.get_value(E=Etmp, L=Ltmp,
                                                    project=project,
                                                    irrep=irrep)
                                    for cob_matrix in cob_matrices:
                                        try:
                                            f_plus_g_tmp =\
                                                (cob_matrix.T)@f_plus_g_tmp\
                                                @ cob_matrix
                                        except ValueError:
                                            pass
                                    g_tmp_entry = f_plus_g_tmp[i][j]
                                    near_pole_mag = np.abs(g_tmp_entry)
                                    pole_found = (near_pole_mag > POLE_CUT)
                                    if (pole_found and
                                        ([i, j, nvecSQs_keep] not in
                                         all_relevant_nvecSQs)):
                                        all_relevant_nvecSQs\
                                            = all_relevant_nvecSQs\
                                            + [[i, j, nvecSQs_keep]]
                                except IndexError:
                                    pass
        all_relevant_nvecSQs = all_relevant_nvecSQs[1:]
        return all_relevant_nvecSQs

    def get_pole_candidate(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                       + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                       + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)
        return pole_candidate

    def get_pole_candidate_eps(self, L, n1vecSQ, n2vecSQ, n3vecSQ, m1, m2, m3):
        pole_candidate_eps = np.sqrt(m1**2+(FOURPI2/L**2)*n1vecSQ)\
                           + np.sqrt(m2**2+(FOURPI2/L**2)*n2vecSQ)\
                           + np.sqrt(m3**2+(FOURPI2/L**2)*n3vecSQ)+EPSILON10
        return pole_candidate_eps

    def get_all_nvecSQs(self, nvecSQs_by_shell):
        all_nvecSQs = []
        for outer_nvecSQ_row in nvecSQs_by_shell:
            for outer_nvecSQ_entry in outer_nvecSQ_row:
                for inner_nvecSQ_row in outer_nvecSQ_entry:
                    for inner_nvecSQ_entry in inner_nvecSQ_row:
                        if len(inner_nvecSQ_entry) != 0:
                            n1vecSQs = inner_nvecSQ_entry[0][0]
                            n2vecSQs = inner_nvecSQ_entry[0][1]
                            n3vecSQs = inner_nvecSQ_entry[0][2]
                            for i in range(len(n1vecSQs)):
                                for j in range(len(n1vecSQs[i])):
                                    nvecSQ_sets = [n1vecSQs[i][j],
                                                   n2vecSQs[i][j],
                                                   n3vecSQs[i][j]]
                                    nvecSQ_sets = list(np.sort(nvecSQ_sets))
                                    if nvecSQ_sets not in all_nvecSQs:
                                        all_nvecSQs = all_nvecSQs+[nvecSQ_sets]
                                    if i == j:
                                        rng = range(-2, 2+1)
                                        mesh = np.meshgrid(*([rng]*3))
                                        nvec_arr = np.vstack([y.flat
                                                              for y in mesh]).T
                                        if n1vecSQs[i][0] == 0:
                                            n3vec = np.array([0, 0, 0])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets =\
                                                    list(
                                                        np.sort([n1SQtmp,
                                                                 n2SQtmp,
                                                                 n3SQtmp]))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 1:
                                            n3vec = np.array([0, 0, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets =\
                                                    list(
                                                        np.sort([n1SQtmp,
                                                                 n2SQtmp,
                                                                 n3SQtmp]))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 2:
                                            n3vec = np.array([0, 1, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 3:
                                            n3vec = np.array([1, 1, 1])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
                                        elif n1vecSQs[i][0] == 4:
                                            n3vec = np.array([0, 0, 2])
                                            for n1vec in nvec_arr:
                                                n2vec = -n1vec-n3vec
                                                n1SQtmp = n1vec@n1vec
                                                n2SQtmp = n2vec@n2vec
                                                n3SQtmp = n3vec@n3vec
                                                nvecSQ_sets = list(np.sort(
                                                    [n1SQtmp, n2SQtmp, n3SQtmp]
                                                    ))
                                                if (nvecSQ_sets not in
                                                   all_nvecSQs):
                                                    all_nvecSQs =\
                                                        all_nvecSQs+[
                                                            nvecSQ_sets]
        return all_nvecSQs

    def get_cob_matrices(self, final_set_for_change_of_basis):
        all_restacks = [[]]
        for dim_shell_counter_all in final_set_for_change_of_basis:
            restack = [[]]
            for shell_index in range(len(dim_shell_counter_all)):
                for dim_shell_counter in dim_shell_counter_all[shell_index]:
                    restack = restack+[[([dim_shell_counter[0][1],
                                          shell_index]), dim_shell_counter[1]]]
            all_restacks = all_restacks+[sorted(restack[1:])]
        all_restacks = all_restacks[1:]
        all_restacks_second = [[]]
        for restack in all_restacks:
            second_restack = []
            for entry in restack:
                second_restack = second_restack+(entry[1])
            all_restacks_second = all_restacks_second+[second_restack]
        all_restacks_second = all_restacks_second[1:]
        cob_matrices = []
        for restack in all_restacks_second:
            cob_matrices = cob_matrices\
                + [(np.identity(len(restack))[restack]).T]
        return cob_matrices

    def get_final_set_for_change_of_basis(self, dim_with_shell_index_all):
        final_set_for_change_of_basis = [[]]
        for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
            dim_shell_counter_all = [[]]
            dim_counter = 0
            for dim_with_shell_index_for_sc in dim_with_shell_index_all:
                dim_shell_counter = [[]]
                for dim_with_shell_index in dim_with_shell_index_for_sc:
                    if dim_with_shell_index[1] <= shell_index:
                        counter_set = []
                        for _ in range(dim_with_shell_index[0]):
                            counter_set = counter_set+[dim_counter]
                            dim_counter = dim_counter+1
                        dim_shell_counter = dim_shell_counter\
                            + [[dim_with_shell_index, counter_set]]
                dim_shell_counter_all = dim_shell_counter_all\
                    + [dim_shell_counter[1:]]
            final_set_for_change_of_basis = final_set_for_change_of_basis\
                + [dim_shell_counter_all[1:]]
        final_set_for_change_of_basis = final_set_for_change_of_basis[1:]
        return final_set_for_change_of_basis

    def get_dim_with_shell_index_all(self, irrep):
        dim_with_shell_index_all = [[]]
        for spectator_channel_index in range(len(self.qcis.fcs.sc_list)):
            dim_with_shell_index_for_sc = [[]]
            ell_set_tmp = self\
                .qcis.fcs.sc_list[spectator_channel_index].ell_set
            ang_mom_dim = 0
            for ell_tmp in ell_set_tmp:
                ang_mom_dim = ang_mom_dim+(2*ell_tmp+1)
            for shell_index in range(len(self.qcis.tbks_list[0][0].shells)):
                shell = self.qcis.tbks_list[0][0].shells[shell_index]
                try:
                    transposed_proj_dict = self.qcis.sc_proj_dicts[
                        spectator_channel_index][irrep][
                        ang_mom_dim*shell[0]:ang_mom_dim*shell[1]].T
                    support_rows = []
                    for row_index in range(len(transposed_proj_dict)):
                        row = transposed_proj_dict[row_index]
                        if (not (row@row < EPSILON10)):
                            support_rows = support_rows\
                                + [row_index]
                    proj_candidate = transposed_proj_dict[support_rows].T
                    # only purpose of the following is to trigger KeyError
                    self.qcis.sc_proj_dicts_by_shell[
                        spectator_channel_index][0][shell_index][irrep]
                    dim_with_shell_index_for_sc = dim_with_shell_index_for_sc\
                        + [[(proj_candidate.shape)[1], shell_index]]
                except KeyError:
                    pass
            dim_with_shell_index_all = dim_with_shell_index_all\
                + [dim_with_shell_index_for_sc[1:]]
        dim_with_shell_index_all = dim_with_shell_index_all[1:]
        return dim_with_shell_index_all

    def grids_and_matrix(self, Emin, Emax, Estep, Lmin, Lmax, Lstep,
                         project, irrep):
        L_grid = np.arange(Lmin, Lmax+EPSILON4, Lstep)
        E_grid = np.arange(Emin, Emax+EPSILON4, Estep)
        interp_mat_shape = (self.get_value(E=Emax, L=Lmax,
                                           project=project,
                                           irrep=irrep)).shape
        interp_mat_dim = interp_mat_shape[0]
        interpolator_matrix = [[]]
        for _ in range(interp_mat_dim):
            interp_mat_row = []
            for _ in range(interp_mat_dim):
                interp_mat_row = interp_mat_row\
                    + [[[BAD_MIN_GUESS, BAD_MAX_GUESS,
                         BAD_MIN_GUESS, BAD_MAX_GUESS], [[]]]]
            interpolator_matrix = interpolator_matrix+[interp_mat_row]
        interpolator_matrix = interpolator_matrix[1:]
        return L_grid, E_grid, interp_mat_dim, interpolator_matrix


class K:
    """
    Class for the two-to-two k matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    """

    def __init__(self, qcis=None):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data

    def _get_masks_and_shells(self, E, nP, L, tbks_entry,
                              cindex, slice_index):
        mask_slices = None
        three_slice_index\
            = self.qcis._get_three_slice_index(cindex)

        if nP@nP == 0:
            slice_entry = tbks_entry.shells[slice_index]
        else:
            masses = self.qcis.fcs.sc_list_sorted[
                self.qcis.fcs.slices_by_three_masses[three_slice_index][0]]\
                    .masses_indexed
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            slices = tbks_entry.shells
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        return mask_slices, slice_entry

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell=0,
                  pcotdelta_function=None, pcotdelta_parameter_list=None,
                  tbks_entry=None,
                  slice_index=None, project=False, irrep=None):
        """Build the K matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qc_impl = self.qcis.fvs.qc_impl
        alpha = self.alpha
        beta = self.beta

        mask_slices, slice_entry\
            = self._get_masks_and_shells(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Kshell = QCFunctions.getK_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell,
                                        pcotdelta_function,
                                        pcotdelta_parameter_list,
                                        alpha, beta,
                                        qc_impl, ts)

        if project:
            try:
                if nP@nP != 0:
                    ibest = self.qcis._get_ibest(E, L)
                    proj_tmp_right = np.array(self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][ibest])[mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_by_shell[
                        sc_ind][0][slice_index][irrep]
                    proj_tmp_left = np.conjugate((proj_tmp_right).T)
            except KeyError:
                return np.array([])
        if project:
            Kshell = proj_tmp_left@Kshell@proj_tmp_right

        return Kshell

    def get_value(self, E=5.0, L=5.0, pcotdelta_parameter_lists=None,
                  project=False, irrep=None):
        """Build the K matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        # n_two_channels = self.qcis.n_two_channels
        if E > Emax:
            raise ValueError("get_value called with E > Emax")
        if L > Lmax:
            raise ValueError("get_value called with L > Lmax")
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError("only n_three_slices = 1 is supported")
        three_slice_index = 0
        cindex = 0
        masses = self.qcis.fcs.sc_list_sorted[
            self.qcis.fcs.slices_by_three_masses[three_slice_index][0]]\
            .masses_indexed
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.shells
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
            if len(self.qcis.tbks_list) > 1:
                raise ValueError("get_value within K assumes tbks_list is "
                                 + "length one.")
            tbks_entry = self.qcis.tbks_list[0][ibest]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            if self.qcis.verbosity >= 2:
                print('mask =')
                print(mask)

            mask_slices = []
            slices = tbks_entry.shells
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        k_final_list = []
        for sc_ind in range(len(self.qcis.fcs.sc_list_sorted)):
            ell_set = self.qcis.fcs.sc_list[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError("only length-one ell_set currently "
                                 + "supported in K")
            ell = ell_set[0]
            pcotdelta_parameter_list = pcotdelta_parameter_lists[sc_ind]
            pcotdelta_function = self.qcis.fcs.sc_list[
                sc_ind].p_cot_deltas[0]
            for slice_index in range(len(slices)):
                k_tmp = self.get_shell(E, L, m1, m2, m3,
                                       cindex,  # only for non-zero P
                                       sc_ind, ell,
                                       pcotdelta_function,
                                       pcotdelta_parameter_list,
                                       tbks_entry,
                                       slice_index, project, irrep)
                if len(k_tmp) != 0:
                    k_final_list = k_final_list+[k_tmp]
        return block_diag(*k_final_list)


class QC:
    """
    Class for the quantization condition.

    Warning: It is up to the user to select values of alphaKSS and C1cut that
    lead to a sufficient estimate of the F matrix.

    :param qcis: quantization-condition index space, specifying all data for
        the class
    :type qcis: QCIndexSpace
    :param alphaKSS: damping factor entering the zeta functions
    :type alphaKSS: float
    :param C1cut: hard cutoff used in the zeta functions
    :type C1cut: int
    """

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0):
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.fplusg = FplusG(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.k = K(qcis=self.qcis)

    def get_value(self, E=None, L=None, k_params=None, project=True,
                  irrep=None, version='kdf_zero_1+',
                  rescale=1.0):
        r"""
        Get value.

        version is drawn from the following:
            'kdf_zero_1+' (defaul)
            'f3'
            'kdf_zero_k2_inv'
            'kdf_zero_f+g_inv'
        """
        if E is None:
            raise TypeError("missing required argument 'E' (float)")
        if L is None:
            raise TypeError("missing required argument 'L' (float)")
        if k_params is None:
            raise TypeError("missing required argument 'k_params'")
        if irrep is None:
            raise TypeError("missing required argument 'irrep'")

        if version == 'f3':
            [pcotdelta_parameter_lists, k3_params] = k_params
            F = self.f.get_value(E, L, project, irrep)/rescale
            G = self.g.get_value(E, L, project, irrep)/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale
            return (F/3 - F @ np.linalg.inv(np.linalg.inv(K)+F+G) @ F)

        if (len(version) >= 8) and (version[:8] == 'kdf_zero'):
            [pcotdelta_parameter_lists, k3_params] = k_params
            if version == 'kdf_zero_1+_fgcombo':
                FplusG = self.fplusg.get_value(E, L, project, irrep)/rescale
                K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                     project, irrep)*rescale
                ident_tmp = np.identity(len(FplusG))
                return np.linalg.det(ident_tmp+(FplusG)@K)

            F = self.f.get_value(E, L, project, irrep)/rescale
            G = self.g.get_value(E, L, project, irrep)/rescale
            K = self.k.get_value(E, L, pcotdelta_parameter_lists,
                                 project, irrep)*rescale

            if version == 'kdf_zero_1+':
                ident_tmp = np.identity(len(G))
                return np.linalg.det(ident_tmp+(F+G)@K)

            if version == 'kdf_zero_k2_inv':
                return np.linalg.det(np.linalg.inv(K)+(F+G))

            if version == 'kdf_zero_f+g_inv':
                return np.linalg.det(np.linalg.inv(F+G)+K)

    def get_qc_curve(self, Emin=None, Emax=None, Estep=None, L=None,
                     k_params=None, project=True, irrep=None,
                     version='kdf_zero_1+_fgcombo', rescale=1.0):
        E_values = np.arange(Emin, Emax, Estep)
        qc_values = []
        for E in E_values:
            qc_value = self.get_value(E=E, L=L, k_params=k_params,
                                      project=project, irrep=irrep,
                                      version=version, rescale=1.0)
            qc_values = qc_values+[qc_value]
        qc_values = np.array(qc_values)
        return E_values, qc_values

    def get_root(self, Elower=None, Eupper=None, L=None,
                 k_params=None, project=True, irrep=None,
                 version='kdf_zero_1+_fgcombo', rescale=1.0):
        root = root_scalar(self.get_value,
                           args=(L, k_params, project, irrep, version,
                                 rescale),
                           bracket=[Elower, Eupper]).root
        return root

    def search_range(self, Emin=None, Emax=None, Estep=None, cutoff=None,
                     L=None, k_params=None, project=True, irrep=None,
                     version='kdf_zero_1+_fgcombo', rescale=1.0):
        E_values = []
        qc_values = []
        for E in np.arange(Emin, Emax, Estep):
            qc = self.get_value(E, L, k_params, project, irrep, version,
                                rescale)
            if np.abs(qc) < cutoff:
                E_values = E_values+[E]
                qc_values = qc_values+[qc]
        E_values = np.array(E_values)
        qc_values = np.array(qc_values)
        return E_values, qc_values

    def get_energy_curve(self, Lstart=None, Lfinish=None, deltaL=None,
                         Estart=None, Eshift=None, dE=None,
                         k_params=None, project=True, irrep=None,
                         version='kdf_zero_1+_fgcombo', rescale=1.0):
        L_vals = np.arange(Lstart, Lfinish, deltaL)
        Estart = Estart+Eshift
        Evalsfinal = []
        Lvalsfinal = []
        for L_val in L_vals:
            try:
                rs = root_scalar(self.get_value,
                                 args=(L_val,
                                       k_params,
                                       project,
                                       irrep,
                                       version,
                                       rescale),
                                 bracket=[Estart-dE, Estart+dE]).root
                dE = np.abs(rs-Estart)*5
                Evalsfinal = Evalsfinal+[rs]
                Lvalsfinal = Lvalsfinal+[L_val]
                Estart = rs
            except ValueError:
                return np.array(Lvalsfinal), np.array(Evalsfinal)
        return np.array(Lvalsfinal), np.array(Evalsfinal)