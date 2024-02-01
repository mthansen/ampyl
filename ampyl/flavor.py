#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# flavor.py
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
from inspect import signature
from copy import deepcopy
from .functions import QCFunctions
from .constants import EPSILON4
from .constants import G_TEMPLATE_DICT
from .constants import bcolors
import warnings
warnings.simplefilter("once")


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
    (default is ``False``)
    :type isospin_multiplet: bool
    :param isospin: isospin of the particle (default is ``None``)
    :type isospin: float

    :raises ValueError: If `isospin_multiplet` is ``True`` but `isospin` is
        ``None``.
    """
    def __init__(self, mass=1., spin=0., flavor='pi',
                 isospin_multiplet=False, isospin=None, verbosity=0):
        if not isospin_multiplet and isospin is not None:
            isospin_multiplet = True
        if isospin_multiplet and isospin is None:
            raise ValueError("isospin cannot be None when isospin_multiplet "
                             "is True")

        self._mass = mass
        self._spin = spin
        self._flavor = flavor
        self._isospin_multiplet = isospin_multiplet
        self._isospin = isospin
        self._verbosity = verbosity

        self.mass = self._mass
        self.spin = self._spin
        self.flavor = self._flavor
        self.isospin_multiplet = self._isospin_multiplet
        self.isospin = self._isospin
        self.verbosity = self._verbosity

        if self._verbosity >= 2:
            self.print_summary()

    def print_summary(self):
        print(f"{bcolors.OKGREEN}Particle initialized with the following "
              "properties:\n"
              f"    mass: {self.mass}\n"
              f"    spin: {self.spin}\n"
              f"    flavor: {self.flavor}\n"
              f"    isospin_multiplet: {self.isospin_multiplet}")
        if self.isospin_multiplet:
            print(f"    isospin: {self.isospin}")
        print(f"{bcolors.ENDC}")

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

    @property
    def verbosity(self):
        """Verbosity of the particle."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the particle."""
        self._check_type(verbosity, 'verbosity', int, 'int')
        self._verbosity = verbosity

    def __eq__(self, other):
        """Check if two Particle objects are equivalent."""
        if not isinstance(other, Particle):
            return False
        return (self.mass == other.mass and
                self.spin == other.spin and
                self.flavor == other.flavor and
                self.isospin_multiplet == other.isospin_multiplet and
                self.isospin == other.isospin)

    def __str__(self):
        """Return a string representation of the Particle object."""
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

    :param n_particles: number of particles in the flavor channel
    :type n_particles: int
    :param particles: particles in the flavor channel. If not specified,
        the channel will be initialized with `n_particles` default Particle
        objects.
    :type particles: list of :class:`Particle` objects, optional
    :param isospin_channel: specifies whether this is an isospin channel
    (Default is ``False``)
    :type isospin_channel: bool, optional
    :param isospin: isospin of the flavor channel (Default is ``None``)
    :type isospin: float, optional

    :ivar masses: masses of the particles in the channel
    :vartype masses: list of floats, set automatically
    :ivar spins: spins of the particles in the channel
    :vartype spins: list of floats, set automatically
    :ivar flavors: flavors of the particles in the channel
    :vartype flavors: list of strings, set automatically
    :ivar isospins: isospins of the particles in the channel
    :vartype isospins: list of floats, set automatically
    :ivar allowed_total_isospins: allowed total isospins for the flavor channel
    :vartype allowed_total_isospins: list of floats, set automatically
    :ivar summary: summary of the flavor channel
    :vartype summary: list, set automatically
    :ivar summary_reduced: reduced summary of the flavor channel
    :vartype summary_reduced: list, set automatically

    :raises ValueError: If `n_particles` is not an int or if `n_particles` is
        less than 2.

    :raises ValueError: If `isospin_channel` is ``True`` but `isospin` is
        ``None``.

    .. note::
        If `particles` is not specified, the channel will be initialized with
        `n_particles` default :class:`Particle` objects.

    :Example:

    >>> import ampyl
    >>> pion = ampyl.Particle(mass=1., spin=0., flavor='pion', isospin=1.)
    >>> fc = ampyl.FlavorChannel(3, particles=[pion, pion, pion],
    ...                          isospin_channel=True, isospin=3.)
    >>> print(fc)
    FlavorChannel with the following details:
        3 particles,
        masses: [1.0, 1.0, 1.0],
        spins: [0.0, 0.0, 0.0],
        flavors: ['pion', 'pion', 'pion'],
        isospin_channel: True,
        isospins: [1.0, 1.0, 1.0],
        allowed_total_isospins: [0.0, 1.0, 2.0, 3.0],
        isospin: 3.0.

    """

    def __init__(self, n_particles, particles=[], isospin_channel=False,
                 isospin=None, verbosity=0):
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
            raise ValueError("isospin cannot be None when isospin_channel "
                             "is True")

        self._particles = particles
        self._isospin_channel = isospin_channel
        self._isospin = isospin
        self._verbosity = verbosity

        self.particles = self._particles
        self.masses = self._get_masses()
        self.spins = self._get_spins()
        self.flavors = self._get_flavors()
        self.isospins = self._get_isospins()

        self.allowed_total_isospins = self._get_allowed_total_isospins()
        self.isospin_channel = self._isospin_channel
        self.isospin = self._isospin
        self.verbosity = self._verbosity
        self.n_particles = self._n_particles

        if self.verbosity >= 2:
            self.print_summary()

    def print_summary(self):
        print(f"{bcolors.OKGREEN}FlavorChannel initialized with the "
              "following properties:\n"
              f"    n_particles: {self.n_particles}\n"
              f"    masses: {self.masses}\n"
              f"    spins: {self.spins}\n"
              f"    flavors: {self.flavors}\n"
              f"    isospin_channel: {self.isospin_channel}\n")
        if self.isospin_channel:
            print(f"    isospins: {self.isospins}\n"
                  f"    allowed_total_isospins:\n"
                  f"{self.allowed_total_isospins}\n"
                  f"    isospin: {self.isospin}\n")
        print(f"{bcolors.ENDC}")

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
        for particle in particles:
            if particle.isospin_multiplet and not self._isospin_channel:
                raise ValueError("none of the particles can be an isospin "
                                 "multiplet if the channel is not an "
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

    @property
    def verbosity(self):
        """Verbosity of the channel."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the channel."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

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

    :param fc: FlavorChannel object used to define the spectator channel
    :type fc: :class:`FlavorChannel` object
    :param indexing: indices of the particles in the FlavorChannel. The first
        entry corresponds to the spectator particle.
    :type indexing: list of ints
    :param sub_isospin: isospin value of the two-particle sub-channel
    :type sub_isospin: float, optional
    :param ell_set: angular momentum values of the channel
    :type ell_set: list of ints, optional
    :param p_cot_deltas: p_cot_delta functions of the channel
    :type p_cot_deltas: list of callables, optional

    :ivar masses_indexed: masses of the particles in the channel with the
        spectator first
    :vartype masses_indexed: list of floats
    :ivar spins_indexed: spins of the particles in the channel with the
        spectator first
    :vartype spins_indexed: list of floats
    :ivar flavors_indexed: flavors of the particles in the channel with the
        spectator first
    :vartype flavors_indexed: list of strings
    :ivar isospins_indexed: isospins of the particles in the channel with the
        spectator first
    :vartype isospins_indexed: list of floats
    :ivar allowed_sub_isospins: allowed sub-channel isospins
    :vartype allowed_sub_isospins: list of floats
    :ivar n_params_set: parameter counts for the channel p_cot_delta functions
    :vartype n_params_set: list of ints

    :raises ValueError: If the `fc` parameter is not a `FlavorChannel` object.

    .. note::
        If `p_cot_deltas` is not specified, it will be set to
        :attr:`QCFunctions.pcotdelta_scattering_length`.

    :Example:

    >>> fc = FlavorChannel(3)
    >>> sc = SpectatorChannel(fc=fc, indexing=[0, 1, 2])

    """

    def __init__(self, fc=FlavorChannel(3), indexing=[0, 1, 2],
                 sub_isospin=None, ell_set=[0], p_cot_deltas=None):

        self.allowed_sub_isospins = None

        self._fc = fc
        self._indexing = indexing
        self._sub_isospin = sub_isospin
        self._ell_set = ell_set
        self._p_cot_deltas = p_cot_deltas

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
        """
        Set the p-cot-delta functions of the spectator channel.

        :param p_cot_deltas: p-cot-delta functions to set
        :type p_cot_deltas: list of callables

        .. note::
            If `p_cot_deltas` is not specified, sets :attr:`ell_set`,
            :attr:`p_cot_deltas`, and :attr:`n_params_set` to None.

        .. warning::
            The number of elements in `p_cot_deltas` must be equal to the
            length of :attr:`ell_set`.

        :raises ValueError: If the length of `p_cot_deltas` is less than the
            length of :attr:`ell_set`.

        """

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
        """Check if two SpectatorChannel objects are equivalent."""
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
              f"    g_templates:\n{self.g_templates}\n"
              f"    g_templates_ell_specific:\n"
              "        Key is built from four entries:\n"
              "        [slice_index_i,  slice_index_j, ell_i, ell_j]"
              "        Entry is built from five entries:\n"
              "         [np.array([[g_template_ij[sc_index_i][sc_index_j]]],\n"
              "sc_index_i, sc_index_j,"
              "collective_index_i,"
              "collective_index_j]")
        for g_temp_key in self.g_templates_ell_specific:
            print(f"        ell = {g_temp_key}:\n"
                  f"        {self.g_templates_ell_specific[g_temp_key]}")
        print(f"{bcolors.ENDC}")

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
