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
from copy import deepcopy
from inspect import signature
from . import flavor_utils
from .functions import QCFunctions
from .constants import bcolors
from .constants import G_TEMPLATE_DICT
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
    :param isospin_multiplet: specifies whether this is an isospin multiplet\
        (default is ``False``)
    :type isospin_multiplet: bool
    :param isospin: isospin of the particle (default is ``None``)
    :type isospin: float

    :raises ValueError: If `isospin_multiplet` is ``True`` but `isospin` is
        ``None``.
    """
    def __init__(self, mass=1., spin=0., flavor='pi',
                 isospin_multiplet=False, isospin=None, verbosity=0):

        self.mass = mass
        self.spin = spin
        self.flavor = flavor

        self._isospin_multiplet = isospin_multiplet
        self._isospin = isospin
        self.isospin_multiplet = self._isospin_multiplet
        self.isospin = self._isospin

        self.verbosity = verbosity
        if self._verbosity >= 2:
            self.print_summary()

    @property
    def mass(self):
        """Mass of the particle."""
        return self._mass

    @mass.setter
    def mass(self, mass):
        """Set the mass of the particle."""
        flavor_utils.check_type(mass, 'mass', float, 'float')
        self._mass = mass

    @property
    def spin(self):
        """Spin of the particle."""
        return self._spin

    @spin.setter
    def spin(self, spin):
        """Set the spin of the particle."""
        flavor_utils.check_type(spin, 'spin', float, 'float')
        self._spin = spin

    @property
    def flavor(self):
        """Flavor of the particle."""
        return self._flavor

    @flavor.setter
    def flavor(self, flavor):
        """Set the flavor of the particle."""
        flavor_utils.check_type(flavor, 'flavor', str, 'str')
        self._flavor = flavor

    @property
    def isospin_multiplet(self):
        """Whether the particle is an isospin multiplet."""
        return self._isospin_multiplet

    @isospin_multiplet.setter
    def isospin_multiplet(self, isospin_multiplet):
        """Set whether the particle is an isospin multiplet."""
        flavor_utils.check_type(isospin_multiplet, 'isospin_multiplet',
                                bool, 'bool')

        if not isospin_multiplet and self._isospin is not None:
            isospin_multiplet = True
        if isospin_multiplet and self._isospin is None:
            raise ValueError("isospin cannot be None when isospin_multiplet "
                             "is True")

        self._isospin_multiplet = isospin_multiplet

    @property
    def isospin(self):
        """Isospin of the particle."""
        return self._isospin

    @isospin.setter
    def isospin(self, isospin):
        """Set the isospin of the particle."""
        flavor_utils.check_type(isospin, 'isospin', float, 'float')

        if not self._isospin_multiplet and isospin is not None:
            self._isospin_multiplet = True
        if isospin is None and self._isospin_multiplet:
            raise ValueError("isospin cannot be None when isospin_multiplet "
                             "is True")

        self._isospin = isospin

    @property
    def verbosity(self):
        """Verbosity of the particle."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the particle."""
        flavor_utils.check_type(verbosity, 'verbosity', int, 'int')
        self._verbosity = verbosity

    def __eq__(self, other):
        if not isinstance(other, Particle):
            return False
        return flavor_utils.particles_equal(self, other)

    def print_summary(self):
        print(f"{bcolors.OKGREEN}Particle initialized with the following "
              "properties:\n"
              f"{self.__str__()}"
              f"{bcolors.ENDC}")

    def __str__(self):
        return flavor_utils.particle_to_string(self)


class FlavorChannel:
    """
    Class used to represent a flavor channel.

    :param n_particles: number of particles in the flavor channel
    :type n_particles: int
    :param particles: particles in the flavor channel. If not specified,\
        the channel will be initialized with `n_particles` default Particle\
        objects.
    :type particles: list of :class:`Particle` objects, optional
    :param isospin_channel: specifies whether this is an isospin channel\
        (Default is ``False``)
    :type isospin_channel: bool, optional
    :param isospin: isospin of the flavor channel (Default is ``None``)
    :type isospin: float, optional

    :raises ValueError: If `n_particles` is not an int or if `n_particles` is
        less than 2.

    :raises ValueError: If `isospin_channel` is ``True`` but `isospin` is
        ``None``.

    .. note::
        Contains particle properties (masses, spins, flavors, isospins) and the
        derived lists of allowed total/sub-isospins plus channel summaries. All
        fields are set automatically.

        If `particles` is not specified, the channel will be initialized with
        `n_particles` default :class:`Particle` objects.

    Example:

    >>> import ampyl
    >>> pion = ampyl.flavor.Particle(isospin=1.)
    >>> fc = ampyl.flavor.FlavorChannel(3, particles=[pion, pion, pion],
    ...                                  isospin_channel=True, isospin=3.)
    >>> print(fc)
    FlavorChannel with the following details:
        3 particles,
        masses: [1.0, 1.0, 1.0],
        spins: [0.0, 0.0, 0.0],
        flavors: ['pi', 'pi', 'pi'],
        isospin_channel: True,
        isospins: [1.0, 1.0, 1.0],
        allowed_total_isospins: 0.0, 1.0, 2.0, 3.0,
        isospin: 3.0.

    """

    def __init__(self, n_particles, particles=[], isospin_channel=False,
                 isospin=None, verbosity=0):
        self.verbosity = verbosity

        self.n_particles = n_particles
        self._isospin = isospin
        # have not yet checked if isospin value is valid
        self.isospin_channel = isospin_channel
        self.particles = particles

        self.masses = self._get_masses()
        self.spins = self._get_spins()
        self.flavors = self._get_flavors()
        self.isospins = self._get_isospins()

        self.allowed_total_isospins\
            = flavor_utils.get_allowed_total_isospins(self)
        self.isospin = self._isospin  # check if value is valid
        self.allowed_sub_isospins\
            = flavor_utils.get_allowed_sub_isospins(self)

        if self.verbosity >= 2:
            self.print_summary()

    def _get_masses(self):
        return [particle.mass for particle in self.particles]

    def _get_spins(self):
        return [particle.spin for particle in self.particles]

    def _get_flavors(self):
        return [particle.flavor for particle in self.particles]

    def _get_isospins(self):
        return [particle.isospin for particle in self.particles]

    @property
    def particles(self):
        """Particles in the channel."""
        return self._particles

    @particles.setter
    def particles(self, particles):
        """Set the particles in the channel."""
        if not isinstance(particles, list):
            raise ValueError("particles must be a list")
        if len(particles) == 0:
            particles = [Particle() for _ in range(self.n_particles)]
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
        flavor_utils.check_type(isospin_channel, 'isospin_channel',
                                bool, 'bool')
        if not isospin_channel and self._isospin is not None:
            isospin_channel = True
        if isospin_channel and self._isospin is None:
            raise ValueError("isospin cannot be None when isospin_channel "
                             "is True")
        self._isospin_channel = isospin_channel

    @property
    def isospin(self):
        """Isospin value of the channel."""
        return self._isospin

    @isospin.setter
    def isospin(self, isospin):
        """Set the isospin value of the channel."""
        flavor_utils.check_type(isospin, "isospin", float, "float")
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
        flavor_utils.check_type(n_particles, "n_particles", int, "int")
        if n_particles < 2:
            raise ValueError("n_particles must be >= 2")
        self._n_particles = n_particles

    @property
    def verbosity(self):
        """Verbosity of the channel."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the channel."""
        flavor_utils.check_type(verbosity, "verbosity", int, "int")
        self._verbosity = verbosity

    def print_summary(self):
        print(f"{bcolors.OKGREEN}FlavorChannel initialized with the following "
              "properties:\n"
              f"{self.__str__()}"
              f"{bcolors.ENDC}")

    def __str__(self):
        return flavor_utils.flavor_channel_to_str(self)


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

        self.set_allowed_sub_isospins()

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
        flavor_utils.check_type(fc, "fc", FlavorChannel, "FlavorChannel")
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
            raise ValueError("for n_particles > 2, sub_isospin must be a "
                             "float")
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
                for _ in range(len(ell_set)-len(self.p_cot_deltas)):
                    self._p_cot_deltas.append(self._p_cot_deltas[-1])
            elif len(self._p_cot_deltas) < len(ell_set):
                for _ in range(len(ell_set)-len(self.p_cot_deltas)):
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

    def set_allowed_sub_isospins(self):
        if self._fc.isospin_channel and self._fc.n_particles > 2:
            allowed_sub_isospins = []
            for entry in self._fc.summary_reduced:
                if self._fc.flavors[self.indexing[0]] == entry[2]:
                    allowed_sub_isospins.append(entry[1])
            self.allowed_sub_isospins = allowed_sub_isospins

    @property
    def n_params_set(self):
        """Parameter counts of the spectator channel p-cot-deltas."""
        return self._n_params_set

    def __eq__(self, other):
        if not isinstance(other, SpectatorChannel):
            return False
        return flavor_utils.spectator_channel_equality(self, other)

    def __str__(self):
        return flavor_utils.spectator_channel_to_string(self)


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
