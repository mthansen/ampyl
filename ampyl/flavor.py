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
from . import flavor_utils
from . import flavor_ope_utils
from . import qc_functions
from .constants import bcolors
import warnings
warnings.simplefilter("once")


class Particle:
    """Represent a single particle and its quantum numbers."""

    def __init__(self, mass=1., spin=0., flavor='pi',
                 isospin_multiplet=False, isospin=None, verbosity=0):
        """Initialize a particle.

        Parameters
        ----------
        mass : float, optional
            Particle mass.
        spin : float, optional
            Particle spin.
        flavor : str, optional
            Particle flavor label.
        isospin_multiplet : bool, optional
            Whether the particle belongs to an isospin multiplet.
        isospin : float, optional
            Particle isospin.
        verbosity : int, optional
            Verbosity level for initialization output.
        """

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
        flavor_utils._check_type(mass, 'mass', float, 'float')
        self._mass = mass

    @property
    def spin(self):
        """Spin of the particle."""
        return self._spin

    @spin.setter
    def spin(self, spin):
        """Set the spin of the particle."""
        flavor_utils._check_type(spin, 'spin', float, 'float')
        self._spin = spin

    @property
    def flavor(self):
        """Flavor of the particle."""
        return self._flavor

    @flavor.setter
    def flavor(self, flavor):
        """Set the flavor of the particle."""
        flavor_utils._check_type(flavor, 'flavor', str, 'str')
        self._flavor = flavor

    @property
    def isospin_multiplet(self):
        """Whether the particle is an isospin multiplet."""
        return self._isospin_multiplet

    @isospin_multiplet.setter
    def isospin_multiplet(self, isospin_multiplet):
        """Set whether the particle is an isospin multiplet."""
        flavor_utils._check_type(isospin_multiplet, 'isospin_multiplet',
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
        flavor_utils._check_type(isospin, 'isospin', float, 'float')

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
        flavor_utils._check_type(verbosity, 'verbosity', int, 'int')
        self._verbosity = verbosity

    def __eq__(self, other):
        """Return whether two particles are equivalent."""
        if not isinstance(other, Particle):
            return False
        return flavor_utils._particles_equal(self, other)

    def print_summary(self):
        """Print a formatted summary of the particle."""
        print(f"{bcolors.OKGREEN}Particle initialized with the following "
              "properties:\n"
              f"{self.__str__()}"
              f"{bcolors.ENDC}")

    def __str__(self):
        """Return a readable string representation of the particle."""
        return flavor_utils._particle_to_string(self)


class FlavorChannel:
    """Represent a flavor channel built from a set of particles."""

    def __init__(self, n_particles, particles=[], isospin_channel=False,
                 isospin=None, verbosity=0):
        """Initialize a flavor channel.

        Parameters
        ----------
        n_particles : int
            Number of particles in the channel.
        particles : list[Particle], optional
            Particles in the channel. When empty, default particles are used.
        isospin_channel : bool, optional
            Whether the channel is treated as an isospin channel.
        isospin : float, optional
            Total isospin assigned to the channel.
        verbosity : int, optional
            Verbosity level for initialization output.
        """
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
            = flavor_utils._get_allowed_total_isospins(self)
        self.isospin = self._isospin  # check if value is valid
        self.allowed_sub_isospins\
            = flavor_utils._get_allowed_sub_isospins(self)

        if self.verbosity >= 2:
            self.print_summary()

    def _get_masses(self):
        """Return the particle masses in channel order."""
        return [particle.mass for particle in self.particles]

    def _get_spins(self):
        """Return the particle spins in channel order."""
        return [particle.spin for particle in self.particles]

    def _get_flavors(self):
        """Return the particle flavors in channel order."""
        return [particle.flavor for particle in self.particles]

    def _get_isospins(self):
        """Return the particle isospins in channel order."""
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
        self.check_particles(particles)
        self._particles = particles

    def check_particles(self, particles):
        """Validate particle content against channel constraints."""
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

    @property
    def isospin_channel(self):
        """Whether the channel is an isospin channel."""
        return self._isospin_channel

    @isospin_channel.setter
    def isospin_channel(self, isospin_channel):
        """Set whether the channel is an isospin channel."""
        flavor_utils._check_type(isospin_channel, 'isospin_channel',
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
        flavor_utils._check_type(isospin, "isospin", float, "float")
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
        flavor_utils._check_type(n_particles, "n_particles", int, "int")
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
        flavor_utils._check_type(verbosity, "verbosity", int, "int")
        self._verbosity = verbosity

    def print_summary(self):
        """Print a formatted summary of the flavor channel."""
        print(f"{bcolors.OKGREEN}FlavorChannel initialized with the following "
              "properties:\n"
              f"{self.__str__()}"
              f"{bcolors.ENDC}")

    def __str__(self):
        """Return a readable string representation of the flavor channel."""
        return flavor_utils._flavor_channel_to_string(self)


class SpectatorChannel:
    """Represent a spectator-channel view of a flavor channel."""

    def __init__(self, fc=FlavorChannel(3), indexing=[0, 1, 2],
                 sub_isospin=None, ell_set=[0], p_cot_deltas=None):
        """Initialize a spectator channel.

        Parameters
        ----------
        fc : FlavorChannel, optional
            Flavor channel used to define the spectator channel.
        indexing : list[int], optional
            Particle ordering with the spectator listed first.
        sub_isospin : float, optional
            Two-particle subchannel isospin.
        ell_set : list[int], optional
            Partial waves included in the channel.
        p_cot_deltas : list[callable], optional
            Functions defining the two-body interaction input.
        """

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

        self._set_cutoff_scheme_data()

        if p_cot_deltas is None:
            p_cot_deltas = []
            for _ in range(len(ell_set)):
                p_cot_deltas.append(qc_functions.pcotdelta_scattering_length)
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
        flavor_utils._check_type(fc, "fc", FlavorChannel, "FlavorChannel")
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
                        qc_functions.pcotdelta_scattering_length)
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
        """Set the p-cot-delta functions for the spectator channel."""

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
        """Populate the allowed sub-isospin values from the parent channel."""
        if self._fc.isospin_channel and self._fc.n_particles > 2:
            allowed_sub_isospins = []
            for entry in self._fc.summary_reduced:
                if self._fc.flavors[self.indexing[0]] == entry[2]:
                    allowed_sub_isospins.append(entry[1])
            self.allowed_sub_isospins = allowed_sub_isospins

    def _set_cutoff_scheme_data(self):
        """Set default pole-cutoff data for this spectator channel."""
        if self.fc.n_particles != 3:
            self.thresholdSQ = None
            self.ESQmin = None
            self.ESQMIN = None
            self.alpha = None
            self.beta = None
            self.scheme_data = None
            return
        mspec = self.masses_indexed[0]
        m1 = self.masses_indexed[1]
        m2 = self.masses_indexed[2]
        m_max = max(m1, m2)
        m_min = min(m1, m2)
        self.thresholdSQ = (m1+m2)**2
        if np.isclose(m1, m2) and mspec < m1:
            self.ESQmin = 4.0*(m1**2-mspec**2)
            self.alpha = 3.0-4.0*mspec**2/m1**2
        else:
            self.ESQmin = m_max**2 - m_min**2
            self.alpha = (3.*m_max - 5.*m_min) / (m_max + m_min)
        self.ESQMIN = self.ESQmin
        self.beta = 0.
        self.scheme_data = [self.alpha, self.beta]

    @property
    def n_params_set(self):
        """Parameter counts of the spectator channel p-cot-deltas."""
        return self._n_params_set

    def __eq__(self, other):
        """Return whether two spectator channels are equivalent."""
        if not isinstance(other, SpectatorChannel):
            return False
        return flavor_utils._spectator_channel_equality(self, other)

    def __str__(self):
        """Return a readable string representation of the spectator channel."""
        return flavor_utils._spectator_channel_to_string(self)


class FlavorChannelSpace:
    """Represent a collection of flavor and spectator channels."""

    def __init__(self, fc_list=[], ni_list=None, verbosity=0):
        """Initialize a flavor-channel space.

        Parameters
        ----------
        fc_list : list[FlavorChannel], optional
            Flavor channels included in the space.
        ni_list : list[FlavorChannel], optional
            Noninteracting channels associated with the space.
        verbosity : int, optional
            Verbosity level for initialization output.
        """
        self.verbosity = verbosity
        self.set_fc_and_ni_lists(fc_list, ni_list)
        flavor_utils._build_sorted_sc_list(self)
        flavor_ope_utils._build_g_templates(self)
        flavor_ope_utils._build_g_templates_ell_specific(self)
        if self.verbosity >= 2:
            self.print_summary()

    @property
    def verbosity(self):
        """Verbosity of the channel space."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the channel space."""
        flavor_utils._check_type(verbosity, "verbosity", int, "int")
        self._verbosity = verbosity

    def set_fc_and_ni_lists(self, fc_list, ni_list):
        """Set the interacting and noninteracting channel lists."""
        self.fc_list = fc_list
        if ni_list is None:
            self.ni_list = fc_list
        else:
            self.ni_list = ni_list
        self.sc_list = []
        for fc in fc_list:
            self.add_flavor_channel(fc)

    def update_g_templates(self):
        """Update the g templates."""
        flavor_ope_utils._build_g_templates(self)
        flavor_ope_utils._build_g_templates_ell_specific(self)
        if self.verbosity >= 2:
            self.print_summary()

    def add_spectator_channel(self, sc):
        """Append a spectator channel to the channel space."""
        self.sc_list.append(sc)

    def add_flavor_channel(self, fc):
        """Expand a flavor channel into spectator channels and add them."""
        if fc.n_particles == 2:
            sc1 = SpectatorChannel(fc, indexing=None)
            self.add_spectator_channel(sc1)
        elif fc.isospin_channel:
            for entry in fc.summary_reduced:
                indexing, sub_isospin, ell_set\
                    = flavor_utils._parse_three_iso_fc_entry(entry, fc)
                sc_tmp = SpectatorChannel(fc, indexing=indexing,
                                          sub_isospin=sub_isospin,
                                          ell_set=ell_set)
                self.add_spectator_channel(sc_tmp)
        else:
            if fc.flavors[0] == fc.flavors[1] == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                self.add_spectator_channel(sc1)
            elif fc.flavors[0] == fc.flavors[1]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[2, 0, 1])
                self.add_spectator_channel(sc1)
                self.add_spectator_channel(sc2)
            elif fc.flavors[0] == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[1, 2, 0])
                self.add_spectator_channel(sc1)
                self.add_spectator_channel(sc2)
            elif fc.flavors[1] == fc.flavors[2]:
                sc1 = SpectatorChannel(fc)
                sc2 = SpectatorChannel(fc, indexing=[1, 2, 0])
                self.add_spectator_channel(sc1)
                self.add_spectator_channel(sc2)
            else:
                sc1 = SpectatorChannel(fc=fc)
                sc2 = SpectatorChannel(fc=fc, indexing=[1, 2, 0])
                sc3 = SpectatorChannel(fc=fc, indexing=[2, 0, 1])
                self.add_spectator_channel(sc1)
                self.add_spectator_channel(sc2)
                self.add_spectator_channel(sc3)

    def print_summary(self):
        """Print a formatted summary of the flavor-channel space."""
        fcs_summary = flavor_utils._generate_flavor_channel_space_summary(self)
        print(f"{bcolors.OKGREEN}{fcs_summary}{bcolors.ENDC}")

    def __str__(self):
        """Return a readable string representation of the channel space."""
        return flavor_utils._flavor_channel_space_to_string(self)
