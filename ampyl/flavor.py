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

