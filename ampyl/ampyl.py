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
import warnings
import functools
from copy import deepcopy
from .group_theory import Groups
from .group_theory import Irreps
from .qc_functions import QCFunctions
from .qc_functions import BKFunctions
warnings.simplefilter('always')

PI = np.pi
TWOPI = 2.*PI
FOURPI2 = 4.0*PI**2
EPSILON = 1.0e-20
PRINT_THRESHOLD_DEFAULT = np.get_printoptions()['threshold']

G_TEMPLATE_DICT = {}
G_TEMPLATE_DICT[0] = np.array([[-1.]])
G_TEMPLATE_DICT[1] = np.array([[1./3., -1./np.sqrt(3.), np.sqrt(5.)/3.],
                               [-1./np.sqrt(3.), 0.5, np.sqrt(15.)/6.],
                               [np.sqrt(5.)/3., np.sqrt(15.)/6., 1./6.]])
G_TEMPLATE_DICT[2] = np.array([[0.5, -np.sqrt(3.)/2.],
                               [-np.sqrt(3.)/2., -0.5]])
G_TEMPLATE_DICT[3] = np.array([[1.]])


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


class FlavorChannel:
    """
    Class used to represent a particular flavor channel.

    Attributes
    ----------
    n_particles : int
        number of particles in the channel.
    masses : list of floats
        mass of each particle in the channel.
    spins : list of ints
        twice the spin of each particle in the channel.
    explicit_flavor_channel : boolean
        specifies whether this is an explicit-flavor channel (as opposed to an
        isospin channel).
    explicit_flavors : list of ints
        explicit flavor of each particle (is None if explicit_flavor_channel
        is False).
    isospin_channel : boolean
        specifies whether this is an isospin channel (as opposed to an
        explicit-flavor channel).
    isospin_value : int
        total isospin of the channel (is None if isospin_channel is False).
    isospin_flavor : int
        extra flavor label that is only relevant in the case of multiple
        isospin channels (is None if isospin_channel is False).
    """

    def __init__(self, n_particles, masses=None, spins=None,
                 explicit_flavor_channel=True, explicit_flavors=None,
                 isospin_channel=False, isospin_value=None,
                 isospin_flavor=None):

        if not isinstance(n_particles, int):
            raise ValueError('n_particles must be an int')
        if n_particles < 2:
            raise ValueError('n_particles must be >= 2')
        self._n_particles = n_particles

        if masses is None:
            masses = n_particles*[1.0]
        if spins is None:
            spins = n_particles*[0]
        if not isospin_channel and isospin_value is not None:
            isospin_channel = True
            explicit_flavor_channel = False
        if not isospin_channel and isospin_flavor is not None:
            isospin_channel = True
            explicit_flavor_channel = False
        if not explicit_flavor_channel and explicit_flavors is not None:
            explicit_flavor_channel = True
        if explicit_flavor_channel and explicit_flavors is None:
            explicit_flavors = n_particles*[1]
        if isospin_channel:
            explicit_flavor_channel = False
            explicit_flavors = None
            if isospin_value is None:
                isospin_value = n_particles
            if isospin_flavor is None:
                isospin_flavor = 0

        self._explicit_flavor_channel = explicit_flavor_channel
        self._explicit_flavors = explicit_flavors
        self._isospin_channel = isospin_channel
        self._isospin_value = isospin_value
        self._isospin_flavor = isospin_flavor

        self.explicit_flavor_channel = explicit_flavor_channel
        self.explicit_flavors = explicit_flavors
        self.isospin_channel = isospin_channel
        self.isospin_value = isospin_value
        self.isospin_flavor = isospin_flavor

        self.masses = masses
        self.spins = spins
        self.n_particles = n_particles

    def _generic_setter(self, var, varstr, enttype, enttypestr):
        if not isinstance(var, list):
            raise ValueError(varstr+' must be a list')
        for obj in var:
            if not isinstance(obj, enttype):
                raise ValueError(varstr+' must be populated with '
                                 + enttypestr+'s')
        if self._n_particles == len(var):
            return var
        if self._n_particles < len(var):
            warnings.warn("\n"+bcolors.WARNING
                          + "length of "+varstr+" must equal n_particles. "
                          + "Last entries will be dropped."
                          + f"{bcolors.ENDC}", stacklevel=2)
            return var[:self._n_particles]
        warnings.warn("\n"+bcolors.WARNING
                      + "length of "+varstr+" must equal n_particles. "
                      + "Last entry will be duplicated."
                      + f"{bcolors.ENDC}", stacklevel=2)
        return var+[var[-1]]*(self._n_particles-len(var))

    @property
    def explicit_flavor_channel(self):
        """Get explicit flavor channel status (boolean)."""
        return self._explicit_flavor_channel

    @explicit_flavor_channel.setter
    def explicit_flavor_channel(self, explicit_flavor_channel):
        if not isinstance(explicit_flavor_channel, bool):
            raise ValueError('explicit_flavor_channel must be a boolean')
        self._explicit_flavor_channel = explicit_flavor_channel
        if explicit_flavor_channel and self._isospin_channel:
            warnings.warn("\n"+bcolors.WARNING
                          + "explicit_flavor_channel is being set to True but "
                          + "isospin_channel is True. Setting it to False."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.isospin_channel = False
        if explicit_flavor_channel and (self._explicit_flavors is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "explicit_flavor_channel is being set to True but "
                          + "explicit_flavors is None. Setting it to default."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._explicit_flavors = (self.n_particles)*[1]
        if ((not explicit_flavor_channel)
           and (self._explicit_flavors is not None)):
            warnings.warn("\n"+bcolors.WARNING
                          + "explicit_flavor_channel is being set to False "
                          + "but explicit_flavors is not None. Setting it to "
                          + "None."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._explicit_flavors = None

    @property
    def explicit_flavors(self):
        """Get explicit flavors (list of ints)."""
        return self._explicit_flavors

    @explicit_flavors.setter
    def explicit_flavors(self, explicit_flavors):
        if explicit_flavors is None:
            self._explicit_flavors = explicit_flavors
        else:
            self._explicit_flavors = self._generic_setter(explicit_flavors,
                                                          'explicit_flavors',
                                                          int, 'int')
        if ((not self.explicit_flavor_channel)
           and (explicit_flavors is not None)):
            warnings.warn("\n"+bcolors.WARNING
                          + "explicit_flavors is being set but "
                          + "explicit_flavor_channel is False. Setting it to "
                          + "True."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.explicit_flavor_channel = True
        if self.explicit_flavor_channel and (explicit_flavors is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "explicit_flavors is being set to None but "
                          + "explicit_flavor_channel is True. Setting it to "
                          + "False."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.explicit_flavor_channel = False

    @property
    def isospin_channel(self):
        """Get isospin channel status (boolean)."""
        return self._isospin_channel

    @isospin_channel.setter
    def isospin_channel(self, isospin_channel):
        if not isinstance(isospin_channel, bool):
            raise ValueError('explicit_flavor_channel must be a boolean')
        self._isospin_channel = isospin_channel
        if isospin_channel and self._explicit_flavor_channel:
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_channel is being set to True but "
                          + "explicit_flavor_channel is True. Setting it to "
                          + "False."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.explicit_flavor_channel = False
        if isospin_channel and (self._isospin_value is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_channel is being set to True but "
                          + "isospin_value is None. Setting it to default."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._isospin_value = self.n_particles
        if isospin_channel and (self._isospin_flavor is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_channel is being set to True but "
                          + "isospin_flavor is None. Setting it to default."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._isospin_flavor = 0
        if (not isospin_channel) and (self._isospin_value is not None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_channel is being set to False but "
                          + "isospin_value is not None. Setting it to None."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._isospin_value = None
        if (not isospin_channel) and (self._isospin_flavor is not None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_channel is being set to False but "
                          + "isospin_flavor is not None. Setting it to None."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._isospin_flavor = None

    @property
    def isospin_value(self):
        """Get isospin value (int)."""
        return self._isospin_value

    @isospin_value.setter
    def isospin_value(self, isospin_value):
        if ((isospin_value is not None)
           and (not isinstance(isospin_value, int))):
            raise ValueError('isospin_value must be an int')
        if ((isospin_value is not None)
           and (isospin_value > self._n_particles)):
            raise ValueError('isospin_value should not exceed n_particles')
        self._isospin_value = isospin_value
        if (not self.isospin_channel) and (isospin_value is not None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_value is being set but isospin_channel "
                          + "is False. Setting it to True."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.isospin_channel = True
        if self.isospin_channel and (isospin_value is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_value is being set to None but "
                          + "isospin_channel is True. Setting it to False."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.isospin_channel = False
            self.isospin_flavor = None

    @property
    def isospin_flavor(self):
        """Get isospin flavor (int)."""
        return self._isospin_flavor

    @isospin_flavor.setter
    def isospin_flavor(self, isospin_flavor):
        if ((isospin_flavor is not None)
           and (not isinstance(isospin_flavor, int))):
            raise ValueError('isospin_flavor must be an int')
        self._isospin_flavor = isospin_flavor
        if (not self.isospin_channel) and (isospin_flavor is not None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_flavor is being set but isospin_channel "
                          + "is False. Setting it to True."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.isospin_channel = True
        if self.isospin_channel and (isospin_flavor is None):
            warnings.warn("\n"+bcolors.WARNING
                          + "isospin_value is being set to None but "
                          + "isospin_channel is True. Setting it to False."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self.isospin_channel = False
            self.isospin_value = None

    @property
    def masses(self):
        """Get masses (list of floats)."""
        return self._masses

    @masses.setter
    def masses(self, masses):
        self._masses = self._generic_setter(masses, 'masses', float, 'float')
        if self.isospin_channel:
            if not (np.array(self._masses) ==
                    np.array([self._masses[0]]*self._n_particles)).all():
                warnings.warn("\n"+bcolors.WARNING
                              + "masses must be equal for isospin "
                              + "channel. "
                              + "Setting to first value."
                              + f"{bcolors.ENDC}", stacklevel=2)
                self._masses = [self._masses[0]]*self._n_particles

    @property
    def spins(self):
        """Get 2xspins (list of ints)."""
        return self._spins

    @spins.setter
    def spins(self, spins):
        self._spins = self._generic_setter(spins, 'spins', int, 'int')
        if self.isospin_channel:
            if not (np.array(self._spins) ==
                    np.array([self._spins[0]]*self._n_particles)).all():
                warnings.warn("\n"+bcolors.WARNING
                              + "spins must be equal for isospin "
                              + "channel. "
                              + "Setting to first value."
                              + f"{bcolors.ENDC}", stacklevel=2)
                self._spins = [self._spins[0]]*self._n_particles

    @property
    def n_particles(self):
        """Get number of particles (int)."""
        return self._n_particles

    @n_particles.setter
    def n_particles(self, n_particles):
        if not isinstance(n_particles, int):
            raise ValueError('n_particles must be an int')
        if n_particles < 2:
            raise ValueError('n_particles must be >= 2')
        else:
            self._n_particles = n_particles
            self.masses = self._masses
            self.spins = self._spins
            self.explicit_flavors = self._explicit_flavors
            if self._isospin_channel and self._isospin_value > n_particles:
                warnings.warn("\n"+bcolors.WARNING
                              + "isospin_value should not exceed n_particles. "
                              + "Setting isospin_value to equal n_particles."
                              + f"{bcolors.ENDC}", stacklevel=2)
                self._isospin_value = n_particles

    def __str__(self):
        """Summary of the flavor channel."""
        strtmp = "FlavorChannel with the following details:\n"
        strtmp = strtmp+"    "+str(self._n_particles)+" particles,\n"
        strtmp = strtmp+"    masses: "+str(self._masses)+",\n"
        strtmp = strtmp+"    spins: "+str(self._spins)+",\n"
        strtmp = strtmp+"    explicit_flavor_channel: "\
            + str(self._explicit_flavor_channel)+",\n"
        strtmp = strtmp+"    explicit_flavors: "\
            + str(self._explicit_flavors)+",\n"
        strtmp = strtmp+"    isospin_channel: "\
            + str(self._isospin_channel)+",\n"
        strtmp = strtmp+"    isospin_value: "+str(self._isospin_value)+",\n"
        strtmp = strtmp+"    isospin_flavor: "+str(self._isospin_flavor)+",\n"
        return strtmp[:-2]+"."


class _SpectatorChannel:

    def __init__(self, fc=FlavorChannel(3),
                 indexing=[0, 1, 2],
                 sub_isospin=None,
                 ell_set=[0],
                 p_cot_deltas=[QCFunctions.pcotdelta_scattering_length],
                 n_params_set=[1]):
        self._fc = fc
        self._indexing = indexing
        self._sub_isospin = sub_isospin
        self._ell_set = ell_set
        self._p_cot_deltas = p_cot_deltas
        self._n_params_set = n_params_set

        self.fc = fc
        self.indexing = indexing
        self.sub_isospin = sub_isospin
        self.ell_set = ell_set
        self.p_cot_deltas = p_cot_deltas
        self.n_params_set = n_params_set

    @property
    def fc(self):
        return self._fc

    @fc.setter
    def fc(self, fc):
        self._fc = fc
        self.indexing = self._indexing
        self.sub_isospin = self._sub_isospin
        self.ell_set = self._ell_set
        self.p_cot_deltas = self._p_cot_deltas
        self.n_params_set = self._n_params_set

    @property
    def indexing(self):
        return self._indexing

    @indexing.setter
    def indexing(self, indexing):
        if (self.fc.n_particles == 2) and (indexing is not None):
            warnings.warn("\n"+bcolors.WARNING
                          + "n_particles == 2 and indexing is not None. "
                          + "Setting it to None."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._indexing = None
        elif (self.fc.n_particles == 2) and (indexing is None):
            pass
        elif self.fc.explicit_flavor_channel:
            if not isinstance(indexing, list):
                raise ValueError('for n_particles > 2, '
                                 + 'indexing must be a list')
            if len(indexing) != self.fc.n_particles:
                raise ValueError('indexing must have length n_particles')
            if (np.sort(indexing) != np.arange(self.fc.n_particles)).any():
                raise ValueError('indexing must be a permuatation of '
                                 + 'ascending integers')
            self._indexing = indexing
        elif self.fc.isospin_channel and indexing is not None:
            warnings.warn("\n"+bcolors.WARNING
                          + "indexing is not None for an isospin_channel. "
                          + "Setting it to None."
                          + f"{bcolors.ENDC}", stacklevel=2)
            self._indexing = None

    @property
    def sub_isospin(self):
        return self._sub_isospin

    @sub_isospin.setter
    def sub_isospin(self, sub_isospin):
        if ((sub_isospin is not None)
           and (self.fc.n_particles == 2)):
            raise ValueError('sub_isospin must be None for n_particles == 2')
        if ((sub_isospin is not None)
           and (not isinstance(sub_isospin, int))):
            raise ValueError('sub_isospin must be an int')
        if ((sub_isospin is not None)
           and (sub_isospin > self.fc.n_particles-1)):
            raise ValueError('sub_isospin should not exceed n_particles-1')
        if (not self.fc.isospin_channel) and (sub_isospin is not None):
            raise ValueError('sub_isospin cannot be set because '
                             + 'isospin_channel is False')
        if (self.fc.isospin_channel and (sub_isospin is None)
           and (self.fc.n_particles != 2)):
            raise ValueError('sub_isospin cannot be set to None because '
                             + 'isospin_channel is True')
        self._sub_isospin = sub_isospin

    @property
    def ell_set(self):
        return self._ell_set

    @ell_set.setter
    def ell_set(self, ell_set):
        self._ell_set = ell_set
        if len(self.p_cot_deltas) != len(ell_set):
            self._p_cot_deltas = [self._p_cot_deltas[0]]*len(ell_set)
        if len(self.n_params_set) != len(ell_set):
            self._n_params_set = [self._n_params_set[0]]*len(ell_set)

    @property
    def p_cot_deltas(self):
        return self._p_cot_deltas

    @p_cot_deltas.setter
    def p_cot_deltas(self, p_cot_deltas):
        self._p_cot_deltas = p_cot_deltas
        if len(self.ell_set) != len(p_cot_deltas):
            raise ValueError('len(ell_set) != len(p_cot_deltas)')

    @property
    def n_params_set(self):
        return self._n_params_set

    @n_params_set.setter
    def n_params_set(self, n_params_set):
        self._n_params_set = n_params_set
        if len(self.ell_set) != len(n_params_set):
            raise ValueError('len(ell_set) != len(n_params_set)')

    def __str__(self):
        strtmp = self.fc.__str__().replace('Flavor', 'Spectator')[:-1]+",\n"
        strtmp = strtmp+"    indexing: "+str(self.indexing)+",\n"
        strtmp = strtmp+"    sub_isospin: "+str(self.sub_isospin)+",\n"
        strtmp = strtmp+"    ell_set: "+str(self.ell_set)+",\n"
        strtmp = strtmp+"    p_cot_deltas: "+str(self.p_cot_deltas)+",\n"
        strtmp = strtmp+"    n_params_set: "+str(self.n_params_set)+",\n"
        return strtmp[:-1]


class FlavorChannelSpace:
    r"""
    Class used to represent a space of multi-hadron channels.

    Attributes
    ----------
    fc_list : list
        flavor channel list (list of instances of FlavorChannel).
    sc_list : list
        spectator channel list (list of instances of _SpectatorChannel).
        _SpectatorChannel is a class to be used only within FlavorChannelSpace.
        It includes extra information relevative to FlavorChannel as
        summarized below.
    qcd_channel_space : boolean
        defines whether the space is a qcd channel space (as opposed to an
        explicit flavor channel space).
    explicit_flavor_channel_space : boolean
        defines whether the space is an explicit flavor channel space (as
        opposed to a qcd channel space).
    sc_compact : list
        compact summary of the relevant spectator-channel properties. The exact
        data depends on whether the space is a qcd channel space or an
        explicit flavor channel space. In both cases sc_compact is a list of
        rank-two np.ndarrays, one for each value of n_particles included in
        the entries of fc_list. If only three-particle channels are included,
        then len(sc_compact) is 1 and it contains a single rank-two np.ndarray.
        Focus on this case. Then len(sc_compact[0]) is the total number of
        three-particle spectator channels.

        In the case of a qcd channel space len(sc_compact[0].T) is 10. Each
        row is populated as follows:
            [3.0, mass1, mass2, mass3, spin1, spin2, spin3, isospin_flavor,
             isospin_value, sub_isospin],
        where the first entry is the number of particles and all values are
        cast to floats.

        In the case of an explicit flavor channel space len(sc_compact[0].T)
        is again 10. Each row is populated as follows:
            [3.0, mass1, mass2, mass3, spin1, spin2, spin3, flavor1, flavor2,
             flavor3],
        where the first entry is the number of particles and all values are
        cast to floats.
    three_index : int
        if the fc_list includes multiple values of n_particles, three_index
        is used to specify the location of the sc_compact entry for the
        three-particle subspace. For the case of only three-particle
        channels, three_index is 0.
    three_slices : list
        list of two-entry (doublet) lists of integers. Each doublet specifies
        a slice of sc_compact[three_index] according to mass values. So, for
        a non-negative integer i < len(three_slices) we can evaluate:
            sc_compact[three_index][three_slices[i][0]:three_slices[i][1]]
        to get a three-particle subspace with fixed mass values.
    n_three_slices : int
        length of three_slices, so the number of different mass values.
    g_templates : list of lists of np.ndarrays
        len(g_templates) is equal to len(g_templates[i]) for any non-negative
        i < len(g_templates). Thus the list of lists is interpreted as a
        square array, with the number of rows and columns also equal to
        n_three_slices. Each entry in g_template gives a template for the
        finite-volume G matrix within each pair of mass-identical subspaces.
        Off diaongal entries are all zeroes if the sorted set of masses is
        distinct but can be non-zero if, for example masses 2.0, 2.0, 1.0 swap
        into masses 1.0, 2.0, 2.0 (where the first entry is the spectator in
        both cases).

    Attributes of _SpectatorChannel
    ------------------------------
    fc : FlavorChannel()
        instance of FlavorChannel, a key property of the spectator channel.
    indexing : list
        permutation of [0, 1, 2], the first entry is the spectator particle
        (indexing is None for a two-particle or an isospin channel).
    sub_isospin : int
        value of the two-particle isospin for the spectator channel
        (is None for a two-particle or an explicit flavor channel).
    ell_set : list
        list of integers specifying the allowed values of orbital angular
        momentum.
    p_cot_deltas : list
        list of functions for the two-particle scattering channels. Same
        length as ell_set.
    n_params_set : list
        number of parameters required for each p_cot_deltas entry. Same length
        as ell_set.
    """

    def __init__(self, fc_list=None):
        if fc_list is None:
            self.fc_list = []
        else:
            self.fc_list = fc_list

        if len(fc_list) > 0:
            self.qcd_channel_space = fc_list[0].isospin_channel
            self.explicit_flavor_channel_space\
                = fc_list[0].explicit_flavor_channel

        if self.qcd_channel_space and self.explicit_flavor_channel_space:
            raise ValueError('FlavorChannelSpace cannot be both a '
                             + 'qcd_channel_space and a '
                             + 'explicit_flavor_channel_space')

        for fc in fc_list:
            if fc.explicit_flavor_channel\
               is not self.explicit_flavor_channel_space:
                raise ValueError('channels do not all have the same status '
                                 + 'of explicit_flavor_channel')
            if fc.isospin_channel is not self.qcd_channel_space:
                raise ValueError('channels do not all have same status of '
                                 + 'isospin_channel')

        self.sc_list = []
        for fc in fc_list:
            self._add_flavor_channel(fc)
        self._build_sorted_sc_list()
        self._build_g_templates()

    def _add_spectator_channel(self, sc):
        self.sc_list.append(sc)

    def _add_flavor_channel(self, fc):
        if fc.n_particles == 2:
            sc1 = _SpectatorChannel(fc, indexing=None)
            self._add_spectator_channel(sc1)
        elif fc.isospin_channel:
            if fc.isospin_value == 3:
                sc1 = _SpectatorChannel(fc, indexing=None, sub_isospin=2)
                self._add_spectator_channel(sc1)
            elif fc.isospin_value == 2:

                sc1 = _SpectatorChannel(fc, indexing=None, sub_isospin=1,
                                        ell_set=[1])
                sc2 = _SpectatorChannel(fc, indexing=None, sub_isospin=2)
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            elif fc.isospin_value == 1:
                sc1 = _SpectatorChannel(fc, indexing=None, sub_isospin=0)
                sc2 = _SpectatorChannel(fc, indexing=None, sub_isospin=1,
                                        ell_set=[1])
                sc3 = _SpectatorChannel(fc, indexing=None, sub_isospin=2)
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
                self._add_spectator_channel(sc3)
            elif fc.isospin_value == 0:
                sc1 = _SpectatorChannel(fc, indexing=None, sub_isospin=1,
                                        ell_set=[1])
                self._add_spectator_channel(sc1)
            else:
                raise ValueError('this isospin_value is not yet supported')
        elif fc.explicit_flavor_channel:
            if fc.explicit_flavors[0] == fc.explicit_flavors[1]\
               == fc.explicit_flavors[2]:
                sc1 = _SpectatorChannel(fc)
                self._add_spectator_channel(sc1)
            elif fc.explicit_flavors[0] == fc.explicit_flavors[1]:
                sc1 = _SpectatorChannel(fc)
                sc2 = _SpectatorChannel(fc, indexing=[2, 0, 1])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            elif fc.explicit_flavors[0] == fc.explicit_flavors[2]:
                sc1 = _SpectatorChannel(fc)
                sc2 = _SpectatorChannel(fc, indexing=[1, 2, 0])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            elif fc.explicit_flavors[1] == fc.explicit_flavors[2]:
                sc1 = _SpectatorChannel(fc)
                sc2 = _SpectatorChannel(fc, indexing=[1, 2, 0])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
            else:
                sc1 = _SpectatorChannel(fc=fc)
                sc2 = _SpectatorChannel(fc=fc, indexing=[1, 2, 0])
                sc3 = _SpectatorChannel(fc=fc, indexing=[2, 0, 1])
                self._add_spectator_channel(sc1)
                self._add_spectator_channel(sc2)
                self._add_spectator_channel(sc3)
        else:
            raise ValueError('either isospin_channel or'
                             + ' explicit_flavor_channel must be True')

    def _build_sorted_sc_list(self):
        n_particles_max = 0
        for fc in self.fc_list:
            if fc.n_particles > n_particles_max:
                n_particles_max = fc.n_particles
        sc_compact = [[[]] for _ in range(n_particles_max-1)]
        for sc in self.sc_list:
            sc_comp_tmp = [sc.fc.n_particles]
            if sc.fc.isospin_channel:
                sc_comp_tmp = sc_comp_tmp+sc.fc.masses
                sc_comp_tmp = sc_comp_tmp+sc.fc.spins
                sc_comp_tmp = sc_comp_tmp+[sc.fc.isospin_flavor]
                sc_comp_tmp = sc_comp_tmp+[sc.fc.isospin_value]
                if sc.fc.n_particles != 2:
                    sc_comp_tmp = sc_comp_tmp+[sc.sub_isospin]
            elif sc.fc.explicit_flavor_channel and sc.indexing is not None:
                sc_comp_tmp = sc_comp_tmp\
                    + list(np.array(sc.fc.masses)[sc.indexing])
                sc_comp_tmp = sc_comp_tmp\
                    + list(np.array(sc.fc.spins)[sc.indexing])
                sc_comp_tmp = sc_comp_tmp\
                    + list(np.array(sc.fc.explicit_flavors)[sc.indexing])
            else:
                return ValueError('something is wrong with channel'
                                  + ' specification.')
            sc_compact[sc.fc.n_particles-2] = sc_compact[sc.fc.n_particles-2]\
                + [sc_comp_tmp]
        for j in range(len(sc_compact)):
            sc_compact[j] = np.array(sc_compact[j][1:])
            len_tmp = len(sc_compact[j].T)
            for i in range(len_tmp):
                if not ((self.qcd_channel_space) and (i == 0)):
                    sc_compact[j] = sc_compact[j][
                        sc_compact[j][:, len_tmp-i-1].argsort(
                            kind='mergesort')]
        self.sc_compact = []
        for sc_compact_entry in sc_compact:
            if len(sc_compact_entry) != 0:
                self.sc_compact = self.sc_compact+[sc_compact_entry]
        self.three_slices = [[]]
        three_index = 0
        for j in range(len(self.sc_compact)):
            if self.sc_compact[j][0][0] == 3.0:
                three_index = j
        self.three_index = three_index
        sc_compact_three_subspace = self.sc_compact[three_index]
        sc_three_masses_previous = sc_compact_three_subspace[0][1:4]
        slice_min = 0
        slice_max = 0
        for sc_compact_entry in sc_compact_three_subspace:
            sc_three_masses_current = sc_compact_entry[1:4]
            if (sc_three_masses_previous == sc_three_masses_current).all():
                slice_max = slice_max+1
            else:
                self.three_slices = self.three_slices+[[slice_min, slice_max]]
                slice_min = slice_max
                slice_max = slice_max+1
                sc_three_masses_previous = sc_three_masses_current
        self.three_slices = self.three_slices+[[slice_min, slice_max]]
        self.three_slices = self.three_slices[1:]
        self.n_three_slices = len(self.three_slices)

    def _build_g_templates(self):
        g_templates_tmp = [[]]
        if self.qcd_channel_space:
            for three_slice in self.three_slices:
                iso_slices = [[]]
                len_tmp = three_slice[1]-three_slice[0]
                zero_point = 0
                i = 0
                iso_pos = 8
                while i < len_tmp:
                    iso_val_tmp = int(self.sc_compact[self.three_index][
                        three_slice[0]+i][iso_pos])
                    if iso_val_tmp == 0:
                        iso_dim_tmp = 1
                    elif iso_val_tmp == 1:
                        iso_dim_tmp = 3
                    elif iso_val_tmp == 2:
                        iso_dim_tmp = 2
                    elif iso_val_tmp == 3:
                        iso_dim_tmp = 1
                    iso_slices = iso_slices+[[zero_point,
                                              zero_point+iso_dim_tmp,
                                              iso_val_tmp]]
                    zero_point = zero_point+iso_dim_tmp
                    i = i+iso_dim_tmp
                iso_slices = iso_slices[1:]
                g_template_tmp = np.zeros((len_tmp, len_tmp))
                for iso_slice in iso_slices:
                    ((g_template_tmp[iso_slice[0]:iso_slice[1]]).T)[
                        iso_slice[0]:iso_slice[1]]\
                        = G_TEMPLATE_DICT[iso_slice[2]]
                g_templates_tmp = g_templates_tmp+[g_template_tmp]
            g_templates_tmp = g_templates_tmp[1:]
            self.g_templates = [[]]
            for i in range(len(g_templates_tmp)):
                g_templates_row = []
                for j in range(len(g_templates_tmp)):
                    if i == j:
                        g_templates_row = g_templates_row+[g_templates_tmp[i]]
                    else:
                        g_templates_row = g_templates_row+[np.zeros(
                            (len(g_templates_tmp[i]), len(g_templates_tmp[j]))
                            )]
                self.g_templates = self.g_templates+[g_templates_row]
            self.g_templates = self.g_templates[1:]
        elif self.explicit_flavor_channel_space:
            self.g_templates = [[]]
            for slice_row in self.three_slices:
                len_row = slice_row[1]-slice_row[0]
                g_templates_row = []
                for slice_col in self.three_slices:
                    len_col = slice_col[1]-slice_col[0]
                    g_template_tmp = np.zeros((len_row, len_col))
                    for i in range(len_row):
                        for j in range(len_col):
                            f_min = 7
                            f_max = 10
                            i_flavs = self.sc_compact[self.three_index][
                                slice_row[0]+i][f_min:f_max]
                            j_flavs = self.sc_compact[self.three_index][
                                slice_col[0]+j][f_min:f_max]
                            if (
                                    ((i_flavs[0] == j_flavs[2])
                                     and (np.sort(i_flavs[1:])
                                          == np.sort(j_flavs[:-1])).all())
                                    or
                                    ((i_flavs[0] == j_flavs[1])
                                     and (np.sort(i_flavs[1:])
                                          == np.sort([j_flavs[0]]
                                                     + [j_flavs[2]])).all())
                                    ):
                                g_template_tmp[i][j] = 1.0
                    g_templates_row = g_templates_row+[g_template_tmp]
                self.g_templates = self.g_templates+[g_templates_row]
            self.g_templates = self.g_templates[1:]
        else:
            raise ValueError("either qcd_channel_space or " +
                             "explicit_flavor_channel_space must be True")


class FiniteVolumeSetup:
    r"""
    Class used to represent the finite-volume set-up.

    The provided data includes the formalism to be used, the total spatial
    momentum, the finite-volume irrep and the qc organization.

    qc_organization is drawn from the following:
        'original'
        'hermitian'
        'real harmonics'
        'hermitian, real harmonics'

    Attributes
    ----------
    formalism : str
        indicates the formalism used. Currently only 'RFT' (relatvistic-field
        theory approach) is supported.
    nP : np.ndarray
        three-vector encoded as a np.ndarray with shape (3,). Populated with
        integers indicating the total spatial momentum in units of 2*PI/L,
        where L is the box length.
    nPSQ : int
        magnitude squared of nP.
    nPmag : float
        magnitude of nP.
    irreps : Irreps(nP)
        instance of the class Irreps() encoding the possible irreducible
        representations of the finite-volume symmetry group for a given
        value of nP.
    qc_organization : str
        specifies how the qc is organized, see list above.
    """

    def __init__(self, formalism='RFT',
                 nP=np.array([0, 0, 0]),
                 qc_organization='hermitian, real harmonics'):
        self.formalism = formalism
        self.qc_organization = qc_organization
        self.nP = nP
        self.irreps = Irreps(nP=self.nP)

    @property
    def nP(self):
        """Get the total three-momentum (np.ndarray, shape is (3,))."""
        return self._nP

    @nP.setter
    def nP(self, nP):
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
    def qc_organization(self):
        """
        Get the qc organization (str).

        qc_organization is drawn from the following:
            'original'
            'hermitian'
            'real harmonics'
            'hermitian, real harmonics'
        """
        return self._qc_organization

    @qc_organization.setter
    def qc_organization(self, qc_organization):
        if not isinstance(qc_organization, str):
            raise ValueError("qc_organization must be a str")
        if ((qc_organization != 'original')
           and (qc_organization != 'hermitian')
           and (qc_organization != 'real harmonics')
           and (qc_organization != 'hermitian, real harmonics')):
            raise ValueError("qc_organization not recognized")
        self._qc_organization = qc_organization

    def __str__(self):
        """Summary of the finite-volume set-up."""
        strtmp = "FiniteVolumeSetup using the "+self.formalism+":\n"
        strtmp = strtmp+"    nP = "+str(self._nP)+",\n"
        strtmp = strtmp+"    qc_organization = "+self.qc_organization+",\n"
        return strtmp[:-2]+"."


class ThreeBodyInteractionScheme:
    r"""
    Class for the meaning and parametrization of the three-body interaction.

    Specifes the cutoff function and the exact meaning of finite-volume
    functions as these affect the exact meaning of the three-body interaction.
    Also specifies the parametrization of the three body interaction on the
    FlavorChannelSpace.

    three_scheme is drawn from the following:
        'relativistic pole'
        'original pole'
    This refers to the type of pole used within the finite-volume G function.
    This choice affects the meaning of the three-body interaction.

    Attributes
    ----------
    fcs : FlavorChannelSpace()
        flavor-channel space, required for building the space of kdf_functions.
    three_scheme : str
        specifies the scheme for the scheme dependent kdf, see the list above.
    scheme_data : list
        two floats, alpha and beta, specifiy the shape of the cutoff function.
        Default is [alpha, beta]=[-1.0, 0.0].
    kdf_functions : list
        list of lists defining a square array, length is the number of flavor
        channels.
    kdf_iso_constant : builtin_function_or_method
        simplest choice for kdf, a single coupling called beta_0, independent
        of all kinematics.
    """

    def __init__(self, fcs=None, Emin=0.0,
                 three_scheme='relativistic pole',
                 scheme_data=[-1.0, 0.0],
                 kdf_functions=None):
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
        """Summary of the three-body interaction scheme."""
        strtmp = "ThreeBodyInteractionScheme with the following data:\n"
        strtmp = strtmp+"    Emin = "+str(self.Emin)+",\n"
        strtmp = strtmp+"    three_scheme = "+self.three_scheme+",\n"
        strtmp = strtmp+"    [alpha, beta] = "+str(self.scheme_data)+",\n"
        strtmp = strtmp+"    kdf_functions as follows:\n"
        for i in range(len(self.fcs.fc_list)):
            strtmp = strtmp+"        "+str(self.kdf_functions[i])+" for\n"
            strtmp = strtmp+"        "+str(self.fcs.fc_list[i]).replace(
                "    ", "            ")[:-1]+",\n"
        return strtmp[:-2]+"."


class ThreeBodyKinematicSpace:
    r"""
    Class encoding spectator-momentum kinematics.

    Attributes
    ----------
    nP : np.ndarray
        three-vector encoded as a np.ndarray with shape (3,). Inherited from
        the finite-volume set-up. Populated with integers indicating the total
        spatial momentum in units of 2*PI/L, where L is the box length.
    build_slice_acc : (boolean)
        determines whether data is prepared to accelerate evaluates of qc.
    nvec_arry : np.ndarray
        array with shape (n, 3), giving the defining list of three-vectors.
    verbosity : int
        determines how verbose the output is.
    """

    def __init__(self, nP=np.array([0, 0, 0]),
                 nvec_arr=np.array([]),
                 build_slice_acc=True,
                 verbosity=0):

        self.build_slice_acc = build_slice_acc
        self.nP = nP
        self.nvec_arr = nvec_arr
        self.verbosity = verbosity

    @property
    def nP(self):
        """Get the total three-momentum (np.ndarray, shape is (3,))."""
        return self._nP

    @nP.setter
    def nP(self, nP):
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
        """Get the array of spectator nvecs (np.ndarray)."""
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

    def _get_slice_sort(self, nvec_arr_first_sort):
        g = Groups(ell_max=0)
        little_group = g.get_little_group(self._nP)
        nvec_arr_copy = np.copy(nvec_arr_first_sort)
        slicedict_nvec_arr = {}
        slicedict_index = 0
        while len(nvec_arr_copy) > 0:
            nvec_tmp = nvec_arr_copy[0].reshape((3, 1))
            nvec_rotations = (little_group*nvec_tmp).sum(1)
            nvec_rotations_unique = np.unique(nvec_rotations, axis=0)
            slicedict_nvec_arr[slicedict_index] = nvec_rotations_unique
            slicedict_index = slicedict_index+1
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
        nvec_arr_slicesort = None
        slices = [[]]
        slices_counter = 0
        for i in range(len(slicedict_nvec_arr)):
            if nvec_arr_slicesort is None:
                nvec_arr_slicesort = slicedict_nvec_arr[i]
            else:
                nvec_arr_slicesort = np.concatenate((nvec_arr_slicesort,
                                                     slicedict_nvec_arr[i]))
            slices = slices+[[slices_counter,
                              slices_counter+len(slicedict_nvec_arr[i])]]
            slices_counter = slices_counter+len(slicedict_nvec_arr[i])
        return nvec_arr_slicesort, slices[1:]

    def _get_stacks(self):
        n1vec_stackdict = {}
        n2vec_stackdict = {}
        n3vec_stackdict = {}
        stackdict_multiplicities = {}
        for i in self.slices:
            for j in self.slices:
                stri = str(i[0])+'_'+str(i[1])
                strj = str(j[0])+'_'+str(j[1])
                n1vec_stackdict[stri+'_'+strj] = 0.0
                n2vec_stackdict[stri+'_'+strj] = 0.0
                n3vec_stackdict[stri+'_'+strj] = 0.0
                stackdict_multiplicities[stri+'_'+strj]\
                    = (i[1]-i[0])*(j[1]-j[0])

        for i in range(len(self.nvecSQ_arr)):
            for j in range(len(self.nvecSQ_arr)):
                shell_index_i = 0
                shell_index_j = 0
                for itmp in range(len(self.slices)):
                    rtmp = self.slices[itmp]
                    if rtmp[0] <= i < rtmp[1]:
                        shell_index_i = itmp
                for jtmp in range(len(self.slices)):
                    rtmp = self.slices[jtmp]
                    if rtmp[0] <= j < rtmp[1]:
                        shell_index_j = jtmp
                stri = str(self.slices[shell_index_i][0])+'_'\
                    + str(self.slices[shell_index_i][1])
                strj = str(self.slices[shell_index_j][0])+'_'\
                    + str(self.slices[shell_index_j][1])
                sizei = self.slices[shell_index_i][1]\
                    - self.slices[shell_index_i][0]
                sizej = self.slices[shell_index_j][1]\
                    - self.slices[shell_index_j][0]
                if (sizei >= sizej and i == self.slices[shell_index_i][0])\
                   or (sizej > sizei and j == self.slices[shell_index_j][0]):
                    if n1vec_stackdict[stri+'_'+strj] == 0.0:
                        n1vec_stackdict[stri+'_'+strj]\
                            = [self.n1vec_mat[i][j]]
                    else:
                        n1vec_stackdict[stri+'_'+strj] =\
                            n1vec_stackdict[stri+'_'+strj]\
                            + [self.n1vec_mat[i][j]]
                    if n2vec_stackdict[stri+'_'+strj] == 0.0:
                        n2vec_stackdict[stri+'_'+strj]\
                            = [self.n2vec_mat[i][j]]
                    else:
                        n2vec_stackdict[stri+'_'+strj]\
                            = n2vec_stackdict[stri+'_'+strj]\
                            + [self.n2vec_mat[i][j]]
                    if n3vec_stackdict[stri+'_'+strj] == 0.0:
                        n3vec_stackdict[stri+'_'+strj]\
                            = [self.n3vec_mat[i][j]]
                    else:
                        n3vec_stackdict[stri+'_'+strj] =\
                            n3vec_stackdict[stri+'_'+strj]\
                            + [self.n3vec_mat[i][j]]
        n1vec_stacked = [[]]
        n2vec_stacked = [[]]
        n3vec_stacked = [[]]
        stack_multiplicities = [[]]
        for i in self.slices:
            n1row = []
            n2row = []
            n3row = []
            mrow = []
            for j in self.slices:
                stri = str(i[0])+'_'+str(i[1])
                strj = str(j[0])+'_'+str(j[1])
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
        if self.build_slice_acc:
            if len(nvec_arr) == 0:
                self._nvec_arr = nvec_arr
            else:
                nvec_arr_first_sort = self._get_first_sort(nvec_arr)
                self._nvec_arr, self.slices\
                    = self._get_slice_sort(nvec_arr_first_sort)
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
        else:
            self._nvec_arr = nvec_arr

    def __str__(self):
        """Summary of the three-body kinematic space."""
        np.set_printoptions(threshold=10)
        strtmp = "ThreeBodyKinematicSpace with the following data:\n"
        strtmp = strtmp+"    nvec_arr="\
            + str(self.nvec_arr).replace("\n", "\n             ")+",\n"
        np.set_printoptions(threshold=PRINT_THRESHOLD_DEFAULT)
        return strtmp[:-2]+"."


class QCIndexSpace:
    r"""
    Class encoding the quantizaiton condition index space.

    Attributes
    ----------
    fcs : FlavorChannelSpace()
        flavor-channel space defining the index space.
    fvs : FiniteVolumeSetup()
        finite-volume set-up defining the index space.
    tbis : ThreeBodyInteractionScheme()
        three-body interaction scheme  defining the index space.
    Emax : float
        maximum energy for building the space.
    Lmax : float
        maximum volume for building the space.
    """

    def __init__(self, fcs=None, fvs=None, tbis=None,
                 Emax=5.0, Lmax=5.0, verbosity=0, ell_max=4):
        self.verbosity = verbosity

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
        self.Emax = Emax
        self.Lmax = Lmax
        self.group = Groups(ell_max=ell_max)

        if self.nPSQ != 0:
            if verbosity == 2:
                print('nPSQ is nonzero, will use grid')
            deltaL = 0.9
            deltaE = 0.9
            Lmin = np.mod(Lmax-2.0, deltaL)+2.0
            Emin = np.mod(Emax-3.0, deltaE)+3.0
            Lvals = np.arange(Lmin, Lmax, deltaL)
            Evals = np.arange(Emin, Emax, deltaE)
            if np.abs(Lvals[-1] - Lmax) > EPSILON:
                Lvals = np.append(Lvals, Lmax)
            if np.abs(Evals[-1] - Emax) > EPSILON:
                Evals = np.append(Evals, Emax)
            self.Lvals = Lvals[::-1]
            self.Evals = Evals[::-1]
            if verbosity == 2:
                print('Lvals =', self.Lvals)
                print('Evals =', self.Evals)
        else:
            self.Lvals = None
            self.Evals = None

        self.param_structure = [[]]
        two_param_struc_tmp = [[]]
        for sc in self.fcs.sc_list:
            tmp_entry = [[]]
            for n_params_tmp in sc.n_params_set:
                tmp_entry = tmp_entry+[[0.0]*n_params_tmp]
            two_param_struc_tmp = two_param_struc_tmp+[tmp_entry[1:]]
        two_param_struc_tmp = two_param_struc_tmp[1:]
        self.param_structure = self.param_structure+[two_param_struc_tmp]
        tmp_entry = []
        for kdf in self.tbis.kdf_functions:
            tmp_entry = tmp_entry+[0.0]
        self.param_structure = self.param_structure+[tmp_entry]
        self.param_structure = self.param_structure[1:]

        self.populate_all_nvec_arr()
        self.ell_sets = self._get_ell_sets()
        self.populate_all_kellm_spaces()
        self.populate_all_proj_dicts()
        self.proj_dict = self.group.get_full_proj_dict(qcis=self)
        self.populate_noninteracting()

    @property
    def nP(self):
        """Get the total three-momentum (np.ndarray, shape is (3,))."""
        return self._nP

    @nP.setter
    def nP(self, nP):
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
        """Get the flavor-channel space (FlavorChannelSpace())."""
        return self._fcs

    @fcs.setter
    def fcs(self, fcs):
        self.n_channels = len(fcs.sc_list)
        tbks_list_tmp = []
        for i in range(self.fcs.n_three_slices):
            tbks_list_tmp = tbks_list_tmp\
                + [ThreeBodyKinematicSpace(nP=self.nP)]
        self.tbks_list = tbks_list_tmp
        self._fcs = fcs

    def _get_nPspecmax(self, three_slice_index):
        sc = self.fcs.sc_list[self.fcs.three_slices[three_slice_index][0]]
        if sc.fc.explicit_flavor_channel:
            mspec = sc.fc.masses[sc.indexing[0]]
        elif sc.fc.isospin_channel:
            mspec = sc.fc.masses[0]
        else:
            raise ValueError('get_nPspecmax was handed a channel that is '
                             + 'neither an isospin-channel nor an explicit '
                             + 'flavor channel.')
        Emax = self.Emax
        EmaxSQ = Emax**2
        nPSQ = self.nP@self.nP
        Lmax = self.Lmax
        EminSQ = self.tbis.Emin**2
        if (EminSQ != 0.0):
            if nPSQ == 0:
                nPspecmax = (Lmax*np.sqrt(
                    Emax**4+(EminSQ-mspec**2)**2-2.*Emax**2*(EminSQ+mspec**2)
                    ))/(2.*Emax*TWOPI)
                return nPspecmax
            else:
                raise ValueError('simultaneous nonzero nP and Emin not'
                                 + ' supported')
        else:
            if nPSQ == 0:
                nPspecmax = Lmax*(EmaxSQ-mspec**2)/(2.0*TWOPI*Emax)
                return nPspecmax
            else:
                nPmag = np.sqrt(nPSQ)
                nPspecmax = (FOURPI2*nPmag*(
                    Lmax**2*(EmaxSQ+mspec**2)-FOURPI2*nPSQ
                    )+np.sqrt(EmaxSQ*FOURPI2*Lmax**2*(
                        Lmax**2*(-EmaxSQ+mspec**2)+FOURPI2*nPSQ
                        )**2))/(2.*FOURPI2*(EmaxSQ*Lmax**2-FOURPI2*nPSQ))
                return nPspecmax

    def populate_nvec_arr_slot(self, three_slice_index):
        """Populate a given nvec_arr slot."""
        if self.fcs.n_three_slices != 1:
            raise ValueError('n_three_slices different from one not yet '
                             + 'supported')
        if three_slice_index != 0:
            raise ValueError('three_slice_index != 0 not yet supported')
        if (self.nP == np.array([0, 0, 0])).all():
            if self.verbosity >= 2:
                print("populating nvec array, three_slice_index = ",
                      str(three_slice_index))
            nPspecmax = self._get_nPspecmax(three_slice_index)
            if isinstance(self.tbks_list[three_slice_index], list):
                tbks_tmp = self.tbks_list[three_slice_index][0]
                if self.verbosity >= 2:
                    print("self.tbks_list[three_slice_index] is a list,",
                          "taking first entry")
            else:
                tbks_tmp = self.tbks_list[three_slice_index]
                if self.verbosity >= 2:
                    print("self.tbks_list[three_slice_index] is not a list")
            tbks_copy = deepcopy(tbks_tmp)
            tbks_copy.verbosity = self.verbosity
            self.tbks_list[three_slice_index] = [tbks_copy]
            nPspec = nPspecmax
            if self.verbosity >= 2:
                print("populating up to nPspecmax = "+str(nPspecmax))
            while nPspec > 0:
                if self.verbosity >= 2:
                    print("nPspec**2 = ", int(nPspec**2))
                rng = range(-int(nPspec), int(nPspec)+1)
                mesh = np.meshgrid(*([rng]*3))
                nvec_arr = np.vstack([y.flat for y in mesh]).T
                carr = (nvec_arr*nvec_arr).sum(1) > nPspec**2
                nvec_arr = np.delete(nvec_arr, np.where(carr), axis=0)
                self.tbks_list[three_slice_index][-1].nvec_arr = nvec_arr
                if self.verbosity >= 2:
                    print(self.tbks_list[three_slice_index][-1])
                tbks_copy = deepcopy(tbks_tmp)
                tbks_copy.verbosity = self.verbosity
                self.tbks_list[three_slice_index] =\
                    self.tbks_list[three_slice_index] + [tbks_copy]
                nPspecSQ = nPspec**2-1.0
                if nPspecSQ >= 0.0:
                    nPspec = np.sqrt(nPspecSQ)
                else:
                    nPspec = -1.0
            self.tbks_list[three_slice_index] =\
                self.tbks_list[three_slice_index][:-1]
        else:
            nPspecmax = self._get_nPspecmax(three_slice_index)
            if isinstance(self.tbks_list[three_slice_index], list):
                tbks_tmp = self.tbks_list[three_slice_index][0]
                if self.verbosity >= 2:
                    print("self.tbks_list[three_slice_index] is a list,",
                          "taking first entry")
            else:
                tbks_tmp = self.tbks_list[three_slice_index]
                if self.verbosity >= 2:
                    print("self.tbks_list[three_slice_index] is not a list")
            rng = range(-int(nPspecmax), int(nPspecmax)+1)
            mesh = np.meshgrid(*([rng]*3))
            nvec_arr = np.vstack([y.flat for y in mesh]).T

            tbks_copy = deepcopy(tbks_tmp)
            tbks_copy.verbosity = self.verbosity
            self.tbks_list[three_slice_index] = [tbks_copy]
            Lmax = self.Lmax
            Emax = self.Emax
            sc_compact_three_subspace\
                = self.fcs.sc_compact[self.fcs.three_index]
            masses = sc_compact_three_subspace[three_slice_index][1:4]
            mspec = masses[0]
            nP = self.nP
            deltaL = 0.9
            deltaE = 0.9
            Lmin = np.mod(Lmax-2.0, deltaL)+2.0
            Emin = np.mod(Emax-3.0, deltaE)+3.0
            Lvals = np.arange(Lmin, Lmax, deltaL)
            Evals = np.arange(Emin, Emax, deltaE)
            if np.abs(Lvals[-1] - Lmax) > EPSILON:
                Lvals = np.append(Lvals, Lmax)
            if np.abs(Evals[-1] - Emax) > EPSILON:
                Evals = np.append(Evals, Emax)
            Lvals = Lvals[::-1]
            Evals = Evals[::-1]
            for Ltmp in Lvals:
                for Etmp in Evals:
                    E2CMSQ = (Etmp-np.sqrt(mspec**2
                                           + FOURPI2/Ltmp**2
                                           * ((nvec_arr**2).sum(axis=1))))**2\
                        - FOURPI2/Ltmp**2*((nP-nvec_arr)**2).sum(axis=1)
                    carr = E2CMSQ < 0.0
                    E2CMSQ = E2CMSQ.reshape((len(E2CMSQ), 1))
                    E2nvec_arr = np.concatenate((E2CMSQ, nvec_arr), axis=1)
                    E2nvec_arr = np.delete(E2nvec_arr, np.where(carr), axis=0)
                    nvec_arr_tmp = ((E2nvec_arr.T)[1:]).T
                    nvec_arr_tmp = nvec_arr_tmp.astype(np.int64)
                    if self.verbosity >= 2:
                        print("L = ", np.round(Ltmp, 10),
                              ", E = ", np.round(Etmp, 10))
                    self.tbks_list[three_slice_index][-1].nvec_arr\
                        = nvec_arr_tmp
                    if self.verbosity >= 2:
                        print(self.tbks_list[three_slice_index][-1])
                    tbks_copy = deepcopy(tbks_tmp)
                    tbks_copy.verbosity = self.verbosity
                    self.tbks_list[three_slice_index] =\
                        self.tbks_list[three_slice_index]\
                        + [tbks_copy]
            self.tbks_list[three_slice_index] =\
                self.tbks_list[three_slice_index][:-1]

    def populate_all_nvec_arr(self):
        """Populate all nvec_arr slots."""
        for three_slice_index in range(self.fcs.n_three_slices):
            self.populate_nvec_arr_slot(three_slice_index)

    def _get_ell_sets(self):
        ell_sets = [[]]
        for cindex in range(self.n_channels):
            ell_set = self.fcs.sc_list[cindex].ell_set
            ell_sets = ell_sets+[ell_set]
        return ell_sets[1:]

    @property
    def ell_sets(self):
        """Get the angular-momentum values (list)."""
        return self._ell_sets

    @ell_sets.setter
    def ell_sets(self, ell_sets):
        self._ell_sets = ell_sets
        tmpset_outer = [[]]
        for ell_set in ell_sets:
            tmpset = [[]]
            for ell in ell_set:
                for mazi in range(-ell, ell+1):
                    tmpset = tmpset+[[ell, mazi]]
            tmpset_outer = tmpset_outer+[tmpset[1:]]
        self.ellm_sets = tmpset_outer[1:]

    def _get_three_slice_index(self, cindex):
        slice_index = 0
        for three_slice in self.fcs.three_slices:
            if cindex > three_slice[1]:
                slice_index = slice_index+1
        return slice_index

    def populate_all_kellm_spaces(self):
        """Populate all kellm spaces."""
        if self.verbosity >= 2:
            print("populating kellm spaces")
            print(self.n_channels, "channels to populate")
        kellm_slices = [[]]
        kellm_spaces = [[]]
        for cindex in range(self.n_channels):
            slice_index = self._get_three_slice_index(cindex)
            tbks_list_tmp = self.tbks_list[slice_index]
            ellm_set = self.ellm_sets[cindex]
            kellm_slices_single = [[]]
            kellm_spaces_single = [[]]
            for tbks_tmp in tbks_list_tmp:
                nvec_arr = tbks_tmp.nvec_arr
                kellm_slice = (len(ellm_set)*np.array(tbks_tmp.slices
                                                      )).tolist()
                kellm_slices_single = kellm_slices_single+[kellm_slice]
                ellm_set_extended = np.tile(ellm_set, (len(nvec_arr), 1))
                nvec_arr_extended = np.repeat(nvec_arr, len(ellm_set),
                                              axis=0)
                kellm_space = np.concatenate((nvec_arr_extended,
                                              ellm_set_extended),
                                             axis=1)
                kellm_spaces_single = kellm_spaces_single+[kellm_space]
            kellm_slices = kellm_slices+[kellm_slices_single[1:]]
            kellm_spaces = kellm_spaces+[kellm_spaces_single[1:]]
        self.kellm_spaces = kellm_spaces[1:]
        self.kellm_slices = kellm_slices[1:]
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
        sc_proj_dicts_sliced = [[]]
        if self.verbosity >= 2:
            print("getting the dict for following qcis:")
            print(self)
        for cindex in range(self.n_channels):
            proj_dict = group.get_channel_proj_dict(qcis=self, cindex=cindex)
            sc_proj_dicts = sc_proj_dicts+[proj_dict]
            sc_proj_dict_single_slice = []
            for kellm_slice in self.kellm_slices[cindex][0]:
                sc_proj_dict_single_slice = sc_proj_dict_single_slice\
                    + [group.get_slice_proj_dict(
                        qcis=self,
                        cindex=cindex,
                        kellm_slice=kellm_slice)]
            sc_proj_dicts_sliced = sc_proj_dicts_sliced\
                + [sc_proj_dict_single_slice]
        self.sc_proj_dicts = sc_proj_dicts
        self.sc_proj_dicts_sliced = sc_proj_dicts_sliced[1:]

    def get_tbks_sub_indices(self, E, L):
        """Get the indices of the relevant three-body kinematics spaces."""
        if E > self.Emax:
            raise ValueError('get_tbks_sub_indices called with E > Emax')
        if L > self.Lmax:
            raise ValueError('get_tbks_sub_indices called with L > Lmax')
        tbks_sub_indices = [0]*len(self.tbks_list)
        if (self.nP)@(self.nP) != 0:
            for slice_index in range(self.fcs.n_three_slices):
                cindex = self.fcs.three_slices[slice_index][0]
                sc = self.fcs.sc_list[cindex]
                if sc.fc.explicit_flavor_channel:
                    mspec = sc.fc.masses[sc.indexing[0]]
                elif sc.fc.isospin_channel:
                    mspec = sc.fc.masses[0]
                else:
                    raise ValueError('get_tbks_sub_indices was handed a'
                                     + ' channel that is neither an'
                                     + ' isospin-channel nor an explicit'
                                     + ' flavor channel.')
                nP = self.nP
                tbkstmp_set = self.tbks_list[cindex]
                still_searching = True
                i = 0
                while still_searching:
                    tbkstmp = tbkstmp_set[i]
                    nvec_arr = tbkstmp.nvec_arr
                    E2CMSQfull = (E-np.sqrt(mspec**2
                                            + FOURPI2/L**2
                                            * ((nvec_arr**2).sum(axis=1)
                                               )))**2\
                        - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
                    still_searching = not (np.sort(E2CMSQfull) > 0.0).all()
                    i = i+1
                i = i-1
                tbkstmp = tbkstmp_set[i]
                nvec_arr = tbkstmp.nvec_arr
                E2CMSQfull = (E-np.sqrt(mspec**2
                                        + FOURPI2/L**2
                                        * ((nvec_arr**2).sum(axis=1)
                                           )))**2\
                    - FOURPI2/L**2*((nP-nvec_arr)**2).sum(axis=1)
                tbks_sub_indices[cindex] = i
            warnings.warn("\n"+bcolors.WARNING
                          + "get_tbks_sub_indices is being called with "
                          + "non_zero nP. This can lead to shells being"
                          + "missed! result is = "+str(tbks_sub_indices)
                          + f"{bcolors.ENDC}", stacklevel=1)
            return tbks_sub_indices
        for slice_index in range(self.fcs.n_three_slices):
            cindex = self.fcs.three_slices[slice_index][0]
            nPspecmax = self._get_nPspecmax(cindex)
            sc = self.fcs.sc_list[cindex]
            if sc.fc.explicit_flavor_channel:
                mspec = sc.fc.masses[sc.indexing[0]]
            elif sc.fc.isospin_channel:
                mspec = sc.fc.masses[0]
            else:
                raise ValueError('get_tbks_sub_indices was handed a channel '
                                 + 'that is neither an isospin-channel nor '
                                 + 'an explicit flavor channel.')
            ESQ = E**2
            nPSQ = self.nP@self.nP
            EminSQ = self.tbis.Emin**2
            if (EminSQ != 0.0):
                if nPSQ == 0:
                    nPspecnew = (L*np.sqrt(
                        E**4+(EminSQ-mspec**2)**2-2.*E**2*(EminSQ+mspec**2)
                        ))/(2.*E*TWOPI)
                else:
                    raise ValueError('nonzero nP and Emin not supported')
            else:
                if nPSQ == 0:
                    nPspecnew = L*(ESQ-mspec**2)/(2.0*TWOPI*E)
                else:
                    nPmag = np.sqrt(nPSQ)
                    nPspecnew = (FOURPI2*nPmag*(
                        L**2*(ESQ+mspec**2)-FOURPI2*nPSQ
                        )+np.sqrt(ESQ*FOURPI2*L**2*(
                            L**2*(-ESQ+mspec**2)+FOURPI2*nPSQ
                            )**2))/(2.*FOURPI2*(ESQ*L**2-FOURPI2*nPSQ))

            nPmaxintSQ = int(nPspecmax**2)
            nPnewintSQ = int(nPspecnew**2)
            tbks_sub_indices[cindex] = nPmaxintSQ - nPnewintSQ
        return tbks_sub_indices

    def populate_noninteracting(self):
        """Populate non-interacting energy vectors."""
        cut = 2
        trust = 30
        rng = range(-cut, cut+1)
        mesh = np.meshgrid(*([rng] * 3))
        n1vec_arr = np.vstack([y.flat for y in mesh]).T
        n2vec_arr = np.vstack([y.flat for y in mesh]).T
        nSQset = [[]]
        for n1vec in n1vec_arr:
            for n2vec in n2vec_arr:
                n3vec = -n1vec-n2vec
                n1SQ = n1vec@n1vec
                n2SQ = n2vec@n2vec
                n3SQ = n3vec@n3vec
                nSQset = nSQset+[[n1SQ+n2SQ+n3SQ]
                                 + list(np.sort([n1SQ, n2SQ, n3SQ]))]
        nSQset = np.array(nSQset[1:])
        nSQset_unique = ((np.unique(nSQset[np.argsort(nSQset[:, 1])],
                                    axis=0)).T)[1:].T[:trust]
        self.nSQset_unique = nSQset_unique

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
        for sc in self.fcs.sc_list:
            for n_params in sc.n_params_set:
                pcotdelta_parameter_list = pcotdelta_parameter_list\
                    + [[0.0]*n_params]
        pcotdelta_parameter_list = pcotdelta_parameter_list[1:]
        k3_params = [0.0]
        return [pcotdelta_parameter_list, k3_params]

    def __str__(self):
        """Summary of the QCIndexSpace."""
        strtmp = "QCIndexSpace containing:\n"
        strtmp = strtmp+"    "+str(self.fcs).replace("\n", "\n    ")+"\n\n"
        strtmp = strtmp+"    "+str(self.fvs).replace("\n", "\n    ")+"\n\n"
        strtmp = strtmp+"    "+str(self.tbis).replace("\n", "\n    ")+"\n\n"
        strtmp = strtmp+"    Parameter input structure:\n"
        strtmp = strtmp+"        "+str(self.param_structure)+"\n\n"
        for tbkstmp in self.tbks_list:
            strtmp = strtmp+"    "+str(tbkstmp[0]).replace("\n", "\n    ")+"\n"
        return strtmp[:-1]


class G:
    r"""
    Class for the finite-volume G matrix (responsible for exchanges).

    Attributes
    ----------
    qcis : QCIndexSpace()
        all data for this QC object is specified by the qc index space.
    """

    def __init__(self, qcis=QCIndexSpace()):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data

    def _get_masks_and_slices(self, E, nP, L, tbks_entry,
                              cindex_row, cindex_col,
                              row_slice_index, col_slice_index):
        mask_row_slices = None
        mask_col_slices = None
        three_slice_index_row\
            = self.qcis._get_three_slice_index(cindex_row)
        three_slice_index_col\
            = self.qcis._get_three_slice_index(cindex_col)
        if not (three_slice_index_row == three_slice_index_col == 0):
            raise ValueError('only one mass slice is supported in G')
        three_slice_index = three_slice_index_row
        if nP@nP == 0:
            row_slice = tbks_entry.slices[row_slice_index]
            col_slice = tbks_entry.slices[col_slice_index]
        else:
            sc_compact_three_subspace\
                = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
            masses = sc_compact_three_subspace[three_slice_index][1:4]
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask_row = (E-omk_arr)**2-PmkSQ_arr > 0.0
            row_slices = tbks_entry.slices
            mask_row_slices = []
            for row_slice in row_slices:
                mask_row_slices = mask_row_slices\
                    + [mask_row[row_slice[0]:row_slice[1]].all()]
            row_slices = list(np.array(row_slices)[mask_row_slices])
            row_slice = row_slices[row_slice_index]

            sc_compact_three_subspace\
                = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
            masses = sc_compact_three_subspace[three_slice_index][1:4]
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask_col = (E-omk_arr)**2-PmkSQ_arr > 0.0
            col_slices = tbks_entry.slices
            mask_col_slices = []
            for col_slice in col_slices:
                mask_col_slices = mask_col_slices\
                    + [mask_col[col_slice[0]:col_slice[1]].all()]
            col_slices = list(np.array(col_slices)[mask_col_slices])
            col_slice = col_slices[col_slice_index]
        return mask_row_slices, mask_col_slices, row_slice, col_slice

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex_row=None, cindex_col=None,  # only for non-zero P
                  sc_row_ind=None, sc_col_ind=None,
                  ell1=0, ell2=0,
                  g_rescale=1.0, tbks_entry=None,
                  row_slice_index=None,
                  col_slice_index=None,
                  project=False, irrep=None):
        """Build the G matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qco = self.qcis.fvs.qc_organization
        alpha = self.alpha
        beta = self.beta

        mask_row_slices, mask_col_slices, row_slice, col_slice\
            = self._get_masks_and_slices(E, nP, L, tbks_entry,
                                         cindex_row, cindex_col,
                                         row_slice_index, col_slice_index)

        Gshell = QCFunctions.getG_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        row_slice, col_slice,
                                        ell1, ell2,
                                        alpha, beta,
                                        qco, ts,
                                        g_rescale)

        if project:
            try:
                if nP@nP != 0:
                    proj_tmp_right = np.array(self.qcis.sc_proj_dicts_sliced[
                        sc_col_ind])[
                            mask_col_slices][
                                col_slice_index][irrep]
                    proj_tmp_left = np.conjugate((
                            np.array(self.qcis.sc_proj_dicts_sliced[
                                sc_row_ind])[
                                    mask_row_slices][
                                        row_slice_index][irrep]
                                        ).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_sliced[
                        sc_col_ind][col_slice_index][irrep]
                    proj_tmp_left = np.conjugate((
                        self.qcis.sc_proj_dicts_sliced[
                            sc_row_ind][row_slice_index][irrep]
                        ).T)
            except KeyError:
                return np.array([])
        if project:
            Gshell = proj_tmp_left@Gshell@proj_tmp_right
        return Gshell

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
                if shtmp == (0,):
                    g_collection[i][j].shape = (rowsizes[i], colsizes[j])

        return g_collection

    def get_value(self, E=5.0, L=5.0, project=False, irrep=None):
        """Build the G matrix in a shell-based way."""
        Lmax = self.qcis.Lmax
        Emax = self.qcis.Emax
        if E > Emax:
            raise ValueError('get_value called with E > Emax')
        if L > Lmax:
            raise ValueError('get_value called with L > Lmax')
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating G using numpy accelerated version')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

            print(self.qcis.tbis.three_scheme, ',',
                  self.qcis.fvs.qc_organization)
            print('cutoff params:', self.alpha, ',', self.beta)

            if self.qcis.tbis.three_scheme == 'original pole':
                sf = '1./(2.*w1*w2*L**3)'
            elif self.qcis.tbis.three_scheme == 'relativistic pole':
                sf = '1./(2.*w1*L**3)\n    * 1./(E-w1-w3+w2)'
            else:
                raise ValueError('three_scheme not recognized')
            if (self.qcis.fvs.qc_organization == 'hermitian')\
                or (self.qcis.fvs.qc_organization
                    == 'hermitian, real harmonics'):
                sf = sf+'\n    * 1./(2.0*w3*L**3)'

            print('G = YY*H1*H2\n    * '+sf+'\n    * 1./(E-w1-w2-w3)\n')

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError('only n_three_slices = 1 is supported')
        cindex_row = cindex_col = 0
        if self.qcis.verbosity >= 2:
            print('representatives of three_slice:')
            print('    cindex_row =', cindex_row,
                  ', cindex_col =', cindex_col)

        if not (irrep in self.qcis.proj_dict.keys()):
            raise ValueError('irrep '+str(irrep)+' not in '
                             + 'qcis.proj_dict.keys()')

        three_compact = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
        masses = three_compact[0][1:4]
        [m1, m2, m3] = masses

        if nP@nP == 0:
            if self.qcis.verbosity >= 2:
                print('nP = [0 0 0] indexing')
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.slices
            if self.qcis.verbosity >= 2:
                print('tbks_sub_indices =', tbks_sub_indices)
                print('tbks_entry =', tbks_entry)
                print('slices =', slices)
        else:
            if self.qcis.verbosity >= 2:
                print('nP != [0 0 0] indexing')
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
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
            slices = tbks_entry.slices
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        g_final = [[]]
        if self.qcis.verbosity <= 2:
            print('iterating over spectator channels, slices')
        for sc_row_ind in range(len(three_compact)):
            g_outer_row = []
            row_ell_set = self.qcis.fcs.sc_list[sc_row_ind].ell_set
            if len(row_ell_set) != 1:
                raise ValueError('only length-one ell_set currently '
                                 + 'supported in G')
            ell1 = row_ell_set[0]
            for sc_col_ind in range(len(three_compact)):
                if self.qcis.verbosity <= 2:
                    print('sc_row_ind, sc_col_ind =', sc_row_ind, sc_col_ind)
                col_ell_set = self.qcis.fcs.sc_list[sc_col_ind].ell_set
                if len(col_ell_set) != 1:
                    raise ValueError('only length-one ell_set currently '
                                     + 'supported in G')
                ell2 = col_ell_set[0]
                g_rescale = self.qcis.fcs.g_templates[0][0][
                    sc_row_ind][sc_col_ind]

                g_inner = [[]]
                for row_slice_index in range(len(slices)):
                    g_inner_row = []
                    for col_slice_index in range(len(slices)):
                        g_tmp = self.get_shell(E, L,
                                               m1, m2, m3,
                                               cindex_row, cindex_col,  # for P
                                               sc_row_ind, sc_col_ind,
                                               ell1, ell2,  # ell vals
                                               g_rescale,
                                               tbks_entry,
                                               row_slice_index,
                                               col_slice_index,
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


class F:
    """
    Class for the F matrix.

    Warning: It is up to the user to select values of alphaKSS and C1cut that
    lead to a sufficient estimate of the F matrix.

    Attributes
    ----------
    qcis : QCIndexSpace()
        all data for this QC object is specified by the qc index space.
    alphaKSS : float
        damping factor entering the zeta functions.
    C1cut : int
        hard cutoff used in the zeta functions.
    """

    def __init__(self, qcis=None, alphaKSS=1.0, C1cut=3):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data
        self.C1cut = C1cut
        self.alphaKSS = alphaKSS

    def _get_masks_and_slices(self, E, nP, L, tbks_entry,
                              cindex, slice_index):
        mask_slices = None
        three_slice_index\
            = self.qcis._get_three_slice_index(cindex)

        if nP@nP == 0:
            slice_entry = tbks_entry.slices[slice_index]
        else:
            sc_compact_three_subspace\
                = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
            masses = sc_compact_three_subspace[three_slice_index][1:4]
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            slices = tbks_entry.slices
            mask_slices = []
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list(np.array(slices)[mask_slices])
            slice_entry = slices[slice_index]
        return mask_slices, slice_entry

    def get_shell(self, E=5.0, L=5.0, m1=1.0, m2=1.0, m3=1.0,
                  cindex=None, sc_ind=None, ell1=0, ell2=0, tbks_entry=None,
                  slice_index=None, project=False, irrep=None):
        """Build the F matrix on a single shell."""
        ts = self.qcis.tbis.three_scheme
        nP = self.qcis.nP
        qco = self.qcis.fvs.qc_organization
        alpha = self.alpha
        beta = self.beta
        C1cut = self.C1cut
        alphaKSS = self.alphaKSS

        mask_slices, slice_entry\
            = self._get_masks_and_slices(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Fshell = QCFunctions.getF_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell1, ell2,
                                        alpha, beta,
                                        C1cut, alphaKSS,
                                        qco, ts)

        if project:
            try:
                if nP@nP != 0:
                    proj_tmp_right = self.qcis.sc_proj_dicts_sliced[
                        sc_ind][
                        mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_sliced[
                        sc_ind][slice_index][irrep]
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
            raise ValueError('get_value called with E > Emax')
        if L > Lmax:
            raise ValueError('get_value called with L > Lmax')
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError('only n_three_slices = 1 is supported')
        three_slice_index = 0
        cindex = 0
        three_compact = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
        masses = three_compact[three_slice_index][1:4]
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.slices
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
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
            slices = tbks_entry.slices
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        f_final_list = []
        for sc_ind in range(len(three_compact)):
            ell_set = self.qcis.fcs.sc_list[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError('only length-one ell_set currently '
                                 + 'supported in F')
            ell1 = ell_set[0]
            ell2 = ell1
            for slice_index in range(len(slices)):
                f_tmp = self.get_shell(E, L,
                                       m1, m2, m3,
                                       cindex,  # for P
                                       sc_ind,
                                       ell1, ell2,
                                       tbks_entry,
                                       slice_index,
                                       project, irrep)
                if len(f_tmp) != 0:
                    f_final_list = f_final_list+[f_tmp]
        return block_diag(*f_final_list)


class K:
    """
    Class for the k matrix.

    Attributes
    ----------
    qcis : QCIndexSpace()
        all data for this QC object is specified by the qc index space.
    """

    def __init__(self, qcis=None):
        self.qcis = qcis
        ts = self.qcis.tbis.three_scheme
        if (ts == 'original pole')\
           or (ts == 'relativistic pole'):
            [self.alpha, self.beta] = self.qcis.tbis.scheme_data

    def _get_masks_and_slices(self, E, nP, L, tbks_entry,
                              cindex, slice_index):
        mask_slices = None
        three_slice_index\
            = self.qcis._get_three_slice_index(cindex)

        if nP@nP == 0:
            slice_entry = tbks_entry.slices[slice_index]
        else:
            sc_compact_three_subspace\
                = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
            masses = sc_compact_three_subspace[three_slice_index][1:4]
            mspec = masses[0]
            kvecSQ_arr = FOURPI2*tbks_entry.nvecSQ_arr/L**2
            kvec_arr = TWOPI*tbks_entry.nvec_arr/L
            omk_arr = np.sqrt(mspec**2+kvecSQ_arr)
            Pvec = TWOPI*nP/L
            PmkSQ_arr = ((Pvec-kvec_arr)**2).sum(axis=1)
            mask = (E-omk_arr)**2-PmkSQ_arr > 0.0
            slices = tbks_entry.slices
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
        qco = self.qcis.fvs.qc_organization
        alpha = self.alpha
        beta = self.beta

        mask_slices, slice_entry\
            = self._get_masks_and_slices(E, nP, L, tbks_entry,
                                         cindex, slice_index)

        Kshell = QCFunctions.getK_array(E, nP, L, m1, m2, m3,
                                        tbks_entry,
                                        slice_entry,
                                        ell,
                                        pcotdelta_function,
                                        pcotdelta_parameter_list,
                                        alpha, beta,
                                        qco, ts)

        if project:
            try:
                if nP@nP != 0:
                    proj_tmp_right = self.qcis.sc_proj_dicts_sliced[
                        sc_ind][
                        mask_slices][slice_index][irrep]
                    proj_tmp_left = np.conjugate(((proj_tmp_right)).T)
                else:
                    proj_tmp_right = self.qcis.sc_proj_dicts_sliced[
                        sc_ind][slice_index][irrep]
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
        if E > Emax:
            raise ValueError('get_value called with E > Emax')
        if L > Lmax:
            raise ValueError('get_value called with L > Lmax')
        nP = self.qcis.nP
        if self.qcis.verbosity >= 2:
            print('evaluating F')
            print('E = ', E, ', nP = ', nP, ', L = ', L)

        if self.qcis.fcs.n_three_slices != 1:
            raise ValueError('only n_three_slices = 1 is supported')
        three_slice_index = 0
        cindex = 0
        three_compact = self.qcis.fcs.sc_compact[self.qcis.fcs.three_index]
        masses = three_compact[three_slice_index][1:4]
        [m1, m2, m3] = masses

        if nP@nP == 0:
            tbks_sub_indices = self.qcis.get_tbks_sub_indices(E=E, L=L)
            tbks_entry = self.qcis.tbks_list[0][
                tbks_sub_indices[0]]
            slices = tbks_entry.slices
        else:
            mspec = m1
            ibest = self.qcis._get_ibest(E, L)
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
            slices = tbks_entry.slices
            for slice_entry in slices:
                mask_slices = mask_slices\
                    + [mask[slice_entry[0]:slice_entry[1]].all()]
            slices = list((np.array(slices))[mask_slices])

        k_final_list = []
        for sc_ind in range(len(three_compact)):
            ell_set = self.qcis.fcs.sc_list[sc_ind].ell_set
            if len(ell_set) != 1:
                raise ValueError('only length-one ell_set currently '
                                 + 'supported in K')
            ell = ell_set[0]
            pcotdelta_parameter_list = pcotdelta_parameter_lists[sc_ind]
            pcotdelta_function = self.qcis.fcs.sc_list[
                sc_ind].p_cot_deltas[0]
            for slice_index in range(len(slices)):
                k_tmp = self.get_shell(E, L, m1, m2, m3,
                                       cindex,  # for P
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

    Attributes
    ----------
    qcis : QCIndexSpace()
        all data for this QC object is specified by the qc index space.
    alphaKSS : float
        damping factor entering the zeta functions.
    C1cut : int
        hard cutoff used in the zeta functions.
    """

    def __init__(self, qcis=None, C1cut=5, alphaKSS=1.0):
        self.qcis = qcis
        self.f = F(qcis=self.qcis, alphaKSS=alphaKSS, C1cut=C1cut)
        self.g = G(qcis=self.qcis)
        self.k = K(qcis=self.qcis)

    def get_value(self, E=None, L=None, k_params=None, project=True,
                  irrep=None, version='kdf_zero_1+',
                  rescale=1.0):
        """Get value."""
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
