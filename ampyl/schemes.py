#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created July 2022.

@author: M.T. Hansen
"""

###############################################################################
#
# schemes.py
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
import functools
from .constants import QC_IMPL_DEFAULTS
from .constants import bcolors
from .flavor import FlavorChannel
from .flavor import FlavorChannelSpace
import warnings
warnings.simplefilter("once")

PRINT_THRESHOLD_DEFAULT = np.get_printoptions()['threshold']


class FiniteVolumeSetup:
    """
    Represent the finite-volume kinematic and QC setup.

    Parameters
    ----------
    formalism : str, optional
        Formalism used to define the quantization condition. Currently only
        ``'RFT'`` is supported.
    nP : numpy.ndarray of int, shape (3,), optional
        Total momentum in finite-volume units.
    qc_impl : dict, optional
        Quantization-condition implementation options. Keys must be drawn from
        :data:`ampyl.constants.QC_IMPL_DEFAULTS`, and values must have the same
        type as the corresponding default.
    spin_half : bool, optional
        Whether to use the spin-half irrep set where supported.
    verbosity : int, optional
        Verbosity level for setup diagnostics.

    Attributes
    ----------
    irrep_set : list of str
        Irreps supported by the little group for ``nP``.
    nPSQ : int
        Squared total momentum, ``nP @ nP``.
    nPmag : float
        Magnitude of the total momentum.

    Raises
    ------
    ValueError
        If ``nP`` is not an integer array with shape ``(3,)``, if ``qc_impl``
        has unsupported keys or value types, or if the requested momentum and
        spin combination is not supported.
    """

    def __init__(self, formalism='RFT', nP=np.array([0, 0, 0]), qc_impl={},
                 spin_half=False, verbosity=0):
        """
        Initialize a finite-volume setup.

        Parameters
        ----------
        formalism : str, optional
            Formalism used to define the quantization condition.
        nP : numpy.ndarray of int, shape (3,), optional
            Total finite-volume momentum.
        qc_impl : dict, optional
            Implementation flags overriding ``QC_IMPL_DEFAULTS`` entries.
        spin_half : bool, optional
            Whether to initialize spin-half irreps.
        verbosity : int, optional
            Verbosity level.
        """
        self.formalism = formalism
        self.spin_half = spin_half
        self.qc_impl = qc_impl
        self.nP = nP
        self.set_irreps()
        self._verbosity = verbosity
        self.verbosity = verbosity

        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}")
            print(self)
            print(f"{bcolors.ENDC}")

    @property
    def verbosity(self):
        """Verbosity of the FiniteVolumeSetup."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the FiniteVolumeSetup."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

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
        for key in qc_impl:
            if key not in QC_IMPL_DEFAULTS:
                raise ValueError("key", key, "not recognized")
        for key in QC_IMPL_DEFAULTS:
            if (key in qc_impl
               and (not isinstance(qc_impl[key],
                                   type(QC_IMPL_DEFAULTS[key])))):
                raise ValueError(f"qc_impl entry {key} mest be a "
                                 f"{type(QC_IMPL_DEFAULTS[key])}")
        self._qc_impl = qc_impl

    def set_irreps(self):
        """
        Set the irreps relevant for the finite-volume setup.

        Raises
        ------
        ValueError
            If ``nP`` or the ``spin_half``/``nP`` combination is not currently
            supported.
        """
        nP_is_zero = (self._nP == np.array([0, 0, 0])).all()
        nP_is_00z = self._nP[0] == 0 and self._nP[1] == 0
        nP_is_0zz = self._nP[0] == 0 and self._nP[1] == self._nP[2]
        if not self.spin_half:
            if nP_is_zero:
                self._set_irreps_nPero_intspin()
            elif nP_is_00z:
                self._set_irreps_nP00z_intspin()
            elif nP_is_0zz:
                self._set_irreps_nP0zz_intspin()
            else:
                self.irrep_set = []
                raise ValueError("unsupported value of nP in irreps: "
                                 + str(self._nP))
        else:
            if nP_is_zero:
                self._set_irreps_nPzero_spinhalf()
            else:
                self.irrep_set = []
                raise ValueError("unsupported value of nP in irreps: "
                                 + str(self._nP))

    def _set_irreps_nPero_intspin(self):
        self.A1PLUS = 'A1PLUS'
        self.A2PLUS = 'A2PLUS'
        self.T1PLUS = 'T1PLUS'
        self.T2PLUS = 'T2PLUS'
        self.EPLUS = 'EPLUS'
        self.A1MINUS = 'A1MINUS'
        self.A2MINUS = 'A2MINUS'
        self.T1MINUS = 'T1MINUS'
        self.T2MINUS = 'T2MINUS'
        self.EMINUS = 'EMINUS'
        self.irrep_set = [self.A1PLUS, self.A2PLUS, self.EPLUS,
                          self.T1PLUS, self.T2PLUS, self.A1MINUS,
                          self.A2MINUS, self.EMINUS, self.T1MINUS,
                          self.T2MINUS]

    def _set_irreps_nP00z_intspin(self):
        self.A1 = 'A1'
        self.A2 = 'A2'
        self.B1 = 'B1'
        self.B2 = 'B2'
        self.E = 'E2'
        self.irrep_set = [self.A1, self.A2, self.B1, self.B2, self.E]

    def _set_irreps_nP0zz_intspin(self):
        self.A1 = 'A1'
        self.A2 = 'A2'
        self.B1 = 'B1'
        self.B2 = 'B2'
        self.irrep_set = [self.A1, self.A2, self.B1, self.B2]

    def _set_irreps_nPzero_spinhalf(self):
        self.A1PLUS = 'A1PLUS'
        self.A2PLUS = 'A2PLUS'
        self.EPLUS = 'EPLUS'
        self.T1PLUS = 'T1PLUS'
        self.T2PLUS = 'T2PLUS'
        self.G1PLUS = 'G1PLUS'
        self.G2PLUS = 'G2PLUS'
        self.HPLUS = 'HPLUS'
        self.A1MINUS = 'A1MINUS'
        self.A2MINUS = 'A2MINUS'
        self.EMINUS = 'EMINUS'
        self.T1MINUS = 'T1MINUS'
        self.T2MINUS = 'T2MINUS'
        self.G1MINUS = 'G1MINUS'
        self.G2MINUS = 'G2MINUS'
        self.HMINUS = 'HMINUS'
        self.irrep_set = [self.A1PLUS, self.A2PLUS, self.EPLUS,
                          self.T1PLUS, self.T2PLUS, self.G1PLUS,
                          self.G2PLUS, self.HPLUS, self.A1MINUS,
                          self.A2MINUS, self.EMINUS, self.T1MINUS,
                          self.T2MINUS, self.G1MINUS, self.G2MINUS,
                          self.HMINUS]

    def __str__(self):
        """Return a string representation of the FiniteVolumeSetup object."""
        finite_volume_setup_str =\
            f"FiniteVolumeSetup using the {self.formalism}:\n"
        finite_volume_setup_str += f"    nP = {self._nP},\n"
        finite_volume_setup_str += f"    qc_impl = {self.qc_impl},\n"
        finite_volume_setup_str += f"    irrep_set = {self.irrep_set},\n"
        return finite_volume_setup_str[:-2]+"."


class ThreeBodyInteractionScheme:
    """
    Represent the three-body interaction and pole-removal scheme.

    Parameters
    ----------
    fcs : FlavorChannelSpace, optional
        Flavor-channel space that defines the domain of ``Kdf``. If omitted,
        a default three-particle flavor channel is used.
    ESQmins : list of float, optional
        Minimum two-body invariant mass squared for each spectator channel.
        If provided, these values override the defaults inferred from the
        flavor-channel space or scheme data.
    three_scheme : str, optional
        Three-body interaction scheme. Currently ``'relativistic pole'`` and
        related alpha-beta pole schemes are supported by the surrounding code.
    scheme_data : list, optional
        Cutoff-scheme data. A single ``[alpha, beta]`` pair is broadcast to all
        channels; otherwise provide one pair per spectator channel.
    kdf_functions : list of callable, optional
        Functions defining the three-body interaction for each flavor channel.
        Defaults to ``kdf_iso_constant`` for every channel.
    use_pv_shift_prescription : list of bool, optional
        Flags selecting whether to use the IPV prescription for removing
        K-matrix poles in each flavor-ell-m component.
    pv_shift_parameters : list, optional
        Parameters for the IPV prescription. Required when any
        ``use_pv_shift_prescription`` entry is true.
    verbosity : int, optional
        Verbosity level for setup diagnostics.

    Attributes
    ----------
    threshSQs : list of float
        Two-particle threshold squared for each spectator channel.
    ESQmins : list of float
        Minimum two-body invariant mass squared for each spectator channel.
    flavor_ell_dim : int
        Number of flavor-angular-momentum entries before expanding magnetic
        components.
    flavor_ellm_dim : int
        Number of flavor-angular-momentum-magnetic entries.

    Raises
    ------
    ValueError
        If ``scheme_data`` has the wrong shape, or if PV-shift options and
        parameters are inconsistent.
    """

    def __init__(self, fcs=None, ESQmins=None, three_scheme='relativistic pole',
                 scheme_data=None, kdf_functions=None,
                 use_pv_shift_prescription=None,
                 pv_shift_parameters=None,
                 verbosity=0):
        """
        Initialize a three-body interaction scheme.

        Parameters
        ----------
        fcs : FlavorChannelSpace, optional
            Flavor-channel space defining the spectator channels.
        ESQmins : list of float, optional
            Override for the minimum two-body invariant mass squared in each
            spectator channel.
        three_scheme : str, optional
            Name of the three-body interaction scheme.
        scheme_data : list, optional
            Either a single ``[alpha, beta]`` pair or a list of such pairs.
        kdf_functions : list of callable, optional
            Three-body interaction functions by flavor channel.
        use_pv_shift_prescription : list of bool, optional
            IPV prescription flags by flavor-ell-m component.
        pv_shift_parameters : list, optional
            IPV prescription parameters.
        verbosity : int, optional
            Verbosity level.
        """
        if fcs is None:
            fcs = FlavorChannelSpace(fc_list=[FlavorChannel(3)])
        self.fcs = fcs

        self.threshSQs = [sc.thresholdSQ for sc in fcs.sc_list_sorted]
        ESQmins_by_channel = [sc.ESQmin for sc in fcs.sc_list_sorted]
        if scheme_data is None:
            scheme_data_by_channel = [
                list(sc.scheme_data) for sc in fcs.sc_list_sorted]
        else:
            scheme_data_by_channel = self._scheme_data_by_channel(
                scheme_data, len(self.threshSQs))
            ESQmins_by_channel = []
            for i, scheme in enumerate(scheme_data_by_channel):
                if not isinstance(scheme, list) or len(scheme) != 2:
                    raise ValueError("scheme_data must be a list of lists "
                                     "with length 2")
                alpha, beta = scheme
                ESQmin_tmp = 0.25*(1.0+alpha)*self.threshSQs[i]
                ESQmins_by_channel.append(ESQmin_tmp)
        if ESQmins is not None:
            self.ESQmins = ESQmins
        else:
            self.ESQmins = ESQmins_by_channel

        self._set_flavor_ellm_dim()
        if use_pv_shift_prescription is None:
            use_pv_shift_prescription = [False]*self.flavor_ellm_dim
        if any(use_pv_shift_prescription):
            if pv_shift_parameters is None:
                raise ValueError("pv_shift_parameters must be provided")
        else:
            if pv_shift_parameters is not None:
                pv_shift_parameters = [None]*self.flavor_ellm_dim
                warnings.warn("pv_shift_parameters provided but "
                              "use_pv_shift_prescription is False. "
                              "Setting pv_shift_parameters to None.")
            if pv_shift_parameters is None:
                pv_shift_parameters = [None]*self.flavor_ellm_dim
        self.use_pv_shift_prescription = use_pv_shift_prescription
        self.pv_shift_parameters = pv_shift_parameters

        wrong_length = False
        if any(use_pv_shift_prescription):
            wrong_length = len(pv_shift_parameters) != self.flavor_ell_dim
        if any(use_pv_shift_prescription) and wrong_length:
            raise ValueError("pv_shift_parameters must have length equal "
                             "to flavor_ellm_dim")
        self.three_scheme = three_scheme
        self.scheme_data_by_channel = scheme_data_by_channel
        self.scheme_data = scheme_data_by_channel[0]
        if kdf_functions is None:
            self.kdf_functions = []
            for fc in self.fcs.fc_list:
                self.kdf_functions = self.kdf_functions+[self.kdf_iso_constant]
        else:
            self.kdf_functions = kdf_functions
        self._verbosity = verbosity
        self.verbosity = verbosity
        if self.verbosity >= 2:
            print(f"{bcolors.OKGREEN}")
            print(self)
            print(f"{bcolors.ENDC}")

    def _scheme_data_by_channel(self, scheme_data, n_channels):
        if len(scheme_data) == 2 and not isinstance(scheme_data[0], list):
            return [scheme_data]*n_channels
        return scheme_data

    def _set_flavor_ellm_dim(self):
        self.flavor_ell_dim = sum(len(sc.ell_set) for sc in self.fcs.sc_list)
        self.flavor_ellm_dim = sum(
            sum(2 * ell + 1 for ell in sc.ell_set) for sc in self.fcs.sc_list
        )

    @property
    def verbosity(self):
        """Verbosity of the ThreeBodyInteractionScheme."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, verbosity):
        """Set the verbosity of the ThreeBodyInteractionScheme."""
        if not isinstance(verbosity, int):
            raise ValueError("verbosity must be an int")
        self._verbosity = verbosity

    def with_str(str_func):
        """
        Decorate a function with custom string behavior.

        Parameters
        ----------
        str_func : callable
            Zero-argument function returning the desired string
            representation.

        Returns
        -------
        callable
            Decorator that wraps the target function in an object preserving
            calls while overriding ``str()``.
        """
        def wrapper(f):
            """Wrap ``f`` in an object with custom string behavior."""
            class FuncType:
                """Callable wrapper that delegates calls and string output."""

                def __call__(self, *args, **kwargs):
                    return f(*args, **kwargs)

                def __str__(self):
                    """Return the custom string representation."""
                    return str_func()
            return functools.wraps(f)(FuncType())
        return wrapper

    def kdf_iso_constant_str():
        """Print behavior for kdf_iso_constant."""
        return "kdf_iso_constant"

    @with_str(kdf_iso_constant_str)
    def kdf_iso_constant(beta_0):
        """
        Evaluate a constant isotropic ``Kdf``.

        Parameters
        ----------
        beta_0 : float
            Constant interaction strength.

        Returns
        -------
        float
            The unchanged input value ``beta_0``.
        """
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
