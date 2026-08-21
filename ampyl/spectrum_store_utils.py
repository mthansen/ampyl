#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created August 2026.

@author: M.T. Hansen
"""

###############################################################################
#
# spectrum_store_utils.py
#
# MIT License
# Copyright (c) 2026 Maxwell T. Hansen
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

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .constants import QC_DICT_DEFAULTS
from .constants import QC_IMPL_DEFAULTS
from .constants import SPECTRUM_STORE_AUTOSAVE
from .constants import SPECTRUM_STORE_DECIMALS
from .version import __version__

ROOTS_PREFIX = 'roots_'
VALUES_PREFIX = 'values_'
STORE_SUFFIX = '.json'


def canonical_value(value, decimals=SPECTRUM_STORE_DECIMALS):
    """
    Return a JSON-serializable, order-independent form of ``value``.

    Dictionaries are rebuilt with string keys in sorted order, arrays and
    tuples become lists, and floats are rounded to ``decimals`` places so
    that values differing only by round-off share a context key.

    Parameters
    ----------
    value : object
        Value to canonicalize.
    decimals : int, optional
        Number of decimal places kept for floating-point entries.

    Returns
    -------
    object
        JSON-serializable representation of ``value``.
    """
    if isinstance(value, dict):
        return {str(key): canonical_value(value[key], decimals)
                for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(entry, decimals) for entry in value]
    if isinstance(value, np.ndarray):
        return canonical_value(value.tolist(), decimals)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(np.round(float(value), decimals))
    if value is None or isinstance(value, str):
        return value
    return str(value)


def context_key(context):
    """Return the short hexadecimal key identifying a context."""
    payload = json.dumps(canonical_value(context), sort_keys=True,
                         separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]


def coordinate_key(coordinate, decimals=SPECTRUM_STORE_DECIMALS):
    """Return the rounded string used to index a determined energy.

    Determined energies are indexed by a rounded volume so that volumes
    differing only by round-off share an entry.
    """
    rounded = float(np.round(float(coordinate), decimals))
    return f"{rounded:.{decimals}f}"


def exact_coordinate_key(coordinate):
    """Return the exact string used to index a memoized QC value.

    QC values are indexed at full precision: a root finder samples
    points far closer together than the rounding of ``coordinate_key``,
    and returning a neighbouring value for one of them would corrupt the
    search rather than accelerate it.
    """
    return repr(float(coordinate))


def policy_summary(qc_dict):
    """
    Summarize the evaluation policy of a ``qc_dict`` as plain data.

    Both the list form and the :class:`~ampyl.ampyl.EvaluationPolicy` form
    are accepted, as is the legacy ``'version'`` entry. The summary keeps
    the per-element energy-volume bounds and component IDs, so a policy
    that gains a new element produces a different context. Elements are
    normalized exactly as ``EvaluationPolicy`` normalizes them, so a
    ``qc_dict`` gives the same summary before and after the QC has
    validated it in place.

    Parameters
    ----------
    qc_dict : dict
        QC evaluation options.

    Returns
    -------
    list[dict]
        Canonicalized policy elements.
    """
    policy = qc_dict.get('policy')
    if policy is None:
        version = qc_dict.get('version', QC_DICT_DEFAULTS['version'])
        elements = [{'version': version}]
    else:
        elements = getattr(policy, 'elements', policy)
        if isinstance(elements, dict):
            elements = [elements]
    summary = []
    for element_id, element in enumerate(elements):
        entry = dict(element)
        entry['id'] = element_id
        for key in ('Lmin', 'Lmax', 'Emin', 'Emax'):
            entry.setdefault(key, None)
        summary.append(canonical_value(entry))
    return summary


def is_interpolation_flag(key):
    """Return whether a QC implementation flag selects an evaluation path."""
    return 'interp' in key


def qc_impl_summary(qc_impl, include_interpolation=False):
    """
    Summarize the QC implementation flags a solution depends on.

    Interpolation flags are excluded by default. A determined energy is a
    property of the quantization condition rather than of the path used
    to evaluate it: the residual dependence on an interpolator is a
    numerical error, controlled by the grid spacings the caller records
    in the context ``extra`` and removed altogether by ``refine_roots``,
    which is kept. Excluding them also keeps a context stable across the
    interpolation flags the solver itself toggles as it runs. Contexts
    for individual QC values do include them, since the same energy and
    volume genuinely give different numbers on different paths.

    Parameters
    ----------
    qc_impl : dict
        QC implementation flags in force.
    include_interpolation : bool, optional
        Whether interpolation flags are kept in the summary.

    Returns
    -------
    dict
        Canonicalized flags, with package defaults filled in.
    """
    keys = set(QC_IMPL_DEFAULTS) | set(qc_impl)
    if not include_interpolation:
        keys = {key for key in keys if not is_interpolation_flag(key)}
    return {key: canonical_value(qc_impl.get(key, QC_IMPL_DEFAULTS.get(key)))
            for key in sorted(keys)}


def qcis_summary(qcis):
    """
    Summarize the index space behind a spectrum calculation.

    The flavor-channel space is reduced to a digest of its string form,
    which captures masses, isospins, angular-momentum sets, and the
    ``p cot(delta)`` parametrizations without bloating the stored context.

    Parameters
    ----------
    qcis : object
        Quantization-condition index space, or ``None``.

    Returns
    -------
    dict
        Canonicalized summary of the index space.
    """
    if qcis is None:
        return {}
    summary = {'Emax': canonical_value(getattr(qcis, 'Emax', None)),
               'Lmax': canonical_value(getattr(qcis, 'Lmax', None))}
    fcs = getattr(qcis, 'fcs', None)
    if fcs is not None:
        digest = hashlib.sha1(str(fcs).encode('utf-8')).hexdigest()[:12]
        summary['fcs_digest'] = digest
    fvs = getattr(qcis, 'fvs', None)
    nP = getattr(fvs, 'nP', None)
    if nP is not None:
        summary['nP'] = canonical_value(nP)
    return summary


def build_context(qc, qc_dict, extra=None):
    """
    Build the context describing what a determined energy depends on.

    Parameters
    ----------
    qc : object
        QC instance whose active index space is being solved.
    qc_dict : dict
        QC evaluation options, including ``'k_params'`` and the policy.
    extra : dict, optional
        Caller-supplied entries, e.g. the interpolation-grid spacings of
        the active tile, which ampyl cannot infer from ``qc_dict``.

    Notes
    -----
    Package defaults are filled in for the optional ``qc_dict`` entries,
    so the context does not change when the QC validates the dictionary
    in place on its first evaluation.

    Returns
    -------
    dict
        Canonicalized context, ready to be hashed by ``context_key``.
    """
    qcis = getattr(qc, 'qcis', None)
    fvs = getattr(qcis, 'fvs', None)
    qc_impl = getattr(fvs, 'qc_impl', None)
    defaults = QC_DICT_DEFAULTS
    context = {
        'k_params': canonical_value(qc_dict.get('k_params')),
        'project': canonical_value(qc_dict.get('project',
                                               defaults['project'])),
        'irrep': canonical_value(qc_dict.get('irrep', defaults['irrep'])),
        'rescale': canonical_value(qc_dict.get('rescale',
                                               defaults['rescale'])),
        'shift': canonical_value(qc_dict.get('shift', defaults['shift'])),
        'policy': policy_summary(qc_dict),
        'qc_impl': qc_impl_summary({} if qc_impl is None else qc_impl),
        'qcis': qcis_summary(qcis),
    }
    if extra is not None:
        context['extra'] = canonical_value(extra)
    return context


class SpectrumValueStore:
    """
    Remember energies, and optionally QC values, determined by a solver.

    The store keeps one table of determined energies per context, indexed
    by band and volume, and writes each context to its own JSON file so
    that a run can be resumed, seeded from a neighbouring parameter set,
    or split across processes without write contention. Passing
    ``directory=None`` keeps everything in memory.

    Parameters
    ----------
    directory : str or pathlib.Path, optional
        Directory holding the JSON files. ``None`` disables disk access.
    autosave_every : int, optional
        Number of recorded energies after which a context is written to
        disk. Zero or a negative value disables autosaving.
    memoize_values : bool, optional
        Whether QC values, not just determined energies, are cached.

    Attributes
    ----------
    directory : pathlib.Path or None
        Directory used for persistence.
    memoize_values : bool
        Whether the QC-value table is active.
    """

    def __init__(self, directory=None, autosave_every=SPECTRUM_STORE_AUTOSAVE,
                 memoize_values=False):
        """Initialize an empty store, optionally backed by a directory."""
        self.directory = None if directory is None else Path(directory)
        self.autosave_every = int(autosave_every)
        self.memoize_values = bool(memoize_values)
        self._contexts = {}
        self._bands = {}
        self._meta = {}
        self._values = {}
        self._pending = {}
        self._loaded = set()
        self._values_loaded = set()

    def register(self, context):
        """
        Register a context and return its key, loading any stored data.

        Parameters
        ----------
        context : dict
            Context describing the calculation, from ``build_context``.

        Returns
        -------
        str
            Key identifying the context.
        """
        key = context_key(context)
        if key not in self._contexts:
            self._contexts[key] = canonical_value(context)
            self._bands.setdefault(key, {})
            self._meta.setdefault(key, {})
            self._pending.setdefault(key, 0)
        self._load_roots(key)
        return key

    def get_meta(self, context):
        """Return the metadata recorded alongside a context."""
        key = self.register(context)
        return dict(self._meta[key])

    def set_meta(self, context, **entries):
        """Update the metadata recorded alongside a context."""
        key = self.register(context)
        self._meta[key].update(canonical_value(entries))
        return self._meta[key]

    def put_root(self, context, band_index, L, E):
        """
        Record one determined energy.

        Parameters
        ----------
        context : dict
            Context describing the calculation.
        band_index : int
            Index of the energy band the solution belongs to.
        L : float
            Box length.
        E : float
            Determined energy. Non-finite values are ignored.
        """
        E = float(E)
        if not np.isfinite(E):
            return
        key = self.register(context)
        band = self._bands[key].setdefault(int(band_index), {})
        band[coordinate_key(L)] = E
        self._pending[key] = self._pending[key]+1
        if 0 < self.autosave_every <= self._pending[key]:
            self.save(context)

    def get_root(self, context, band_index, L):
        """Return one determined energy, or ``None`` if it is not stored."""
        key = self.register(context)
        band = self._bands[key].get(int(band_index))
        if band is None:
            return None
        return band.get(coordinate_key(L))

    def clear(self, context):
        """Discard every energy recorded against a context."""
        key = self.register(context)
        self._bands[key] = {}
        self._meta[key] = {}
        self._pending[key] = 0
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._save_roots(key)

    def band_count(self, context):
        """Return the number of bands recorded for a context."""
        key = self.register(context)
        return len(self._bands[key])

    def get_curves(self, context):
        """
        Return the stored energies as volume and energy band lists.

        Returns
        -------
        tuple[list[list[float]], list[list[float]]] or None
            Volume and energy values of each band, sorted by volume, or
            ``None`` when nothing is stored for the context.
        """
        key = self.register(context)
        bands = self._bands[key]
        if len(bands) == 0:
            return None
        L_vals = []
        E_vals = []
        for band_index in sorted(bands):
            band = bands[band_index]
            pairs = sorted((float(L_key), E) for L_key, E in band.items())
            L_vals.append([pair[0] for pair in pairs])
            E_vals.append([pair[1] for pair in pairs])
        return L_vals, E_vals

    def get_value(self, context, E, L):
        """Return a memoized QC value, or ``None`` if it is not stored."""
        if not self.memoize_values:
            return None
        key = context_key(context)
        self._load_values(key)
        table = self._values.get(key)
        if table is None:
            return None
        return table.get(self._value_key(E, L))

    def put_value(self, context, E, L, value):
        """Record one QC value for later lookup."""
        if not self.memoize_values:
            return
        key = context_key(context)
        if key not in self._contexts:
            self._contexts[key] = canonical_value(context)
            self._bands.setdefault(key, {})
            self._meta.setdefault(key, {})
            self._pending.setdefault(key, 0)
        self._load_values(key)
        self._values.setdefault(key, {})[self._value_key(E, L)] = float(value)

    def nearest_context(self, context, metric):
        """
        Return the stored context closest to ``context`` under ``metric``.

        Only contexts already registered, or read in by ``load``, are
        searched.

        Parameters
        ----------
        context : dict
            Reference context, excluded from the search.
        metric : callable
            Maps a stored context to a distance, or to ``None`` when the
            stored context is not comparable.

        Returns
        -------
        dict or None
            Closest stored context holding at least one energy.
        """
        reference_key = context_key(context)
        best_context = None
        best_distance = None
        for key in sorted(self._contexts):
            if key == reference_key:
                continue
            self._load_roots(key)
            if len(self._bands.get(key, {})) == 0:
                continue
            distance = metric(self._contexts[key])
            if distance is None or not np.isfinite(distance):
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_context = self._contexts[key]
        return best_context

    def contexts(self):
        """Return every context known to the store, keyed by context key."""
        return dict(self._contexts)

    def merge(self, other):
        """Merge the tables of another store into this one."""
        for key, context in other._contexts.items():
            self._contexts.setdefault(key, context)
            self._meta.setdefault(key, {})
            self._pending.setdefault(key, 0)
            bands = self._bands.setdefault(key, {})
            for band_index, band in other._bands.get(key, {}).items():
                bands.setdefault(band_index, {}).update(band)
            self._meta[key].update(other._meta.get(key, {}))
            values = other._values.get(key)
            if values is not None:
                self._values.setdefault(key, {}).update(values)
            self._pending[key] = self._pending[key]+1
        return self

    def load(self):
        """Load every context found in the directory."""
        if self.directory is None or not self.directory.is_dir():
            return self
        pattern = f"{ROOTS_PREFIX}*{STORE_SUFFIX}"
        for path in sorted(self.directory.glob(pattern)):
            key = path.name[len(ROOTS_PREFIX):-len(STORE_SUFFIX)]
            self._load_roots(key, path=path)
        return self

    def save(self, context=None):
        """
        Write one context, or every context, to the directory.

        Parameters
        ----------
        context : dict, optional
            Context to write. All contexts are written when omitted.
        """
        if self.directory is None:
            return
        if context is None:
            keys = sorted(self._contexts)
        else:
            keys = [context_key(context)]
        self.directory.mkdir(parents=True, exist_ok=True)
        for key in keys:
            self._save_roots(key)
            if self.memoize_values and key in self._values:
                self._save_values(key)
            self._pending[key] = 0

    def _value_key(self, E, L):
        """Return the string indexing one QC value."""
        return f"{exact_coordinate_key(E)}|{exact_coordinate_key(L)}"

    def _roots_path(self, key):
        """Return the path holding the determined energies of a context."""
        return self.directory/f"{ROOTS_PREFIX}{key}{STORE_SUFFIX}"

    def _values_path(self, key):
        """Return the path holding the memoized QC values of a context."""
        return self.directory/f"{VALUES_PREFIX}{key}{STORE_SUFFIX}"

    def _load_roots(self, key, path=None):
        """Load one context's energies from disk, at most once."""
        if key in self._loaded or self.directory is None:
            return
        self._loaded.add(key)
        if path is None:
            path = self._roots_path(key)
        if not path.is_file():
            return
        with open(path, 'r') as file_handle:
            payload = json.load(file_handle)
        self._contexts.setdefault(key, payload.get('context', {}))
        self._pending.setdefault(key, 0)
        meta = self._meta.setdefault(key, {})
        for meta_key, meta_value in payload.get('meta', {}).items():
            meta.setdefault(meta_key, meta_value)
        bands = self._bands.setdefault(key, {})
        for band_index, band in payload.get('bands', {}).items():
            stored = bands.setdefault(int(band_index), {})
            for L_key, E in band.items():
                stored.setdefault(L_key, float(E))

    def _load_values(self, key):
        """Load one context's memoized QC values from disk, at most once."""
        if key in self._values_loaded or self.directory is None:
            return
        self._values_loaded.add(key)
        path = self._values_path(key)
        if not path.is_file():
            return
        with open(path, 'r') as file_handle:
            payload = json.load(file_handle)
        table = self._values.setdefault(key, {})
        for value_key, value in payload.get('values', {}).items():
            table.setdefault(value_key, float(value))

    def _save_roots(self, key):
        """Write one context's energies to disk."""
        bands = {str(band_index): self._bands[key][band_index]
                 for band_index in sorted(self._bands.get(key, {}))}
        payload = {'ampyl_version': __version__,
                   'context': self._contexts.get(key, {}),
                   'meta': self._meta.get(key, {}),
                   'bands': bands}
        _write_json(self._roots_path(key), payload)

    def _save_values(self, key):
        """Write one context's memoized QC values to disk."""
        payload = {'ampyl_version': __version__,
                   'context': self._contexts.get(key, {}),
                   'values': self._values.get(key, {})}
        _write_json(self._values_path(key), payload)


def _write_json(path, payload):
    """Write ``payload`` to ``path`` through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name+f".tmp{os.getpid()}")
    with open(tmp_path, 'w') as file_handle:
        json.dump(payload, file_handle, sort_keys=True, indent=1)
    os.replace(tmp_path, path)


def load_store(directory, memoize_values=False,
               autosave_every=SPECTRUM_STORE_AUTOSAVE):
    """Return a store populated from every context file in ``directory``."""
    store = SpectrumValueStore(directory=directory,
                               autosave_every=autosave_every,
                               memoize_values=memoize_values)
    return store.load()
