"""Helper for the KKpi ground-state example (A1PLUS irrep)."""
import hashlib
from pathlib import Path

import ampyl

try:
    import dill
except ImportError:
    dill = None


MPI = 1.0
MK = 0.09698/0.06906  # lattice values a*m_K = 0.09698, a*m_pi = 0.06906
EMIN = 2.0*MK+MPI


def build_qc(qcis, grid, project, irrep, precomputed_dir=None):
    """Return a QC whose F+G matrix is spline-interpolated.

    The interpolator covers the energy-volume window specified by
    ``grid`` (keys ``Emin``, ``Emax``, ``dE``, ``Lmin``, ``Lmax`` and
    ``dL``). When ``precomputed_dir`` is given and ``dill`` is
    installed, the interpolator is cached there after the first build
    and re-loaded on later calls; otherwise it is rebuilt each time.
    """
    cache_path = None
    cached_fplusg = None
    if precomputed_dir is not None and dill is not None:
        cache_path = _cache_path(precomputed_dir, qcis, grid,
                                 project, irrep)
        cached_fplusg = _load_cache(cache_path)

    qcis.fvs.qc_impl['fplusg_interpolate'] = False
    qc = ampyl.QC(qcis=qcis)
    if cached_fplusg is not None:
        qc.fplusg = qc.fplusg_list.replace(0, cached_fplusg)
        qc.matrix_builder.fplusg = qc.fplusg
    else:
        qc.fplusg.build_interpolator(
            grid['Emin'], grid['Emax'], grid['dE'],
            grid['Lmin'], grid['Lmax'], grid['dL'],
            project, irrep, name='ground_state')
        if cache_path is not None:
            _save_cache(cache_path, qc.fplusg)
    qcis.fvs.qc_impl['fplusg_interpolate'] = True
    return qc


def make_qc_dict(qc, scattering_lengths, project, irrep):
    """Build a qc_dict with one scattering length per spectator channel.

    The evaluation policy has a single element: the Kdf-zero version
    ``'kdf_zero_1+_fgcombo'``, evaluated with the F+G interpolator
    built by ``build_qc``.
    """
    k_params = qc.qcis.default_k_params()
    if len(scattering_lengths) != len(k_params[0]):
        raise ValueError(
            "scattering_lengths must have one entry per spectator "
            f"channel ({len(k_params[0])})")
    for channel_params, scattering_length in zip(k_params[0],
                                                 scattering_lengths):
        for param_index in range(len(channel_params)):
            channel_params[param_index] = scattering_length
    policy = [{'version': 'kdf_zero_1+_fgcombo',
               'fplusg_interpolator': True,
               'fplusg_interpolator_id': 0}]
    return {'k_params': k_params, 'project': project, 'irrep': irrep,
            'policy': policy}


def _cache_path(directory, qcis, grid, project, irrep):
    payload = repr((sorted(grid.items()), project, irrep,
                    qcis.Emax, qcis.Lmax))
    key = hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]
    return Path(directory)/f"fplusg_{key}.dill"


def _load_cache(path):
    if not path.is_file():
        return None
    with open(path, 'rb') as file_handle:
        return dill.load(file_handle)


def _save_cache(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as file_handle:
        dill.dump(obj, file_handle)
