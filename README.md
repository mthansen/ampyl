[![Build/Test](https://github.com/mthansen/ampyl/workflows/Build/Test/badge.svg)](https://github.com/mthansen/ampyl/actions?query=workflow%3ABuild%2FTest)
![Docs](https://github.com/mthansen/ampyl/workflows/Docs/badge.svg)
[![codecov](https://codecov.io/gh/mthansen/ampyl/branch/main/graph/badge.svg?token=IR43OJAV6T)](https://codecov.io/gh/mthansen/ampyl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="/doc/ampyl-logo.png" width="25%">

A Python package to relate finite-volume data to amplitudes.

The name AmPyL ("am-pie-ell") stands for **Am**plitdues via **Py**thon from finite-volume (**L**) data. The package requires python version 3.9.x or newer.

- **Website:** https://github.com/mthansen/ampyl
- **Documentation:** https://mthansen.github.io/ampyl/
- **Examples:** [examples](./examples)
- **Bug reports:** https://github.com/mthansen/ampyl/issues

### Authors

Copyright (C) 2022, Maxwell T. Hansen

## Installation

The NumPy package is required to use AmPyL. The latest version AmPyL can be installed locally, e.g. using `pip`:

```python
git clone https://github.com/mthansen/ampyl
cd ampyl
pip install .

python
import numpy
import ampyl
```

## Documentation

The documentation is generated from the package and docstrings with
pdoc. To build it locally, install the documentation dependencies and run:

```bash
pip install -e ".[docs]"
make -C doc html
```

The generated site is written to `doc/_build/html/index.html`. The same pdoc
build runs on GitHub Actions for pull requests and publishes from `main` with
GitHub Pages.

## Example

```python
import ampyl
from ampyl.flavor import FlavorChannel
from ampyl.flavor import FlavorChannelSpace
from ampyl.spaces import QCIndexSpace
from scipy.optimize import root_scalar

# single 3-particle channel
fc = FlavorChannel(3)
fcs = FlavorChannelSpace(fc_list=[fc])
qcis = QCIndexSpace(fcs=fcs,
                    Emax=5.0, Lmax=6.0)
qcis.populate()
qc = ampyl.QC(qcis=qcis)
k_params = qcis.default_k_params()
# k_params default is [[[0.0]], [0.0]]
# first entry is the scattering length
# second entry is kdf (3-body
# interaction):
L = 5.
k_params[0][0][0] = 0.1  # scattering length
qc_dict = {'k_params': k_params,
           'project': True,
           'irrep': ('A1PLUS', 0)}
args = (L, qc_dict)
bracket = [3.001, 3.1]
print(root_scalar(qc.get_value, args=args,
                  bracket=bracket).root
     )
# Returns ground state energy
# around 3.031816
```

## Evaluation policy

The example above passes no `'version'` and no `'policy'`. That is the common
case: `qc.get_value` then evaluates the default quantization condition,
`'kdf_zero_1+'`, using the matrix objects that `ampyl.QC` builds for itself.

A **policy** is the part of `qc_dict` that is allowed to depend on the point
`(E, L)` being evaluated. It does not describe the physical system --- that
lives in `FlavorChannel`, `FlavorChannelSpace` and `QCIndexSpace` --- it
selects *how* a value is computed: which QC version, which registered matrix
object, and whether an interpolator is used.

A policy is a list of dictionaries (a single dictionary is promoted to a
one-element list). Every element requires `'version'`; the remaining keys are
optional:

```python
policy = [{'version': 'kdf_zero_1+_fgcombo',
           'fplusg_interpolator': True,
           'fplusg_interpolator_id': 0,
           'Emin': 3.0, 'Emax': 4.5,
           'Lmin': 3.0, 'Lmax': 6.0},
          {'version': 'kdf_zero_1+'}]  # default element, no bounds
qc_dict = {'k_params': k_params,
           'project': True,
           'irrep': ('A1PLUS', 0),
           'policy': policy}
```

Elements are scanned in list order and the first one whose `(E, L)` box
contains the evaluation point is used. The last element is the fallback and
must leave all four bounds unset. An element is only eligible to match if all
four of `Emin`, `Emax`, `Lmin` and `Lmax` are given; an element with a partial
range never matches and falls through to the fallback.

Keys that may be set per element:

| Key | Effect |
| --- | --- |
| `'version'` | QC expression to evaluate (required) |
| `'shift'` | Constant subtracted from the determinant, for versions that use it |
| `'f_id'`, `'f_name'` (likewise `g`, `fplusg`, `ftwo`, `k`, `kdf`, `ktwo`) | Which registered matrix object to use; defaults to id `0` |
| `'f_interpolator'`, `'f_interpolator_id'`, `'f_interpolator_name'` (likewise `g`, `fplusg`, `ftwo`) | Which interpolator to evaluate through |

The remaining `qc_dict` keys --- `'k_params'`, `'project'`, `'irrep'` and
`'rescale'` --- are global to the call and cannot vary by element.

### When a policy is needed

- **An interpolator was built.** `qc.fplusg.build_interpolator(...)` creates a
  spline but does not by itself route evaluations through it; the
  `'fplusg_interpolator'` key does. See
  [examples/KKpi](./examples/KKpi).
- **More than one matrix object is registered.** `qc.add_fplusg(qcis_id=0,
  name='coarse')` and friends append to per-component registries. Without a
  policy element every component resolves to id `0`, so the extra objects are
  unreachable.
- **Different formulas are wanted in different windows**, for example an
  interpolated evaluation over the bulk of a scan and an exact one near
  threshold.

Otherwise the policy can be ignored; `'version': ...` in `qc_dict` is simply a
shorthand for a one-element policy with no bounds, and omitting both selects
the default version. Note that `'policy'` takes precedence over `'version'`
when both are supplied, and that `FVSpectrum` sizes its scan window from the
version of the fallback element.
