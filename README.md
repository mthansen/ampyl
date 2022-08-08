[![Build/Test](https://github.com/mthansen/ampyl/workflows/Build/Test/badge.svg)](https://github.com/mthansen/ampyl/actions?query=workflow%3ABuild%2FTest)
![Build Doc](https://github.com/mthansen/ampyl/workflows/Build%20Doc/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<img src="/doc/ampyl-logo.png" width="25%">

A Python library to relate finite-volume data to amplitudes.

The name AmPyL stands for **Am**plitdues via **Py**thon from finite-volume (**L**). The package requires python version 3.8.x or newer.

- **Website:** https://github.com/mthansen/ampyl
- **Documentation:** https://mthansen.github.io/ampyl/
- **Examples:** [tutorials](./tutorials)
- **Bug reports:** https://github.com/mthansen/ampyl/issues

### Authors

Copyright (C) 2022, Maxwell T. Hansen

## Installation

The NumPy package is required to use AmPyL. The latest version AmPyL can be installed locally, e.g. using `pip`:

```
git clone https://github.com/mthansen/ampyl
cd ampyl
pip install .

python
import numpy
import ampyl
```

## Example

```
import numpy as np
import ampyl

# single 3-particle channel
fc = ampyl.FlavorChannel(3)
fcs = ampyl.FlavorChannelSpace(fc_list=[fc])

fvs = ampyl.FiniteVolumeSetup()
tbis = ampyl.ThreeBodyInteractionScheme()

qcis = ampyl.QCIndexSpace(fcs=fcs, fvs=fvs,
                          tbis=tbis,
                          Emax=5.0, Lmax=6.0)
qc = ampyl.QC(qcis=qcis)

k_params = qcis.default_k_params()

L = 5.0
E_vals = np.arange(3.01, 4., 0.1)
qc_vals = np.array([])
for E in Evals:
    qc_tmp = qc.get_value(E, L, k_params, irrep='A1PLUS')
    qc_vals = np.append(qc_vals, qc_tmp)
print(np.array([E_vals, qc_vals].T))
```
