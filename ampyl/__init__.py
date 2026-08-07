r'''
# What is ampyl?

`ampyl` is a Python package for relating finite-volume spectra to scattering
amplitudes. It provides tools for building flavor-channel spaces, organizing
finite-volume matrix indices, evaluating finite-volume functions, and solving
quantization conditions.

The name AmPyL ("am-pie-ell") stands for **Am**plitudes via **Py**thon from
finite-volume (**L**) data.

The package is designed around composable objects that keep the physics setup,
kinematic spaces, interaction schemes, and quantization-condition evaluation
separate. This makes it possible to define a system once, inspect the resulting
index structure, and reuse the same setup across scans in energy, volume, and
model parameters.

## Installation

Install the current local checkout with pip:

```bash
git clone https://github.com/mthansen/ampyl
cd ampyl
python -m pip install .
```

## Basic example

```python
import ampyl
from scipy.optimize import root_scalar

fc = ampyl.FlavorChannel(3)
fcs = ampyl.FlavorChannelSpace(fc_list=[fc])
qcis = ampyl.QCIndexSpace(fcs=fcs, Emax=5.0, Lmax=6.0)
qcis.populate()

qc = ampyl.QC(qcis=qcis)
k_params = qcis.default_k_params()
k_params[0][0][0] = 0.1

qc_dict = {'k_params': k_params,
           'project': True,
           'irrep': ("A1PLUS", 0)}
root = root_scalar(
    qc.get_value,
    args=(5.0, qc_dict),
    bracket=[3.001, 3.1],
).root
```

## Core classes

- `ampyl.flavor.Particle`
- `ampyl.flavor.FlavorChannel`
- `ampyl.flavor.SpectatorChannel`
- `ampyl.flavor.FlavorChannelSpace`
- `ampyl.spaces.ThreeBodyKinematicSpace`
- `ampyl.spaces.QCIndexSpace`
- `ampyl.ampyl.QC`
- `ampyl.ampyl.FVSpectrum`

The full API reference below is generated directly from the package docstrings.
'''

__all__ = []
from .version import __version__, __version_full__
__all__.extend(["__version__"])
from .ampyl import *
from .flavor import FlavorChannel, FlavorChannelSpace
from .spaces import QCIndexSpace
__all__.extend(["FlavorChannel", "FlavorChannelSpace", "QCIndexSpace"])
from . import kinematic_functions
from . import qc_functions
