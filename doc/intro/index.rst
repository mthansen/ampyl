Introduction
============

AmPyL is a Python package for relating finite-volume spectra to scattering
amplitudes. It provides tools for building flavor-channel spaces, organizing
finite-volume matrix indices, evaluating finite-volume functions, and solving
quantization conditions.

The package is designed around composable objects that keep the physics setup,
kinematic spaces, interaction schemes, and quantization-condition evaluation
separate. This makes it possible to define a system once, inspect the resulting
index structure, and reuse the same setup across scans in energy, volume, and
model parameters.

The API reference is generated directly from the package docstrings. Functions
and classes therefore document their arguments, return values, and expected
array shapes next to the implementation that defines them.
