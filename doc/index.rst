.. |logo| image:: ./ampyl-logo.png
    :width: 60pt

ampyl |release|
================

A Python package to relate finite-volume data to amplitudes.

.. The current stable version is |release|

The software is hosted on `GitHub <https://github.com/mthansen/ampyl>`__ and is distributed
under the MIT license.

Features
--------

* object-oriented for easy manipulation of...

  -- flavor-channel and other index spaces

  -- finite-volume matrices

  -- quantization conditions

  -- other relevant finite-volume and scattering quantities

Documentation
-------------

.. toctree::
   :maxdepth: 4

   intro/index
   ampyl/index

Class documentation
~~~~~~~~~~~~~~~~~~~

* ``ampyl.py``

  * :class:`ampyl.ampyl.IdentifiedObjectList`
  * :class:`ampyl.ampyl.EvaluationPolicy`
  * :class:`ampyl.ampyl.QCMatrixBuilder`
  * :class:`ampyl.ampyl.QCVersionEvaluator`
  * :class:`ampyl.ampyl.QC`
  * :class:`ampyl.ampyl.FVSpectrum`

* ``cuts.py``

  * :class:`ampyl.cuts.G`
  * :class:`ampyl.cuts.F`
  * :class:`ampyl.cuts.FplusG`

* ``flavor.py``

  * :class:`ampyl.flavor.Particle`
  * :class:`ampyl.flavor.FlavorChannel`
  * :class:`ampyl.flavor.SpectatorChannel`
  * :class:`ampyl.flavor.FlavorChannelSpace`

* ``groups.py``

  * :class:`ampyl.groups.Groups`

* ``interpolable.py``

  * :class:`ampyl.interpolable.Interpolable`

* ``k_matrices.py``

  * :class:`ampyl.k_matrices.K`
  * :class:`ampyl.k_matrices.Kdf`

* ``schemes.py``

  * :class:`ampyl.schemes.FiniteVolumeSetup`
  * :class:`ampyl.schemes.ThreeBodyInteractionScheme`

* ``spaces.py``

  * :class:`ampyl.spaces.ThreeBodyKinematicSpace`
  * :class:`ampyl.spaces.QCIndexSpace`

Authors
-------

Maxwell T. Hansen, Copyright (C) 2022
