.. _concepts.ops:

===============
Core Operations
===============

.. index:: !core

A diffractometer's ``.core`` provides most of its functionality.
The ``core`` conducts transactions with the |solver| on behalf of the
diffractometer. These transactions include the ``forward()`` and ``inverse()``
coordinate transformations, at the core of scientific measurements using
a diffractometer.

=============================================   ==============
Python class                                    Purpose
=============================================   ==============
:class:`~hklpy2.diffract.DiffractometerBase`    ophyd `PseudoPositioner <https://blueskyproject.io/ophyd/user/reference/positioners.html#pseudopositioner>`_
:class:`~hklpy2.ops.Core`                       The class for a diffractometer's ``.core`` operations.
:class:`~hklpy2.backends.base.SolverBase`       Code for diffractometer geometries and capabilities.
=============================================   ==============

In addition to |solver| transactions, the ``.core`` manages all
details involving the set of samples and their lattices & reflections.

.. seealso:: :ref:`guide.diffract` for step-by-step instructions on creating
   and using a diffractometer object.

.. index:: presets

Core concepts
-------------

The table below summarizes the main topics managed by ``.core`` and points to
where each is described in more detail.

.. list-table::
    :header-rows: 1
    :widths: 20 45 35

    * - Concept
      - Summary
      - More detail
    * - Solver
      - Backend library that implements diffractometer geometries, modes, and
        the ``forward()`` / ``inverse()`` transformations.
      - :ref:`concepts.solvers`
    * - Geometry & mode
      - The physical arrangement of axes (e.g. E4CV) and the computation mode
        (e.g. ``bissector``, ``constant_phi``) that controls how the solver
        assigns axis values.
      - :ref:`concepts.diffract`
    * - Wavelength
      - Beam wavelength used in ``forward()`` / ``inverse()`` calculations;
        defaults to 1 Å.
      - :ref:`concepts.wavelength`
    * - Sample & lattice
      - Crystal sample with its unit-cell lattice parameters; multiple samples
        can be registered and switched between.
      - :ref:`concepts.sample`, :ref:`concepts.lattice`
    * - Reflections & UB matrix
      - Measured orientation reflections used to compute the UB orientation
        matrix via ``calc_UB()``.
      - :ref:`concepts.reflection`
    * - Constraints
      - Limits on real-axis ranges that filter the solutions returned by
        ``forward()``.  Act *after* computation.
      - :ref:`concepts.constraints`
    * - Axis names
      - Cross-reference map between diffractometer axis names and solver axis
        names, set by ``assign_axes()``.
      - :ref:`diffract_axes`
    * - Presets
      - Per-mode constant-axis values the solver assumes during ``forward()``
        instead of the current motor positions; do not move any motor.
        Act *before* computation.
      - :ref:`concepts.presets`
