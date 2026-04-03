.. _concepts.ops:

===============
Core Operations
===============

.. TODO: How much is guide or example?  This should be a concepts doc. Brief.

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

Here, we use the :func:`~hklpy2.diffract.creator()` to create a simulated 4-circle
diffractometer with the :ref:`E4CV <geometries-hkl_soleil-e4cv>` geometry.

EXAMPLE::

    >>> from hklpy2 import creator
    >>> e4cv = creator(name="e4cv")
    >>> e4cv.sample
    Sample(name='cubic', lattice=Lattice(a=1, b=1, c=1, alpha=90.0, beta=90.0, gamma=90.0))
    >>> e4cv.core.solver
    HklSolver(name='hkl_soleil', version='v5.0.0.3434', geometry='E4CV', engine='hkl', mode='bissector')
    >>> e4cv.sample.reflections
    {}
    >>> e4cv.add_reflection((1, 0, 0), (10, 0, 0, 20), name="r1")
    Reflection(name='r1', geometry='E4CV', pseudos={'h': 1, 'k': 0, 'l': 0}, reals={'omega': 10, 'chi': 0, 'phi': 0, 'tth': 20}, wavelength=1.0)
    >>> e4cv.add_reflection((0, 1, 0), (10, -90, 0, 20), name="r2")
    Reflection(name='r2', geometry='E4CV', pseudos={'h': 0, 'k': 1, 'l': 0}, reals={'omega': 10, 'chi': -90, 'phi': 0, 'tth': 20}, wavelength=1.0)
    >>> e4cv.sample.reflections
    {'r1': {'name': 'r1', 'geometry': 'E4CV', 'pseudos': {'h': 1, 'k': 0, 'l': 0}, 'reals': {'omega': 10, 'chi': 0, 'phi': 0, 'tth': 20}, 'wavelength': 1.0, 'order': 0}, 'r2': {'name': 'r2', 'geometry': 'E4CV', 'pseudos': {'h': 0, 'k': 1, 'l': 0}, 'reals': {'omega': 10, 'chi': -90, 'phi': 0, 'tth': 20}, 'wavelength': 1.0, 'order': 1}}

    Using the ``e4cv`` simulator, the property is::

        >>> e4cv.core.solver_signature
        "HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')"

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
      - Limits and conditions on real-axis values that filter the solutions
        returned by ``forward()``.
      - :ref:`concepts.constraints`
    * - Axis names
      - Cross-reference map between diffractometer axis names and solver axis
        names, set by ``assign_axes()``.
      - :ref:`diffract_axes`
    * - Presets
      - Per-mode constant-axis values the solver assumes during ``forward()``
        instead of the current motor positions; do not move any motor.
      - :ref:`how_presets`
