.. _overview:

========
Overview
========

|hklpy2| provides `ophyd <https://blueskyproject.io/ophyd>`_ diffractometer
devices.  Each diffractometer is a positioner which may be used with `bluesky
<https://blueskyproject.io/bluesky>`_ plans.

Any diffractometer may be provisioned with simulated axes; motors from an EPICS
control system are not required to use |hklpy2|.

Built from :class:`~hklpy2.diffract.DiffractometerBase()`, each diffractometer is
an `ophyd.PseudoPositioner
<https://blueskyproject.io/ophyd/positioners.html#pseudopositioner>`_ that
defines all the components of a diffractometer. The diffractometer
:ref:`geometry <geometries>` defines the names and order for the real motor
axes. Geometries are defined by backend  :ref:`concepts.solvers`. Some solvers
support different calculation engines (other than :math:`hkl`). It is common for a
geometry to support several operating *modes*.

.. _overview.architecture:

Package Architecture
--------------------

The diagrams below show how the major components of |hklpy2| fit together.
The first diagram gives a high-level overview; the following three provide
detail for each major section.

.. figure:: ../_static/hklpy2-overview.svg
   :alt: hklpy2 package architecture overview
   :align: center

   **Overview** -- major sections left to right: External, User-facing, Core,
   Solvers, Backend libraries.

.. figure:: ../_static/hklpy2-user.svg
   :alt: hklpy2 user-facing components
   :align: center

   **User-facing** -- Bluesky plans and EPICS hardware connect to
   :class:`~hklpy2.diffract.DiffractometerBase` and
   :class:`~hklpy2.wavelength.WavelengthBase`; ``creator()`` and
   ``hklpy2.user`` provide convenience wrappers.

.. figure:: ../_static/hklpy2-core.svg
   :alt: hklpy2 core components
   :align: center

   **Core** -- :class:`~hklpy2.ops.Core` manages the seven block classes
   (Sample, Lattice, Reflection, Constraints, Presets, Zone, Configuration)
   and delegates calculations to the solver.

.. figure:: ../_static/hklpy2-solvers.svg
   :alt: hklpy2 solver components
   :align: center

   **Solvers** -- :class:`~hklpy2.solvers.base.SolverBase` is the adapter
   interface; built-in solvers (HklSolver, ThTthSolver, NoOpSolver) and
   additional solvers registered via entry points all subclass it.

.. seealso:: :ref:`glossary`
