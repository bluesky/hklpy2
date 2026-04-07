.. _guides:

==========
Guides
==========

.. toctree::
   :glob:
   :hidden:

   guides/*

Guides, how-to documents, notebooks and tutorials.

Getting started
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :ref:`guide.diffract`
     - Define a diffractometer class, connect it to a solver, and use it
       with the bluesky RunEngine.
   * - :ref:`guide.solvers`
     - List, select, and instantiate solvers; understand solver entry points.

Diffractometer axes
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :doc:`guides/how_extra_reals_and_pseudos`
     - Add extra real motors or pseudo axes beyond the solver defaults.
   * - :doc:`guides/how_additional_parameters`
     - Pass extra solver parameters (e.g. ``psi``) to the backend.

Computation
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :ref:`how_forward_solution`
     - Choose which ``forward()`` solution the diffractometer uses by default.
   * - :ref:`how_presets`
     - Hold a real axis at a fixed value during ``forward()`` without moving
       any motor (SPEC ``freeze``/``unfreeze`` equivalent).
   * - :doc:`guides/var_engines`
     - Switch between calculation engines (e.g. ``hkl``, ``q``) on the same
       geometry.

Configuration and solvers
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :doc:`guides/configuration_save_restore`
     - Save a full diffractometer configuration (orientation, samples,
       reflections) and restore it later.
   * - :ref:`how_creator_from_config`
     - Create a simulated diffractometer directly from a saved config file.
   * - :ref:`howto.solvers.write`
     - Write and register a new solver plugin using Python entry points.

Reference and background
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Description
   * - :ref:`spec_commands_map`
     - Cross-reference of common SPEC commands to their |hklpy2| equivalents.
   * - :ref:`guide.migration_from_hklpy_v1`
     - How code written for |hklpy| (v1) maps to |hklpy2|.
   * - :ref:`guide.design`
     - Design rationale and architectural decisions behind |hklpy2|.
   * - :ref:`v2_checklist`
     - Feature checklist tracking the v2 build (historical reference).
