.. _examples:

==========
Examples
==========

These notebooks are **worked demonstrations** — they show |hklpy2| working
on specific use cases with real (or simulated) hardware configurations.
They are not tutorials: they assume you already know the basic workflow.
If you are new to |hklpy2|, start with the :ref:`tutorial` first.

The notebooks are available for download from the source code website:
https://github.com/bluesky/hklpy2/docs/source/examples.

.. toctree::
   :glob:
   :hidden:

   examples/*

Getting started
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/example-4-circle-creator`
     - Simulated 4-circle diffractometer built with :func:`~hklpy2.diffract.creator`.
   * - :doc:`examples/example-4-circle-custom-class`
     - Same diffractometer built as a custom Python subclass.
   * - :doc:`examples/_api_demo`
     - Quick tour of the |hklpy2| API.

Eulerian geometries (hkl_soleil)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/hkl_soleil-e4ch`
     - E4CH (horizontal scattering) 4-circle geometry.
   * - :doc:`examples/hkl_soleil-e4cv`
     - E4CV (vertical scattering) 4-circle geometry.
   * - :doc:`examples/hkl_soleil-e4cv+epics`
     - E4CV with real EPICS motor PVs.
   * - :doc:`examples/hkl_soleil-e6c-psi`
     - E6C with the extra ``psi`` axis.
   * - :doc:`examples/hkl_soleil-e6c-test_calculations`
     - E6C forward/inverse calculation validation.

Kappa geometries (hkl_soleil)
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/hkl_soleil-k4cv`
     - K4CV kappa 4-circle geometry.

Orientation matrix
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/hkl_soleil-ub_calc`
     - Calculate the UB matrix from two orientation reflections.
   * - :doc:`examples/hkl_soleil-ub_set`
     - Set the UB matrix directly.
   * - :doc:`examples/hkl_soleil-lattice_refine`
     - Refine lattice parameters from three or more reflections.

Constraints
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :ref:`examples.constraints`
     - Filter ``forward()`` solutions by real-axis range; use presets to
       hold axes fixed during computation.

Advanced topics
---------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/virtual-axis`
     - Add a virtual (computed) axis to a diffractometer.
   * - :doc:`examples/zone-scan`
     - Scan along a crystallographic zone axis.
   * - :doc:`examples/tst_e4cv_fourc`
     - Cross-validate |hklpy2| results against SPEC ``fourc``.

Real instrument examples
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Example
     - Description
   * - :doc:`examples/aps-isn-e6c`
     - APS ISN 6-circle diffractometer with renamed axes.
   * - :doc:`examples/nslsii-tardis`
     - NSLS-II TARDIS diffractometer.
