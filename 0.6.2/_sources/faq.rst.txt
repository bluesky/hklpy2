.. index:: ! FAQ

.. _FAQ:

===
FAQ
===

.. rubric:: Frequently Asked Questions

**General**

#. :ref:`faq.ophyd-version`
#. :ref:`faq.simulation`
#. :ref:`faq.units`
#. :ref:`faq.import-gi`

**Computation**

5. :ref:`faq.no-solutions`
#. :ref:`faq.constraints-vs-presets`
#. :ref:`faq.ub-wrong`

**SPEC users**

8. :ref:`faq.spec-wh-pa`
#. :ref:`faq.azimuthal`

----

.. _faq.ophyd-version:

Is |hklpy2| ophyd v1 only, or can it be used with ophyd async?
---------------------------------------------------------------

|hklpy2| uses ophyd v1 (synchronous).  Ophyd async (v2) is not yet supported.
See :issue:`334` for a feasibility analysis of migrating to ophyd-async.

----

.. _faq.simulation:

Is there a way to use |hklpy2| without real hardware?
------------------------------------------------------

Yes.  Simulators are easy to create for any defined solver and geometry.
Use :func:`~hklpy2.diffract.creator` without supplying EPICS PV names — all
real axes default to soft (simulated) positioners::

    import hklpy2
    e4cv = hklpy2.creator(name="e4cv")

.. seealso::

   :ref:`tutorial` — full guided walkthrough using a simulated E4CV diffractometer.

   :doc:`/examples/example-4-circle-creator` — worked demonstration.

----

.. _faq.units:

What units does |hklpy2| use?
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Quantity
     - Default unit
     - Notes
   * - Wavelength
     - Å (angstroms)
     - Configurable; energy conversion available for X-rays via
       :class:`~hklpy2.incident.WavelengthXray`.
   * - Real axes (angles)
     - degrees
     - All motor positions reported and accepted in degrees.
   * - Reciprocal-space axes (h, k, l)
     - reciprocal of wavelength unit (Å\ :sup:`-1`)
     - Dimensionless Miller indices when wavelength is in Å.
   * - Lattice parameters (a, b, c)
     - Å (angstroms)
     - Must match the wavelength unit.
   * - Lattice angles (α, β, γ)
     - degrees
     - —

----

.. _faq.import-gi:

``import gi`` raises an ``ImportError`` when starting the bluesky queue-server.
--------------------------------------------------------------------------------

This is tracked as :issue:`69`.  The problem occurs on linux 64-bit systems
when the ``libgobject`` shared library is not on ``LD_LIBRARY_PATH``.  Either
of these approaches works around it until it is resolved upstream:

1. Set the environment variable when starting the queue-server::

       LD_LIBRARY_PATH=${CONDA_PREFIX}/lib start-re-manager <options>

2. Configure the conda environment to set this on activation::

       conda env config vars set LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

----

.. _faq.no-solutions:

``forward()`` (or ``cahkl()``) returns no solutions when I expect some.
------------------------------------------------------------------------

The most common causes, in order of likelihood:

1. **Constraints are too narrow.**  Check ``diffractometer.core.constraints``.
   Each real axis has a ``low_limit`` and ``high_limit``; the solver silently
   discards any candidate solution where an axis falls outside those limits.
   Widen the range for the blocking axis::

       e4cv.core.constraints["omega"].limits = -180, 180

2. **The requested** :math:`hkl` **is outside the Ewald sphere** at the current
   wavelength.  Use a shorter wavelength (higher energy) or request a lower-angle
   reflection.

3. **Axes out of order in** ``axes_xref``\ **.**  If you supplied real-axis names
   in a different order than the solver expects, the cross-reference map silently
   swaps axes.  A swapped detector angle and sample rotation angle will produce
   Q = 0, yielding no valid solutions.  Inspect
   ``diffractometer.core.axes_xref`` and, if necessary, supply
   ``_real=["name1", "name2", ...]`` to :func:`~hklpy2.diffract.creator` in
   solver order.  See :ref:`diffract_axes.reals-out-of-order`.

.. seealso::

   :ref:`how_constraints` — set limits, cut points, and write custom constraints.

   :ref:`concepts.constraints`, :ref:`concepts.presets`

----

.. _faq.constraints-vs-presets:

What is the difference between constraints and presets?
--------------------------------------------------------

Both involve holding real axes at specific values, but they act at different
points in the ``forward()`` computation:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - :ref:`Presets <concepts.presets>`
     - :ref:`Constraints <concepts.constraints>`
   * - **When**
     - *Before* computation — the solver *assumes* this value for a constant
       axis instead of the current motor position.
     - *After* computation — solutions where an axis falls outside the allowed
       range are discarded.
   * - **Effect on motors**
     - None — no motor moves.
     - None — no motor moves.
   * - **Typical use**
     - SPEC ``freeze``: hold ``phi=0`` while the solver finds ``omega``,
       ``chi``, ``tth``.
     - Restrict ``chi`` to ±90° to avoid sample collision.

.. seealso::

   :ref:`how_constraints` — set axis limits and cut points.

   :ref:`how_presets` — freeze an axis at a fixed value during ``forward()``.

----

.. _faq.ub-wrong:

The UB matrix looks wrong, or ``pa()`` and ``configuration`` show different values.
------------------------------------------------------------------------------------

Three common causes:

1. **Axis mapping is swapped.**  If ``axes_xref`` has detector angles wired to
   sample-rotation slots (or vice versa), the UB calculation receives the wrong
   angles and produces a degenerate or incorrect matrix.  Verify
   ``diffractometer.core.axes_xref`` matches your physical wiring.
   See :ref:`diffract_axes.reals-out-of-order`.

2. **Display precision.**  ``pa()`` rounds values for readability; the stored
   matrix retains full floating-point precision.  A sign difference (e.g.
   ``+0.0101`` vs. ``-0.0101``) on a near-zero element is a display rounding
   artefact.  Retrieve the full matrix via ``diffractometer.sample.UB``.

3. **The UB is :term:`stale UB <stale ub>`.**  An orienting reflection was
   added, removed, reordered, or edited after ``calc_UB()`` last ran, so
   the stored matrix no longer reflects the chosen pair.  ``pa()`` shows
   ``UB stale: True`` and ``forward()`` / ``inverse()`` emit a
   ``UserWarning``.  Check
   ``diffractometer.sample.UB_is_stale``; refresh by calling
   ``calc_UB()`` again.  See :ref:`how_calc_ub.stale`.

.. seealso::

   :ref:`how_calc_ub` — add reflections, compute, inspect, and refine the UB matrix.

----

.. _faq.spec-wh-pa:

What are the |hklpy2| equivalents of SPEC's ``wh`` and ``pa``?
---------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - SPEC command
     - |hklpy2| equivalent
     - Notes
   * - ``wh``
     - ``diffractometer.wh()``
     - Shows current pseudos and reals.  Pass ``full=True`` for more detail.
   * - ``pa``
     - ``diffractometer.core.pa()``
     - Full orientation report: lattice, reflections, UB matrix, constraints,
       presets.
   * - ``freeze`` / ``unfreeze``
     - :ref:`how_presets`
     - Use ``diffractometer.core.presets``.
   * - ``cuts`` (cut points)
     - :ref:`concepts.constraints`
     - Use ``diffractometer.core.constraints``.
   * - ``ca`` / ``cahkl``
     - ``diffractometer.forward(h, k, l)``
     - Returns all solutions; the diffractometer's ``forward()`` picks one.
   * - ``ubr`` / ``or0`` / ``or1``
     - :meth:`~hklpy2.diffract.DiffractometerBase.add_reflection`
     - Record an orientation reflection.

See :ref:`spec_commands_map` for a comprehensive cross-reference.

----

.. _faq.azimuthal:

Can I perform azimuthal (ψ) scans?
------------------------------------

Azimuthal scans — rotating the sample about the scattering vector
:math:`\mathbf{Q}` at fixed :math:`hkl` — are supported using
:meth:`~hklpy2.diffract.DiffractometerBase.scan_extra` with the
``psi_constant`` mode of the ``hkl_soleil`` solver.

.. seealso::

   :doc:`guides/how_psi_scan` — step-by-step how-to guide for E4CV,
   including realistic motor constraints and pre-scan verification.

   :doc:`/examples/hkl_soleil-e6c-psi` — worked E6C demonstration,
   including the inverse case (reading ψ from real motor positions).

   :issue:`188` — tracking issue; a convenience wrapper ``scan_psi()``
   may be added in a future release.
