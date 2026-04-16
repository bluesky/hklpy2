.. _how_performance:

=================================================
Understanding forward() and inverse() Performance
=================================================

.. index::
    !performance
    forward() performance
    inverse() performance
    fly scan
    throughput

|hklpy2| targets a minimum throughput of **2,000** :meth:`~hklpy2.diffract.DiffractometerBase.forward`
and :meth:`~hklpy2.diffract.DiffractometerBase.inverse` operations per second.
This matters most for fly scans, where ``forward()`` pre-computes motor
trajectories and ``inverse()`` labels encoder positions with reciprocal-space
coordinates at high repetition rates.

``forward()`` and ``inverse()`` are **purely computational** — they perform
no hardware communication and do not move any motors.  For local solvers
(such as the default ``hkl_soleil``), throughput depends only on the CPU and
the solver library, not on the state of the hardware control system, EPICS
IOCs, or motor controllers.  A network-based solver (such as a future SPEC
backend) would add network round-trip latency to every call.

The actual throughput you observe depends on several factors, grouped below
into those within |hklpy2|'s control and those outside it.

Factors outside hklpy2's control
---------------------------------

Workstation and OS
    CPU speed, memory bandwidth, and OS scheduling all affect raw Python
    execution time.  A heavily loaded workstation or one with many competing
    processes will deliver lower throughput than a quiet one.  Background
    tasks such as file indexing, anti-virus scans, or other data-acquisition
    processes can cause intermittent slowdowns.

Solver library
    |hklpy2| delegates the core crystallographic computation to an external
    solver.  For the default ``hkl_soleil`` solver this is the C library
    ``libhkl``.  The solver's own speed is outside |hklpy2|'s control.

    ``inverse()`` (angles → *hkl*) is generally much faster than
    ``forward()`` (*hkl* → angles) because the solver computes one result,
    whereas ``forward()`` must enumerate all mathematically valid solutions
    for the given geometry before |hklpy2| can apply constraints and pick one.

Number of solutions returned by the solver
    Some geometry/mode combinations return many theoretical solutions (e.g.
    ``E4CV bissector`` returns up to 18).  Each solution requires unit
    conversion and constraint checking, so modes that return more solutions
    cost proportionally more time in ``forward()``.  Modes that constrain the
    solution space more tightly (e.g. ``constant_phi``) return fewer solutions
    and are therefore faster.

Factors within hklpy2's control
---------------------------------

Unit conversion overhead
    Every axis value passed to or received from the solver goes through a unit
    conversion step.  When the diffractometer and solver use the same units
    (the common case), the conversion is skipped entirely.  Earlier versions
    of |hklpy2| performed the full ``pint`` conversion even for identical
    units, which accounted for more than 90 % of ``forward()`` time.

    If you write a custom solver, declare its units to match the diffractometer
    units wherever possible.

Mode and geometry
    Different solver modes return different numbers of solutions.  Choose a
    mode that is appropriate for your experiment; a more constrained mode
    (fewer free axes) will typically be faster as well as less ambiguous.

Number of axes
    Diffractometers with more real axes (e.g. 6-circle) have more unit
    conversions per call than 4-circle geometries, and the solver must
    enumerate a larger solution space.

Performance target
-------------------

The project benchmark (``test_i221.py``) measures throughput using
:func:`~hklpy2.run_utils.creator_from_config` with representative
configuration files and reports operations per second for both
``forward()`` and ``inverse()``.  The target is met when **all** parameter
sets in that test pass.

Measured baselines on a typical beamline workstation (``hkl_soleil`` solver):

.. list-table::
    :header-rows: 1
    :widths: 30 20 20 20 20

    * - Configuration
      - Operation
      - Mode
      - Before fix (ops/sec)
      - After fix (ops/sec)
    * - E4CV vibranium
      - ``forward()``
      - bissector
      - ~183
      - >2,000
    * - E4CV vibranium
      - ``inverse()``
      - bissector
      - ~2,700
      - >10,000
    * - APS POLAR
      - ``forward()``
      - 4-circles const-phi
      - ~376
      - >2,000
    * - APS POLAR
      - ``inverse()``
      - 4-circles const-phi
      - ~1,899
      - >10,000

.. seealso::

    :ref:`how_forward_solution` — choose which ``forward()`` solution the
    diffractometer uses; some pickers add overhead proportional to the number
    of solutions.

    :ref:`guide.solvers` — list and select solvers; understand solver entry
    points and how to write a custom one.
