.. _how_zone_scan:

===========================
How to Perform a Zone Scan
===========================

.. index::
    !zone scan
    zone axis
    scan_zone
    move_zone
    OrthonormalZone

A :term:`zone` is a set of crystal lattice planes all parallel to a single
line — the *zone axis*.  Scanning along a zone axis constrains the
diffractometer to a crystallographic plane in reciprocal space, which is a
common technique for measuring diffuse scattering, structured diffuse
scattering, or mapping regions of reciprocal space systematically.

.. seealso::

   :term:`zone` in the :ref:`glossary`.

   :mod:`hklpy2.blocks.zone` — API reference for all zone functions.

   :doc:`/examples/zone-scan` — worked demonstration notebook.

   :ref:`spec_commands_map` — SPEC equivalents: ``cz``, ``mz``, ``pl``, ``sz``.

Prerequisites
-------------

- A diffractometer object with a sample, lattice, and computed UB matrix.
  See :ref:`tutorial` and :ref:`how_calc_ub`.
- A running bluesky :class:`~bluesky.run_engine.RunEngine` (``RE``).
- At least one detector (or use ``ophyd.sim.noisy_det`` for testing).

The examples below use a simulated E4CV diffractometer with the Vibranium
sample from the tutorial.  Substitute your own diffractometer, sample, and
hkl positions as needed.

.. code-block:: python

   import hklpy2
   from ophyd.sim import noisy_det
   import bluesky
   import bluesky.plan_stubs as bps

   RE = bluesky.RunEngine()
   fourc = hklpy2.creator(name="fourc")
   # ... define sample, add reflections, compute UB matrix ...

How to Move to a Position in the Zone
--------------------------------------

Use :func:`~hklpy2.blocks.zone.move_zone` to move the diffractometer to
a single reciprocal-space position.  This is the |hklpy2| equivalent of
the SPEC ``mz`` command.

.. code-block:: python

   from hklpy2 import move_zone

   # Move to (1, 0, 0)
   RE(move_zone(fourc, (1, 0, 0)))

:func:`~hklpy2.blocks.zone.move_zone` calls
:meth:`~hklpy2.diffract.DiffractometerBase.forward` internally and moves
all real axes to the computed positions.

How to Scan Along a Zone
-------------------------

Use :func:`~hklpy2.blocks.zone.scan_zone` to scan from one reciprocal-space
position to another along the zone defined by their cross product.  This is
the |hklpy2| equivalent of the SPEC ``scanzone`` command.

.. code-block:: python

   from hklpy2 import scan_zone

   # Scan from (1,0,0) to (0,1,0) in 11 steps, recording noisy_det
   (uid,) = RE(scan_zone([noisy_det], fourc, (1, 0, 0), (0, 1, 0), 11))

The ``start`` and ``finish`` vectors define the zone axis implicitly via
their cross product.  All intermediate points lie in the same zone.
Points where :meth:`~hklpy2.diffract.DiffractometerBase.forward` finds no
valid solution are logged at ``DEBUG`` level and skipped — the run
continues with the remaining points.

.. note::

   The ``num`` argument counts the total number of points **including** both
   endpoints.  Pass ``num=11`` for 11 points (9 intermediate + 2 endpoints).

How to Inspect Zone Positions Before Scanning
---------------------------------------------

Use :func:`~hklpy2.blocks.zone.zone_series` to print a table of pseudos
and reals along the zone without running a scan.  This is useful for
verifying that all positions are reachable before committing to a scan.

.. code-block:: python

   from hklpy2.blocks.zone import zone_series

   zone_series(fourc, (1, 0, 0), (0, 1, 0), 11)

Example output (columns depend on your geometry's axis names)::

   hkl_1=(1, 0, 0) hkl_2=(0, 1, 0) n=11
   ======= ======= ======= ========= ======= ======= =========
   h       k       l       omega     chi     phi     tth
   ======= ======= ======= ========= ======= ======= =========
   1.0000  0.0000  0.0000  ...       ...     ...     ...
   ...
   ======= ======= ======= ========= ======= ======= =========

How to Define a Zone Axis Explicitly
--------------------------------------

:class:`~hklpy2.blocks.zone.OrthonormalZone` can be used directly when you
want to define the zone axis explicitly or inspect zone membership before
scanning.

**From two vectors** (cross product):

.. code-block:: python

   from hklpy2.blocks.zone import OrthonormalZone

   # Zone axis = (1,0,0) × (0,1,0) = (0,0,1) — the [001] zone
   zone = OrthonormalZone(b1=(1, 0, 0), b2=(0, 1, 0))
   print(zone.axis)          # array([0., 0., 1.])

**Directly** (SPEC ``sz`` equivalent):

.. code-block:: python

   zone = OrthonormalZone(axis=(0, 0, 1))

**Check whether a vector is in the zone:**

.. code-block:: python

   zone.in_zone((1, 1, 0))   # True  — (110) is in the [001] zone
   zone.in_zone((0, 0, 1))   # False — (001) is the zone axis itself

Common Pitfalls
---------------

- **No forward solutions** — if one or more points along the zone cannot
  be reached (constraints, geometry limits, or wavelength), those points
  are silently skipped by :func:`~hklpy2.blocks.zone.scan_zone`.  Use
  :func:`~hklpy2.blocks.zone.zone_series` first to verify coverage, and
  check constraints with :meth:`~hklpy2.diffract.DiffractometerBase.forward`
  before running the scan.

- **Parallel vectors** — passing two parallel vectors as ``b1`` and ``b2``
  (e.g. ``(1,0,0)`` and ``(2,0,0)``) raises a ``ValueError`` because their
  cross product is zero and no zone axis can be defined.

- **UB matrix not set** — zone calculations use the sample's UB matrix to
  transform between reciprocal space and real motor positions.  Always
  compute the UB matrix before scanning.  See :ref:`how_calc_ub`.
