.. _tutorial:

========================================
Tutorial: Your First Diffractometer
========================================

.. index:: tutorial

By the end of this tutorial you will be able to:

- create a simulated 4-circle diffractometer in Python
- define a crystal sample and its lattice parameters
- add two orientation reflections and compute the :math:`UB` matrix
- verify the orientation with ``forward()`` and ``inverse()``
- move to a reciprocal-space position
- run a simple scan in reciprocal space

No hardware is required.  Everything runs in a simulated environment.
The same workflow applies to other diffractometer geometries — you
simply substitute a different geometry name when creating the
diffractometer.

.. note::

   This tutorial uses the E4CV (4-circle vertical) geometry with the
   ``hkl_soleil`` solver and silicon as the sample crystal.  If you are
   working with a different geometry or crystal, the steps are identical —
   only the axis names, lattice parameters, and reflection angles change.

.. seealso::

   :ref:`concepts.diffract` — conceptual overview of the diffractometer.

   :ref:`geometries` — full table of available geometries and solvers.

   :ref:`concepts.constraints` — how constraints filter ``forward()`` solutions.


Prerequisites
-------------

Install |hklpy2| and its dependencies following the
:doc:`installation guide <../quickstart>`.  Then start Python or a
Jupyter notebook in an environment where |hklpy2| is installed.


Step 1 — Create the diffractometer
-----------------------------------

We use :func:`~hklpy2.misc.creator` to build a diffractometer object.
It handles all the wiring between the Python object, the solver, and
the simulated motor axes.

.. code-block:: python

   import hklpy2

   fourc = hklpy2.creator(name="fourc", geometry="E4CV", solver="hkl_soleil")

The ``name`` keyword is a label for this diffractometer object — by
convention it matches the Python variable name.  ``geometry="E4CV"``
selects the 4-circle vertical geometry.  ``solver="hkl_soleil"`` selects
the Hkl/Soleil backend library.

.. tip::

   Run :func:`~hklpy2.misc.solvers` to see all installed solvers, and
   inspect ``fourc.core.solver.geometries`` to see the geometries your
   solver supports.


Step 2 — Import convenience functions and set the active diffractometer
------------------------------------------------------------------------

The :mod:`hklpy2.user` module provides interactive convenience functions
modelled on common SPEC commands.  We import the ones we need and tell
|hklpy2| which diffractometer is the *active* one:

.. code-block:: python

   from hklpy2.user import (
       add_sample,
       calc_UB,
       cahkl,
       cahkl_table,
       pa,
       set_diffractometer,
       setor,
       wh,
   )

   set_diffractometer(fourc)

All subsequent calls to ``pa()``, ``wh()``, ``setor()``, etc. will
operate on ``fourc`` until you call ``set_diffractometer()`` again with
a different object.


Step 3 — Add a sample
----------------------

A *sample* pairs a name with a crystal lattice.  |hklpy2| includes the
silicon lattice parameter as a built-in constant:

.. code-block:: python

   add_sample("silicon", a=hklpy2.SI_LATTICE_PARAMETER)

Silicon is cubic so only one lattice parameter ``a`` is needed.
Notice that ``add_sample()`` prints a confirmation:

.. code-block:: text

   Sample(name='silicon', lattice=Lattice(a=5.431, system='cubic'))

For a non-cubic crystal you would supply additional parameters, for
example ``a=3.0, c=5.0, gamma=120`` for a hexagonal crystal.


Step 4 — Set the wavelength
----------------------------

The wavelength of the incident X-rays is a property of the beam, not
the crystal.  We set it once and it applies to all subsequent
calculations:

.. code-block:: python

   fourc.beam.wavelength.put(1.54)  # Angstroms — Cu K-alpha

Notice we use ``.put()`` because
:attr:`~hklpy2.wavelength.WavelengthBase.wavelength` is an ophyd Signal.
At a real beamline this signal would be connected to the monochromator
control system, which may work in either wavelength or energy units —
:class:`~hklpy2.wavelength.WavelengthXray` supports both.


Step 5 — Add orientation reflections
--------------------------------------

The :math:`UB` matrix encodes how the crystal is mounted on the
diffractometer.  To compute it we need at least two measured
*orientation reflections* — positions where we know both the
Miller indices :math:`(h, k, l)` and the motor angles.

We use :func:`~hklpy2.user.setor` ("set orienting reflection"):

.. code-block:: python

   r1 = setor(4, 0, 0, tth=69.0966, omega=-145.451, chi=0, phi=0)
   r2 = setor(0, 4, 0, tth=69.0966, omega=-145.451, chi=90, phi=0)

``r1`` is the :math:`(4, 0, 0)` reflection measured at those four motor
angles.  ``r2`` is the :math:`(0, 4, 0)` reflection.

.. note::

   In a real experiment these angles come from your diffractometer
   control system — you physically drive to a known Bragg peak, read
   the motor positions, and record them here.  In this tutorial the
   values are pre-calculated for silicon at Cu K-alpha.


Step 6 — Compute the UB matrix
--------------------------------

With two reflections recorded, we ask |hklpy2| to compute the
:math:`UB` orientation matrix:

.. code-block:: python

   calc_UB(r1, r2)

The function returns and prints the :math:`3 \times 3` matrix.  The
exact numbers depend on the geometry and crystal orientation — what
matters is that the computation succeeded without error.

Now call :func:`~hklpy2.user.pa` ("print all") to see the full
diffractometer state:

.. code-block:: python

   pa()

You will see the solver, sample, reflections, :math:`UB` matrix,
constraints, mode, wavelength, and current position all in one place.
Notice the :math:`U` and :math:`UB` matrices are now populated.


Step 7 — Verify the orientation
---------------------------------

Before moving any motors it is good practice to verify that
``forward()`` and ``inverse()`` give consistent results with the
orientation reflections.

First, narrow the constraints so that only physically sensible solutions
are returned — keeping :math:`2\theta` positive and :math:`\omega`
negative:

.. code-block:: python

   fourc.core.constraints["tth"].limits = -0.001, 180
   fourc.core.constraints["omega"].limits = (-180, 0.001)

Now check ``inverse()`` — given the angles of the first reflection,
do we recover :math:`(4, 0, 0)`?

.. code-block:: python

   fourc.inverse((-145.451, 0, 0, 69.0966))
   # → Hklpy2DiffractometerPseudoPos(h=3.9999, k=0, l=0)  ✓

And ``forward()`` — given :math:`(4, 0, 0)`, do we get back angles
close to the measured reflection?

.. code-block:: python

   fourc.core.mode = "bissector"   # omega = tth / 2
   fourc.forward(4, 0, 0)
   # → Hklpy2DiffractometerRealPos(omega=-34.5491, chi=0.0, phi=-110.9011, tth=69.0982)

.. note::

   The ``forward()`` answer may differ from the measured reflection angles
   — both are valid positions for :math:`(4, 0, 0)`.  There are often
   multiple geometrically equivalent solutions.  Use
   :func:`~hklpy2.user.cahkl_table` to see all of them:

   .. code-block:: python

      cahkl_table((4, 0, 0), (0, 4, 0))

   Each row is a valid solution.  The constraints we set above have
   already filtered out solutions outside the motor ranges.


Step 8 — Move to a reciprocal-space position
---------------------------------------------

Once the orientation is verified, moving to an accessible :math:`(h, k,
l)` position is straightforward.  Not every position is reachable —
physical motor limits, the Ewald sphere, the current constraints, and the
wavelength all restrict which reflections the diffractometer can reach.
The wavelength sets the radius of the Ewald sphere and therefore determines
which reciprocal-lattice points are in range at all; changing the wavelength
(or equivalently the energy) shifts that boundary.  If ``forward()``
returns no solutions, the position is inaccessible under the current
configuration:

.. code-block:: python

   fourc.move(4, 0, 0)

This drives all four motors simultaneously to the angles that correspond
to :math:`(4, 0, 0)`.  Check the current position:

.. code-block:: python

   wh()

You will see the current :math:`(h, k, l)` pseudo-position and the
real motor angles together.  Notice that the *pseudo* position reports
the reciprocal-space coordinate and the *real* positions are the motor
angles — these are the two coordinate spaces described in
:ref:`guide.design`.


Step 9 — Scan in reciprocal space
-----------------------------------

A reciprocal-space scan works like any Bluesky scan — you specify start
and stop values in :math:`(h, k, l)` and |hklpy2| converts each step to
motor angles automatically.

Set up a minimal Bluesky RunEngine first:

.. code-block:: python

   import bluesky.plans as bp
   from bluesky import RunEngine
   from bluesky.callbacks.best_effort import BestEffortCallback

   bec = BestEffortCallback()
   bec.disable_plots()

   RE = RunEngine({})
   RE.subscribe(bec)

Then scan :math:`h` from 3.9 to 4.1 around the :math:`(4, 0, 0)` reflection:

.. code-block:: python

   fourc.move(4, 0, 0)
   RE(bp.scan([fourc], fourc.h, 3.9, 4.1, 5))

The scan table will show :math:`h`, :math:`k`, :math:`l` and all four
motor angles at each step.  Notice that as :math:`h` changes, all four
motors move together to track the reciprocal-space trajectory — this is
what makes a diffractometer different from a simple multi-axis stage.


What you have learned
----------------------

In this tutorial we:

1. Created a simulated 4-circle diffractometer with :func:`~hklpy2.misc.creator`
2. Added a silicon sample with :func:`~hklpy2.user.add_sample`
3. Set the X-ray wavelength
4. Recorded two orientation reflections with :func:`~hklpy2.user.setor`
5. Computed the :math:`UB` orientation matrix with :func:`~hklpy2.user.calc_UB`
6. Verified the orientation with ``forward()`` and ``inverse()``
7. Moved to a reciprocal-space position with :meth:`~hklpy2.diffract.DiffractometerBase.move`
8. Ran a Bluesky scan along a reciprocal-space direction


Where to go next
-----------------

- :ref:`how_constraints` — set axis limits and cut points to control
  which ``forward()`` solutions are accepted
- :ref:`how_presets` — hold a real axis at a fixed value during
  ``forward()`` computations
- :ref:`how_forward_solution` — choose which ``forward()`` solution the
  diffractometer uses by default
- :ref:`examples` — worked demonstrations for specific geometries,
  EPICS connections, and advanced use cases
- :ref:`concepts.diffract` — conceptual background on the diffractometer
  object
