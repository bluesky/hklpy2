.. _how_calc_ub:

=========================================
How to Compute and Set the UB Matrix
=========================================

.. index::
    !UB matrix; how-to
    see: orientation matrix; UB matrix
    see: calc_UB; UB matrix

The :math:`UB` orientation matrix encodes how the crystal is mounted on
the diffractometer and is required before any
:meth:`~hklpy2.diffract.DiffractometerBase.forward` calculation can succeed.  This guide covers the common tasks: computing :math:`UB`
from reflections, setting it manually, inspecting and refining it, and
resetting it.

.. seealso::

   :ref:`tutorial` — step-by-step walkthrough of the full orientation
   workflow for a new user.

   :ref:`concepts.constraints` — filter :meth:`~hklpy2.diffract.DiffractometerBase.forward`
   solutions after the UB matrix is set.

   :ref:`guide.design` — explanation of the :math:`B`, :math:`U`, and
   :math:`UB` matrices.

   :doc:`/examples/hkl_soleil-ub_calc` — executable notebook: compute
   :math:`UB` from two reflections using the ``hkl_soleil`` solver.

   :doc:`/examples/hkl_soleil-ub_set` — executable notebook: set
   :math:`UB` directly from a known matrix.

Setup
-----

All examples use a simulated 4-circle diffractometer with silicon::

    >>> import hklpy2
    >>> from hklpy2.user import (
    ...     add_sample, calc_UB, or_swap, pa,
    ...     remove_reflection, set_diffractometer, setor,
    ... )
    >>> fourc = hklpy2.creator(name="fourc", geometry="E4CV", solver="hkl_soleil")
    >>> set_diffractometer(fourc)
    >>> add_sample("silicon", a=hklpy2.SI_LATTICE_PARAMETER)
    >>> fourc.beam.wavelength.put(1.54)

How do I add orientation reflections?
--------------------------------------

Use :func:`~hklpy2.user.setor` to record a reflection — the measured
motor angles for a known :math:`(h, k, l)` position::

    >>> r1 = setor(4, 0, 0, tth=69.0966, omega=-145.451, chi=0, phi=0)
    >>> r2 = setor(0, 4, 0, tth=69.0966, omega=-145.451, chi=90, phi=0)

:func:`~hklpy2.user.setor` uses the diffractometer's current wavelength
at the time it is called.  To record a reflection measured at a different
wavelength, pass ``wavelength=`` explicitly::

    >>> r3 = setor(0, 0, 4, tth=69.0966, omega=-145.451, chi=90, phi=90,
    ...            wavelength=1.00)

For lower-level control, :meth:`~hklpy2.ops.Core.add_reflection` on
``fourc.core`` accepts the same arguments and returns a
:class:`~hklpy2.blocks.reflection.Reflection` object directly.

How do I compute UB from two reflections?
------------------------------------------

Call :func:`~hklpy2.user.calc_UB` with the two reflection objects
returned by :func:`~hklpy2.user.setor`::

    >>> calc_UB(r1, r2)
    [[-1.4134285e-05, -1.4134285e-05, -1.156906937382],
     [0.0, -1.156906937469, 1.4134285e-05],
     [-1.156906937469, 1.73e-10, 1.4134285e-05]]

The matrix is stored on the sample and pushed to the solver
automatically.  You can also pass reflection *names* instead of objects::

    >>> calc_UB(r1.name, r2.name)

Or call the equivalent method directly on ``Core``::

    >>> fourc.core.calc_UB(r1, r2)

How do I swap the two orienting reflections?
---------------------------------------------

:func:`~hklpy2.user.or_swap` swaps the first two reflections in the
orientation list and recomputes :math:`UB`.  This is equivalent to
SPEC's ``or_swap`` command::

    >>> or_swap()

The new :math:`UB` matrix is returned and stored automatically.

.. _how_calc_ub.set_manually:

How do I set UB manually from a known matrix?
----------------------------------------------

Assign directly to :attr:`~hklpy2.blocks.sample.Sample.UB` on the
sample.  A plain nested Python list is sufficient — a numpy array is
accepted but not required::

    >>> fourc.core.sample.UB = [
    ...     [0.0,       0.0,      -1.1569],
    ...     [0.0,      -1.1569,    0.0   ],
    ...     [-1.1569,   0.0,       0.0   ],
    ... ]

The solver is updated automatically on the next
:meth:`~hklpy2.diffract.DiffractometerBase.forward` or
:meth:`~hklpy2.diffract.DiffractometerBase.inverse` call.

How do I inspect the current UB matrix?
-----------------------------------------

Call :func:`~hklpy2.user.pa` to see the full diffractometer state
including the :math:`U` and :math:`UB` matrices::

    >>> pa()

Or access the matrix directly::

    >>> fourc.core.sample.UB

How do I remove a reflection?
------------------------------

Use :func:`~hklpy2.user.remove_reflection` with the reflection's name::

    >>> remove_reflection(r1.name)

To remove all reflections and reset the sample to defaults::

    >>> fourc.core.reset_samples()

.. note::

    :meth:`~hklpy2.ops.Core.reset_samples` removes *all* samples and
    reflections and recreates a single default sample.  Use
    :func:`~hklpy2.user.remove_reflection` if you only want to discard
    one reflection while keeping others.

.. _how_calc_ub.stale:

How do I tell if my UB is stale?
---------------------------------

.. index::
    pair: UB matrix; stale
    see: UB_is_stale; stale UB

After :func:`~hklpy2.user.calc_UB`, the stored :math:`UB` matrix
reflects the two orienting reflections (the first two entries of
``sample.reflections.order``) *as they were at the moment*
:func:`~hklpy2.user.calc_UB` *ran*.  Several common edits leave the
stored matrix inconsistent with the current orienting pair — a
:term:`stale UB`.

Triggers for stale UB include:

* Removing one of the two orienting reflections (for example,
  ``remove_reflection(r1.name)`` above).
* Reassigning ``sample.reflections.order`` so a different reflection
  becomes ``order[0]`` or ``order[1]``.
* Editing the ``pseudos``, ``reals``, or ``wavelength`` of either
  orienting reflection in place.

Edits that do **not** invalidate the UB include: adding or removing a
reflection that is not in ``order[:2]``, editing a non-orienting
reflection, and assigning :attr:`~hklpy2.blocks.sample.Sample.U` /
:attr:`~hklpy2.blocks.sample.Sample.UB` directly (the user has taken
ownership of the matrix; see :ref:`how_calc_ub.set_manually`).

Check the current state via
:attr:`~hklpy2.blocks.sample.Sample.UB_is_stale`::

    >>> fourc.sample.UB_is_stale
    False
    >>> remove_reflection(r1.name)
    >>> fourc.sample.UB_is_stale
    True

When :meth:`~hklpy2.diffract.DiffractometerBase.forward` or
:meth:`~hklpy2.diffract.DiffractometerBase.inverse` runs against a
stale UB, a single ``UserWarning`` is emitted and :func:`pa`
annotates its output with ``UB stale: True``.  Refresh the matrix by
running :func:`~hklpy2.user.calc_UB` again with the desired
orienting pair::

    >>> r4 = setor(0, 4, 0, tth=69.0966, omega=-145.451, chi=90, phi=0)
    >>> calc_UB(r4, r2)
    >>> fourc.sample.UB_is_stale
    False

How do I refine the lattice parameters?
-----------------------------------------

If the solver supports it, :meth:`~hklpy2.ops.Core.refine_lattice`
uses three or more reflections to refine the lattice parameters::

    >>> r3 = setor(0, 0, 4, tth=69.0966, omega=-145.451, chi=90, phi=90)
    >>> fourc.core.refine_lattice(r1, r2, r3)

The refined :class:`~hklpy2.blocks.lattice.Lattice` is returned and
stored on the sample.  Not all solvers implement lattice refinement —
check the solver documentation if this raises ``NotImplementedError``.

How do I verify the UB matrix is correct?
-------------------------------------------

The standard check is to confirm that
:meth:`~hklpy2.diffract.DiffractometerBase.inverse` at the reflection
angles recovers the expected :math:`(h, k, l)`, and that
:meth:`~hklpy2.diffract.DiffractometerBase.forward` at the expected
:math:`(h, k, l)` returns angles close to the measured ones::

    >>> # inverse: angles → hkl
    >>> fourc.inverse(dict(omega=-145.451, chi=0, phi=0, tth=69.0966))
    Hklpy2DiffractometerPseudoPos(h=3.9999, k=0, l=0)   # ≈ (4, 0, 0) ✓

    >>> # forward: hkl → angles
    >>> fourc.forward(4, 0, 0)
    Hklpy2DiffractometerRealPos(omega=-34.5491, chi=0.0, phi=-110.9011, tth=69.0982)

Small differences (last decimal place) are normal floating-point
rounding.  Large differences indicate the orientation reflections were
recorded incorrectly or the wrong geometry was selected.
