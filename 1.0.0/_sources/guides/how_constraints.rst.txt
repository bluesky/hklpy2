.. _how_constraints:

==============================
How to Use Constraints
==============================

.. index::
    !constraint; how-to
    see: limits; constraint
    see: cut point; constraint

**Constraints** filter the candidate solutions returned by
:meth:`~hklpy2.diffract.DiffractometerBase.forward`.  Every real axis on a
diffractometer has a default
:class:`~hklpy2.blocks.constraints.LimitsConstraint` that accepts any
finite angle.  This guide shows how to tighten those constraints and how to
write a custom one.

.. seealso::

   :ref:`concepts.constraints` — explains what constraints are and how they
   work internally.

   :ref:`concepts.presets` — hold a real axis at a fixed value *before*
   ``forward()`` runs (complementary to constraints).

Setup
-----

All examples use a simulated 4-circle diffractometer::

    >>> import hklpy2
    >>> e4cv = hklpy2.creator(name="e4cv")

Inspect the default constraints::

    >>> e4cv.core.constraints
    {'omega': LimitsConstraint(label='omega', low=-180.0, high=180.0, cut=-180.0),
     'chi':   LimitsConstraint(label='chi',   low=-180.0, high=180.0, cut=-180.0),
     'phi':   LimitsConstraint(label='phi',   low=-180.0, high=180.0, cut=-180.0),
     'tth':   LimitsConstraint(label='tth',   low=-180.0, high=180.0, cut=-180.0)}

How do I set a limit on one axis?
----------------------------------

Set :attr:`~hklpy2.blocks.constraints.LimitsConstraint.limits` to a
``(low, high)`` tuple::

    >>> e4cv.core.constraints["chi"].limits = (0, 100)
    >>> e4cv.core.constraints["chi"].limits
    (0.0, 100.0)

Now any ``forward()`` solution where ``chi`` falls outside ``[0, 100]``
degrees is discarded.

.. tip::

   ``limits`` always sorts the pair automatically, so ``(100, 0)`` and
   ``(0, 100)`` are equivalent.

How do I change the cut point?
-------------------------------

A **cut point** controls which 360-degree window is used to express an
angle — it does not accept or reject solutions, only changes the
representation.  The default cut point is ``-180``, giving angles in
``[-180, +180)``.

To express angles in ``[0, 360)`` instead::

    >>> e4cv.core.constraints["chi"].cut_point = 0

To express angles in ``[-90, +270)``::

    >>> e4cv.core.constraints["omega"].cut_point = -90

How do I use cut points and limits together?
---------------------------------------------

A common scenario: a motor can physically travel from 0° to 290°, and you
want angles expressed in ``[0, 360)`` so the values are easy to compare
against hardware limits.

Set the cut point first (representation), then the limits (filtering)::

    >>> e4cv.core.constraints["chi"].cut_point = 0
    >>> e4cv.core.constraints["chi"].limits = (0, 290)

After wrapping, every ``chi`` solution is in ``[0, 360)``.  Then the
limits filter further to ``[0, 290]``, rejecting solutions in
``(290, 360)``.

.. note::

   The cut point is applied *first*, then limits are checked on the wrapped
   value.  See :ref:`concepts.constraints.cut_points` for the pipeline
   diagram.

How do I reset constraints to defaults?
----------------------------------------

:meth:`~hklpy2.ops.Core.reset_constraints` restores all axes to ``±180``
limits and a cut point of ``-180``::

    >>> e4cv.core.reset_constraints()
    >>> e4cv.core.constraints["chi"].limits
    (-180.0, 180.0)
    >>> e4cv.core.constraints["chi"].cut_point
    -180.0

How do I write a custom constraint?
-------------------------------------

Subclass :class:`~hklpy2.blocks.constraints.ConstraintBase` and implement
the :meth:`~hklpy2.blocks.constraints.ConstraintBase.valid` method.  The
method receives the full set of real-axis positions as keyword arguments and
must return ``True`` to keep a solution or ``False`` to reject it.

Example: keep only solutions where ``tth`` is positive (upper hemisphere
only)::

    from hklpy2.blocks.constraints import ConstraintBase

    class PositiveTthConstraint(ConstraintBase):
        """Accept only solutions where tth > 0."""

        def valid(self, **values):
            return values.get("tth", 0.0) > 0.0

Install it on the diffractometer by replacing the default constraint::

    >>> e4cv.core.constraints["tth"] = PositiveTthConstraint(label="tth")

.. important::

   The ``label`` argument must match the real axis name exactly.
   :meth:`~hklpy2.blocks.constraints.ConstraintBase.valid` looks up the
   axis value from its ``**values`` keyword arguments using ``self.label``
   as the key.  A mismatch raises
   :exc:`~hklpy2.misc.ConstraintsError`.

A more involved example: reject solutions where two axes simultaneously
reach their limits (useful for avoiding mechanical conflicts)::

    class NoSimultaneousLimitConstraint(ConstraintBase):
        """Reject solutions where both omega and chi are at their maximum."""

        def __init__(self, omega_max, chi_max, **kwargs):
            super().__init__(**kwargs)
            self.omega_max = omega_max
            self.chi_max = chi_max

        def valid(self, **values):
            at_omega_limit = abs(values.get("omega", 0.0) - self.omega_max) < 1e-3
            at_chi_limit   = abs(values.get("chi",   0.0) - self.chi_max)   < 1e-3
            return not (at_omega_limit and at_chi_limit)

    >>> e4cv.core.constraints["omega"] = NoSimultaneousLimitConstraint(
    ...     omega_max=180.0, chi_max=180.0, label="omega"
    ... )
