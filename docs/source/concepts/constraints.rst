.. _concepts.constraints:

======================
Constraints
======================

.. index:: !constraint

**Constraints** filter the solutions returned by a
:meth:`~hklpy2.diffract.DiffractometerBase.forward()` computation.

The solver can return many candidate sets of real-axis angles for a given
:math:`hkl` position.  One or more constraints
(:class:`~hklpy2.blocks.constraints.ConstraintBase`), together with a choice
of operating **mode**, narrow those candidates by:

* Accepting only solutions where each real axis falls within a specified range.

.. tip:: Constraints act *after* the solver computes solutions.  If you want
   the solver to *assume* a specific value for a constant axis *before*
   computation, use :ref:`concepts.presets` instead.

.. rubric:: Examples

Many of the :ref:`examples` show how to adjust :ref:`constraints <examples.constraints>`.

.. seealso::

   :ref:`concepts.presets` — define constant-axis values used before
   ``forward()`` computation.

   :ref:`glossary`

.. _concepts.constraints.cut_points:

Cut Points
==========

.. index:: cut point

A **cut point** is the angle at which a motor's reported position "wraps
around."  It sets the start of the 360-degree window used to express the
angle.  The physics is unchanged — it is the same motor position either
way — only how the number is written down changes.

Two common choices:

- ``cut_point = -180`` (default): angles are reported in the range
  −180 up to (but not including) +180.
- ``cut_point = 0``: angles are reported in the range 0 up to (but not
  including) 360.

Technically, a cut point ``c`` maps any computed angle to its equivalent
in the range from ``c`` up to (but not including) ``c + 360``.

.. rubric:: Cut point vs. constraint — the key distinction

+---------------------+--------------------------------------------------+
| **Cut point**       | **LimitsConstraint**                             |
+=====================+==================================================+
| Controls the        | Filters out solutions whose axis value falls     |
| *representation* of | *outside* a physically acceptable range.         |
| an angle — which    |                                                  |
| 360-degree window   |                                                  |
| it is expressed in. |                                                  |
+---------------------+--------------------------------------------------+
| Applied **first**,  | Applied **after** cut-point wrapping.            |
| before limit        |                                                  |
| checking.           |                                                  |
+---------------------+--------------------------------------------------+
| Does not accept or  | Accepts or rejects solutions.                    |
| reject solutions.   |                                                  |
+---------------------+--------------------------------------------------+
| Equivalent to SPEC  | No direct SPEC equivalent — SPEC uses cut points |
| ``cuts``,           | for both wrapping *and* as a proxy for limits.   |
| diffcalc ``setcut`` |                                                  |
+---------------------+--------------------------------------------------+

.. rubric:: When to use each

* **Need angles in** ``[-180, +180)``? — That is the default cut point
  (``-180``).  No change required.

* **Need angles in** ``[0, 360)``? — Set ``cut_point = 0``.

* **Motor has a physical travel limit** (e.g. ``chi`` can only reach
  ``[0, 100]``)? — Set ``limits = (0, 100)`` on the constraint.  The
  cut point controls representation; the limits control which solutions
  are physically reachable.

* **Both**: a motor in ``[0, 360)`` representation with physical travel
  of ``[0, 290]`` needs ``cut_point = 0`` *and* ``limits = (0, 290)``.

In practice: if your motor can only travel from 0° to 290°, setting
``cut_point = 0`` means all reported angles are positive numbers that
are easy to compare against your travel limits.  With the default cut
of −180, a position of 181° still reads as 181° — so in that case it
makes no difference.  The cut point matters when the motor's usable
range straddles the wrap boundary.

.. rubric:: Pipeline order in ``forward()``

.. code-block:: text

   solver.forward(pseudos)
     └─→ for each solution:
           for each axis:
             apply_cut(value)         ← wraps into [cut_point, cut_point+360)
           constraints.valid(...)     ← checks wrapped value against limits
           if valid: keep solution

.. rubric:: Example — using cut points

.. code-block:: python

   # Default: chi expressed in [-180, +180).
   diffractometer.core.constraints["chi"].cut_point   # -180.0

   # Change to [0, 360).
   diffractometer.core.constraints["chi"].cut_point = 0

   # Restrict physical travel as well.
   diffractometer.core.constraints["chi"].limits = (0, 290)

   # Reset everything to defaults.
   diffractometer.core.reset_constraints()

.. seealso::

   :term:`cut point` in the glossary.

   :ref:`spec_commands_map` — SPEC ``cuts`` maps to ``cut_point``.

.. _concepts.constraints.internals:

How Constraints Work Internally
================================

.. _concepts.constraints.valid:

The ``valid()`` Method
-----------------------

Every constraint class inherits from
:class:`~hklpy2.blocks.constraints.ConstraintBase` and must implement the
abstract method :meth:`~hklpy2.blocks.constraints.ConstraintBase.valid`.

.. code-block:: python

   def valid(self, **values: dict) -> bool:
       ...

The method receives the **full set of real-axis positions** (as keyword
arguments, ``axis_name=value``) and returns ``True`` when the constraint is
satisfied or ``False`` when it is not.  The values passed to ``valid()``
have already been cut-point-wrapped by :meth:`~hklpy2.blocks.constraints.LimitsConstraint.apply_cut`.

The built-in implementation,
:class:`~hklpy2.blocks.constraints.LimitsConstraint`, checks whether the
axis value falls within the configured ``[low_limit, high_limit]`` range:

.. code-block:: python

   # simplified from hklpy2/blocks/constraints.py
   def valid(self, **values):
       value = values[self.label]   # already cut-point-wrapped
       return (
           (value + ENDPOINT_TOLERANCE) >= self.low_limit
           and (value - ENDPOINT_TOLERANCE) <= self.high_limit
       )

A small tolerance (``ENDPOINT_TOLERANCE = 1e-4``) is applied at each
endpoint so that solver solutions that land exactly on a limit boundary
are not rejected due to floating-point rounding.

.. _concepts.constraints.forward_call:

How ``valid()`` Is Called During ``forward()``
----------------------------------------------

After the solver returns its candidate solutions,
:meth:`~hklpy2.ops.Core.forward` iterates over every solution, applies
cut-point wrapping, then calls
:meth:`~hklpy2.blocks.constraints.RealAxisConstraints.valid` on the
collection of constraints:

.. code-block:: python

   # simplified from hklpy2/ops.py  Core.forward()
   for solution in self.solver.forward(pseudos):
       reals = {axis: <computed_value>, ...}   # full set of real-axis values

       # Step 1: apply cut-point wrapping (new in #296)
       for name, constraint in self.constraints.items():
           reals[name] = constraint.apply_cut(reals[name])

       # Step 2: check limits on wrapped values
       if self.constraints.valid(**reals):
           solutions.append(reals)             # solution passes all constraints
       # solutions that fail are discarded (and logged at INFO level)

:meth:`~hklpy2.blocks.constraints.RealAxisConstraints.valid` in turn calls
:meth:`~hklpy2.blocks.constraints.LimitsConstraint.valid` on each individual
constraint and returns ``True`` only when **all** constraints are satisfied.
Solutions that fail at least one constraint are silently discarded; the
reasons are recorded at the ``logging.INFO`` level.

.. seealso::

   :ref:`concepts.presets` — supply a fixed value for a constant axis
   *before* the solver runs, rather than filtering solutions *after*.

.. _concepts.constraints.label:

The ``LimitsConstraint`` Label
-------------------------------

The ``label`` attribute of a
:class:`~hklpy2.blocks.constraints.LimitsConstraint` **must match the real
axis name** as it appears on the diffractometer.  This is because
``valid()`` looks up the axis value from the ``**values`` keyword-argument
dictionary using ``self.label`` as the key:

.. code-block:: python

   # from LimitsConstraint.valid()
   value = values[self.label]   # KeyError / ConstraintsError if label is wrong

The same name is used as the dictionary key in
``diffractometer.core.constraints``, which is a
:class:`~hklpy2.blocks.constraints.RealAxisConstraints` instance (a ``dict``
subclass):

.. code-block:: python

   diffractometer.core.constraints
   # {
   #   "omega": LimitsConstraint(label="omega", ...),
   #   "chi":   LimitsConstraint(label="chi",   ...),
   #   "phi":   LimitsConstraint(label="phi",   ...),
   #   "tth":   LimitsConstraint(label="tth",   ...),
   # }

Constraints are created automatically for every real axis when the
diffractometer is initialised (or when
:meth:`~hklpy2.ops.Core.reset_constraints` is called).  The ``label`` is set
to the axis name at that point.  You do not normally need to set the label
manually; if you create a :class:`~hklpy2.blocks.constraints.LimitsConstraint`
directly you must supply a ``label`` that matches a real axis name, otherwise
:meth:`~hklpy2.blocks.constraints.LimitsConstraint.valid` will raise a
:exc:`~hklpy2.misc.ConstraintsError`.

.. rubric:: Example — adjusting a constraint

.. code-block:: python

   # Restrict chi to the range [0, 90] degrees.
   diffractometer.core.constraints["chi"].limits = (0, 90)

   # Set chi's cut point to [0, 360).
   diffractometer.core.constraints["chi"].cut_point = 0

   # Reset all constraints to defaults (±180 degrees, cut at -180).
   diffractometer.core.reset_constraints()
