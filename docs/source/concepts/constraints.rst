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

.. index:: cut points
.. note:: **Relation to** *cut points* **in other software** (SPEC ``cuts``,
   diffcalc): In SPEC, a *cut point* is a branch point that sets where an
   angle wraps around — for example, a cut point of ``-180`` means the motor
   is represented in the interval ``[-180, +180)``.  Its primary purpose is
   controlling motor travel direction and angle representation, not filtering
   solutions.  hklpy2 ``LimitsConstraint`` is a *post-computation filter*: it
   discards solver solutions whose axis values fall outside a given range.
   The two concepts overlap in effect when the constraint window matches the
   cut-point interval, but they are not the same mechanism.  See
   :ref:`spec_commands_map` for the SPEC-to-hklpy2 mapping.

.. tip:: Constraints act *after* the solver computes solutions.  If you want
   the solver to *assume* a specific value for a constant axis *before*
   computation, use :ref:`concepts.presets` instead.

.. rubric:: Examples

Many of the :ref:`examples` show how to adjust :ref:`constraints <examples.constraints>`.

.. seealso::

   :ref:`concepts.presets` — define constant-axis values used before
   ``forward()`` computation.

   :ref:`glossary`

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
satisfied or ``False`` when it is not.

The built-in implementation,
:class:`~hklpy2.blocks.constraints.LimitsConstraint`, checks whether the
axis value falls within the configured ``[low_limit, high_limit]`` range:

.. code-block:: python

   # simplified from hklpy2/blocks/constraints.py
   def valid(self, **values):
       value = values[self.label]
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
:meth:`~hklpy2.ops.Core.forward` iterates over every solution and calls
:meth:`~hklpy2.blocks.constraints.RealAxisConstraints.valid` on the
collection of constraints:

.. code-block:: python

   # simplified from hklpy2/ops.py  Core.forward()
   for solution in self.solver.forward(pseudos):
       reals = {axis: <computed_value>, ...}   # full set of real-axis values
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

   # Reset all constraints to defaults (±180 degrees).
   diffractometer.core.reset_constraints()
