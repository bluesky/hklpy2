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
.. tip:: *Constraints* are implemented as *cut points* in other software.
   Similar in concept yet not entirely identical in implementation.

.. tip:: Constraints act *after* the solver computes solutions.  If you want
   the solver to *assume* a specific value for a constant axis *before*
   computation, use :ref:`concepts.presets` instead.

.. rubric:: Examples

Many of the :ref:`examples` show how to adjust :ref:`constraints <examples.constraints>`.

.. seealso::

   :ref:`concepts.presets` — define constant-axis values used before
   ``forward()`` computation.

   :ref:`glossary`
