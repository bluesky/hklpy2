.. _concepts.constraints:

======================
Constraints
======================

.. index:: !constraint

Computation of :meth:`~hklpy2.diffract.DiffractometerBase.forward()` can have
many solutions.  One or more constraints
(:class:`~hklpy2.blocks.constraints.ConstraintBase`) (a.k.a, cut points),
together with a choice of operating **mode**, can be applied to:

* Limit the range of :meth:`~hklpy2.diffract.DiffractometerBase.forward()`
  solutions accepted for that positioner.
* Future possibilities derived from
  :class:`~hklpy2.blocks.constraints.ConstraintBase`

.. index:: cut points
.. tip:: *Constraints* are implemented as *cut points* in other software.
    Similar in concept yet not entirely identical in implementation.

.. rubric:: Examples

Many of the :ref:`examples` show how to adjust :ref:`constraints <examples.constraints>`.

.. seealso:: :ref:`glossary`
