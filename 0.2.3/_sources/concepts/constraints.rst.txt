.. _concepts.constraints:

======================
Constraints
======================

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

.. TODO: describe how the constraint class works (the ``valid()`` method)

.. TODO: state clearly that LimitsConstraint label must match real axis name
    and same name is used in the .core.constraints dictionary.

.. rubric:: Examples

Many of the :ref:`examples` show how to adjust :ref:`constraints <examples.constraints>`.

.. seealso:: :ref:`glossary`
