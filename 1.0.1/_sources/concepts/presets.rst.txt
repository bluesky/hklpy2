.. _concepts.presets:

=======
Presets
=======

.. index:: !presets
   see: freeze; presets

**Presets** define values for the constant real axes the |solver| uses during
a :meth:`~hklpy2.diffract.DiffractometerBase.forward()` computation.

Some diffractometer modes hold one or more real axes at a fixed value while
the solver computes the remaining axes needed to reach a requested
:math:`hkl` position.  A preset supplies the value the solver *assumes* for
such a constant axis — it does **not** move any motor.

.. tip:: Presets act *before* the solver computes solutions.  If you want
   to narrow which solutions are accepted *after* computation, use
   :ref:`concepts.constraints` instead.

Each mode stores its own independent preset dictionary, defaulting to ``{}``.
Switching modes switches to that mode's presets.

.. seealso::

   :ref:`concepts.constraints` — filter ``forward()`` solutions after
   computation.

   :ref:`how_presets` — step-by-step guide to setting and clearing presets.

   :ref:`glossary` — definition of *preset*.
