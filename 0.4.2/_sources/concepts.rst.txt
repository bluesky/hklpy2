.. _concepts:

==========
Concepts
==========

.. toctree::
   :glob:
   :hidden:

   concepts/*
   concepts/planning/*

.. figure:: _static/hklpy2-block-diagram.png
   :alt: hklpy2 block diagram

   Functional blocks in  |hklpy2|.

Overview
========

The |hklpy2| *diffractometer* [#]_ is an ophyd *PseudoPositioner*, [#]_
providing both *pseudo* coordinates and *real* coordinates.  Movement in one
coordinate space updates the coordinates of the other space.  For
diffractometers, the orientation matrix (:math:`UB`) enables transformation between the
two spaces.

.. note:: A monochromatic [#]_ radiation source is expected.  See
   :ref:`concepts.wavelength` for more details.

Coordinates
===========

:pseudos:
   Virtual coordinates (crystallographic `h`, `k`, `l`), computed from *reals*.
:reals:
   Positioners (rotation motors) in physical coordinates.

An ophyd *PseudoPositioner* relies on the
:meth:`~hklpy2.diffract.DiffractometerBase.forward()` and
:meth:`~hklpy2.diffract.DiffractometerBase.inverse()` methods.

Transformations
===============

==========  =========   ============== ================
from        to          solution(s)     transformation
==========  =========   ============== ================
*reals*     *pseudos*   1              :meth:`hklpy2.diffract.DiffractometerBase.inverse()`
*pseudos*   *reals*     1 or exception :meth:`hklpy2.diffract.DiffractometerBase.forward()`
*pseudos*   *reals*     0, 1 or more   :meth:`hklpy2.ops.Core.forward()`
==========  =========   ============== ================

.. note:: The diffractometer's `forward()` method picks the default solution
   from the list returned from ``core.forward()``.  Initially, the first
   solution in the list is chosen.
   See :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution`.

Solvers
===============

A |solver| provides computational support for one or more diffractometer
geometries. Each geometry has a specific set of *pseudos*, *reals*, and other
terms which support the ``forward()`` and ``inverse()`` transformations.  See
:ref:`concepts.solvers` for more details.

Core Operations
===============

The :class:`~hklpy2.diffract.DiffractometerBase` class provides the ophyd
*PseudoPositioner*.  This class relies on :class:`~hklpy2.ops.Operations` to
provide most features (sample, lattice, reflections, ...) and to connect with
the diffractometer's chosen |solver|.  See
:ref:`concepts.ops` for more details.

``hklpy2.creator()``
====================

The ``creator()`` function reduces the effort to create all but the most
complex diffractometer objects.  See :ref:`concepts.creator` for more details.

``DiffractometerBase()``
========================

All diffractometers are created as subclasses of
:class:`!hklpy2.diffract.DiffractometerBase`.  This base class defines a
diffractometer as an ``ophyd.PseudoPositioner``.  See :ref:`concepts.diffract`
for more details.

----

.. seealso:: :ref:`glossary`

.. rubric:: Footnotes

.. [#] https://en.wikipedia.org/wiki/Diffractometer
.. [#] https://blueskyproject.io/ophyd/user/how-to/pseudopositioner.html
.. [#] *monochromatic*: The variation of wavelengths in the source is negligible
   for scientific interpretation.
