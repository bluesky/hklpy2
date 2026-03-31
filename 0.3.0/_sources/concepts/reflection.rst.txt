.. _concepts.reflection:

======================
Orientation Reflection
======================

An orientation :index:`reflection` is a peak intensity (typically an
observation) due to diffraction from a specific orientation of a crystalline
sample with respect to the incident radiation. The reflection occurs for a
precise alignment of the sample's crystalline planes and diffractometer axes
with the incoming radiation.  Information for each reflection is stored in
an instance of :class:`hklpy2.blocks.reflection.Reflection()`.

The coordinates of two (or more) reflections are used to compute a sample's
:math:`UB` :index:`orientation matrix`.  :math:`UB` transforms between the
sample's crystallographic axes (:index:`reciprocal-space`) and the
diffractometer's physical axes (:index:`real-space`).

Each reflection is defined by:

* full set of diffractometer angles - *real axes*
* reciprocal-space coordinates - *pseudo axes*
* wavelength of the incident radiation

Note that the engineering units for the pseudo axes are the reciprocal of the
units of the wavelength (where *angstroms* is typical).  The angles are reported
in *degrees*.

Reflections are stored as part of the :ref:`sample <concepts.sample>`.

.. rubric:: Examples

Many of the :ref:`examples` (such as :doc:`E4CH </examples/hkl_soleil-e4ch>`,
:doc:`E4CV </examples/hkl_soleil-e4cv>`, and :doc:`K4CV
</examples/hkl_soleil-k4cv>`) show how to create or add a reflection.

.. seealso:: :ref:`glossary`
