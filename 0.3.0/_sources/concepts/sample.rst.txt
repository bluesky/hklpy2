.. _concepts.sample:

======
Sample
======

In |hklpy2|, each sample's name, :ref:`lattice <concepts.lattice>`,
:ref:`reflections <concepts.reflection>`, and *orientation* are stored in the
:class:`~hklpy2.blocks.sample.Sample()` class.  The complete list of all a
diffractometer's samples (a Python dictionary) is managed by the core
:class:`~hklpy2.ops.Core()` class.

You can find the samples at ``fourc.samples``, a shortcut to
``fourc.core.samples`` (substitute your diffractometer's name here).

It's easy to switch between samples when there are choices.  For example, select
the *vibranium* sample: ``fourc.sample = "vibranium"``

The operations ``.core`` sends information *from* the selected sample *to* the
solver when calling a solver function (such as
:meth:`~hklpy2.backends.base.SolverBase.calculate_UB()`). When the solver
computes :math:`U` and :math:`UB`, those orientation matrices are stored in the sample
structure.

.. rubric:: Examples

Many of the :ref:`examples` (such as :doc:`E4CH </examples/hkl_soleil-e4ch>`,
:doc:`E4CV </examples/hkl_soleil-e4cv>`, and :doc:`K4CV
</examples/hkl_soleil-k4cv>`) show how to create or add a sample.

.. seealso:: :ref:`glossary`
