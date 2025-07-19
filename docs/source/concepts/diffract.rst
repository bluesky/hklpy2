.. _concepts.diffract:

==================
Diffractometer
==================

.. TODO: How much is guide or example?  This should be a concepts doc. Brief.

.. index:: !design; diffractometer

Diffractometers are built as a subclass of
:class:`~hklpy2.diffract.DiffractometerBase()`, adding a variety of
positioners as ophyd Components.  In an instance of that subclass, user
sets :attr:`~hklpy2.diffract.DiffractometerBase.backend_solver` by calling
:func:`~hklpy2.misc.solver_factory`.  In this call, the user specifies the solver,
the geometry, and defines which Components (of the diffractometer) are to be used as
pseudos and reals.  The backend implements
:meth:`~hklpy2.backends.base.SolverBase.forward`,
:meth:`~hklpy2.backends.base.SolverBase.inverse`, and all related support, for
only the pseudos and reals that are identified.

.. grid:: 2

    .. grid-item-card:: :material-outlined:`check_box;3em` A |solver| is not needed to:

      - define subclass of :class:`~hklpy2.diffract.DiffractometerBase()` and create instance
      - create instance of :class:`~hklpy2.blocks.sample.Sample()`
      - create instance of :class:`~hklpy2.blocks.lattice.Lattice()`
      - create instance of :class:`~hklpy2.incident._WavelengthBase()` subclass
      - create instance of :class:`~hklpy2.backends.base.SolverBase()` subclass
      - set :attr:`~hklpy2.incident._WavelengthBase.wavelength`
      - list *available* solvers: (:func:`~hklpy2.misc.solvers`)
      - review saved orientation details

    .. grid-item-card:: :material-outlined:`rule;3em` A |solver| is needed to:

      - list available |solver| :attr:`~hklpy2.backends.base.SolverBase.geometries`
      - list a |solver| geometry's required
        :attr:`~hklpy2.backends.base.SolverBase.pseudo_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.real_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.extra_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.modes`
      - create instance of :class:`~hklpy2.blocks.reflection.Reflection()`
      - define or compute a :math:`UB` matrix
        (:meth:`~hklpy2.backends.base.SolverBase.calculateOrientation`)
      - :meth:`~hklpy2.backends.base.SolverBase.forward`
        and :meth:`~hklpy2.backends.base.SolverBase.inverse`
      - determine the diffractometer :attr:`~hklpy2.diffract.DiffractometerBase.position`
      - save or restore orientation details
      - refine lattice parameters

.. TODO: This is a guide section

Steps to define a diffractometer object
=======================================

#. Identify the geometry.
#. Find its |solver|, geometry, and other parameters in :ref:`geometries`.
#. Create a custom subclass for the diffractometer.
#. (optional) Identify the EPICS PVs for the real positioners.
#. (optional) Connect energy to the control system.
#. Define the diffractometer object using ``hklpy2.creator()``.

.. TODO: This section should point to the concepts, not re-explain.

A `Diffractometer` object
=========================

name
----

The ``name`` of a :class:`~hklpy2.diffract.DiffractometerBaseBase()` instance is
completely at the choice of the user and conveys no specific information to
the underlying Python support code.

One important *convention* is that the name given on the left side of the ``=``
matches the name given by the ``name="..."`` keyword, such as this example::

    e4cv = hklpy2.creator(name="e4cv")

geometry
--------

The geometry describes the physical arrangement of real positioners, pseudo
axes, and extra parameters that make up the diffractometer.  The choices are
limited to those geometries provided the chosen |solver|.

core
--------

All operations are coordinated through :ref:`concepts.ops`.  This is ``fourc.core``.

wavelength (and energy)
-----------------------

The `diffractometer.beam` describes the radiation source using the
:class:`~hklpy2.incident._WavelengthBase` class (or subclass).  Wavelength is
the term common to both neutron and X-ray diffractometer users.
:class:`~hklpy2.incident.WavelengthXray` is the default. This supports
conversion between wavelength and X-ray photon energy.

.. tip:: Neutron users would make a similar class with different calculations
  between wavelength and energy.

.. note:: It is more common for X-ray users to describe the *energy*
   of the incident radiation than its *wavelength*.  The
   ``MonochromaticXrayWavelength()`` class allows the X-ray photon energy
   to be expressed in any engineering units
   that are convertible to the expected units (`keV`).

   ..
    An offset may be
    applied, which is useful when connecting the diffractometer energy
    with a control system variable.

.. note:: The wavelength, commonly written as :math:`\lambda`,
   cannot be named in Python code as `"lambda"`.
   Python reserves `lambda` as a type of expression:
   `reserved <https://docs.python.org/3/reference/expressions.html#lambda>`_

sample
------

The purpose of a diffractometer is to position a sample for scientific
measurements. The ``sample`` attribute is an instance of
:class:`~hklpy2.blocks.sample.Sample`. Behind the scenes, the
:class:`~hklpy2.ops.Core` class maintains a *dictionary* of samples (keyed
by ``name``), each with its own :class:`~hklpy2.blocks.lattice.Lattice` and
orientation :class:`~hklpy2.blocks.reflection.Reflection` information.

lattice
+++++++

Crystal samples have :class:`~hklpy2.blocks.lattice.Lattice` parameters defined by
unit cell lengths and angles.  (Units here are angstroms and degrees.)

.. index:: !vibranium

This table describes the lattice of crystalline Vibranium [#vibranium]_:

========= ============  ============   ============   ===== ====  =====
sample    a             b              c              alpha beta  gamma
========= ============  ============   ============   ===== ====  =====
vibranium :math:`2\pi`  :math:`2\pi`   :math:`2\pi`   90    90    90
========= ============  ============   ============   ===== ====  =====

.. [#vibranium] Vibranium (https://en.wikipedia.org/wiki/Vibranium)
   is a fictional metal.  Here, we have decided it has a cubic lattice
   with lattice parameter of exactly :math:`2\pi`.

orientation
+++++++++++

The **UB** matrix describes the :meth:`~hklpy2.diffract.DiffractometerBase.forward()`
and :meth:`~hklpy2.diffract.DiffractometerBase.inverse()` transformations that allow
precise positioning of a crystalline sample's atomic planes in the laboratory
reference system of the diffractometer.  It is common to compute the **UB** matrix
from two orientation reflections using :meth:`~hklpy2.ops.Core.calc_UB()`.

orientation reflections
~~~~~~~~~~~~~~~~~~~~~~~

An orientation reflection consists of a set of matching pseudos
and reals at a specified wavelength.  These values may be
measured or computed.

There are several use cases for a set of reflections:

* Computation of the :math:`UB` matrix (for 2 or more non-parallel reflections).
* Documentation of observed (or theoretical) reflection settings.
* Reference settings so as to re-position the diffractometer.
* Define a crystallographic zone or axis to guide the diffractometer for measurements.

Here is an example of three orientation reflections for a sample of crystalline
vibranium [#vibranium]_ as mounted on a diffractometer with
:ref:`E4CV <geometries-hkl_soleil-e4cv>` geometry:

= === === === ======== ==== ==== ======= ========== =======
# h   k   l   omega    chi  phi  tth     wavelength orient?
= === === === ======== ==== ==== ======= ========== =======
1 4.0 0.0 0.0 -145.451 0.0  0.0  69.0966 1.54       False
2 0.0 4.0 0.0 -145.451 0.0  90.0 69.0966 1.54       True
3 0.0 0.0 4.0 -145.451 90.0 0.0  69.0966 1.54       True
= === === === ======== ==== ==== ======= ========== =======

mode
----

The ``forward()`` transformation can have many solutions.  The
diffractometer is set to a mode (chosen from a list specified by the
diffractometer geometry) that controls how values for each of the real
positioners will be controlled. A mode can control relationships between
real positioners in addition to limiting the motion of a real positioner.
Further, a mode can specify an additional reflection which will be used to
determine the outcome of the ``forward()`` transformation.

=====================  =======================
object                 meaning
=====================  =======================
``DFRCT.core.mode``    mode selected now
``DFRCT.core.modes``   list of possible modes
=====================  =======================

Here, ``DFRCT`` is the diffractometer object (such as ``e4cv`` above).

Parts of `DiffractometerBase`
=============================

A :class:`~hklpy2.diffract.DiffractometerBase` object has several parts:

The :class:`~hklpy2.diffract.DiffractometerBase()` class should
be a thin interface. Most real diffractometer capability should be
provided in the :class:`~hklpy2.ops.Core()` class (or one of
its attributes, such as :attr:`~hklpy2.ops.Core.solver`
and :attr:`~hklpy2.ops.Core.sample`)

.. rubric:: Core-related methods and properties
.. autosummary::

    ~hklpy2.diffract.DiffractometerBase.forward (method)
    ~hklpy2.diffract.DiffractometerBase.inverse (method)
    ~hklpy2.diffract.DiffractometerBase.position (method)
    ~hklpy2.diffract.DiffractometerBase.pseudo_axis_names (property)
    ~hklpy2.diffract.DiffractometerBase.real_axis_names (property)
    ~hklpy2.diffract.DiffractometerBase.wh (method)

.. rubric:: Sample-related methods and properties
.. autosummary::

    ~hklpy2.diffract.DiffractometerBase.add_reflection (method)
    ~hklpy2.diffract.DiffractometerBase.add_sample (method)
    ~hklpy2.diffract.DiffractometerBase.sample (property)
    ~hklpy2.diffract.DiffractometerBase.samples (property)

.. rubric:: Related methods and properties from other classes
.. autosummary::

    ~hklpy2.ops.Core.assign_axes (method)
    ~hklpy2.backends.base.SolverBase.extra_axis_names (property)
    ~hklpy2.blocks.sample.Sample.lattice (property)
    ~hklpy2.blocks.sample.Sample.refine_lattice (method)
    ~hklpy2.ops.Core.refine_lattice (method)
    ~hklpy2.blocks.sample.Sample.reflections (property)
    ~hklpy2.ops.Core.set_solver (method)
    ~hklpy2.blocks.sample.Sample.U (property)
    ~hklpy2.blocks.sample.Sample.UB (property)


Use a Diffractometer with the bluesky RunEngine
===============================================

The positioners of a :class:`~hklpy2.diffract.DiffractometerBase` object may be
used with the `bluesky RunEngine
<https://blueskyproject.io/bluesky/generated/bluesky.run_engine.RunEngine.html?highlight=runengine>`_
with any of the `pre-assembled plans
<https://blueskyproject.io/bluesky/plans.html#pre-assembled-plans>`_ or
in custom plans of your own.

   .. code-block:: Python
      :linenos:

      from hklpy2.misc import ConfigurationRunWrapper

      fourc = hklpy2.creator(name="fourc")

      # Save configuration with every run
      crw = ConfigurationRunWrapper(fourc)
      RE.preprocessors.append(crw.wrapper)

      # steps not shown here:
      #   define a sample & orientation reflections, and compute UB matrix

      # record the diffractometer metadata to a run
      RE(bp.count([fourc]))

      # relative *(h00)* scan
      RE(bp.rel_scan([scaler, fourc], fourc.h, -0.1, 0.1, 21))

      # absolute *(0kl)* scan
      RE(bp.scan([scaler, fourc], fourc.k, 0.9, 1.1, fourc.l, 2, 3, 21))

      # absolute ``chi`` scan
      RE(bp.scan([scaler, fourc], fourc.chi, 30, 60, 31))

Keep in mind these considerations:

1. Use the :class:`hklpy2.misc.ConfigurationRunWrapper` to save configuration
   as part of every run.  Here's an example:

   .. code-block:: Python
     :linenos:

     from hklpy2.misc import ConfigurationRunWrapper
     crw = ConfigurationRunWrapper(fourc)
     RE.preprocessors.append(crw.wrapper)

   .. seealso:: :doc:`/guides/configuration_save_restore`

2. Don't mix axis types (pseudos *v.* reals) in a scan.  You can only
   scan with either *pseudo* axes (``h``, ``k``, ``l``, ``q``, ...) or *real*
   axes (``omega``, ``tth``, ``chi``, ...) at one time.  You cannot scan with
   both types (such as ``h`` and ``tth``) in a single scan (because the
   :meth:`~hklpy2.diffract.DiffractometerBase.forward()` and
   :meth:`~hklpy2.diffract.DiffractometerBase.inverse()` methods cannot
   resolve).  Example:

   .. code-block:: Python
      :linenos:

      # Cannot scan both ``k`` and ``chi`` at the same time.
      # This will raise a `ValueError` exception.
      RE(bp.scan([scaler, fourc], fourc.k, 0.9, 1.1, fourc.chi, 2, 3, 21))


3. When scanning with pseudo axes (``h``, ``k``, ``l``, ``q``, ...), first
   check that all steps in the scan can be computed successfully with
   the :meth:`~hklpy2.diffract.DiffractometerBase.forward()` computation::

        fourc.forward(1.9, 0, 0)

4. Only restore orientation reflections from a **matching**
   diffractometer geometry (such as ``E4CV``).  Mismatch will trigger an exception.
