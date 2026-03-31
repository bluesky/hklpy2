.. _guide.diffract:

===========================
Diffractometer How-To Guide
===========================

.. seealso:: :ref:`concepts.diffract` for a concept overview.

Steps to Define a Diffractometer Object
========================================

#. Identify the geometry.
#. Find its |solver|, geometry, and other parameters in :ref:`geometries`.
#. Create a custom subclass for the diffractometer.
#. (optional) Identify the EPICS PVs for the real positioners.
#. (optional) Connect wavelength to the control system.
#. Define the diffractometer object using ``hklpy2.creator()``.

A Diffractometer Object
========================

name
----

The ``name`` of a :class:`~hklpy2.diffract.DiffractometerBase()` instance is
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
----

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

.. index:: vibranium

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

.. _diffract_axes:

Diffractometer Axis Names
=========================

In |hklpy2|, the names of diffractometer axes (pseudos and reals) are
not required to match any particular |solver| library.  Users are free
to use any names allowed by ophyd.

User-defined axis names
-----------------------

Let's see examples of diffractometers built with user-defined names.

* :ref:`diffract_axes.diffractometer-factory` with automatic mapping
* :ref:`diffract_axes.direct-assign` with directed mapping

.. _diffract_axes.diffractometer-factory:

Diffractometer Creator
+++++++++++++++++++++++++++++++

The :func:`~hklpy2.diffract.creator()` function constructs a diffractometer object using the
supplied `reals={}` to define their names.  These are mapped to the names used
by the |solver|.  Let's show this cross-reference map with just a few commands::

    >>> import hklpy2

    >>> twoc = hklpy2.creator(
        name="twoc",
        geometry="TH TTH Q",
        solver="th_tth",
        reals={"sample": None, "detector": None},
    )

    >>> twoc.core.axes_xref
    {'q': 'q', 'sample': 'th', 'detector': 'tth'}

    >>> twoc.wh()
    q=0
    wavelength=1.0
    sample=0, detector=0

.. _diffract_axes.custom-assign:

Custom Diffractometer class
+++++++++++++++++++++++++++++++++++++

Construct a 2-circle diffractometer, one axis for the sample and one axis for
the detector.

In addition to defining the diffractometer axes, we name the |solver| to use
with our diffractometer. The ``th_tth`` |solver| has a
:class:`~hklpy2.backends.th_tth_q.ThTthSolver` with a ``"TH TTH Q"`` geometry
that fits our design. We set that up in the ``__init__()`` method of our new
class.

The :ref:`TH TTH Q <geometries-th_tth-th-tth-q>` geometry has real axes named
``th`` and ``tth``. Even though we are using different names, it is not
necessary to define ``_real`` (as shown in :ref:`diffract_axes.direct-assign`)
as long as:

* We define the *same number of pseudos* as the solver expects.
* We define the *same number of reals* as the solver expects.
* We *specify each in the order expected by the solver*.

.. code-block:: Python
    :linenos:

    import hklpy2
    from hklpy2.diffract import Hklpy2PseudoAxis
    from ophyd import Component, SoftPositioner

    class S1D1(hklpy2.DiffractometerBase):

        q = Component(Hklpy2PseudoAxis, "", kind=H_OR_N)

        sample = Component(SoftPositioner, init_pos=0)
        detector = Component(SoftPositioner, init_pos=0)

        # Alias 'sample' to 'th', 'detector' to 'tth'
        _real = ["sample", "detector"]

        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                solver="th_tth",                # solver name
                geometry="TH TTH Q",            # solver geometry
                **kwargs,
            )

Create a Python object that uses this class:

.. code-block:: Python

    twoc = S1D1(name="twoc")

.. tip:: Use the :func:`hklpy2.diffract.creator()` instead:

    .. code-block:: Python

        twoc = hklpy2.creator(
            name="twoc",
            geometry="TH TTH Q",
            solver="th_tth",
            reals=dict(sample=None, detector=None)
        )

Show the mapping between user-defined axes and axis names used by the |solver|::

    >>> print(twoc.core.axes_xref)
    {'q': 'q', 'sample': 'th', 'detector': 'tth'}

.. _diffract_axes.direct-assign:

Custom Diffractometer with additional axes
++++++++++++++++++++++++++++++++++++++++++++++++

Consider this example for a two-circle class (with additional axes).
The ``"TH TTH Q"`` |solver| geometry expects ``q`` as
the only pseudo axis and ``th`` and ``tth`` as the two real axes
(no extra axes).

We construct this example so that we'll need to override the automatic
assignment of axes (lots of extra pseudo and real axes, none of them in the
order expected by the solver). Look for the ``_pseudo=["q"]`` and
``_real=["theta", "ttheta"]`` parts where we define the mapping.

.. code-block:: Python
    :linenos:

    import hklpy2
    from hklpy2.diffract import Hklpy2PseudoAxis
    from ophyd import Component, SoftPositioner

    class MyTwoC(hklpy2.DiffractometerBase):

        # sorted alphabetically for this example
        another = Component(Hklpy2PseudoAxis)
        horizontal = Component(SoftPositioner, init_pos=0)
        q = Component(Hklpy2PseudoAxis)
        theta = Component(SoftPositioner, init_pos=0)
        ttheta = Component(SoftPositioner, init_pos=0)
        vertical = Component(SoftPositioner, init_pos=0)

        _pseudo = ["q"]
        _real = ["theta", "ttheta"]

        def __init__(self, *args, **kwargs):
            super().__init__(
              *args,
              solver="th_tth",
              geometry="TH TTH Q",
              **kwargs
              )

Create the diffractometer:

.. code-block:: Python

    twoc = MyTwoC(name="twoc")

What are the axes names used by this diffractometer?

.. code-block:: Python

    >>> twoc.pseudo_axis_names
    ['another', 'q']
    >>> twoc.real_axis_names
    ['horizontal', 'theta', 'ttheta', 'vertical']

Show the ``twoc`` diffractometer's |solver|:

.. code-block:: Python

    >>> twoc.core.solver
    ThTthSolver(name='th_tth', version='0.0.14', geometry='TH TTH Q')

What are the axes expected by this |solver|?

.. code-block:: Python

    >>> twoc.core.solver_pseudo_axis_names
    ['q']
    >>> twoc.core.solver_real_axis_names
    ['th', 'tth']
    >>> twoc.core.solver_extra_axis_names
    []

Show the cross-reference mapping from diffractometer
to |solver| axis names (as defined in our MyTwoC class above):

.. code-block:: Python

    >>> twoc.core.axes_xref
    {'q': 'q', 'theta': 'th', 'ttheta': 'tth'}

.. _diffract_axes.pseudos-out-of-order:

Pseudos supplied in a different order than the solver expects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. index:: _pseudo; pseudos out of order

The ``_pseudo`` keyword has two related uses:

1. **Pseudos out of order** — custom names defined in a different order than
   the |solver| expects.
2. **Additional pseudos** — extra pseudo axes are present and only a subset
   should be mapped to the |solver|.

Without ``_pseudo``, :func:`~hklpy2.diffract.creator()` zips pseudo names
positionally against the solver's pseudo axis order.  If orders differ, or
if extra pseudos are present, the mapping will be incorrect.

**Use 1: pseudos out of order**

An E4CV diffractometer with custom pseudo names supplied in a different order
than the solver's ``h, k, l``:

.. tabs::

    .. tab:: Wrong (missing ``_pseudo``)

        Without ``_pseudo``, names are zipped positionally: ``ll`` (1st)
        maps to solver ``h`` (1st) and ``hh`` (3rd) maps to solver ``l``
        (3rd) — silently swapped:

        .. code-block:: python

            sim = hklpy2.creator(
                name="sim",
                solver="hkl_soleil",
                geometry="E4CV",
                pseudos=["ll", "kk", "hh"],
            )
            sim.core.axes_xref
            # {'ll': 'h', 'kk': 'k', 'hh': 'l', ...}   ← swapped

    .. tab:: Correct (with ``_pseudo``)

        Supply ``_pseudo`` to declare which local name maps to each solver
        pseudo axis slot, independent of the ``pseudos`` list order:

        .. code-block:: python

            sim = hklpy2.creator(
                name="sim",
                solver="hkl_soleil",
                geometry="E4CV",
                pseudos=["ll", "kk", "hh"],
                _pseudo=["hh", "kk", "ll"],
            )
            sim.core.axes_xref
            # {'hh': 'h', 'kk': 'k', 'll': 'l', ...}   ← correct

**Use 2: additional pseudos**

When extra pseudo axes are added alongside the solver's pseudos, ``_pseudo``
selects exactly which names map to the |solver|.  The remaining pseudos are
available as diffractometer attributes but are not included in the solver
mapping:

.. code-block:: python

    sim = hklpy2.creator(
        name="sim",
        solver="hkl_soleil",
        geometry="E4CV",
        pseudos=["hh", "kk", "ll", "extra"],
        _pseudo=["hh", "kk", "ll"],   # select these 3 for solver mapping
    )
    sim.core.axes_xref
    # {'hh': 'h', 'kk': 'k', 'll': 'l', ...}  ← 'extra' not mapped
    sim.pseudo_axis_names
    # ['hh', 'kk', 'll']                        ← 'extra' excluded

.. _diffract_axes.reals-out-of-order:

Reals supplied in a different order than the solver expects
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. index:: _real; reals out of order

When hardware motor names are wired or named in a different order than the
|solver| expects, the ``_real`` keyword is required to declare which local
axis name corresponds to each |solver| axis slot.

Without ``_real``, :func:`~hklpy2.diffract.creator()` (and
:func:`~hklpy2.diffract.diffractometer_class_factory()`) zip the ``reals``
dict keys positionally against the solver's axis order.  If these orders
differ, axes are silently swapped in ``axes_xref``, which can cause
:meth:`~hklpy2.ops.Core.calc_UB` to fail with a degenerate U matrix.

**Example:** an APS POLAR 6-circle diffractometer whose hardware motors are
wired in a different order than the solver expects
(solver order: ``tau, mu, chi, phi, gamma, delta``):

.. tabs::

    .. tab:: Wrong (missing ``_real``)

        Without ``_real``, ``reals`` dict keys are zipped positionally to
        solver axes. ``gamma`` (3rd key) maps to solver ``chi`` (3rd slot)
        and ``chi`` (5th key) maps to solver ``gamma`` (5th slot) — silently
        swapped:

        .. code-block:: python

            cradle = hklpy2.creator(
                name="cradle",
                solver="hkl_soleil",
                geometry="APS POLAR",
                reals=dict(tau="m73", mu="m4", gamma="m19", delta="m20", chi="m37", phi="m38"),
            )
            cradle.core.axes_xref
            # {'tau': 'tau', 'mu': 'mu', 'gamma': 'chi', 'delta': 'phi',
            #  'chi': 'gamma', 'phi': 'delta'}   ← swapped

    .. tab:: Correct (with ``_real``)

        Supply ``_real`` to declare which local name maps to each solver axis
        slot, independent of the ``reals`` dict key order:

        .. code-block:: python

            cradle = hklpy2.creator(
                name="cradle",
                solver="hkl_soleil",
                geometry="APS POLAR",
                reals=dict(tau="m73", mu="m4", gamma="m19", delta="m20", chi="m37", phi="m38"),
                _real="tau mu chi phi gamma delta".split(),
            )
            cradle.core.axes_xref
            # {'tau': 'tau', 'mu': 'mu', 'chi': 'chi', 'phi': 'phi',
            #  'gamma': 'gamma', 'delta': 'delta'}  ← correct
