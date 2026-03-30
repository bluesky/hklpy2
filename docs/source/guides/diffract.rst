.. _diffract_axes:

=========================
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

When custom pseudo axis names are defined in a different order than the
|solver| expects, the ``_pseudo`` keyword declares which local name maps to
each |solver| pseudo axis slot.

Without ``_pseudo``, :func:`~hklpy2.diffract.creator()` zips the pseudo
names positionally against the solver's pseudo axis order.  If these orders
differ, pseudo axes are silently swapped in ``axes_xref``.

**Example:** an E4CV diffractometer with custom pseudo names supplied in a
different order than the solver's ``h, k, l``:

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
