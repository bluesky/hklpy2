.. _how_axis_positions:

===============================================
Ways to Specify Real and Pseudo Axis Positions
===============================================

.. index::
    !axis positions
    forward() axis positions
    inverse() axis positions
    pseudo axis positions
    real axis positions

|hklpy2| accepts several equivalent forms when specifying axis positions to
:meth:`~hklpy2.diffract.DiffractometerBase.forward`,
:meth:`~hklpy2.diffract.DiffractometerBase.inverse`,
:meth:`~hklpy2.ops.Core.forward`,
:meth:`~hklpy2.ops.Core.inverse`,
:meth:`~hklpy2.ops.Core.add_reflection`, and related methods.

The examples use an E4CV diffractometer restored from a saved configuration::

    import hklpy2
    sim = hklpy2.simulator_from_config("e4cv-config.yml")

Pseudo axis positions (for ``diffractometer.forward()``)
---------------------------------------------------------

Pseudo axes are the reciprocal-space coordinates (*h*, *k*, *l* for an HKL
geometry).

.. tabs::

    .. tab:: Positional args

        Pass values in the order the solver expects (``h``, ``k``, ``l``
        for E4CV):

        .. code-block:: python

            sim.forward(1, 0, 0)

    .. tab:: Keyword args

        Name each axis explicitly — order does not matter:

        .. code-block:: python

            sim.forward(h=1, k=0, l=0)
            sim.forward(l=0, h=1, k=0)   # same result

    .. tab:: Dictionary

        Pass a ``dict`` mapping axis names to values:

        .. code-block:: python

            sim.forward({"h": 1, "k": 0, "l": 0})

    .. tab:: Named tuple

        Use the diffractometer's ``PseudoPosition`` named tuple:

        .. code-block:: python

            pp = sim.PseudoPosition(h=1, k=0, l=0)
            sim.forward(pp)

    .. tab:: Bare tuple

        Values are mapped positionally in solver order (``h``, ``k``, ``l``):

        .. code-block:: python

            sim.forward((1, 0, 0))

Real axis positions (for ``diffractometer.inverse()``)
-------------------------------------------------------

Real axes are the physical motor angles (``omega``, ``chi``, ``phi``,
``tth`` for E4CV).

.. tabs::

    .. tab:: Positional args

        Pass values in the order the solver expects:

        .. code-block:: python

            sim.inverse(-145, 0, 0, 69)

    .. tab:: Keyword args

        Name each axis explicitly:

        .. code-block:: python

            sim.inverse(omega=-145, chi=0, phi=0, tth=69)

    .. tab:: Dictionary

        Pass a ``dict`` mapping axis names to values:

        .. code-block:: python

            sim.inverse({"omega": -145, "chi": 0, "phi": 0, "tth": 69})

    .. tab:: Named tuple

        Use the diffractometer's ``RealPosition`` named tuple:

        .. code-block:: python

            rp = sim.RealPosition(omega=-145, chi=0, phi=0, tth=69)
            sim.inverse(rp)

    .. tab:: Bare tuple

        Values are mapped positionally in solver order
        (``omega``, ``chi``, ``phi``, ``tth``):

        .. code-block:: python

            sim.inverse((-145, 0, 0, 69))

Lower-level: ``core.forward()`` and ``core.inverse()``
------------------------------------------------------

The :meth:`~hklpy2.ops.Core.forward` and :meth:`~hklpy2.ops.Core.inverse`
methods on the ``core`` object are lower-level and behave differently:

- ``core.forward()`` returns **all** solutions as a list (unfiltered by
  constraints or the forward-solution picker).
- ``core.inverse()`` returns a plain **dict** (not a named tuple).
- Positional args and keyword args are **not** accepted.

.. tabs::

    .. tab:: Positional args

        Not available for ``core.forward()`` or ``core.inverse()``.

    .. tab:: Keyword args

        Not available for ``core.forward()`` or ``core.inverse()``.

    .. tab:: Dictionary

        .. code-block:: python

            # forward — returns all solutions
            solutions = sim.core.forward({"h": 1, "k": 0, "l": 0})

            # inverse — returns a dict
            hkl = sim.core.inverse({"omega": -145, "chi": 0, "phi": 0, "tth": 69})

    .. tab:: Named tuple

        .. code-block:: python

            # forward
            pp = sim.PseudoPosition(h=1, k=0, l=0)
            solutions = sim.core.forward(pp)

            # inverse
            rp = sim.RealPosition(omega=-145, chi=0, phi=0, tth=69)
            hkl = sim.core.inverse(rp)

    .. tab:: Bare tuple

        Values are mapped positionally in solver order:

        .. code-block:: python

            # forward: (h, k, l)
            solutions = sim.core.forward((1, 0, 0))

            # inverse: (omega, chi, phi, tth)
            hkl = sim.core.inverse((-145, 0, 0, 69))

Use ``diffractometer.forward()`` and ``diffractometer.inverse()`` for everyday
work — they apply constraints, pick the best solution, and return named tuples.
Use ``core.forward()`` when you need access to all solutions before filtering.

Reflections: ``core.add_reflection()``
---------------------------------------

:meth:`~hklpy2.ops.Core.add_reflection` accepts **Dictionary**,
**Named tuple**, and **Bare tuple** for both the pseudo (``hkl``) and real
(motor angles) position arguments — **Positional args** and **Keyword args**
are not available:

.. code-block:: python

    # Dictionary (recommended)
    sim.core.add_reflection(
        dict(h=4, k=0, l=0),
        dict(omega=-145.451, chi=0, phi=0, tth=69.066),
    )

    # Named tuple
    sim.core.add_reflection(
        sim.PseudoPosition(h=4, k=0, l=0),
        sim.RealPosition(omega=-145.451, chi=0, phi=0, tth=69.066),
    )

    # Bare tuple (positionally mapped in solver order)
    sim.core.add_reflection((4, 0, 0), (-145.451, 0, 0, 69.066))

Reflections: ``user.setor()`` / ``user.add_reflection()``
----------------------------------------------------------

The :func:`~hklpy2.user.setor` function (also aliased as
:func:`~hklpy2.user.add_reflection`) has a distinct and more flexible
interface.  Pseudo positions are always **Positional args** (``h``, ``k``,
``l``).  Real positions can be provided in three ways:

.. tabs::

    .. tab:: Positional args

        Real positions in solver order, appended after ``h``, ``k``, ``l``:

        .. code-block:: python

            from hklpy2.user import setor
            setor(4, 0, 0, -145.451, 0, 0, 69.066)

    .. tab:: Keyword args

        Real positions as named keyword arguments — any order:

        .. code-block:: python

            from hklpy2.user import setor
            setor(4, 0, 0, omega=-145.451, chi=0, phi=0, tth=69.066)

    .. tab:: Dictionary

        Not available for ``setor()``/``user.add_reflection()``.

    .. tab:: Named tuple

        Not available for ``setor()``/``user.add_reflection()``.

    .. tab:: Bare tuple

        Not available for ``setor()``/``user.add_reflection()``.

    .. tab:: Omitted

        When no real positions are given, the current motor positions are used:

        .. code-block:: python

            from hklpy2.user import setor
            setor(4, 0, 0)   # uses current omega, chi, phi, tth

Which form to use
-----------------

.. list-table::
    :header-rows: 1
    :stub-columns: 1
    :widths: 30 14 14 14 14 14

    * - Method
      - Positional args
      - Keyword args
      - Dictionary
      - Named tuple
      - Bare tuple
    * - ``diffractometer.forward()``
      - ✓
      - ✓
      - ✓
      - ✓
      - ✓
    * - ``diffractometer.inverse()``
      - ✓
      - ✓
      - ✓
      - ✓
      - ✓
    * - ``core.forward()``
      - ✗
      - ✗
      - ✓
      - ✓
      - ✓
    * - ``core.inverse()``
      - ✗
      - ✗
      - ✓
      - ✓
      - ✓
    * - ``core.add_reflection()``
      - ✗
      - ✗
      - ✓
      - ✓
      - ✓
    * - ``user.setor()``
      - ✓ (reals)
      - ✓ (reals)
      - ✗
      - ✗
      - ✗

.. seealso::

    :ref:`diffract_axes` — axis naming, ordering, and custom axis names.

    :ref:`how_forward_solution` — choosing which ``forward()`` solution is
    returned.
