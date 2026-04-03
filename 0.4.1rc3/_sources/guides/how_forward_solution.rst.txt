.. _how_forward_solution:

=============================================
How to Choose the Default forward() Solution
=============================================

.. index::
    !forward() solution
    pick_first_solution
    pick_closest_solution
    _forward_solution

The :meth:`~hklpy2.diffract.DiffractometerBase.forward` method may return
multiple valid real-axis positions for a given set of pseudo-axis coordinates.
A *solution picker* function selects one solution from that list.

This guide explains the built-in pickers, how to switch between them, and how
to write a custom one.

Built-in solution pickers
--------------------------

Two pickers are provided:

.. list-table::
   :header-rows: 1

   * - Function
     - Behavior
   * - :func:`~hklpy2.misc.pick_first_solution` *(default)*
     - Returns the first solution in the list as supplied by the solver.
   * - :func:`~hklpy2.misc.pick_closest_solution`
     - Returns the solution whose real-axis values are closest (RMS distance)
       to the current motor positions.

The default is :func:`~hklpy2.misc.pick_first_solution`.

Check the current picker
-------------------------

::

    >>> e4cv._forward_solution
    <function pick_first_solution at 0x...>

Switch to pick_closest_solution at runtime
-------------------------------------------

Assign directly to :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution`::

    >>> from hklpy2.misc import pick_closest_solution
    >>> e4cv._forward_solution = pick_closest_solution

Switch back::

    >>> from hklpy2.misc import pick_first_solution
    >>> e4cv._forward_solution = pick_first_solution

Set the picker at creation time
---------------------------------

Pass ``forward_solution_function`` as a dotted name string to
:func:`~hklpy2.diffract.creator`::

    >>> import hklpy2
    >>> e4cv = hklpy2.creator(
    ...     name="e4cv",
    ...     forward_solution_function="hklpy2.misc.pick_closest_solution",
    ... )

Or pass a callable directly to
:class:`~hklpy2.diffract.DiffractometerBase.__init__` when subclassing::

    >>> from hklpy2.misc import pick_closest_solution
    >>> e4cv = MyDiffractometerClass(
    ...     "",
    ...     name="e4cv",
    ...     forward_solution_function=pick_closest_solution,
    ... )

Write a custom picker
----------------------

A picker function must accept two arguments and return one solution:

.. code-block:: python

    from typing import NamedTuple

    def my_picker(position: NamedTuple, solutions: list[NamedTuple]) -> NamedTuple:
        """Return the solution with the smallest omega value."""
        from hklpy2.misc import NoForwardSolutions
        if not solutions:
            raise NoForwardSolutions("No solutions.")
        return min(solutions, key=lambda s: abs(s.omega))

Assign it at runtime or at creation::

    >>> e4cv._forward_solution = my_picker

    >>> e4cv = hklpy2.creator(
    ...     name="e4cv",
    ...     forward_solution_function="mymodule.my_picker",
    ... )

The picker interface
---------------------

All pickers must follow this interface:

``position`` : named tuple
    The current real-axis position of the diffractometer (e.g.
    ``RealPosition(omega=..., chi=..., phi=..., tth=...)``).

``solutions`` : list of named tuples
    All valid solutions returned by :meth:`~hklpy2.ops.Core.forward`.
    Each element is a named tuple with the same fields as ``position``.

**Returns:** one named tuple from ``solutions``.

**Raises:** :exc:`~hklpy2.misc.NoForwardSolutions` if ``solutions`` is empty.

.. seealso::

    :func:`~hklpy2.misc.pick_first_solution`,
    :func:`~hklpy2.misc.pick_closest_solution` — built-in pickers.

    :attr:`~hklpy2.diffract.DiffractometerBase._forward_solution` — the
    attribute that holds the active picker.

    :meth:`~hklpy2.ops.Core.forward` — returns all solutions before the
    picker is applied.
