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

.. rubric:: The four-stage forward pipeline

Computing a single motor position from pseudo-axis coordinates (e.g.
``h, k, l``) involves four distinct stages across three layers of
|hklpy2|.  This guide covers **Stage 4** — the solution picker.

.. graphviz::
    :caption: Four-stage forward() pipeline in hklpy2 (equivalent stages in SPEC and diffcalc shown in parentheses).
    :align: center

    digraph forward_pipeline {
        graph [rankdir=TB, splines=ortho, nodesep=0.6, ranksep=0.5,
               fontname="sans-serif", bgcolor="transparent"]
        node  [shape=box, style="rounded,filled", fontname="sans-serif",
               fontsize=11, margin="0.15,0.08"]
        edge  [fontname="sans-serif", fontsize=10]

        pseudos [label="pseudos\n(h, k, l)", shape=ellipse,
                 fillcolor="#e8f4e8", color="#4a7c4a"]

        s1 [label="Stage 1: SolverBase.forward()\nBackend engine returns ALL theoretical\nsolutions for the pseudos, geometry, mode.\nSPEC/diffcalc: geometry engine\nlibhkl: pseudo_axis_values_set()",
            fillcolor="#dce8f8", color="#3a6898"]

        s2 [label="Stage 2: apply_cut()  [Core]\nEach axis angle is mapped into\n[cut_point, cut_point+360).\nControls representation, not validity.\nSPEC: cuts  |  diffcalc: _cut_angles()",
            fillcolor="#fdf3dc", color="#a07820"]

        s3 [label="Stage 3: LimitsConstraint.valid()  [Core]\nSolutions whose wrapped axis values\nfall outside configured limits are discarded.\nSPEC: lm  |  diffcalc: is_position_within_limits()",
            fillcolor="#fdf3dc", color="#a07820"]

        s4 [label="Stage 4: solution picker  [DiffractometerBase]\nOne solution is selected from survivors\nfor motor motion (ophyd PseudoPositioner).",
            fillcolor="#f0e8f8", color="#6a3a98",
            style="rounded,filled,bold"]

        result [label="single real-axis position\nfor motor motion",
                shape=ellipse, fillcolor="#e8f4e8", color="#4a7c4a"]

        pseudos -> s1 [label="all theoretical solutions"]
        s1      -> s2 [label="wrapped solutions"]
        s2      -> s3 [label="filtered solutions"]
        s3      -> s4 [label="one solution"]
        s4      -> result
    }

The number of solutions entering Stage 4 depends on the backend
|solver|'s capabilities.  Some engines analytically enumerate all
mathematically valid solutions; others return only one.  A
single-element list is valid.  See also
:ref:`howto.solvers.write.forward_contract`.

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
