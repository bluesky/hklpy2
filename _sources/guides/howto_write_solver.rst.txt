.. _howto.solvers.write:

How to write a new Solver
=========================

An |hklpy2| |solver| is an adapter [#adapter_pattern]_ for a backend
diffractometer computation library.

.. reference:
    https://deepwiki.com/search/describe-the-steps-to-write-a_90411934-3765-4bb8-b4da-bc1672c09b96?mode=fast

    .. Collected considerations for Solvers
    - https://github.com/bluesky/hklpy/issues/14
    - https://github.com/bluesky/hklpy/issues/161
    - https://github.com/bluesky/hklpy/issues/162
    - https://github.com/bluesky/hklpy/issues/163
    - https://github.com/bluesky/hklpy/issues/165
    - https://github.com/bluesky/hklpy/issues/244
    - https://xrayutilities.sourceforge.io/
    - https://cohere.readthedocs.io
    - https://github.com/AdvancedPhotonSource/cohere-scripts/tree/main/scripts/beamlines/aps_34idc
    - https://xrayutilities.sourceforge.io/_modules/xrayutilities/experiment.html#QConversion
    - https://github.com/DiamondLightSource/diffcalc
    - SPEC server mode
    - https://github.com/prjemian/pyub
    - https://github.com/prjemian/chewacla

Steps
-----

.. sidebar:: Overview

    * Create a new Python package.
    * Create a Solver class.
    * Register as Entry Point.
    * Install and test.

To write a new solver for |hklpy2|, you need to create a Python class
that inherits from :class:`~hklpy2.backends.base.SolverBase` and
register it as an entry point.

.. tip:: Create a new project [#new_python_gh_project]_ for this work.

Here are the essential steps:

Step 1. Create a Solver Class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a Python class that inherits from
:class:`~hklpy2.backends.base.SolverBase` and implement all required abstract
methods (methods marked with decorator ``@abstractmethod``
[#abstractmethod_decorator]_):

.. code-block:: Python
    :linenos:

    from hklpy2.backends.base import SolverBase
    from hklpy2.misc import IDENTITY_MATRIX_3X3
    from hklpy2.misc import KeyValueMap
    from hklpy2.misc import Matrix3x3
    from hklpy2.misc import NamedFloatDict

    class MySolver(SolverBase):
        name = "my_solver"
        version = "1.0.0"

        def __init__(self, geometry: str, **kwargs):
            super().__init__(geometry, **kwargs)

        # Required abstract methods
        def addReflection(self, reflection: KeyValueMap) -> None:
            """Add an observed diffraction reflection."""
            pass  # TODO: send to your library

        def calculate_UB(self, r1: KeyValueMap, r2: KeyValueMap) -> Matrix3x3:
            """Calculate the UB matrix with two reflections."""
            return IDENTITY_MATRIX_3X3  # TODO: calculate with your library

        def forward(self, pseudos: NamedFloatDict) -> list[NamedFloatDict]:
            """Compute list of solutions(reals) from pseudos."""
            return [{}]  # TODO: calculate with your library

        def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
            """Compute pseudos from reals."""
            return {}  # TODO: calculate with your library

        def refineLattice(self, reflections: list[KeyValueMap]) -> NamedFloatDict:
            """Refine lattice parameters from reflections."""
            return {}  # TODO: calculate with your library

        def removeAllReflections(self) -> None:
            """Remove all reflections."""
            pass  # TODO: use your library

        # Required properties
        @property
        def extra_axis_names(self) -> list[str]:
            """Ordered list of extra axis names."""
            return []

        @classmethod
        def geometries(cls) -> list[str]:
            """Supported diffractometer geometries."""
            return ["MY_GEOMETRY"]

        @property
        def modes(self) -> list[str]:
            """Available operating modes."""
            return []

        @property
        def pseudo_axis_names(self) -> list[str]:
            """Ordered list of pseudo axis names (h, k, l)."""
            return []

        @property
        def real_axis_names(self) -> list[str]:
            """Ordered list of real axis names (omega, chi, phi, tth)."""
            return []

Step 2. Register as Entry Point
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a ``[project.entry-points."hklpy2.solver"]`` section in your project's
``pyproject.toml`` file and declare your solver.  Here's an example:

.. code-block:: toml
    :linenos:

    [project.entry-points."hklpy2.solver"]
    my_solver = "my_package.my_solver:MySolver"

Step 3. Install and Test
^^^^^^^^^^^^^^^^^^^^^^^^

Install your package to make the solver discoverable. Then test that it
loads correctly:

.. code-block:: Python
    :linenos:

    import hklpy2

    # List available solvers
    print(hklpy2.solvers())

    # Load your solver class
    SolverClass = hklpy2.get_solver("my_solver")

    # Create an instance
    solver = SolverClass("MY_GEOMETRY")

Use the :func:`~hklpy2.diffract.creator()` factory to create a diffractometer
with your solver and test it.  Here's a suggested start:

.. code-block:: Python
    :linenos:

    import hklpy2

    sim = hklpy2.creator(name="sim", solver="my_solver")
    sim.wh()

Key Implementation Details
---------------------------

Required Methods Contract
^^^^^^^^^^^^^^^^^^^^^^^^^

All solvers must implement these attributes, methods, and properties:

==============================  ==================
method (or property)            description
==============================  ==================
``name``                        (string attribute) Name of this solver.
``version``                     (string attribute) Version of this solver.
``addReflection(reflection)``   Add an observed diffraction reflection.
``calculate_UB(r1, r2)``        Calculate the UB matrix with two reflections.
``extra_axis_names``            Returns list of any extra axes in the current *mode*.
``forward(pseudos)``            Compute list of solutions(reals) from pseudos.  A single-element list is acceptable (see :ref:`forward() contract <howto.solvers.write.forward_contract>`).
``geometries``                  ``@classmethod`` [#classmethod_decorator]_ : Returns list of all geometries support by this solver.
``inverse(reals)``              Compute pseudos from reals.
``modes``                       Returns list of all modes support by this geometry.
``pseudo_axis_names``           Returns list of all pseudos support by this geometry.
``real_axis_names``             Returns list of all reals support by this geometry.
``refineLattice(reflections)``  Return refined lattice parameters given reflections.
``removeAllReflections()``      Clears sample of all stored reflections.
==============================  ==================

.. _howto.solvers.write.forward_contract:

``forward()`` Contract
^^^^^^^^^^^^^^^^^^^^^^

The ``forward()`` method appears at three layers, each with a distinct
role and return type:

:meth:`SolverBase.forward(pseudos) <hklpy2.backends.base.SolverBase.forward>`
    Returns ``list[NamedFloatDict]`` — all valid real-axis solutions the
    backend engine can find for the given pseudo-axis values, geometry,
    and mode.  **A single-element list is a valid return value.**

:meth:`Core.forward(pseudos) <hklpy2.ops.Core.forward>`
    Calls the solver's ``forward()``, then applies
    :ref:`constraint filtering <concepts.constraints>` to each solution.
    Returns the filtered list — the full set of candidate motor angle
    combinations that satisfy all constraints.

:meth:`DiffractometerBase.forward(pseudos) <hklpy2.diffract.DiffractometerBase.forward>`
    The :class:`ophyd.PseudoPositioner` interface, called by motion
    commands during bluesky plans.  Calls ``Core.forward()``, then
    applies a :ref:`solution picker <how_forward_solution>` to select
    one solution for motor motion.  Returns a single ``NamedTuple``.

When writing a solver, only ``SolverBase.forward()`` needs to be
implemented.  The number of solutions depends on the backend library's
capabilities.  An empty list (or raising
:exc:`~hklpy2.misc.NoForwardSolutions`) signals that no solution
exists for the requested pseudo-axis values.

.. rubric:: The four-stage forward pipeline

All three backend libraries (|libhkl|, diffcalc, SPEC) follow the same
pattern: the engine returns **all** theoretical solutions; post-processing
stages then wrap, filter, and select.  hklpy2 implements the same
four stages.  A solver adapter is responsible only for **Stage 1**;
Stages 2–4 are handled by the |hklpy2| Core and DiffractometerBase
layers.

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
            fillcolor="#f0e8f8", color="#6a3a98"]

        result [label="single real-axis position\nfor motor motion",
                shape=ellipse, fillcolor="#e8f4e8", color="#4a7c4a"]

        pseudos -> s1 [label="all theoretical solutions"]
        s1      -> s2 [label="wrapped solutions"]
        s2      -> s3 [label="filtered solutions"]
        s3      -> s4 [label="one solution"]
        s4      -> result
    }

.. _howto.solvers.write.backend_requirements:

Backend Library Requirements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A |solver| is an adapter for a backend computation library.  The backend
library must provide (or enable the adapter to implement) the following
capabilities.

Required
""""""""

.. index:: backend requirements; required

**Geometry-aware rotation chain.**
    The library must know the physical axis directions and stacking order
    for each geometry it supports.  This is the irreducible foundation
    for all diffractometer calculations.

**Forward transform** (pseudos to reals).
    Given pseudo-axis values, orientation matrix, and wavelength, compute
    real-axis angles.  This is geometry- and mode-specific.

**Inverse transform** (reals to pseudos).
    Given real-axis angles, lattice parameters, orientation matrix, and
    wavelength, compute pseudo-axis values.

**UB matrix calculation.**
    Given two measured reflections (each with known pseudos, measured
    angles, and wavelength), compute the orientation matrix.  This
    requires the geometry's rotation chain to convert measured angles
    into lab-frame scattering vectors.  ``calculate_UB()``,
    ``forward()``, and ``inverse()`` all depend on the same rotation
    chain and cannot be separated.

**Reflection management.**
    Store and retrieve measured reflections for UB matrix calculation.
    The backend library must manage reflections because UB calculation
    consumes them — the solver adapter passes reflections through to the
    backend, not buffering them in the adapter layer.

Optional
""""""""

.. index:: backend requirements; optional

The following capabilities enhance a solver but are not required.
Solvers that lack these features remain valid — the :class:`~hklpy2.ops.Core`
layer handles their absence gracefully.

**Lattice refinement.**
    Refine lattice parameters from multiple reflections.  Solvers that
    lack this capability return ``None`` from ``refineLattice()``; Core
    raises an informative error to the user.

**Multi-solution enumeration.**
    Return multiple valid angle solutions for a given set of pseudo-axis
    values.  A single-element list is a valid return from ``forward()``
    (see :ref:`howto.solvers.write.forward_contract`).

**Operating modes.**
    Named configurations (e.g., bisector, constant-phi) that define
    which axes will be modified by ``forward()``.

.. _howto.solvers.write.design_rationale:

Design Rationale
""""""""""""""""

.. index:: separation of concerns

The required capabilities above are coupled through the geometry's
rotation chain: the same axis definitions that drive ``forward()`` and
``inverse()`` are needed to compute lab-frame scattering vectors for
``calculate_UB()``, and to manage the reflections that feed it.  A
library that can compute ``forward()`` and ``inverse()`` necessarily
has the geometry knowledge to also compute ``calculate_UB()``.  A
library that lacks this knowledge cannot serve as a complete |hklpy2|
solver backend.

This coupling is why these capabilities cannot be factored into
:class:`~hklpy2.backends.base.SolverBase`.  A "generic" implementation
would require embedding a full geometry engine — effectively becoming a
backend library itself, violating the separation of concerns between
the adapter layer and the computation engine.

The |solver| should be a thin adapter: it translates
:class:`~hklpy2.backends.base.SolverBase` method calls into whatever
the backend library needs.  Computation, data management (including
reflections), and transport (for remote backends) belong in the
backend or its solver adapter, not in the base class.

Backend library APIs vary widely.  For example,
:meth:`HklSolver.forward() <hklpy2.backends.hkl_soleil.HklSolver.forward>`
calls ``engine.pseudo_axis_values_set()`` — a |libhkl| GObject
introspection binding that sets pseudo-axis values and returns a
``GeometryList`` of solutions as a side effect.  The backend function
name gives no indication it is computing forward solutions.  This is
exactly why the solver adapter exists: to present a consistent
``forward(pseudos)`` interface regardless of how the backend library
exposes its capabilities.

Engineering Units System
^^^^^^^^^^^^^^^^^^^^^^^^

Define your solver's internal units via class constants:

* ``ANGLE_UNITS``: Default "degrees"
* ``LENGTH_UNITS``: Default "angstrom"

Example Reference
^^^^^^^^^^^^^^^^^

Compare with these |Solver| classes:

=======================================================  =====================================
class                                                    description
=======================================================  =====================================
:class:`~hklpy2.backends.hkl_soleil.HklSolver`           Full-featured (Linux x86_64 only)
:class:`~hklpy2.backends.no_op.NoOpSolver`               No-operation (demonstration & testing)
:class:`~hklpy2.backends.th_tth_q.ThTthSolver`           Minimal pure-Python (demonstration)
``TrivialSolver()`` [#TrivialSolver]_                    Minimal requirements, non-functional (internal testing)
=======================================================  =====================================

Notes
-----

* |hklpy2| identifies a |Solver| [#solver_system_analysis]_ as a plugin using
  Python's *entry point* [#entry_point]_ support.
* All solvers must inherit from
  :class:`~hklpy2.backends.base.SolverBase` which enforces a consistent
  interface.
* The :class:`~hklpy2.ops.Core` class handles unit conversion between
  diffractometer and solver units.
* Solvers can be platform-specific (such as
  :class:`~hklpy2.backends.hkl_soleil.HklSolver` which is C code compiled only
  for Linux x86_64 architectures).
* Consider using :class:`~hklpy2.backends.no_op.NoOpSolver` or
  ``TrivialSolver()`` [#TrivialSolver]_ as starting references for testing
  infrastructure.

Footnotes
^^^^^^^^^

.. [#abstractmethod_decorator] The ``@abstractmethod`` decorator, from the
    Python standard library module  :mod:`abc`, enforces that subclasses
    implement the decorated method. See `@abstractmethod
    <https://docs.python.org/3/library/abc.html#abc.abstractmethod>`_ for more
    details or this `tutorial
    <https://coderivers.org/blog/abstract-method-python/>`_.
.. [#adapter_pattern] *Adapter pattern* ( or *wrapper*) is a software design pattern.
    For more details, see this `explanation <https://en.wikipedia.org/wiki/Adapter_pattern>`_.
.. [#classmethod_decorator] The ``@classmethod_decorator`` decorator, from the
    Python standard library module  :mod:`abc`, enforces that subclasses
    implement the decorated method. See `@classmethod_decorator
    <https://docs.python.org/3/library/abc.html#abc.classmethod_decorator>`_ for more
    details.
.. [#solver_system_analysis] Analysis of |hklpy2| *Solver*
    `backend <https://deepwiki.com/bluesky/hklpy2/3.4-solver-backend-system>`_
.. [#entry_point] `Entry Points
    <https://packaging.python.org/en/latest/specifications/entry-points/>`_:
    *Entry points are a way for Python packages to advertise components they
    provide to be discovered and used by other packages at runtime.*
.. [#new_python_gh_project] `Guide
    <https://coderivers.org/blog/github-python-projects/#creating-a-python-project-on-github>`_
    to *Create a Python Project on GitHub*.
.. [#TrivialSolver] ``TrivialSolver()``: Source code in the test `suite
    <https://github.com/bluesky/hklpy2/blob/main/hklpy2/backends/tests/test_base.py>`_.
