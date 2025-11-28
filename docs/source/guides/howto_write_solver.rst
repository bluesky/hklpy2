.. _howto.solvers.write:

How to write a new Solver
=========================

An |hklpy2| |solver| is an adapter [#adapter_pattern]_ for the backend
diffractometer computation library it supports.

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

    from hklpy2.backends.base import NamedFloatDict
    from hklpy2.backends.base import SolverBase
    from hklpy2.backends.base import SolverLattice
    from hklpy2.backends.base import SolverMatrix3x3
    from hklpy2.backends.base import SolverReflection
    from hklpy2.misc import IDENTITY_MATRIX_3X3

    class MySolver(SolverBase):
        name = "my_solver"
        version = "1.0.0"

        def __init__(self, geometry: str, **kwargs):
            super().__init__(geometry, **kwargs)

        # Required abstract methods
        def addReflection(self, reflection: SolverReflection) -> None:
            """Add coordinates of a diffraction condition."""
            pass  # TODO: send to your library

        def calculate_UB(
            self, r1: SolverReflection, r2: SolverReflection
        ) -> SolverMatrix3x3:
            """Calculate the UB matrix with two reflections."""
            return IDENTITY_MATRIX_3X3  # TODO: calculate with your library

        def forward(self, pseudos: NamedFloatDict) -> list[NamedFloatDict]:
            """Compute list of solutions(reals) from pseudos."""
            return [{}]  # TODO: calculate with your library

        def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
            """Compute pseudos from reals."""
            return {}  # TODO: calculate with your library

        def refineLattice(self, reflections: list[SolverReflection]) -> NamedFloatDict:
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

Add your solver to the ``[project.entry-points."hklpy2.solver"]``
section in your project's ``pyproject.toml`` file.  Here's an example:

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

Use the :func:`~hklpy2.diffract.creator()` factory to create a
diffractometer with your solver and its default geometry:

.. code-block:: Python
    :linenos:

    import hklpy2

    sim = hklpy2.creator(name="sim", solver="my_solver")
    sim.wh()

Key Implementation Details
---------------------------

Required Methods Contract
^^^^^^^^^^^^^^^^^^^^^^^^^

All solvers must implement these methods with specific contracts:

=============================   ==================
method                          description
=============================   ==================
``forward(pseudos)``            Returns ``list[dict]`` of all possible real angle solutions
``inverse(reals)``              Returns single ``dict`` of pseudo coordinates
``calculate_UB(r1, r2)``        Returns 3×3 UB matrix using Busing & Levy method
``addReflection(reflection)``   Updates (current) sample with new reflection
``removeAllReflections()``      Clears (current) sample all stored reflections
=============================   ==================

Engineering Units System
^^^^^^^^^^^^^^^^^^^^^^^^

Define your solver's internal units via class constants:

* ``ANGLE_UNITS``: Default "degrees"
* ``LENGTH_UNITS``: Default "angstrom"

Example Reference
^^^^^^^^^^^^^^^^^

See these example solver implementation classes:

=======================================================  =====================================
class                                                    description
=======================================================  =====================================
:class:`~hklpy2.backends.th_tth_q.ThTthSolver`           Minimal pure-Python solver
:class:`~hklpy2.backends.no_op.NoOpSolver`               No-operation solver for testing
:class:`~hklpy2.backends.hkl_soleil.HklSolver`           Production solver (Linux x86_64 only)
``TrivialSolver()`` [#TrivialSolver]_                    basic template (in tests)
=======================================================  =====================================

Notes
-----

* The solver system [#solver_system_analysis]_ uses Python's *entry point*
  mechanism [#entry_point]_ for runtime discovery.
* All solvers must inherit from
  :class:`~hklpy2.backends.base.SolverBase` which enforces a consistent
  interface.
* The :class:`~hklpy2.ops.Core` class handles unit conversion between
  diffractometer and solver units.
* Solvers can be platform-specific (like
  :class:`~hklpy2.backends.hkl_soleil.HklSolver` which is Linux x86_64
  only).
* Consider using :class:`~hklpy2.backends.no_op.NoOpSolver` or
  ``TrivialSolver()`` [#TrivialSolver]_ as starting references for testing
  infrastructure.

Footnotes
^^^^^^^^^

.. [#abstractmethod_decorator] The ``@abstractmethod`` decorator, from the
    Python standard library module  :mod:`abc`, enforces that subclasses
    implement the decorated methods. See `@abstractmethod
    <https://docs.python.org/3/library/abc.html#abc.abstractmethod>`_ for more
    details or this `tutorial
    <https://coderivers.org/blog/abstract-method-python/>`_.
.. [#adapter_pattern] *Adapter pattern* ( or *wrapper*) is a software design pattern.
    For more details, see this `explanation <https://en.wikipedia.org/wiki/Adapter_pattern>`_.
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
