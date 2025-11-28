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
    from hklpy2.backends.base import SolverReflectionType
    from hklpy2.misc import IDENTITY_MATRIX_3X3
    from hklpy2.misc import Matrix3x3
    from hklpy2.misc import NamedFloatDict

    class MySolver(SolverBase):
        name = "my_solver"
        version = "1.0.0"

        def __init__(self, geometry: str, **kwargs):
            super().__init__(geometry, **kwargs)

        # Required abstract methods
        def addReflection(self, reflection: SolverReflectionType) -> None:
            """Add an observed diffraction reflection."""
            pass  # TODO: send to your library

        def calculate_UB(self, r1: SolverReflectionType, r2: SolverReflectionType) -> Matrix3x3:
            """Calculate the UB matrix with two reflections."""
            return IDENTITY_MATRIX_3X3  # TODO: calculate with your library

        def forward(self, pseudos: NamedFloatDict) -> list[NamedFloatDict]:
            """Compute list of solutions(reals) from pseudos."""
            return [{}]  # TODO: calculate with your library

        def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
            """Compute pseudos from reals."""
            return {}  # TODO: calculate with your library

        def refineLattice(self, reflections: list[SolverReflectionType]) -> NamedFloatDict:
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
``forward(pseudos)``            Compute list of solutions(reals) from pseudos.
``geometries``                  ``@classmethod`` [#classmethod_decorator]_ : Returns list of all geometries support by this solver.
``inverse(reals)``              Compute pseudos from reals.
``modes``                       Returns list of all modes support by this geometry.
``pseudo_axis_names``           Returns list of all pseudos support by this geometry.
``real_axis_names``             Returns list of all reals support by this geometry.
``refineLattice(reflections)``  Return refined lattice parameters given reflections.
``removeAllReflections()``      Clears sample of all stored reflections.
==============================  ==================

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
