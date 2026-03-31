.. _concepts.solvers:

==================
Solvers
==================

.. TODO: How much is guide or example?  This should be a concepts doc. Brief.

.. TODO:
    - Describe the responsibilities of a |solver|.
    - Define the terms expected (add to glossary.).
    - Note that solvers provide different features: additions and not availables

.. index:: !design; solver

A |solver| is a Python class that connects |hklpy2| with a (backend) library
that provides diffractometer capabilities, including:

* definition(s) of physical diffractometer **geometries**

  * axes (angles and reciprocal space)
  * operating **modes** for the axes (angles and reciprocal space)

* calculations that convert:

  * **forward**: reciprocal space coordinates into diffractometer angles
  * **inverse**: diffractometer angles into reciprocal space coordinates

* support blocks include:

  * calculate the UB matrix
  * refine the crystal lattice
  * sample definition

    * name
    * crystal lattice parameters: :math:`a, b, c, \alpha, \beta, \gamma`
    * list of orientation reflections

.. index:: entry point

A |solver| class is written as a plugin for |hklpy2| and is connected by an `entry point
<https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins>`_
using the ``"hklpy2.solver"`` group.  Here's an example from |hklpy2|'s
``pyproject.toml`` file for two such |solver| classes::

    [project.entry-points."hklpy2.solver"]
    hkl_soleil = "hklpy2.backends.hkl_soleil:HklSolver"
    th_tth = "hklpy2.backends.th_tth_q:ThTthSolver"


.. _api.solvers.set:

How to select a Solver
----------------------

To list all available |solver| classes (by their entry point name),
call :func:`~hklpy2.backends.base.solvers()`.
This example shows the |solver| classes supplied with |hklpy2|::

    >>> from hklpy2 import solvers
    >>> solvers()
    {'hkl_soleil': 'hklpy2.backends.hkl_soleil:HklSolver',
     'th_tth': 'hklpy2.backends.th_tth_q:ThTthSolver'}

This is a dictionary, keyed by the solver names.  To create an instance
of a specific |solver| class, use :func:`~hklpy2.misc.solver_factory`.
In the next example (Linux-only), the first argument, `hkl_soleil`, picks the
:class:`~hklpy2.backends.hkl_soleil.HklSolver`, the `geometry` keyword
picks the Eulerian 4-circle geometry with the *hkl* engine:

.. code-block: Python
    :linenos:

    >>> from hklpy2 import solver_factory
    >>> solver = solver_factory("hkl_soleil", "E4CV")
    >>> print(solver)
    HklSolver(name='hkl_soleil', version='v5.0.0.3434', geometry='E4CV', engine='hkl')

To select a |solver| class without creating an instance, call
:func:`~hklpy2.misc.get_solver`. This example
selects the |libhkl| |solver| (using its entry point name:
``"hkl_soleil"``):

.. code-block: Python
    :linenos:

    >>> from hklpy2 import get_solver
    >>> Solver = get_solver("hkl_soleil")
    >>> print(f"{Solver=}")
    Solver=<class 'hklpy2.backends.hkl_soleil.HklSolver'>

Solver: hkl_soleil
~~~~~~~~~~~~~~~~~~~~~~

*Hkl* (`documentation <https://people.debian.org/~picca/hkl/hkl.html>`_), from
Synchrotron Soleil, is used as a backend library to convert between real-space
motor coordinates and reciprocal-space crystallographic coordinates.  Here, we
refer to this library as **hkl_soleil** to clarify and distinguish from other
use of of the term *hkl*.  Multiple source code repositories exist. |hklpy2|
uses the `active development repository <https://repo.or.cz/hkl.git>`_.

.. caution:: At this time, it is only compiled for 64-bit Linux.  Not Windows, not Mac OS.

Solver: no_op
~~~~~~~~~~~~~~~~~~~~~~

This solver was built for testing the |hklpy2| code.  It provides no useful
geometries for diffractometer users.

Solver: th_tth
~~~~~~~~~~~~~~~~~~~~~~

This solver was built as a demonstration of a minimal all Python solver.  It
provides basic support for :math:`\theta, 2\theta` geometry with a :math:`Q`
pseudo axis. It can be used on any OS where Python runs.

How to write a Solver
----------------------

.. seealso:: :ref:`howto.solvers.write`
