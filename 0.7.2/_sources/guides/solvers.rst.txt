.. _guide.solvers:

=============
Solvers Guide
=============

.. seealso:: :ref:`concepts.solvers` for a concept overview.

How to list available Solvers
=============================

To list all available |solver| classes (by their entry point name),
call :func:`~hklpy2.backends.base.solvers()`.
This example shows the |solver| classes supplied with |hklpy2|::

    >>> from hklpy2 import solvers
    >>> solvers()
    {'hkl_soleil': 'hklpy2.backends.hkl_soleil:HklSolver',
     'th_tth': 'hklpy2.backends.th_tth_q:ThTthSolver'}

How to select a Solver
=======================

To create an instance of a specific |solver| class, use
:func:`~hklpy2.misc.solver_factory`.  The first argument is the entry point
name; the ``geometry`` keyword picks the geometry.  In the next example
(Linux-only), ``hkl_soleil`` picks the
:class:`~hklpy2.backends.hkl_soleil.HklSolver` and ``"E4CV"`` selects the
Eulerian 4-circle geometry with the *hkl* engine:

.. code-block:: pycon

    >>> from hklpy2 import solver_factory
    >>> solver = solver_factory("hkl_soleil", "E4CV")
    >>> print(solver)
    HklSolver(name='hkl_soleil', version='v5.0.0.3434', geometry='E4CV', engine='hkl')

To select a |solver| class without creating an instance, call
:func:`~hklpy2.misc.get_solver`:

.. code-block:: pycon

    >>> from hklpy2 import get_solver
    >>> Solver = get_solver("hkl_soleil")
    >>> print(f"{Solver=}")
    Solver=<class 'hklpy2.backends.hkl_soleil.HklSolver'>

How to register a new Solver
=============================

A |solver| class is registered as a plugin via an `entry point
<https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins>`_
using the ``"hklpy2.solver"`` group.  Here is an example from |hklpy2|'s
``pyproject.toml``::

    [project.entry-points."hklpy2.solver"]
    hkl_soleil = "hklpy2.backends.hkl_soleil:HklSolver"
    th_tth = "hklpy2.backends.th_tth_q:ThTthSolver"

.. seealso:: :ref:`howto.solvers.write`

Solver descriptions
===================

Solver: hkl_soleil
------------------

*Hkl* (`documentation <https://people.debian.org/~picca/hkl/hkl.html>`_), from
Synchrotron Soleil, is used as a backend library to convert between real-space
motor coordinates and reciprocal-space crystallographic coordinates.  Here, we
refer to this library as **hkl_soleil** to clarify and distinguish from other
use of the term *hkl*.  Multiple source code repositories exist. |hklpy2|
uses the `active development repository <https://repo.or.cz/hkl.git>`_.

.. caution:: At this time, it is only compiled for 64-bit Linux.  Not Windows, not Mac OS.

Solver: no_op
-------------

This solver was built for testing the |hklpy2| code.  It provides no useful
geometries for diffractometer users.

Solver: th_tth
--------------

This solver was built as a demonstration of a minimal all-Python solver.  It
provides basic support for :math:`\theta, 2\theta` geometry with a :math:`Q`
pseudo axis. It can be used on any OS where Python runs.
