.. _concepts.creator:

======================
Diffractometer creator
======================

.. index:: factory; creator, creator

The :func:`~hklpy2.diffract.creator()`, a `factory function
<https://en.wikipedia.org/wiki/Factory_(object-oriented_programming)>`_, reduces
the effort to create all but the most complex diffractometer objects. It uses
the :func:`~hklpy2.diffract.diffractometer_class_factory()` to build the Python
class for the chosen |solver| and geometry.  Here's an example :ref:`6-circle
<geometries-hkl_soleil-k6c>` diffractometer in kappa geometry with simulated
motors::

    k6c = hklpy2.creator(name="k6c", solver="hkl_soleil", geometry="K6C")

.. tip:: It's always possible to define your own subclass of
    :class:`~hklpy2.diffract.DiffractometerBase()` when you need more control than
    provided by :func:`~hklpy2.diffract.creator()`. See
    :ref:`example.e4cv.custom-class`.

.. index:: factory; diffractometer_class_factory, diffractometer_class_factory

The :func:`~hklpy2.diffract.diffractometer_class_factory()` gets information
(the pseudo and real axes) from the |solver| about the requested geometry, then
constructs a Python class with this structure.  The class includes
``ophyd.EpicsMotor`` Components when EPICS PVs are provided for the real axes,
otherwise ``ophyd.SoftPositioner`` (simulated motor) Components are used.  Names for
these Components, when provided by the caller are mapped in order to the reals
expected by the |solver|.  All real axis name conversion between user-provided
and |solver|-defined is handled in the :ref:`concepts.ops`.

.. index:: factory; solver_factory, solver_factory

Another factory, :func:`~hklpy2.misc.solver_factory()`, locates if the selected
|solver| has been installed and then creates an instance of the corresponding
:class:`~hklpy2.backends.base.SolverBase` subclass with the selected geometry
and any additional keyword arguments provided by the caller.

.. tip:: |hklpy2| is extensible.  A new |solver| can be added to the Python
    environment using the ``"hklpy2.solver"`` :index:`entry point`.  (See
    :ref:`concepts.solvers` for more details.)

.. rubric:: Examples

See :ref:`examples` for many documents which use the
:func:`~hklpy2.diffract.creator()` factory function.
Here's a sampler:

=============================== ====================
diffractometer description      example
=============================== ====================
simple                          :doc:`/examples/hkl_soleil-e4cv`
geometry                        :doc:`/examples/hkl_soleil-k4cv`
solver                          :doc:`/examples/_api_demo`
engine (via ``solver_kwargs``)  :doc:`/guides/var_engines`
EPICS PVs                       :doc:`/examples/hkl_soleil-e4cv+epics`
renamed axes                    :doc:`/examples/aps-isn-e6c`
additional reals                :doc:`/examples/hkl_soleil-e6c-psi`
auto assign                     tbd
reals out of order              tbd
extras                          :doc:`/examples/hkl_soleil-e6c-psi`
=============================== ====================
