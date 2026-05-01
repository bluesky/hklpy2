.. _concepts.solvers:

==================
Solvers
==================

.. index::
    !design; solver
    !solver

A |solver| is a Python class that connects |hklpy2| with a backend library
providing diffractometer capabilities, including:

* definition(s) of physical diffractometer **geometries**

  * axes (angles and reciprocal space)
  * operating **modes** for the axes

* calculations that convert:

  * **forward**: reciprocal space coordinates into diffractometer angles
  * **inverse**: diffractometer angles into reciprocal space coordinates

* support blocks include:

  * calculate the UB matrix
  * refine the crystal lattice
  * sample definition (name, lattice parameters, orientation reflections)

.. index:: entry point

A |solver| class is written as a plugin for |hklpy2| and is connected by an
`entry point
<https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins>`_
using the ``"hklpy2.solver"`` group.

Solvers provided with |hklpy2|:

.. list-table::
   :header-rows: 1

   * - Name
     - Description
   * - ``hkl_soleil``
     - |libhkl| backend (Linux 64-bit only). Supports many geometry types.
   * - ``no_op``
     - No-op solver for testing. Provides no useful geometries.
   * - ``th_tth``
     - Pure-Python minimal solver: :math:`\theta, 2\theta` geometry with
       :math:`Q` pseudo axis. Runs on any OS.

.. grid:: 2

    .. grid-item-card:: :material-outlined:`check_box;3em` A |solver| is not needed to:

      - define subclass of :class:`~hklpy2.diffract.DiffractometerBase()` and create instance
      - create instance of :class:`~hklpy2.blocks.sample.Sample()`
      - create instance of :class:`~hklpy2.blocks.lattice.Lattice()`
      - create instance of :class:`~hklpy2.incident._WavelengthBase()` subclass
      - create instance of :class:`~hklpy2.backends.base.SolverBase()` subclass
      - set :attr:`~hklpy2.incident._WavelengthBase.wavelength`
      - list *available* solvers: (:func:`~hklpy2.misc.solvers`)
      - review saved orientation details

    .. grid-item-card:: :material-outlined:`rule;3em` A |solver| is needed to:

      - list available |solver| :attr:`~hklpy2.backends.base.SolverBase.geometries`
      - list a |solver| geometry's required
        :attr:`~hklpy2.backends.base.SolverBase.pseudo_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.real_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.extra_axis_names`,
        :attr:`~hklpy2.backends.base.SolverBase.modes`
      - create instance of :class:`~hklpy2.blocks.reflection.Reflection()`
      - define or compute a :math:`UB` matrix
        (:func:`~hklpy2.user.calc_UB`, :meth:`~hklpy2.ops.Core.calc_UB`)
      - :meth:`~hklpy2.backends.base.SolverBase.forward`
        and :meth:`~hklpy2.backends.base.SolverBase.inverse`
      - determine the diffractometer :attr:`~hklpy2.diffract.DiffractometerBase.position`
      - save or restore orientation details
      - refine lattice parameters

.. seealso::

   :ref:`guide.solvers` for how to list, select, and instantiate solvers.

   :ref:`howto.solvers.write` for how to write a new solver plugin.
