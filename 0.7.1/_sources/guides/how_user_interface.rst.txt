.. _how_user_interface:

=============================================
How to Use the hklpy2.user Interface
=============================================

.. index::
    !hklpy2.user; how-to
    see: interactive interface; hklpy2.user
    see: SPEC commands; hklpy2.user

The :mod:`hklpy2.user` module provides a set of convenience functions
designed for interactive use at the Python or IPython prompt.  They are
modelled on familiar SPEC commands and operate on a single *active*
diffractometer set with :func:`~hklpy2.user.set_diffractometer`.

.. seealso::

   :ref:`tutorial` — step-by-step walkthrough using these functions together.

   :ref:`how_calc_ub` — detailed guide for the UB matrix workflow.

   :ref:`spec_commands_map` — full cross-reference of SPEC commands to
   their |hklpy2| equivalents.

Setup
-----

All examples assume a diffractometer has been created and set as active::

    >>> import hklpy2
    >>> from hklpy2.user import *
    >>> fourc = hklpy2.creator(name="fourc", geometry="E4CV", solver="hkl_soleil")
    >>> set_diffractometer(fourc)

The ``from hklpy2.user import *`` form is convenient at the interactive
prompt.  In scripts, import only the functions you need.

Setting the active diffractometer
-----------------------------------

All :mod:`hklpy2.user` functions operate on the *active* diffractometer.
Use :func:`~hklpy2.user.set_diffractometer` to select it::

    >>> set_diffractometer(fourc)

To query which diffractometer is currently active::

    >>> get_diffractometer()

Reporting the diffractometer state
------------------------------------

:func:`~hklpy2.user.pa` ("print all") prints the full diffractometer
state — solver, sample, reflections, :math:`UB` matrix, constraints,
mode, wavelength, and current position::

    >>> pa()

:func:`~hklpy2.user.wh` ("where") prints a brief summary of the current
position — wavelength, pseudo axes (:math:`h, k, l`), and real motor
angles::

    >>> wh()

Managing samples
-----------------

Add a named sample with lattice parameters::

    >>> add_sample("silicon", a=hklpy2.SI_LATTICE_PARAMETER)

List all samples on the active diffractometer (the current sample is
shown with a ``>`` prefix)::

    >>> list_samples()

Change the lattice of the current sample::

    >>> set_lattice(5.431)                       # cubic: a only
    >>> set_lattice(3.0, c=5.0, gamma=120)       # hexagonal: a, c, gamma
    >>> set_lattice(5.0, 6.0, 7.0, gamma=109.235)    # monoclinic: a, b, c, gamma

Remove a sample by name::

    >>> remove_sample("silicon")

Managing the wavelength
------------------------

Set the wavelength of the active diffractometer's beam::

    >>> set_wavelength(1.54)                     # Angstroms (default)
    >>> set_wavelength(154.0, units="pm")        # picometres

.. note::

   :func:`~hklpy2.user.set_wavelength` requires write access to the
   beam wavelength signal.  The wavelength is read-only when the source
   has a fixed wavelength — for example an X-ray tube or rotating anode.
   At a tunable synchrotron beamline, wavelength is set via the
   monochromator control system and the signal has write access.

Orientation reflections and UB matrix
---------------------------------------

Record an orientation reflection (motor angles for a known
:math:`(h, k, l)`)::

    >>> r1 = setor(4, 0, 0, tth=69.0966, omega=-145.451, chi=0, phi=0)
    >>> r2 = setor(0, 4, 0, tth=69.0966, omega=-145.451, chi=90, phi=0)

Compute the :math:`UB` matrix from two reflections::

    >>> calc_UB(r1, r2)

Swap the two orienting reflections and recompute :math:`UB` (equivalent
to SPEC's ``or_swap``)::

    >>> or_swap()

Remove a reflection by name::

    >>> remove_reflection(r1.name)

See :ref:`how_calc_ub` for the full UB matrix workflow.

Calculating motor positions
----------------------------

Calculate motor positions for a given :math:`(h, k, l)` **without
moving** any motors::

    >>> cahkl(4, 0, 0)

Show a table of *all* allowed solutions for one or more reflections::

    >>> cahkl_table((4, 0, 0), (0, 4, 0))

Moving to a reciprocal-space position
---------------------------------------

Move the diffractometer to a given :math:`(h, k, l)` position directly::

    >>> fourc.move(4, 0, 0)

Or from a Bluesky plan::

    >>> import bluesky.plan_stubs as bps
    >>> RE(bps.mv(fourc, (4, 0, 0)))

.. tip::

   :func:`~hklpy2.user.cahkl` and :func:`~hklpy2.user.wh` are useful
   before and after a move to verify the expected and actual positions.

Inspecting solver modes and geometries
----------------------------------------

Print a summary of all engines, modes, and axes for the active solver::

    >>> solver_summary()

Set the calculation mode::

    >>> fourc.core.mode = "bissector"

List all available modes::

    >>> fourc.core.modes

Scanning with extra parameters fixed
--------------------------------------

:func:`~hklpy2.user.scan_extra` is a Bluesky plan generator that scans
diffractometer axes while holding pseudo axes, real axes, or extra
parameters fixed::

    >>> from bluesky import RunEngine
    >>> RE = RunEngine({})
    >>> # scan omega while holding h=4, k=0, l=0 fixed
    >>> RE(scan_extra([fourc], fourc.omega, -40, -30, num=11,
    ...               pseudos=dict(h=4, k=0, l=0)))

Quick reference
----------------

The table below maps common SPEC commands to their :mod:`hklpy2.user`
equivalents.  See :ref:`spec_commands_map` for the full list.

.. list-table::
   :header-rows: 1
   :widths: 20 35 45

   * - SPEC command
     - hklpy2.user
     - Description
   * - ``pa``
     - :func:`~hklpy2.user.pa`
     - Print all diffractometer settings
   * - ``wh``
     - :func:`~hklpy2.user.wh`
     - Print current position (where)
   * - ``ca h k l``
     - :func:`~hklpy2.user.cahkl`
     - Calculate angles for hkl, no move
   * - ``br h k l``
     - :meth:`~hklpy2.diffract.DiffractometerBase.move`
     - Move to hkl position
   * - ``or0`` / ``or1``
     - :func:`~hklpy2.user.setor`
     - Set an orienting reflection
   * - ``or_swap``
     - :func:`~hklpy2.user.or_swap`
     - Swap the two orienting reflections
   * - ``setlat``
     - :func:`~hklpy2.user.set_lattice`
     - Set crystal lattice parameters
   * - ``setmode``
     - ``fourc.core.mode = "..."``
     - Set the forward() calculation mode
   * - ``reflex``
     - :meth:`~hklpy2.ops.Core.refine_lattice`
     - Refine lattice from 3+ reflections
