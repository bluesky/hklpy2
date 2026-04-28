.. _how_simulator_from_config:

==========================================
How to Create a Simulator from a Config
==========================================

.. index::
    !simulator_from_config
    simulator; from configuration
    configuration; restore as simulator

:func:`~hklpy2.misc.simulator_from_config` creates a fully configured
simulated diffractometer — with no hardware connections — from a previously
saved configuration file or dictionary.  All real-axis positioners are soft
(simulated) positioners regardless of how they were defined in the original
diffractometer.

This is useful for:

* Reproducing an orientation offline for data analysis.
* Debugging a configuration without access to the hardware.
* Computing :math:`hkl` positions from a saved orientation.

Setup
-----

A configuration file is produced by exporting a live diffractometer::

    >>> import hklpy2
    >>> e4cv = hklpy2.creator(name="e4cv")
    >>> # ... define sample, reflections, compute UB ...
    >>> e4cv.export("e4cv-config.yml")

Create a simulator from that file
----------------------------------

Pass the path to the YAML file::

    >>> import hklpy2
    >>> sim = hklpy2.simulator_from_config("e4cv-config.yml")
    >>> sim.wh()

Or pass a configuration dictionary directly::

    >>> config = hklpy2.misc.load_yaml_file("e4cv-config.yml")
    >>> sim = hklpy2.simulator_from_config(config)

Or pass the ``configuration`` property of an existing diffractometer to
create a simulator with the same orientation — no file needed::

    >>> sim = hklpy2.simulator_from_config(k6c.configuration)

The simulator preserves
------------------------

* solver, geometry, and engine
* real-axis names and their solver-expected order
* sample(s), lattice parameters, and orientation reflections
* the UB matrix
* wavelength
* constraints
* presets

No hardware is connected — all real axes are soft positioners.

Compute positions offline
--------------------------

With the orientation restored, use the simulator exactly as you would a live
diffractometer::

    >>> sim.forward(1, 0, 0)
    >>> sim.wh()

Custom-named axes are preserved
---------------------------------

If the original diffractometer used custom axis names (e.g. ``theta`` instead
of ``omega``), those names are preserved in the simulator::

    >>> sim = hklpy2.simulator_from_config("fourc-config.yml")
    >>> sim.real_axis_names     # e.g. ['theta', 'chi', 'phi', 'ttheta']
    ['theta', 'chi', 'phi', 'ttheta']

Round-trip: simulator to simulator
------------------------------------

A simulator's configuration can itself be used to create another simulator::

    >>> sim1 = hklpy2.simulator_from_config("e4cv-config.yml")
    >>> sim2 = hklpy2.simulator_from_config(sim1.configuration)

.. seealso::

    :func:`~hklpy2.diffract.creator` — create a diffractometer from scratch.

    :doc:`/guides/configuration_save_restore` — how to export and restore
    diffractometer configurations.
