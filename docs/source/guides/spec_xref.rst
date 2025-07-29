.. index:: SPEC; commands

.. _spec_commands_map:

========================
SPEC commands in hklpy2
========================

Make it easier for users (especially |spec| users) to learn and remember
the tools in Bluesky's |hklpy2| package.

.. index:: !Quick Reference Table
.. rubric:: Quick Reference Table

===============  =============================================================  ============
|spec|           |hklpy2|                                                       description
===============  =============================================================  ============
--               :func:`~hklpy2.user.set_diffractometer`                        Select the default diffractometer.
``pa``           :func:`~hklpy2.user.pa`                                        Report (full) diffractometer settings.  (pa: print all)
``wh``           :func:`~hklpy2.user.wh`                                        Report (brief) diffractometer settings. (wh: where)
``br h k l``     ``diffractometer.move(h, k, l)``                               (command line) Move motors of ``diffractometer`` to the given :math:`h, k, l`.
``br h k l``     ``yield from bps.mv(diffractometer, (h, k, l))``               (bluesky plan) Move motors of ``diffractometer`` to the given :math:`h, k, l`.
``ca h k l``     :func:`~hklpy2.user.cahkl`                                     Prints calculated motor settings for the given :math:`h, k, l`.
``or_swap``      :func:`~hklpy2.user.or_swap()`                                 Exchange primary & secondary orientation reflections.
``or0``          :func:`~hklpy2.user.setor`                                     Define a crystal reflection and its motor positions.
``or1``          :func:`~hklpy2.user.setor`                                     Define a crystal reflection and its motor positions.
``reflex``       :func:`~hklpy2.blocks.sample.Sample.refine_lattice()`          Refinement of lattice parameters from list of 3 or more reflections
``reflex_beg``   not necessary                                                  Initializes the reflections file
``reflex_end``   not necessary                                                  Closes the reflections file
``setlat``       :meth:`~hklpy2.blocks.sample.Sample.lattice`                   Update current sample lattice.
``setmode``      ``diffractometer.core.mode = "psi_constant``                   Set the diffractometer mode for the `forward()` computation.
--               ``diffractometer.core.constraints``                            Show the current set of constraints (cut points).
``cuts``         :meth:`~hklpy2.blocks.sample.Sample.lattice`                   Add constraints to the diffractometer `forward()` computation.
``freeze``       :meth:`~hklpy2.blocks.sample.Sample.lattice`                   Hold an axis constant during the diffractometer `forward()` computation.
``unfreeze``     :meth:`~hklpy2.blocks.sample.Sample.lattice`                   Undo the most-recent constraints applied.
--               :func:`~hklpy2.user.calc_UB`                                   Compute the UB matrix with two reflections.
--               ``diffractometer.sample = "vibranium"``                        Pick a known sample to be the current selection.
--               ``diffractometer.samples``                                     List all defined crystal samples.
--               :func:`~hklpy2.user.add_sample`                                Define a new crystal sample.
``setaz h k l``  :attr:`~hklpy2.ops.Core.extras`                                Set the azimuthal reference vector to the given :math:`h, k, l`.
``setsector``    Not yet implemented.                                           Select a sector.
``cz``           Not yet implemented.                                           Calculate zone from two reflections
``mz``           Not yet implemented.                                           Move zone
``pl``           Not yet implemented.                                           Set the scattering plane
``sz``           Not yet implemented.                                           Set zone
===============  =============================================================  ============
