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
``setmode``      ``diffractometer.core.mode = "psi_constant``                   Set the diffractometer mode for the ``forward()`` computation.
--               ``diffractometer.core.modes``                                  List all available diffractometer modes.
--               :func:`~hklpy2.user.add_sample`                                Define a new crystal sample.
``setlat``       :meth:`~hklpy2.blocks.sample.Sample.lattice`                   Update current sample lattice.
--               ``diffractometer.sample = "vibranium"``                        Pick a known sample to be the current selection.
--               ``diffractometer.samples``                                     List all defined crystal samples.
``or0``          :func:`~hklpy2.user.setor`                                     Define a crystal reflection and its motor positions.
``or1``          :func:`~hklpy2.user.setor`                                     Define a crystal reflection and its motor positions.
``or_swap``      :func:`~hklpy2.user.or_swap()`                                 Exchange primary & secondary orientation reflections.
``br h k l``     ``diffractometer.move(h, k, l)``                               (command line) Move motors of ``diffractometer`` to the given :math:`h, k, l`.
``br h k l``     ``yield from bps.mv(diffractometer, (h, k, l))``               (bluesky plan) Move motors of ``diffractometer`` to the given :math:`h, k, l`.
``ca h k l``     :func:`~hklpy2.user.cahkl`                                     Prints calculated motor settings for the given :math:`h, k, l`.
``reflex``       :func:`~hklpy2.blocks.sample.Sample.refine_lattice()`          Refinement of lattice parameters from list of 3 or more reflections
``reflex_beg``   not necessary                                                  Initializes the reflections file
``reflex_end``   not necessary                                                  Closes the reflections file
--               ``diffractometer.core.constraints``                            Show the current set of constraints (cut points).
``cuts``         See :meth:`~hklpy2.blocks.constraints.LimitsConstraint`        Add constraints to the diffractometer ``forward()`` computation.
``freeze``       Move axis to value, Choose mode that does not update *axis*.   Hold an axis constant during the diffractometer ``forward()`` computation.
``unfreeze``     Choose mode that updates *axis*.                               Allow axis to be updated by ``forward()`` computation.
--               :func:`~hklpy2.user.calc_UB`                                   Compute the UB matrix with two reflections.
``setaz h k l``  :attr:`~hklpy2.ops.Core.extras`                                Set the azimuthal reference vector to the given :math:`h, k, l`.
``setsector``    ``diffractometer.core.constraints``                                          Select a sector.
``cz``           See :meth:`~hklpy2.blocks.zone.scan_zone`                      Calculate zone from two reflections: TODO
``mz``           See :meth:`~hklpy2.blocks.zone.scan_zone`                      Move zone: TODO
``pl``           See :meth:`~hklpy2.blocks.zone.scan_zone`                      Set the scattering plane: TODO
``sz``           See :meth:`~hklpy2.blocks.zone.Zone`                           Set zone: TODO
===============  =============================================================  ============
