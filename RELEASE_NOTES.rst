..
    This file describes user-visible changes between the versions.

    subsections could include these headings (in this order), omit if no content

    Notice
    Breaking Changes
    New Features
    Enhancements
    Fixes
    Maintenance
    Deprecations
    New Contributors

.. _release_notes:

========
Releases
========

Brief notes describing each release and what's new.

Project `milestones <https://github.com/bluesky/hklpy2/milestones>`_
describe future plans.

.. comment

    1.0.0
    #####

    Release expected 2025-Q4.

    0.2.0
    #####

    Release expected 2025-H2.

0.1.6
#####

Released TBA.

New Features
-----------

* '@crw_decorator()':  Decorator for the ConfigurationRunWrapper

Fixes
-----------

* Allow update of '.core.extras["h2"] = 1.0'
* Energy was not changed initially by 'wavelength.put(new_value)'.
* DiffractometerError raised by 'hklpy2.user.wh()''
* TypeError from 'diffractometer.wh()' and EPICS.
* TypeError when diffractometer was not connected.

Maintenance
-----------

* Add advice files for virtual AI chat agents.
* Add and demonstrate SPEC-style pseudoaxes.
* Add inverse transformation to DiffractometerBase.scan_extra() method.
* Add virtual axes.
* Complete 'refineLattice()' for Core and Sample.
* Compute B from sample lattice.
* Control displayed precision of position tuples using 'position_display_digits'.
* Consistent response when no forward solutions are found.
* Engineering units throughout
    * Solver defines the units it uses.
    * Conversion to/from solver's units.
    * Lattice, beam, rotational axes can all have their own units.
    * Ensure unit cell edge length units match wavelength units.
* Extend 'creator()' factory for custom real axis specifications.
* Improve code coverage.
* New GHA jobs cancel in in-progress jobs.
* Pick the 'forward()' solution closest to the current angles.
* Simple math for reflections: r1+r2, r1-r2, r1 == r2, ...
* Update table with SPEC comparison

0.1.5
#####

Released 2025-07-21.

Fixes
-----------

* Resolve TypeError raised from auxiliary pseudo position.

Maintenance
-----------

* Cancel in-progress GHA jobs when new one is started.
* Remove diffractometer solver_signature component.

0.1.4
#####

Released 2025-07-18.

New Features
------------

* Added FAQ document.
* Added 'pick_closest_solution()' as alternative 'forward()' decision function.
* Added 'VirtualPositionerBase' base class.

Maintenance
-----------

* Completed 'refineLattice()' method for both Core and Sample classes.
* Utility function 'check_value_in_list()' not needed at package level.

0.1.3
#####

Released 2025-04-16.

Notice
------

* Move project to bluesky organization on GitHub.
    * home: https://blueskyproject.io/hklpy2/
    * code: https://github.com/bluesky/hklpy2

Fixes
-----

* core.add_reflection() should define when wavelength=None

0.1.2
#####

Released 2025-04-14.

Fixes
-----

* Do not package unit test code.
* Packaging changes in ``pyproject.toml``.
* Unit test changes affecting hklpy2/__init__.py.

0.1.0
#####

Released 2025-04-14.

Initial project development complete.

Notice
------

- Ready for relocation to Bluesky organization on GitHub.
- See :ref:`concepts` for more details about how this works.
- See :ref:`v2_checklist` for progress on what has been planned.
- For those familiar with SPEC, see :ref:`spec_commands_map`.
