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

    Release expected 2025-Q3.

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
