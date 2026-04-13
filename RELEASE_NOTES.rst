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

    Release expected 2026-Q3.

    0.5.0
    #####

    Release expected by 2026-H1.

    Documentation
    -------------

    * Document ``ConstraintBase.valid()`` internals, the ``forward()`` call
      sequence, and the ``LimitsConstraint`` label requirement; clarify that
      hklpy2 constraints are post-computation filters distinct from SPEC/diffcalc
      cut points. (:issue:`275`)

0.4.3
#####

Released 2026-04-07.

Enhancements
------------

* Add custom autoapi templates for concise page titles (short name as
  heading, full import path in monospace ``Import:`` line below), and add
  ``Import:`` line for classes and functions. (:issue:`251`)
* Add ``Full API Reference`` card to the User Guide index. (:issue:`251`)
* Hide ``sig-prename`` via CSS; style ``import-path`` container. (:issue:`251`)

Maintenance
-----------

* Add ``concepts/presets.rst`` concepts page; update ``concepts/constraints.rst``
  with the presets/constraints distinction and mutual cross-references.
  (:issue:`259`)
* Move design and checklist planning docs from ``concepts/planning/`` to
  ``guides/``; enrich ``guides.rst`` and ``examples.rst`` landing pages
  with categorized content tables. (:issue:`259`)
* Expand FAQ with units table, no-solutions diagnosis, constraints vs.
  presets comparison, UB-matrix troubleshooting, SPEC command equivalents,
  and azimuthal scan note. (:issue:`259`)
* Revise concepts documents to be brief and purely conceptual; move
  ``migration.rst`` from ``concepts/`` to ``guides/`` and remove
  guide-like example content from ``ops.rst``. (:issue:`259`)

0.4.2
#####

Released 2026-04-06.

New Features
------------

* Add :func:`~hklpy2.blocks.zone.move_zone` plan (SPEC ``mz`` equivalent):
  move diffractometer to a pseudo position in the zone. (:issue:`273`)
* Export :class:`~hklpy2.blocks.zone.OrthonormalZone` and
  :func:`~hklpy2.blocks.zone.move_zone` from the top-level ``hklpy2``
  package. (:issue:`273`)

Fixes
-----

* Fix :meth:`~hklpy2.backends.th_tth_q.ThTthSolver.removeAllReflections`
  to clear reflections list and reset wavelength instead of raising
  ``NotImplementedError``. (:issue:`267`)
* Remove unused ``INPUT_VECTOR`` and ``NUMERIC`` re-exports from
  ``misc.py``; nothing imported them from there. (:issue:`267`)
* Fix metadata dict bug in :func:`~hklpy2.blocks.zone.scan_zone`: the
  ``dict(...).update(...)`` pattern always returned ``None``; replaced
  with unpacking syntax. (:issue:`273`)

Maintenance
-----------

* Move ``from .typing import ...`` statements in ``misc.py`` to the
  top-level import section; drop all ``# noqa: E402, F401`` suppressions
  from those lines; remove the backward-compatibility re-export comment
  block and docstring mention. (:issue:`283`)
* Update ``spec_xref.rst`` to replace ``TODO`` placeholders for ``cz``,
  ``mz``, ``pl``, and ``sz`` SPEC zone commands with correct hklpy2
  references. (:issue:`273`)
* Expand ``test_summary_dict`` and ``test_summary`` in
  ``test_hkl_soleil.py`` to assert keys, values, columns, and rows.
  (:issue:`269`)
* Add reflection comparison assertions (pseudo, real, wavelength) in
  ``test_configure.py`` restore test. (:issue:`270`)
* Address CodeQL findings:

  - Unused local variables: ``lattice.py``, ``test_lattice.py``;
    restructure try/except in ``lattice.py`` to initialize
    ``beta_val``/``gamma_val`` before the block.
  - Unused imports: add ``# noqa: F401`` to ``TYPE_CHECKING``-guarded
    imports in ``misc.py``, ``ops.py``; remove unused
    ``TYPE_CHECKING``/``Core`` block from ``reflection.py`` (replaced
    annotation with ``Optional[Any]``); restore accidentally removed
    ``INPUT_VECTOR``, ``NamedFloatDict``, ``NUMERIC`` re-exports in
    ``misc.py``; remove unused import from ``hkl_soleil-python_api.py``;
    fix mixed ``import`` and ``from ... import`` of ``hklpy2`` in
    ``test_isn_libhkl.py``.
  - Mixed explicit/implicit returns: update return-type annotations and
    fix fall-through returns in ``base.py``, ``hkl_soleil.py``,
    ``no_op.py``, ``ops.py``, ``th_tth_q.py``, and ``user.py``.
  - Variable defined multiple times: fix loop variables in ``ops.py``,
    ``user.py``, ``test_backends.py``, ``test_diffract.py``, and
    ``test_sample.py``.
  - Modification of parameter with mutable default: fix ``beam_kwargs``
    and ``extras`` in ``diffract.py`` and ``user.py``; suppress B006
    with ``# noqa`` for read-only mutable defaults where changing the
    default would alter runtime semantics.
  - Use of return value of a procedure: drop ``tbl =`` assignment for
    ``pa()`` and ``wh()`` calls in ``test_user.py``.
  - Unused global variable: remove ``_I243_REAL_AXES`` from
    ``test_i210.py``; add ``logger.debug()`` calls throughout
    ``misc.py`` to activate the previously unused ``logger``.
  - File not always closed: use a ``with`` block in
    ``load_yaml_file()`` in ``misc.py``.
  - ``__eq__`` not overridden when adding attributes: add ``__eq__``
    to ``ReflectionsDict`` in ``reflection.py``.
  - Redundant comparison: remove from ``test_lattice.py``
    ``test_equal()``.
  - Assert with side-effect: separate ``sys.path.pop()`` from
    ``assert`` in ``test_init.py``.
  - Non-standard exception in special method: return ``NotImplemented``
    instead of raising ``TypeError`` in ``Reflection.__sub__()``.
  - Unused exception object: add missing ``raise`` in
    ``hkl_soleil-python_api.py``.
* Additional cleanup found during CodeQL review:

  - Remove redundant outer loop in ``test_misc.py`` ``test_axes_to_dict()``.
  - Rename overwritten loop variable ``solution`` in ``test_diffract.py``.
  - Remove unnecessary intermediate assignments before ``return`` in
    ``hkl_soleil.py``, ``incident.py``, ``misc.py``, ``ops.py``, and
    ``user.py`` (RET504).
* Review TODO & FIXME markers: remove resolved comments, open new issues
  for remaining concerns. (:issue:`260`)

0.4.1
#####

Released 2026-04-03.

Fixes
-----

* Fix docs workflow race condition: ``switcher.json`` update now pulls
  with rebase before pushing to ``gh-pages``, preventing ``fetch first``
  push failures when a concurrent deploy has already advanced the branch.
  (:issue:`263`)

Maintenance
-----------

* Add ``scripts/release.sh`` interactive release checklist: automates
  ``switcher.json`` update, pre-commit, commit, tag, and GitHub Release
  steps in the correct order; supports ``--dry-run`` to preview all
  actions without making changes. (:issue:`263`)

0.4.0
#####

Released 2026-04-03.

New Features
------------

* Add :func:`~hklpy2.misc.creator_from_config` to create a simulated
  diffractometer (no hardware) from a saved configuration file or dict.
  (:issue:`210`)

Enhancements
------------

* Add ``TypedDict`` subclasses for structured solver and configuration
  dicts: :class:`~hklpy2.backends.typing.ReflectionDict`,
  :class:`~hklpy2.backends.typing.SampleDict`, and
  :class:`~hklpy2.backends.typing.SolverMetadataDict` in
  ``backends/typing.py``; :class:`~hklpy2.typing.ConfigHeaderDict` in
  ``hklpy2/typing.py``; and ``HklSolverMetadataDict`` in
  ``hkl_soleil.py`` as the pattern for solver-specific metadata
  extensions. (:issue:`233`)
* Consolidate type aliases (``KeyValueMap``, ``NamedFloatDict``,
  ``Matrix3x3``, ``NUMERIC``, ``AxesDict``, ``AxesArray``, ``AxesList``,
  ``AxesTuple``, ``AnyAxesType``, ``BlueskyPlanType``, ``INPUT_VECTOR``)
  from :mod:`hklpy2.misc` into :mod:`hklpy2.typing`; backward-compatible
  re-exports remain in :mod:`hklpy2.misc`. (:issue:`252`)
* Export :func:`~hklpy2.misc.get_run_orientation` and
  :func:`~hklpy2.misc.list_orientation_runs` at the top-level ``hklpy2``
  namespace, consistent with :class:`~hklpy2.misc.ConfigurationRunWrapper`.
  (:issue:`231`)
* Add how-to guide for choosing the default ``forward()`` solution picker
  (``pick_first_solution``, ``pick_closest_solution``, or custom).
  (:issue:`224`)
* Publish versioned docs: ``main`` branch to ``latest/``, each tag to
  ``<version>/``; add version-switcher dropdown to navbar; auto-update
  ``switcher.json`` on new tags. (:issue:`213`)
* Show a banner on dev/pre-release doc pages noting that a stable
  version is available, with a link to it. (:issue:`213`)

Fixes
-----

* Fix ``diffractometer.configuration = config`` setter to delegate to
  :meth:`~hklpy2.diffract.DiffractometerBase.restore`, applying geometry
  validation, beam/wavelength restoration, and state clearing consistently
  with calling ``restore()`` directly. (:issue:`231`)

* Fix stale lattice parameters not reaching the solver before ``calc_UB()``
  computes the orientation matrix; lattice changes now propagate immediately
  via ``Lattice.__setattr__`` → ``Sample`` callback → solver update, so
  subsequent ``forward()`` / ``inverse()`` calls use the correct lattice.
  (:issue:`240`, :pr:`244`)
* Fix ``LimitsConstraint.valid()`` rejecting solver solutions that land just
  outside a limit boundary due to floating-point arithmetic; increase
  ``ENDPOINT_TOLERANCE`` from ``1e-7`` to ``1e-4``. (:issue:`242`)
* Fix :func:`~hklpy2.misc.creator_from_config` restoring reflections with
  wrong axis values when YAML serialises ``reals`` dict keys alphabetically
  instead of in physical axis order. (:issue:`243`)
* Fix error message bugs: missing f-string prefix in ``hkl_soleil.py``,
  typo ``"must by"`` → ``"must be"`` in ``sample.py``, trailing comma in
  ``user.py`` ``set_wavelength()`` message; standardize
  ``NoForwardSolutions`` message; deduplicate ``_header`` key message
  into a constant; fix capitalization inconsistency. (:issue:`199`)
* Sort glossary alphabetically; fix ``:real:`` → ``:virtual:`` typo in
  glossary entry. (:issue:`235`)

Maintenance
-----------

* Add ``re.escape()`` to all ``pytest.raises(match=...)`` calls that were
  using raw strings. (:issue:`232`)
* Clarify the complementary roles of ``standardize_pseudos`` /
  ``standardize_reals`` (solver/Core layer, returns ``AxesDict``) vs. the
  ophyd ``@pseudo_position_argument`` / ``@real_position_argument``
  decorators (diffractometer layer, returns namedtuple); remove three
  redundant pre-normalisation calls in ``diffract.py`` where the ophyd
  decorator already handles flexible input. (:issue:`247`)
* Fix type annotations: ``*reals`` in :func:`~hklpy2.user.setor` from
  ``AnyAxesType`` to ``NUMERIC``; ``**kwargs`` in
  :func:`~hklpy2.misc.dict_device_factory` from ``KeyValueMap`` to ``Any``;
  ``Matrix3x3`` from deprecated ``typing.List`` to built-in ``list``.
  (:issue:`230`)
* Remove deprecated ``assert_context_result()`` helper and all 117 call
  sites; fold error message strings into ``match=re.escape(...)`` on
  ``pytest.raises()``; convert bare list param sets to ``pytest.param()``
  with ``id=``. (:issue:`232`)
* Backfill workflow no longer patches ``conf.py`` from ``main``; legacy
  tag builds use their own ``conf.py`` so the displayed version is correct
  and no switcher dropdown appears in builds that predate it. (:issue:`213`)
* Refactor ``scan_extra()`` into smaller methods: extract input validation,
  mover construction, metadata assembly, and inner plan helpers; fix latent
  bug where ``dict.update()`` returned ``None`` for run metadata.
  (:issue:`229`)
* Remove resolved TODO comment in ``__init__.py``; ``scan_extra()`` is
  implemented in :class:`~hklpy2.diffract.DiffractometerBase` and available
  via ``from hklpy2.user import *``. (:issue:`254`)

0.3.1
##########

Released 2026-03-31.

New Features
------------

* Add ``presets`` dict to supply constant-axis values for ``forward()``
  without moving motors; presets are stored per mode. (:issue:`190`)

Enhancements
------------

* Add ``preset`` Glossary entry; alphabetize Glossary entries. (:issue:`202`)
* ``presets`` setter now **replaces** the preset dictionary for the current
  mode (standard Python assignment semantics) rather than merging into it.
  Use ``presets = {}`` to clear; ``clear_presets()`` is removed.
  (:issue:`219`)
* Document ``presets`` setter replace behavior, per-mode storage, and
  effect on ``forward()`` solutions. Add how-to guide and Core concepts
  summary table. (:issue:`200`, :issue:`219`)
* Improve Sphinx Index consistency: primary ``!term`` entries now point to
  the most substantive page for each term rather than the Glossary.
  (:issue:`202`)
* Trim ``concepts/diffract.rst`` to a brief concept overview; move
  guide and example content to ``guides/diffract.rst``. (:issue:`204`)
* Trim ``concepts/solvers.rst`` to a brief concept overview; move
  how-to content to new ``guides/solvers.rst``; add ``entry point``
  Glossary entry. (:issue:`205`)
* Use US spelling throughout docs and source. (:issue:`202`)

Fixes
-----

* ``%wa`` bluesky magic raises ``TypeError`` for diffractometers under
  numpy 2.x; this is a bluesky upstream bug triggered by numpy 2.0
  tightening ``ndarray.__format__``. (:issue:`201`)
* ``cahkl()`` now returns solutions at motor positions that previously
  yielded no results. (:issue:`193`)
* ``calc_UB()`` now raises a clear ``ValueError`` with diagnostic hints
  when libhkl returns a degenerate U matrix, rather than a cryptic
  downstream error. (:issue:`207`)
* Constraints example: rename "Freeze an axis" to "Limited range";
  add "Preset (frozen) axes" section; update SPEC ``freeze``/``unfreeze``
  cross-references to point to presets. (:issue:`212`)
* ``forward()`` solutions no longer use wrong angle values from orientation
  reflections when computing constant-axis modes. (:issue:`195`)

Maintenance
-----------

* Add ``sphinx.ext.extlinks`` to Sphinx config; define ``:issue:`` and
  ``:pr:`` roles pointing to GitHub. (:issue:`203`)
* Bump ``actions/upload-artifact`` from v6 to v7. (:pr:`194`)
* Skip CI unit tests for pull requests that change only documentation.

0.3.0
#####

Released 2026-01-12.

New Features
-------------

* Support (define, scan) crystallographic zones.

Fixes
-----

* creator() not connecting with EPICS

Maintenance
-----------

* UB computed from 2 reflections should match expected value.

0.2.3
#####

Released 2025-12-02.

Maintenance
-----------

* Add Python type annotations to source code.
* Add reference to DeepWiki site.
* Relocate source code from 'hklpy2' to 'src/hklpy2'.
* Transition from databroker to tiled.

0.2.2
#####

Released 2025-11-25.

Maintenance
-----------

* 'creator()' and 'diffractometer_class_factory()': replace 'aliases' with
  '_pseudo' and '_real'.

0.2.1
#####

Released 2025-11-22.

Fixes
-----------

* Resolve inconsistent forward() results.
* WavelengthXray: Synchronize & initialize energy when wavelength changes.

Maintenance
-----------

* Add APS ISN diffractometer notebook.
* Lattice: 0.0 and None are different.
* Remove unit tests for tardis diffractometer.

0.2.0
#####

Released 2025-10-10.

New Features
------------

* Compute lattice B matrix.
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
* Control displayed precision of position tuples using 'DIFFRACTOMETER.digits' property.
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
* 'scan_extra' plan now supports one or more extras (similar to bp.scan).
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
