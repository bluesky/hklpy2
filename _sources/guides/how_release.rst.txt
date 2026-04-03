.. index:: release; process

.. _how_release:

=======================
How to Make a Release
=======================

Releases are managed with an interactive shell script that walks through
every step in the correct order and prevents common mistakes (such as
pushing a tag before updating :file:`switcher.json`).

Script location
===============

.. code-block:: bash

    scripts/release.sh

Usage
=====

.. code-block:: bash

    bash scripts/release.sh [--dry-run] [VERSION]

Options
-------

``VERSION``
    The version string to release (e.g. ``0.4.1`` or ``0.4.1rc1``).
    Prompted interactively if omitted.

``--dry-run``
    Walk through all steps and show what *would* happen without writing
    any files, creating commits, pushing, or tagging.  Network calls
    (``git fetch``) are also skipped.  Useful for previewing a release
    or verifying the script before the real run.

``-h``, ``--help``
    Print usage and exit immediately.

Typical workflow
================

Preview the release first:

.. code-block:: bash

    bash scripts/release.sh --dry-run 0.4.1

Then run for real:

.. code-block:: bash

    bash scripts/release.sh 0.4.1

The script is interactive: it confirms each destructive action before
proceeding and pauses at manual steps (milestone triage, RELEASE_NOTES
editing, waiting for CI).

Steps performed
===============

.. list-table::
   :header-rows: 1
   :widths: 5 30 65

   * - Step
     - Name
     - What happens
   * - 0
     - Preflight checks
     - Verifies branch, clean working tree, sync with origin, and that
       the tag does not already exist.
   * - 1
     - Milestone triage
     - Lists open milestones; prompts to close or defer any remaining
       issues before tagging.
   * - 2
     - RELEASE_NOTES.rst
     - Reminds you to move the release section out of the ``.. comment``
       block and add the release date.
   * - 3
     - Update switcher.json
     - Adds the new version to
       :file:`docs/source/_static/switcher.json`.
       Pre-releases (``rcN``, ``aN``, ``bN``) are added without
       changing ``"preferred"``; stable releases clear all
       ``"preferred"`` flags and mark the new version preferred.
   * - 4
     - pre-commit
     - Runs ``pre-commit run --all-files`` and prompts to fix any
       failures before continuing.
   * - 5
     - Test suite
     - Optionally runs ``pytest`` locally.
   * - 6
     - Commit
     - Stages and commits all release-prep changes (RELEASE_NOTES,
       switcher.json).
   * - 7
     - Push
     - Pushes the release-prep commit to ``origin``.
   * - 8
     - CI gate
     - Pauses so you can confirm all CI workflows are green before
       tagging.
   * - 9
     - Tag
     - Creates an annotated tag and pushes it, triggering the docs and
       PyPI publish workflows.
   * - 10
     - GitHub Release
     - Creates the GitHub Release via ``gh release create``.
   * - 11
     - Monitor CI
     - Reminds you to watch the ``docs.yml`` and ``pypi.yml`` runs.
       For stable tags, ``docs.yml`` automatically removes pre-release
       doc directories and switcher entries for the same base version.
   * - 12
     - conda-forge
     - Reminds you to review and merge the automated feedstock PR.

Pre-release and stable tags
===========================

Pre-release tags (``0.4.1rc1``, ``0.4.1a1``, etc.)
    - Docs published to ``gh-pages/0.4.1rc1/``.
    - Added to the version switcher **without** ``"preferred": true``
      — the current stable release stays preferred.

Stable tags (``0.4.1``)
    - Docs published to ``gh-pages/0.4.1/``.
    - All pre-release directories for the same base version
      (``0.4.1rc1/``, ``0.4.1a1/``, …) are **deleted** from
      ``gh-pages`` automatically by ``docs.yml``.
    - Corresponding entries removed from the live ``switcher.json``
      on ``gh-pages``.
    - ``"preferred": true`` moved to the new stable version.

Piping output
=============

The script detects when its output is not connected to a terminal and
suppresses ANSI colour codes automatically, so it is safe to pipe
through :command:`less` or :command:`tee`:

.. code-block:: bash

    bash scripts/release.sh --dry-run 0.4.1 2>&1 | less
    bash scripts/release.sh 0.4.1 2>&1 | tee release.log
