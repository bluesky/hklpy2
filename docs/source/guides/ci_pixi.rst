.. _guide.ci_pixi:

=====================
Maintain CI with Pixi
=====================

The unit-test workflow uses Pixi to keep its supported dependency environments
reproducible. The committed ``pixi.lock`` file is the CI baseline.

Locked baseline
===============

Normal CI installs the named Pixi environment with ``--locked``. A manifest
change without a matching lockfile update fails installation rather than
silently resolving a new environment.

The supported environments are ``py311``, ``py312``, ``py313``, and ``py314``.
The ``py315-dev`` environment is an allowed-failure upstream compatibility
canary. It may fail when conda-forge packages are not yet available for Python
3.15.

The Tiled package source is environment-specific:

* Python 3.11 through 3.13 and the default environment use conda-forge Tiled
  0.2.20.
* Python 3.14 uses PyPI Tiled 0.2.20 because the corresponding conda-forge
  dependency graph is not currently solvable there.
* Python 3.15 remains an upstream canary while its dependency support develops.

Updating the baseline
=====================

A scheduled workflow runs ``pixi update`` and opens a pull request containing
``pixi.lock`` and a dependency diff. Review the generated package changes and
allow the complete locked test matrix to run before merging the update.

The update workflow is also available through manual dispatch after it is
registered on the default branch. Do not update ``pixi.lock`` directly on
``main``.

Latest-dependency canary
========================

A separate scheduled/manual workflow updates each named environment and runs
the test matrix without using the committed baseline. This detects upstream
compatibility problems before they enter the supported lockfile. Its Python
3.15 job is allowed to fail.

Updating Python support
=======================

Python support changes are coordinated maintenance changes, not automatic
lockfile updates. Check the `official Python status and release schedule
<https://devguide.python.org/versions/>`_ when planning a transition.

When a supported Python version reaches end of support, update these together:

* Change ``requires-python`` in ``pyproject.toml``.
* Remove the retired Pixi feature and environment.
* Promote the development environment for the newly supported Python version.
* Add the next ``-dev`` Pixi feature and environment.
* Update the matrices in ``.github/workflows/code.yml`` and
  ``.github/workflows/pixi-canary.yml``.
* Regenerate ``pixi.lock`` and run the complete matrix in the pull request.
* Update this guide and the release notes.

For example, when Python 3.11 is retired and Python 3.15 becomes supported,
remove ``py311``, promote ``py315-dev`` to ``py315``, and add ``py316-dev`` as
the new allowed-failure canary. Do not let a routine dependency update silently
change the supported Python policy.

Local development
=================

Use the pinned Pixi version from CI when reproducing an environment:

.. code-block:: bash

   pixi install --locked -e py313
   pixi run --locked -e py313 pytest -q .

On NFS-backed home directories, put the Pixi package cache on local storage
before solving or installing:

.. code-block:: bash

   export RATTLER_CACHE_DIR=/tmp/pixi-cache

Use a local filesystem workspace copy for long lockfile solves when necessary.
