.. index:: !install

.. _install:

================
Installation
================

There are several possible ways to install this package:

Conda - conda-forge
====================

Install from conda-forge:

``conda install -c conda-forge hklpy2``

.. seealso:: https://anaconda.org/conda-forge/hklpy2

Pip - PyPI
==========

Install from the Python Package Index (PyPI) repository:

``pip install hklpy2``

.. seealso:: https://pypi.org/project/hklpy2/

Pip - Source
===============

Install directly from the GitHub source code repository.  Either:

``pip install https://github.com/bluesky/hklpy2/archive/main.zip``

or editable install (this is for developers) from local clone of source code.

First, clone the repository (into directory ``hklpy2``):

``git clone https://github.com/bluesky/hklpy2``

Then, install from the new ``hklpy2`` directory:

``pip install -e hklpy2 --no-deps``

.. seealso:: https://github.com/bluesky/hklpy2

Development
===========

Follow these steps (in bash shell) to setup a conda environment for development
and testing:

.. code-block:: bash
    :linenos:

    export HKLPY2_ENV=hklpy2
    conda create -y -n "${HKLPY2_ENV}" pyepics hkl tiled bson python pandoc
    conda activate "${HKLPY2_ENV}"
    pip install --pre tiled
    pip install --pre -e .[all]
