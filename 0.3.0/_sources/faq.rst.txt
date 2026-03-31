.. index:: ! FAQ

.. _FAQ:

===
FAQ
===

.. rubric:: Frequently Asked Questions

#. Is |hklpy2| ophyd v1 only, or can it be used with with ophyd async?

   hklpy2 is ophyd v1 (sync, not async) only

#. Is there a way to use it in some kind of simulation mode?

   Simulators are easy to create for any defined solver and geometry. This
   notebook demonstrates an :doc:`example </examples/hkl_soleil-e4cv>` of a
   4-circle diffractometer that might be used at a synchrotron beam line.

#. `import gi` raises an `ImportError` exception when starting in the bluesky queueserver.

   Tracked this `problem <https://github.com/bluesky/hklpy2/issues/69>`_
   to a situation (on linux 64-bit OS).  Either of these approaches were
   able to work around this problem until it is resolved upstream:

   1. Set an environment variable when starting the queue-server::

       LD_LIBRARY_PATH=${CONDA_PREFIX}/lib start-re-manager <options>

   2. Configure the conda environment to set this on activation::

       conda env config vars set LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
