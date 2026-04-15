.. _user_guide.quickstart:

==========
Quickstart
==========

This page verifies that |hklpy2| is installed and working.  If you are new
to |hklpy2|, continue to the :ref:`tutorial` after this page for a full
guided walkthrough: sample setup, orientation reflections, UB matrix
computation, and reciprocal-space scanning.

.. seealso::

   :ref:`tutorial` — start here for a complete first experience.

   :ref:`guide.diffract` — how to define and work with a diffractometer.

   :ref:`geometries` — available geometries and solvers.

   :ref:`guides` — task-oriented how-to guides for common workflows.

Installation
------------

Install |hklpy2| and its dependencies:

.. code-block:: bash

   pip install hklpy2

Then confirm the installation:

.. code-block:: python

   import hklpy2
   print(hklpy2.__version__)

Create a Simulated Diffractometer
----------------------------------

Use :func:`~hklpy2.diffract.creator` to build a simulated 6-circle
diffractometer (no hardware required):

.. code-block:: python

   >>> import hklpy2
   >>> sixc = hklpy2.creator(name="sixc", geometry="E6C", solver="hkl_soleil")

Make it the *default* diffractometer for the interactive interface:

.. code-block:: python

   >>> from hklpy2.user import *
   >>> set_diffractometer(sixc)

Show the brief position report:

.. code-block:: python

   >>> wh()  # wh: "WHere"
   h=0, k=0, l=0
   wavelength=1.0
   mu=0, omega=0, chi=0, phi=0, gamma=0, delta=0

Show the full orientation report:

.. code-block:: python

   >>> pa()  # pa: "Print All"
   diffractometer='sixc'
   HklSolver(name='hkl_soleil', version='5.1.2', geometry='E6C', engine_name='hkl', mode='bissector_vertical')
   Sample(name='sample', lattice=Lattice(a=1, system='cubic'))
   Orienting reflections: []
   U=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
   UB=[[6.28318530718, -0.0, -0.0], [0.0, 6.28318530718, -0.0], [0.0, 0.0, 6.28318530718]]
   constraint: -180.0 <= mu <= 180.0
   constraint: -180.0 <= omega <= 180.0
   constraint: -180.0 <= chi <= 180.0
   constraint: -180.0 <= phi <= 180.0
   constraint: -180.0 <= gamma <= 180.0
   constraint: -180.0 <= delta <= 180.0
   h=0, k=0, l=0
   wavelength=1.0
   mu=0, omega=0, chi=0, phi=0, gamma=0, delta=0

Calculate angles for :math:`hkl=(1\ \bar{1}\ 0)` without moving motors:

.. code-block:: python

   >>> cahkl(1, -1, 0)
   Hklpy2DiffractometerRealPos(mu=0, omega=-45.000000066239, chi=44.999999876575, phi=-89.999999917768, gamma=0, delta=-90.000000132477)
   >>> wh()
   h=0, k=0, l=0
   wavelength=1.0
   mu=0, omega=0, chi=0, phi=0, gamma=0, delta=0

Note this was only a calculation — the motors did not move.  Move now:

.. code-block:: python

   >>> sixc.move(1, -1, 0)
   MoveStatus(done=True, pos=sixc, elapsed=0.0, success=True, settle_time=0.0)
   >>> wh()
   h=1.0, k=-1.0, l=0
   wavelength=1.0
   mu=0, omega=-45.0, chi=45.0, phi=-90.0, gamma=0, delta=-90.0

If this works, |hklpy2| is installed correctly.  Continue to the
:ref:`tutorial` to learn how to define a crystal sample, add orientation
reflections, and compute the UB matrix.
