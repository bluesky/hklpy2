.. _user_guide.quickstart:

==========
Quickstart
==========

Get started with a diffractometer object.  For example, use the diffractometer
:func:`~hklpy2.diffract.creator()` function to build a 6-circle diffractometer
(:ref:`E6C <geometries-hkl_soleil-e6c>` geometry). Use simulated motor
axes.

.. code-block:: python
   :linenos:

   >>> import hklpy2
   >>> sixc = hklpy2.creator(name="sixc", geometry="E6C", solver="hkl_soleil")

Make it the *default* diffractometer and show its current settings:

.. code-block:: python
   :linenos:

   >>> from hklpy2.user import *
   >>> set_diffractometer(sixc)

Show its current settings (the brief report):

.. code-block:: python
   :linenos:

   >>> wh()  # wh: "WHere"
   h=0, k=0, l=0
   wavelength=1.0
   mu=0, omega=0, chi=0, phi=0, gamma=0, delta=0

Show the full report:

.. code-block:: python
   :linenos:

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

Calculate the angles for :math:`hkl=(1\ \bar{1}\ 0)`, using
:func:`~hklpy2.user.cahkl()`:

.. code-block:: python
   :linenos:

   >>> cahkl(1, -1, 0)
   Hklpy2DiffractometerRealPos(mu=0, omega=-45.000000066239, chi=44.999999876575, phi=-89.999999917768, gamma=0, delta=-90.000000132477)
   >>> wh()
   h=0, k=0, l=0
   wavelength=1.0
   mu=0, omega=0, chi=0, phi=0, gamma=0, delta=0

Note this was only a calculation.  The motors did not move.  Do that now.

.. code-block:: python
   :linenos:

   >>> sixc.move(1, -1, 0)
   MoveStatus(done=True, pos=sixc, elapsed=0.0, success=True, settle_time=0.0)
   >>> wh()
   h=1.0, k=-1.0, l=0
   wavelength=1.0
   mu=0, omega=-45.0, chi=45.0, phi=-90.0, gamma=0, delta=-90.0
