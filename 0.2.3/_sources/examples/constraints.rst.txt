.. _examples.constraints:

======================
Constraints
======================

:ref:`Constraints <concepts.constraints>` are used to filter
acceptable solutions computed by a |solver| ``forward()`` method.
One or more constraints
(:class:`~hklpy2.blocks.constraints.ConstraintBase`) (a.k.a, cut points),
together with a choice of operating **mode** are used to control
the over-determined transformation from :math:`hkl` to motor angles.

.. index:: cut points
.. tip:: *Constraints* are implemented as *cut points* in other software.
    Similar in concept yet not entirely identical in implementation.

Show the current constraints
----------------------------

Start with a diffractometer.  This example starts with
:ref:`E6C <geometries-hkl_soleil-e6c>`, as shown in the
:ref:`user_guide.quickstart`.

.. code-block:: python
   :linenos:

   >>> import hklpy2
   >>> sixc = hklpy2.creator(name="sixc", geometry="E6C", solver="hkl_soleil")

Show the constraints:

.. code-block:: python
   :linenos:

   >>> sixc.core.constraints
   ['-180.0 <= mu <= 180.0', '-180.0 <= omega <= 180.0', '-180.0 <= chi <= 180.0', '-180.0 <= phi <= 180.0', '-180.0 <= gamma <= 180.0', '-180.0 <= delta <= 180.0']

Change a constraint
-------------------

Only accept ``forward()`` solutions where ``omega`` :math:`>= 0`.

.. code-block:: python
   :linenos:

   >>> sixc.core.constraints["omega"].low_limit
   -180.0
   >>> sixc.core.constraints["omega"].low_limit = 0
   >>> sixc.core.constraints["omega"]
   0 <= omega <= 180.0

Apply axis cuts
~~~~~~~~~~~~~~~~~~

Only accept ``forward()`` solutions where ``chi`` is between :math:`\\pm90`:

.. code-block:: python
   :linenos:

   >>> sixc.core.constraints["chi"].limits
   (-180.0, 180.0)
   >>> sixc.core.constraints["chi"].limits = -90, 90
   >>> sixc.core.constraints["chi"].limits
   (-90.0, 90.0)

Freeze an axis
~~~~~~~~~~~~~~~~~~

Only accept ``forward()`` solutions where ``mu`` is zero:

.. code-block:: python
   :linenos:

   >>> sixc.core.constraints["mu"].limits
   (-180.0, 180.0)
   >>> sixc.core.constraints["mu"].limits = 0, 0
   >>> sixc.core.constraints["mu"].limits
   (0.0, 0.0)
