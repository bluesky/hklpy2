.. _examples.constraints:

======================
Constraints
======================

:ref:`Constraints <concepts.constraints>` are used to filter
acceptable solutions computed by a |solver| ``forward()`` method.
One or more constraints
(:class:`~hklpy2.blocks.constraints.ConstraintBase`),
together with a choice of operating **mode**, are used to control
the over-determined transformation from :math:`hkl` to motor angles.

.. index:: cut points
.. note:: **Relation to** *cut points* **in other software** (SPEC ``cuts``):
    In SPEC, a *cut point* is a branch point that sets where an angle wraps
    around (e.g., ``-180`` → ``[-180, +180)``), controlling motor travel
    direction rather than filtering solutions.  hklpy2 ``LimitsConstraint``
    is a *post-computation filter* that discards solutions outside a range.
    The two overlap in effect when the constraint window matches the
    cut-point interval, but the mechanisms differ.

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

Limited range
~~~~~~~~~~~~~

.. index:: freeze

Only accept ``forward()`` solutions where ``mu`` is near zero.  Setting
both limits to the same value (or very close values) effectively restricts
that axis to a single position:

.. code-block:: python
   :linenos:

   >>> sixc.core.constraints["mu"].limits
   (-180.0, 180.0)
   >>> sixc.core.constraints["mu"].limits = -0.01, 0.01   # narrow range
   >>> sixc.core.constraints["mu"].limits
   (-0.01, 0.01)

.. tip:: Constraints *filter* solutions after the solver computes them.
    If you want the solver to *use* a specific value for a constant axis
    during ``forward()`` computation (rather than the current motor
    position), use :ref:`presets <how_presets>` instead.

Preset (frozen) axes
--------------------

.. index:: freeze; presets, presets; freeze

In |spec|, the ``freeze`` command holds an axis at a fixed value during
``forward()`` computation.  In |hklpy2|, this is done with
:ref:`presets <how_presets>`.

A *preset* tells the solver which value to use for a constant axis instead
of the current motor position.  Presets do not move any motor.

Choose a mode where the axis is constant, then set a preset:

.. code-block:: python
   :linenos:

   >>> sixc.core.mode = "bissector_vertical"   # mu and gamma are constant
   >>> sixc.core.constant_axis_names
   ['mu', 'gamma']
   >>> sixc.core.presets = {"gamma": 0}         # solver uses gamma=0
   >>> sixc.forward(1, 0, 0)                    # no motor is moved

To release a single preset (|spec| ``unfreeze``):

.. code-block:: python
   :linenos:

   >>> sixc.core.presets.pop("gamma")           # remove one preset
   0
   >>> sixc.core.presets
   {}

To release all presets at once:

.. code-block:: python
   :linenos:

   >>> sixc.core.presets = {}                  # remove all presets for current mode

.. seealso::

    :ref:`how_presets` — complete how-to guide for presets.
