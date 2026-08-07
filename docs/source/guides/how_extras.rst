.. _how_extras:

=========================
How to Set Extras
=========================

.. index::
    !extra; how-to
    see: extra parameter; extra

**Extras** are optional parameters provided by some solver backends and modes.
They are not part of the standard diffractometer axes (reals or pseudos) but are
sometimes needed for certain calculations or operations. This guide shows how to
read and set extras.

.. seealso::

   :ref:`concepts.solvers` — explains what solvers are and how extras fit into
   the larger solver architecture.

   :doc:`/examples/hkl_soleil-e6c-psi` — demonstrates using extras with the psi
   angle example.

Setup
-----

All examples use a simulated K6C diffractometer with the `constant_incidence` mode,
which has several extra parameters::

    >>> import hklpy2
    >>> k6c = hklpy2.creator(name="k6c", geometry="K6C", solver="hkl_soleil")
    >>> k6c.core.mode = "constant_incidence"
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

How do I read the current extras?
---------------------------------

Access the :attr:`~hklpy2.ops.Core.extras` property, which returns a dictionary
of all extras available in the current mode::

    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

To get a specific extra value::

    >>> k6c.core.extras["x"]
    0.0

Or use the dict ``.get()`` method with a default fallback::

    >>> k6c.core.extras.get("nonexistent", 0.0)
    0.0

How do I set a single extra parameter?
--------------------------------------

Set an individual extra by key-assignment::

    >>> k6c.core.extras["x"] = 0.5
    >>> k6c.core.extras["x"]
    0.5

The value is immediately validated against the current mode's allowed extras.
Invalid keys raise a :exc:`~hklpy2.exceptions.ConfigurationError`::

    >>> k6c.core.extras["invalid_key"] = 0.5
    Traceback (most recent call last):
      ...
    ConfigurationError: Unexpected extra axis name(s) ['invalid_key'].
      Expected names: ['x', 'y', 'z', 'incidence', 'azimuth'].

How do I set multiple extras at once?
-------------------------------------

Use the ``.update()`` method to set multiple extras::

    >>> k6c.core.extras.update({"x": 0.5, "y": 0.25})
    >>> k6c.core.extras
    {'x': 0.5, 'y': 0.25, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

You can also use keyword arguments::

    >>> k6c.core.extras.update(x=0.1, y=0.2)
    >>> k6c.core.extras
    {'x': 0.1, 'y': 0.2, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

Like key-assignment, ``.update()`` validates all keys before applying changes.
If any key is invalid, the entire update is rejected::

    >>> k6c.core.extras.update(x=0.5, invalid_key=0.1)
    Traceback (most recent call last):
      ...
    ConfigurationError: Unexpected extra axis name(s) ['invalid_key'].
      Expected names: ['x', 'y', 'z', 'incidence', 'azimuth'].

How do I set a default value for an extra?
-------------------------------------------

Use the ``.setdefault()`` method. It sets a value only if the key is not already set,
and returns the value that was there::

    >>> k6c.core.extras["x"] = 0.5
    >>> result = k6c.core.extras.setdefault("x", 0.1)
    >>> result  # Returns existing value, not the default
    0.5
    >>> k6c.core.extras["x"]  # Unchanged
    0.5

For a key that is not set::

    >>> result = k6c.core.extras.setdefault("y", 0.25)
    >>> result  # Returns the default since key wasn't set
    0.25
    >>> k6c.core.extras["y"]
    0.25

How do I reset all extras to their default value?
-------------------------------------------------

Use the ``.clear()`` method to reset all extras in the current mode to 0.0::

    >>> k6c.core.extras.update(x=0.5, y=0.25, z=0.1)
    >>> k6c.core.extras
    {'x': 0.5, 'y': 0.25, 'z': 0.1, 'incidence': 0.0, 'azimuth': 0.0}
    >>> k6c.core.extras.clear()
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

How do I set extras using the old dict-assignment pattern?
----------------------------------------------------------

For compatibility, you can still replace the entire extras dictionary using
the setter::

    >>> k6c.core.extras = {"x": 0.5, "y": 0.25}
    >>> k6c.core.extras
    {'x': 0.5, 'y': 0.25, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

This validates all keys in the new dict, and unspecified extras retain their
current values (as shown above, `z`, `incidence`, and `azimuth` remain at 0.0).

Can I pop or delete an extra?
-----------------------------

Popping a key from the dict view is allowed::

    >>> view = k6c.core.extras
    >>> view.pop("x", None)
    0.5
    >>> "x" in view  # Removed from this view
    False

However, getting a fresh view of extras shows the key is still there in Core's
internal state::

    >>> k6c.core.extras["x"]
    0.5

This is intentional: extras for a mode are fixed (determined by the solver),
so deletion is not permitted. The `pop()` method only modifies the temporary
view you accessed, not the underlying state.

**Explicit deletion via ``del`` is forbidden** and raises a :exc:`TypeError`::

    >>> del k6c.core.extras["x"]
    Traceback (most recent call last):
      ...
    TypeError: Deletion of extras is not allowed.

What happens to extras when I change the mode?
----------------------------------------------

Changing the mode changes which extras are available. Each mode has its own
set of extras, and switching between modes resets the extras to their default
values::

    >>> k6c.core.mode = "constant_incidence"
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}
    >>> k6c.core.extras["x"] = 0.5
    >>> k6c.core.mode = "eulerians"  # Switch mode
    >>> k6c.core.extras  # Different extras for this mode
    {'solutions': 0.0}

If you switch back to the original mode, the extras will have been reset::

    >>> k6c.core.mode = "constant_incidence"
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

Do all modes have extras?
-------------------------

No. Some modes have no extras at all. For example, the E4CV geometry's
``bissector`` mode has no extras::

    >>> e4cv = hklpy2.creator(name="e4cv", geometry="E4CV")
    >>> e4cv.core.mode = "bissector"
    >>> e4cv.core.extras
    {}
    >>> len(e4cv.core.extras)
    0

It is safe to call any dict method on an empty extras dict::

    >>> e4cv.core.extras.get("anything")
    >>> e4cv.core.extras.clear()  # No-op on empty
    >>> e4cv.core.extras.update({})  # Also safe

But attempting to set a key when the mode has no extras raises
:exc:`~hklpy2.exceptions.ConfigurationError`::

    >>> e4cv.core.extras["x"] = 0.5
    Traceback (most recent call last):
      ...
    ConfigurationError: Unexpected extra axis name(s) ['x'].
      Expected names: [].

What is an ExtrasDict?
----------------------

The :attr:`~hklpy2.ops.Core.extras` property returns an instance of
:class:`~hklpy2.ops.ExtrasDict`, a custom dictionary subclass. It behaves like
a normal Python :class:`dict`, supporting all standard dict methods and
operations:

- **Read-only methods**: ``get()``, ``keys()``, ``values()``, ``items()``,
  ``__contains__`` (``in`` operator), ``__len__`` (``len()``), ``copy()``, etc.

- **Mutation methods**: ``__setitem__`` (``[]=``), ``update()``,
  ``setdefault()``, ``pop()``, ``clear()``

- **Validation**: All mutations validate that keys belong to the current mode's
  extras, and that values are valid numbers. Invalid mutations raise
  :exc:`~hklpy2.exceptions.ConfigurationError`.

- **Solver updates**: All mutations automatically flag the solver to update,
  ensuring changes are propagated correctly.

- **Deletion forbidden**: ``__delitem__`` (``del``) and other destructive
  operations that would remove keys are forbidden by design.

See the examples in this guide for typical usage patterns.

Examples
--------

**Example 1: Adjust incidence angle for a surface-sensitive measurement**

Set the incidence angle to a shallow value for surface-sensitive scattering::

    >>> k6c = hklpy2.creator(name="k6c", geometry="K6C")
    >>> k6c.core.mode = "constant_incidence"
    >>> k6c.core.extras["incidence"] = 0.2  # 0.2 degrees
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.2, 'azimuth': 0.0}

**Example 2: Apply offset correction**

Apply X-Y-Z offsets all at once::

    >>> k6c.core.extras.update(x=0.1, y=0.05, z=0.02)
    >>> k6c.core.extras
    {'x': 0.1, 'y': 0.05, 'z': 0.02, 'incidence': 0.2, 'azimuth': 0.0}

**Example 3: Reset and reconfigure**

Clear all extras and set a fresh configuration::

    >>> k6c.core.extras.clear()
    >>> k6c.core.extras.update(incidence=0.15, azimuth=90.0)
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.15, 'azimuth': 90.0}

**Example 4: Check what extras are available in current mode**

List all available extras and their current values::

    >>> for key, value in k6c.core.extras.items():
    ...     print(f"  {key}: {value}")
      x: 0.0
      y: 0.0
      z: 0.0
      incidence: 0.15
      azimuth: 90.0

See Also
--------

- :meth:`~hklpy2.ops.Core.forward` — Computes real-space motor positions using
  the current extras settings.
- :meth:`~hklpy2.ops.Core.inverse` — Computes pseudo-space positions; extras
  may be needed depending on the mode.
- :class:`~hklpy2.ops.ExtrasDict` — The custom dict class returned by the
  :attr:`~hklpy2.ops.Core.extras` property.
- :doc:`/examples/hkl_soleil-e6c-psi` — Advanced example using extras with the
  psi (ψ) angle.
