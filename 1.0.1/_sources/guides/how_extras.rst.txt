.. _how_extras:

=================
How to Use Extras
=================

.. index::
    !extra; how-to
    see: extra parameter; extra

Some solver modes define **extras**: optional named parameters that are not
part of the diffractometer's real or pseudo axes, but still affect solver
calculations.  This guide shows how to inspect and change them through
:attr:`~hklpy2.ops.Core.extras`.

Setup
-----

Create a simulated K6C diffractometer and choose a mode with extras::

    >>> import hklpy2
    >>> k6c = hklpy2.creator(name="k6c", geometry="K6C", solver="hkl_soleil")
    >>> k6c.core.mode = "constant_incidence"
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

Reading extras
--------------

The getter returns a dictionary-like view of the extras available in the
current mode::

    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}
    >>> k6c.core.extras["x"]
    0.0
    >>> list(k6c.core.extras)
    ['x', 'y', 'z', 'incidence', 'azimuth']

Setting one extra
-----------------

Set a single extra by key assignment::

    >>> k6c.core.extras["x"] = 0.5
    >>> k6c.core.extras["x"]
    0.5

Unknown names are rejected::

    >>> k6c.core.extras["invalid_key"] = 0.5
    Traceback (most recent call last):
      ...
    ConfigurationError: Unexpected extra axis name(s) ['invalid_key'].
      Expected names: {'x': 0, 'y': 0, 'z': 0, 'incidence': 0, 'azimuth': 0}.

Setting several extras at once
------------------------------

Use ``update()`` to change multiple extras together::

    >>> k6c.core.extras.update({"x": 0.5, "y": 0.25})
    >>> k6c.core.extras
    {'x': 0.5, 'y': 0.25, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

Keyword arguments are supported too::

    >>> k6c.core.extras.update(x=0.1, y=0.2)
    >>> k6c.core.extras
    {'x': 0.1, 'y': 0.2, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

If any supplied key is invalid, the update is rejected before any change is
applied::

    >>> k6c.core.extras.update(x=0.5, invalid_key=0.1)
    Traceback (most recent call last):
      ...
    ConfigurationError: Unexpected extra axis name(s) ['invalid_key'].
      Expected names: {'x': 0, 'y': 0, 'z': 0, 'incidence': 0, 'azimuth': 0}.

Compatibility with whole-dict assignment
----------------------------------------

The existing setter still works::

    >>> k6c.core.extras = {"x": 0.5, "y": 0.25}
    >>> k6c.core.extras
    {'x': 0.5, 'y': 0.25, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

This is useful when you already have a dictionary of values and want to apply
it in one step.

Resetting extras
----------------

Use ``clear()`` to reset every extra in the current mode to its default value
(``0.0``)::

    >>> k6c.core.extras.update(x=0.5, y=0.25, z=0.1)
    >>> k6c.core.extras.clear()
    >>> k6c.core.extras
    {'x': 0.0, 'y': 0.0, 'z': 0.0, 'incidence': 0.0, 'azimuth': 0.0}

``clear()`` is also safe when the mode has no extras.

What ``setdefault()`` does here
-------------------------------

Extras are pre-populated for the current mode, so ``setdefault()`` usually
returns the value that is already present::

    >>> k6c.core.extras.setdefault("x", 1.0)
    0
    >>> k6c.core.extras["x"]
    0.0

In other words, ``setdefault()`` follows normal dictionary semantics, but the
keys already exist in the extras view for the active mode.

What ``pop()`` does here
------------------------

``pop()`` only removes a key from the temporary dictionary view you accessed;
it does not delete the underlying extra from Core state::

    >>> extras = k6c.core.extras
    >>> extras["x"] = 0.5
    >>> extras.pop("x")
    0.5
    >>> "x" in extras
    False
    >>> k6c.core.extras["x"]
    0.5

This is intentional: the set of extras for a mode is defined by the solver and
is not user-removable.

Deletion is forbidden
---------------------

Explicit deletion with ``del`` is not allowed::

    >>> del k6c.core.extras["x"]
    Traceback (most recent call last):
      ...
    TypeError: Deletion of extras is not allowed.

Modes with no extras
--------------------

Some modes expose no extras at all::

    >>> e4cv = hklpy2.creator(name="e4cv", geometry="E4CV")
    >>> e4cv.core.mode = "bissector"
    >>> e4cv.core.extras
    {}

You may still read the empty dict or call methods such as ``clear()`` and
``update({})`` on it.

.. seealso::

    :ref:`guide.solvers` — overview of solver backends and their mode-specific
    capabilities.

    :class:`~hklpy2.ops.Core` — full API reference for the Core object.
