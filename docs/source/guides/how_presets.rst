.. _how_presets:

=======================
How to Use Presets
=======================

.. index::
    !presets
    see: freeze; presets

Some diffractometer modes hold one or more real axes at a fixed value while
the solver computes the remaining axes needed to reach a requested :math:`hkl`
position.  **Presets** are the values the solver assumes for those constant
axes during ``forward()`` computations.

This guide walks through common preset operations using a simulated E4CV
diffractometer.

Setup
-----

Create a simulated 4-circle diffractometer::

    >>> import hklpy2
    >>> e4cv = hklpy2.creator(name="e4cv")

Do presets move motors?
-----------------------

**No.** Setting a preset never commands a motor to move. The preset only tells
the solver which value to *assume* for a constant axis when computing
``forward()`` solutions. The motor position is unaffected.

This is useful when you want to compute :math:`hkl` solutions at a specific
constant-axis angle without physically driving the motor there first.

Which axes can have presets?
----------------------------

Only axes that are **constant** (held fixed) in the current mode accept
presets. Values supplied for computed axes are silently ignored.

Check which axes are constant for the active mode::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.constant_axis_names
    ['phi']

    >>> e4cv.core.mode = "constant_omega"
    >>> e4cv.core.constant_axis_names
    ['omega']

Setting a preset
----------------

Assign a dictionary to ``presets`` while the desired mode is active::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.core.presets
    {'phi': 45.0}

A list or tuple in ``local_real_axes`` order is also accepted::

    >>> e4cv.core.presets = [0.0, 0.0, 45.0, 0.0]   # omega, chi, phi, tth

Only the constant-axis values are stored; the rest are silently dropped.

.. important::

    The setter **replaces** the preset dictionary for the current mode — it
    does **not** merge with existing presets.  Each assignment is a fresh
    replacement.

::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.core.presets = {"phi": 90.0}   # replaces previous assignment
    >>> e4cv.core.presets
    {'phi': 90.0}

Presets are per-mode
--------------------

Each mode stores its own independent preset dictionary. The default for every
mode is an empty dictionary ``{}``. Switching modes switches to that mode's
presets; switching back restores them::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}

    >>> e4cv.core.mode = "constant_omega"
    >>> e4cv.core.presets          # omega mode starts empty
    {}
    >>> e4cv.core.presets = {"omega": 30.0}

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets          # phi preset restored
    {'phi': 45.0}

Updating a single preset key
-----------------------------

Because the setter replaces the entire dictionary, update a single key by
modifying the live dict returned by the getter directly::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.core.presets["phi"] = 90.0   # update one key in place
    >>> e4cv.core.presets
    {'phi': 90.0}

Or use a read-modify-write pattern when building from the current state::

    >>> new = dict(e4cv.core.presets)     # copy current presets
    >>> new["phi"] = 90.0                 # change one entry
    >>> e4cv.core.presets = new           # replace with updated dict

.. _how_presets.pop:

Removing a single preset key
-----------------------------

Use ``.pop()`` on the live dictionary returned by the getter::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.core.presets.pop("phi")
    45.0
    >>> e4cv.core.presets
    {}

Clearing all presets for the current mode
------------------------------------------

Assign an empty dictionary::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.core.presets = {}
    >>> e4cv.core.presets
    {}

Checking presets with ``wh()``
-------------------------------

When presets are active, :meth:`~hklpy2.diffract.DiffractometerBase.wh` with
``full=True`` includes them in its output::

    >>> e4cv.core.mode = "constant_phi"
    >>> e4cv.core.presets = {"phi": 45.0}
    >>> e4cv.wh(full=True)
    ...
    presets: {'phi': 45.0}
    ...

.. seealso::

    :ref:`concepts.ops` — overview of Core concepts, including presets.

    :class:`~hklpy2.ops.Core` — full API reference.
