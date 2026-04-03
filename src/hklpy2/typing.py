"""
Package-level type aliases and typed dictionary structures for |hklpy2|.

Consolidates the type aliases previously defined in :mod:`hklpy2.misc` (see
:issue:`252`) alongside the ``TypedDict`` subclasses introduced in
:issue:`233`.  :mod:`hklpy2.misc` re-exports everything defined here for
backward compatibility.

.. rubric:: Simple type aliases

.. autosummary::

    ~AnyAxesType
    ~AxesArray
    ~AxesDict
    ~AxesList
    ~AxesTuple
    ~BlueskyPlanType
    ~INPUT_VECTOR
    ~KeyValueMap
    ~Matrix3x3
    ~NamedFloatDict
    ~NUMERIC

.. rubric:: TypedDict structures

.. autosummary::

    ~ConfigHeaderDict

Solver-internal typed structures (``ReflectionDict``, ``SampleDict``,
``SolverMetadataDict``) live in :mod:`hklpy2.backends.typing`.
Solver-specific metadata extensions (e.g. ``HklSolverMetadataDict``) live
in the solver's own module (e.g. :mod:`hklpy2.backends.hkl_soleil`).
"""

from collections.abc import Iterator
from collections.abc import Sequence
from typing import Any
from typing import Mapping
from typing import TypedDict
from typing import Union

import numpy as np
import numpy.typing as npt
from bluesky.utils import Msg

__all__ = [
    "AnyAxesType",
    "AxesArray",
    "AxesDict",
    "AxesList",
    "AxesTuple",
    "BlueskyPlanType",
    "ConfigHeaderDict",
    "INPUT_VECTOR",
    "KeyValueMap",
    "Matrix3x3",
    "NamedFloatDict",
    "NUMERIC",
]

# ---------------------------------------------------------------------------
# Simple type aliases
# ---------------------------------------------------------------------------

BlueskyPlanType = Iterator[Sequence[Msg]]
"""Type of a bluesky plan."""

KeyValueMap = Mapping[str, Any]
"""Dictionary for configuration and other."""

NUMERIC = Union[float, int]
"""Either integer or real number."""

INPUT_VECTOR = Union[
    list[NUMERIC],
    Mapping[str, NUMERIC],
    npt.NDArray[np.floating],
    Sequence[NUMERIC],
]
"""Acceptable forms of vector input for zones, ..."""

AxesArray = npt.NDArray[np.floating]
"""Numpy array of axes values."""

AxesDict = dict[str, NUMERIC]
"""Dictionary of axes names and values."""

AxesList = list[NUMERIC]
"""List of axes values."""

AxesTuple = tuple[NUMERIC, ...]
"""Tuple of axes values."""

AnyAxesType = Union[AxesArray, AxesDict, AxesList, AxesTuple]
"""
Any of these types are used to describe both pseudo and real axes.

=============   =========================   ====================
description     example                     type annotation
=============   =========================   ====================
dict            {"h": 0, "k": 1, "l": -1}   AxesDict
namedtuple      (h=0.0, k=1.0, l=-1.0)      AxesTuple
numpy array     numpy.array([0, 1, -1])     AxesArray
ordered list    [0, 1, -1]                  AxesList
ordered tuple   (0, 1, -1)                  AxesTuple
=============   =========================   ====================
"""

Matrix3x3 = list[list[float]]
"""Python type annotation: mutable orientation & rotation matrices."""

NamedFloatDict = Mapping[str, NUMERIC]
"""Python type annotation: dictionary of named floats."""


# ---------------------------------------------------------------------------
# TypedDict structures
# ---------------------------------------------------------------------------


class ConfigHeaderDict(TypedDict):
    """
    Typed structure for the configuration ``_header`` block.

    Written by :meth:`~hklpy2.ops.Core._asdict` to record provenance
    information alongside saved diffractometer configurations.

    Keys
    ----
    datetime : str
        ISO-8601 timestamp when the configuration was saved.
    hklpy2_version : str
        Version of |hklpy2| used to write the configuration.
    python_class : str
        ``__class__.__name__`` of the diffractometer.
    """

    datetime: str
    hklpy2_version: str
    python_class: str
