"""
Package-level typed dictionary structures for |hklpy2|.

This module is the future consolidation point for all type aliases and
``TypedDict`` subclasses currently scattered across the codebase (see
:issue:`252` for the full ``misc.py`` migration plan).

At present it holds structures that belong above the backends layer:

.. autosummary::

    ~ConfigHeaderDict

Solver-internal typed structures (``ReflectionDict``, ``SampleDict``,
``SolverMetadataDict``) live in :mod:`hklpy2.backends.typing`.
Solver-specific metadata extensions (e.g. ``HklSolverMetadataDict``) live
in the solver's own module (e.g. :mod:`hklpy2.backends.hkl_soleil`).
"""

from typing import TypedDict

__all__ = [
    "ConfigHeaderDict",
]


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
