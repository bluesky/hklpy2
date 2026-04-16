"""
Custom exception classes for |hklpy2|.

All package-specific exceptions inherit from :class:`Hklpy2Error`, which
itself inherits from :exc:`Exception`.  Catching :class:`Hklpy2Error`
will catch any exception raised intentionally by |hklpy2|.

.. autosummary::

    ~Hklpy2Error
    ~ConfigurationError
    ~ConstraintsError
    ~CoreError
    ~DiffractometerError
    ~LatticeError
    ~NoForwardSolutions
    ~ReflectionError
    ~SampleError
    ~SolverError
"""

__all__ = [
    "ConfigurationError",
    "ConstraintsError",
    "CoreError",
    "DiffractometerError",
    "Hklpy2Error",
    "LatticeError",
    "NoForwardSolutions",
    "ReflectionError",
    "SampleError",
    "SolverError",
]


class Hklpy2Error(Exception):
    """Any exception from the |hklpy2| package."""


class ConfigurationError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.configure`."""


class ConstraintsError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.constraints`."""


class CoreError(Hklpy2Error):
    """Custom exceptions from :class:`hklpy2.ops.Core`."""


class DiffractometerError(Hklpy2Error):
    """Custom exceptions from :class:`hklpy2.diffract.DiffractometerBase`."""


class LatticeError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.lattice`."""


class NoForwardSolutions(Hklpy2Error):
    """A solver did not find any ``forward()`` solutions."""


class ReflectionError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.reflection`."""


class SampleError(Hklpy2Error):
    """Custom exceptions from :mod:`hklpy2.blocks.sample`."""


class SolverError(Hklpy2Error):
    """Custom exceptions from a |solver|."""
