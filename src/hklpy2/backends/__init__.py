"""
A Solver connects |hklpy2| with a backend calculation library.

A :index:`!Solver` class (also described as a *backend*) is a
Python class that connects |hklpy2| with a library
that provides diffractometer geometries & calculations.
See the API documentation for details.

.. autosummary::

    ~hkl_soleil
    ~no_op
    ~th_tth_q
    ~base
"""

from .base import SolverBase  # noqa: F401
