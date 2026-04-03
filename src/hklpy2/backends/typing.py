"""
Typed dictionary structures shared across all solver backends.

Replaces the broad ``KeyValueMap = Mapping[str, Any]`` alias at call-sites
where the dict structure is well-known and fixed, enabling static type
checking and serving as inline documentation of expected dict shapes.

Solver-specific metadata extensions (e.g. ``HklSolverMetadataDict``) belong
in their respective solver modules, not here.  To add bespoke metadata keys
for a new solver, subclass :class:`SolverMetadataDict` with ``total=False``
and place the subclass in that solver's module.

Structures that belong above the backends layer live in
:mod:`hklpy2.typing` (e.g. ``ConfigHeaderDict``).

.. autosummary::

    ~ReflectionDict
    ~SampleDict
    ~SolverMetadataDict
"""

from typing import Dict
from typing import List
from typing import TypedDict

__all__ = [
    "ReflectionDict",
    "SampleDict",
    "SolverMetadataDict",
]


class ReflectionDict(TypedDict):
    """
    Typed structure for a single diffraction reflection.

    Keys
    ----
    name : str
        Label for this reflection (e.g. ``"r1"``).
    pseudos : Dict[str, float]
        Pseudo-axis values at this reflection, keyed by pseudo axis name
        (e.g. ``{"h": 1.0, "k": 0.0, "l": 0.0}``).
    reals : Dict[str, float]
        Real-axis values at this reflection, keyed by solver axis name
        (e.g. ``{"omega": 10.0, "chi": 0.0, "phi": 0.0, "tth": 20.0}``).
    wavelength : float
        Incident wavelength (Å) used when the reflection was measured.
    """

    name: str
    pseudos: Dict[str, float]
    reals: Dict[str, float]
    wavelength: float


class SampleDict(TypedDict):
    """
    Typed structure for a crystalline sample.

    Keys
    ----
    name : str
        Human-readable label for the sample.
    lattice : Dict[str, float]
        Unit-cell parameters ``{"a", "b", "c", "alpha", "beta", "gamma"}``.
    reflections : Dict[str, ReflectionDict]
        Mapping of reflection name → :class:`ReflectionDict`.
    order : List[str]
        Reflection names in the order they should be added to the solver
        (controls which two are used for UB calculation).
    """

    name: str
    lattice: Dict[str, float]
    reflections: Dict[str, "ReflectionDict"]
    order: List[str]


class SolverMetadataDict(TypedDict):
    """
    Typed structure for the **common** solver summary metadata.

    Returned by :attr:`~hklpy2.backends.base.SolverBase._metadata` and
    stored under the ``"solver"`` key in the configuration dictionary.

    This class defines only the keys that every solver is required to
    provide.  Solver-specific extra fields should be added by subclassing
    with ``total=False`` in the solver's own module.  Example::

        # in backends/hkl_soleil.py
        class HklSolverMetadataDict(SolverMetadataDict, total=False):
            engine: str

    Keys
    ----
    name : str
        Solver name (e.g. ``"no_op"``, ``"hkl_soleil"``).
    description : str
        Human-readable repr string.
    geometry : str
        Selected geometry name (e.g. ``"E4CV"``).
    real_axes : List[str]
        Ordered list of real axis names.
    version : str
        Version string for the solver library.
    """

    name: str
    description: str
    geometry: str
    real_axes: List[str]
    version: str
