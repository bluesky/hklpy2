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

    ~GeometryDescriptor
    ~ReflectionDict
    ~SampleDict
    ~SolverMetadataDict
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import TypedDict

from deprecated.sphinx import versionadded

__all__ = [
    "GeometryDescriptor",
    "ReflectionDict",
    "SampleDict",
    "SolverMetadataDict",
]


@versionadded(version="0.5.0", reason="Dynamic geometry registration support.")
@dataclass
class GeometryDescriptor:
    """
    Describes a diffractometer geometry independently of any solver backend.

    A :class:`GeometryDescriptor` captures the static facts about a geometry —
    its axis names, available modes, and optional description — decoupled from
    the solver library that implements the mathematics.  It serves two roles:

    1. **Registry entry** — stored in a solver class's
       :attr:`~hklpy2.backends.base.SolverBase._geometry_registry` so that
       :meth:`~hklpy2.backends.base.SolverBase.geometries` can enumerate
       available geometries without creating a solver instance.

    2. **Dispatch table** — used by pure-Python solvers (e.g.
       :class:`~hklpy2.backends.th_tth_q.ThTthSolver`) to look up axis lists
       and modes by geometry name, replacing hard-coded ``if self.geometry ==
       ...`` branching in every property.

    Parameters
    ----------
    name : str
        Canonical geometry name, e.g. ``"TH TTH Q"`` or ``"E4CV"``.  Used as
        the dictionary key in the registry and must match the string passed to
        the solver's constructor.
    pseudo_axis_names : list of str
        Ordered pseudo-axis names, e.g. ``["h", "k", "l"]`` or ``["q"]``.
        Order is significant — solvers must not sort this list.
    real_axis_names : list of str
        Ordered real-axis names, e.g. ``["omega", "chi", "phi", "tth"]`` or
        ``["th", "tth"]``.  Order is significant — solvers must not sort.
    modes : list of str
        All valid operating mode names for this geometry.
    default_mode : str
        Mode to use when ``mode=""`` is requested.  Must be one of
        :attr:`modes` or ``""`` (meaning the solver chooses).
    description : str
        Optional human-readable description of this geometry.

    Examples
    --------
    >>> from hklpy2.backends.typing import GeometryDescriptor
    >>> geo = GeometryDescriptor(
    ...     name="TH TTH Q",
    ...     pseudo_axis_names=["q"],
    ...     real_axis_names=["th", "tth"],
    ...     modes=["bissector"],
    ...     default_mode="bissector",
    ...     description="theta / two-theta powder diffractometer",
    ... )
    >>> geo.name
    'TH TTH Q'
    >>> geo.real_axis_names
    ['th', 'tth']
    """

    name: str
    pseudo_axis_names: List[str]
    real_axis_names: List[str]
    modes: List[str]
    default_mode: str = ""
    description: str = ""
    extra_axis_names: Dict[str, List[str]] = field(default_factory=dict)
    """Per-mode extra parameter names: ``{mode_name: [param_names, ...]}``.

    Used by solvers that expose additional named parameters (e.g. azimuthal
    angle) for specific modes.  Defaults to an empty dict (no extras).
    """


@versionadded(version="0.4.0", reason="Typed reflection dictionary.")
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


@versionadded(version="0.4.0", reason="Typed sample dictionary.")
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


@versionadded(version="0.4.0", reason="Typed solver metadata dictionary.")
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
