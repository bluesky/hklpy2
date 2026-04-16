"""
"th_tth" example solver in Python.

Transformations between :math:`\\theta,2\\theta` and :math:`Q`.

Example::

    import hklpy2
    SolverClass = hklpy2.get_solver("th_tth")
    solver = SolverClass()

.. autosummary::

    ~ThTthSolver
"""

import logging
import math
from typing import List

from .. import __version__
from ..misc import IDENTITY_MATRIX_3X3
from ..exceptions import SolverError
from ..typing import Matrix3x3
from ..typing import NamedFloatDict
from .base import SolverBase
from .typing import GeometryDescriptor
from .typing import ReflectionDict

logger = logging.getLogger(__name__)
TH_TTH_Q_GEOMETRY = "TH TTH Q"
BISECTOR_MODE = "bissector"  # spelled same as in E4CV

#: Built-in geometry descriptor for the theta/two-theta/Q geometry.
_TH_TTH_Q_DESCRIPTOR = GeometryDescriptor(
    name=TH_TTH_Q_GEOMETRY,
    pseudo_axis_names=["q"],
    real_axis_names=["th", "tth"],
    modes=[BISECTOR_MODE],
    default_mode=BISECTOR_MODE,
    description="theta / two-theta powder diffractometer (Q transform)",
)


class ThTthSolver(SolverBase):
    """
    ``"th_tth"`` (any OS) :math:`\\theta,2\\theta` and :math:`Q`.

    ============== =================
    transformation equation
    ============== =================
    ``forward()``  :math:`\\theta = \\sin^{-1}(q\\lambda / 4\\pi)`
    ``inverse()``  :math:`q = (4\\pi / \\lambda) \\sin(\\theta)`
    ============== =================

    Wavelength is specified either directly (``solver.wavelength = 1.0``) or
    by adding at least one :index:`reflection` (see
    :class:`~hklpy2.blocks.reflection.Reflection`).  All
    reflections must have the same :index:`wavelength`.

    No orientation matrix is used in this geometry.

    New geometries can be added at runtime via
    :meth:`~hklpy2.backends.base.SolverBase.register_geometry`.

    .. rubric:: Python Methods

    .. autosummary::

        ~addReflection
        ~calculate_UB
        ~extra_axis_names
        ~forward
        ~geometries
        ~inverse
        ~pseudo_axis_names
        ~real_axis_names
        ~refineLattice
        ~removeAllReflections

    .. rubric:: Python Properties

    .. autosummary::

        ~geometry
        ~lattice
        ~mode
        ~modes
        ~sample
    """

    name = "th_tth"
    version = __version__

    # Each ThTthSolver subclass (or the class itself) has its own registry,
    # independent of SolverBase._geometry_registry.
    _geometry_registry = {}

    def __init__(self, geometry: str, **kwargs) -> None:
        super().__init__(geometry, **kwargs)
        self._reflections = []
        self._wavelength = None

    def _descriptor(self) -> GeometryDescriptor:
        """Return the GeometryDescriptor for the current geometry, or None."""
        return self._geometry_registry.get(self.geometry)

    def addReflection(self, value: ReflectionDict) -> None:
        """Add coordinates of a diffraction condition (a reflection)."""
        if not isinstance(value, dict):
            raise TypeError(f"Must supply SolverReflection (dict), received {value!r}")
        self._reflections.append(value)

        # validate: all reflections must have same wavelength
        wavelengths = [r["wavelength"] for r in self._reflections]
        if min(wavelengths) != max(wavelengths):
            self._reflections.pop(-1)
            raise SolverError(
                f"All reflections must have same wavelength. Received: {wavelengths!r}"
            )
        self.wavelength = wavelengths[0]

    def calculate_UB(self, r1: ReflectionDict, r2: ReflectionDict) -> Matrix3x3:
        return IDENTITY_MATRIX_3X3

    @property
    def extra_axis_names(self) -> List[str]:
        desc = self._descriptor()
        if desc is None:
            return []
        # extra_axis_names is a per-mode dict; return all unique names
        all_extras: List[str] = []
        for names in desc.extra_axis_names.values():
            all_extras += names
        return sorted(set(all_extras))

    def forward(self, pseudos: NamedFloatDict) -> List[NamedFloatDict]:
        """Transform pseudos to list of reals."""
        if not isinstance(pseudos, dict):
            raise TypeError(f"Must supply dict, received {pseudos!r}")

        desc = self._descriptor()
        solutions = []
        if desc is not None and self.geometry == TH_TTH_Q_GEOMETRY:
            q = pseudos.get("q")
            if q is None:
                raise SolverError(f"'q' not defined. Received {pseudos!r}.")
            if self.wavelength is None:
                raise SolverError("Wavelength is not set. Add a reflection.")
            if self.mode == BISECTOR_MODE:
                th = math.degrees(math.asin(q * self.wavelength / 4 / math.pi))
                solutions.append({"th": th, "tth": 2 * th})

        return solutions

    @classmethod
    def geometries(cls) -> List[str]:
        """Sorted list of registered geometry names."""
        return sorted(cls._geometry_registry.keys())

    def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
        """Transform reals to pseudos."""
        if not isinstance(reals, dict):
            raise TypeError(f"Must supply dict, received {reals!r}")

        desc = self._descriptor()
        pseudos = {}
        if desc is not None and self.geometry == TH_TTH_Q_GEOMETRY:
            tth = reals.get("tth")
            if tth is None:
                raise SolverError(f"'tth' not defined. Received {reals!r}.")
            if self.wavelength is None:
                raise SolverError("Wavelength is not set. Add a reflection.")
            if self.mode == BISECTOR_MODE:
                q = (4 * math.pi) / self.wavelength
                q *= math.sin(math.radians(tth / 2))
                pseudos["q"] = q
        return pseudos

    @property
    def modes(self) -> List[str]:
        desc = self._descriptor()
        if desc is None:
            return []
        return list(desc.modes)

    @property
    def pseudo_axis_names(self) -> List[str]:
        desc = self._descriptor()
        if desc is None:
            return []
        return list(desc.pseudo_axis_names)

    @property
    def real_axis_names(self) -> List[str]:
        desc = self._descriptor()
        if desc is None:
            return []
        return list(desc.real_axis_names)

    def refineLattice(self, reflections: list[ReflectionDict]) -> NamedFloatDict | None:
        """No lattice refinement in this |solver|."""
        return None

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        self._reflections.clear()
        self._wavelength = None

    @property
    def wavelength(self) -> float:
        """Diffractometer wavelength, for forward() and inverse()."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Must supply number, received {value!r}")
        if value <= 0:
            raise ValueError(f"Must supply positive number, received {value!r}")
        self._wavelength = value


# Register the built-in TH TTH Q geometry at import time.
ThTthSolver.register_geometry(_TH_TTH_Q_DESCRIPTOR)
