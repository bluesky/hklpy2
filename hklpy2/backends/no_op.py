"""
"no_op" solver for testing.

no reciprocal-space conversions

Example::

    import hklpy2
    SolverClass = hklpy2.get_solver("no_op")
    noop_solver = SolverClass()

.. autosummary::

    ~NoOpSolver
"""

import logging
from typing import Dict
from typing import List

from .. import __version__
from ..misc import IDENTITY_MATRIX_3X3
from .base import Lattice
from .base import Reflection
from .base import SolverBase

logger = logging.getLogger(__name__)


class NoOpSolver(SolverBase):
    """
    ``"no_op"`` (any OS) no transformations.

    |solver| that has no reciprocal space transformations.

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

    name = "no_op"
    version = __version__

    def __init__(self, geometry: str, **kwargs) -> None:
        super().__init__(geometry, **kwargs)

    def addReflection(self, reflection: Reflection) -> None:
        return None

    def calculate_UB(self, r1: Reflection, r2: Reflection) -> List[List[float]]:
        return IDENTITY_MATRIX_3X3

    @property
    def extra_axis_names(self) -> List[str]:
        return []

    def forward(self, pseudos: Dict[str, float]) -> List[Dict[str, float]]:
        return [{}]

    @classmethod
    def geometries(cls) -> List[str]:
        return []

    def inverse(self, reals: Dict[str, float]) -> Dict[str, float]:
        return {}

    @property
    def modes(self) -> List[str]:
        return []

    @property
    def pseudo_axis_names(self) -> List[str]:
        return []  # no axes

    @property
    def real_axis_names(self) -> List[str]:
        return []  # no axes

    def refineLattice(self, reflections: List[Reflection]) -> Lattice:
        """No refinement."""
        return None

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        pass
