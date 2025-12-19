"""
A Crystalline Sample.

.. autosummary::

    ~Sample
"""

import logging
import math
from typing import Mapping
from typing import Union

import numpy as np
from numpy.linalg import norm

from ..misc import Matrix3x3
from ..misc import unique_name
from .lattice import Lattice
from .lattice import LatticeDictType
from .reflection import Reflection
from .reflection import ReflectionsDict

logger = logging.getLogger(__name__)

SampleDictType = Mapping[
    str,
    Union[
        int,
        list[Reflection],
        Matrix3x3,
        ReflectionsDict,
        str,
        Union[Lattice, LatticeDictType],
    ],
]


class Sample:
    """
    A crystalline sample mounted on a diffractometer.

    .. note:: Internal use only.

       It is expected this class is called from a method of
       :class:`~hklpy2.ops.Core`, not directly by the user.

    .. rubric:: Python Methods

    .. autosummary::

        ~refine_lattice
        ~_validate_matrices

    .. rubric:: Python Properties

    .. autosummary::

        ~_asdict
        ~lattice
        ~name
        ~reflections
        ~remove_reflection
        ~U
        ~UB
    """

    def __init__(
        self,
        core: object,
        name: str,
        lattice: Lattice,
    ) -> None:
        from ..misc import IDENTITY_MATRIX_3X3
        from ..ops import Core

        if not isinstance(core, Core):
            raise TypeError(f"Unexpected type {core=!r}, expected Core")
        self.name = name or unique_name()
        self.core = core
        self.lattice = lattice
        self.U = IDENTITY_MATRIX_3X3
        # Consider: UB = self.U @ self.lattice.B
        self.UB = ((2 * math.pi / self.lattice.a) * np.array(self.U)).tolist()
        self.core._solver_needs_update = True
        self.reflections = ReflectionsDict()

    def __repr__(self) -> str:
        """Brief text representation."""
        return f"Sample(name={self.name!r}, lattice={self.lattice!r})"

    def _asdict(self) -> SampleDictType:
        """Describe the sample as a dictionary."""
        return {
            "name": self.name,
            "lattice": self.lattice._asdict(),
            "reflections": self.reflections._asdict(),
            "reflections_order": self.reflections.order,
            "U": self.U,
            "UB": self.UB,
            "digits": self.digits,
        }

    def _fromdict(self, config: SampleDictType, core=None):
        """Redefine sample from a (configuration) dictionary."""
        self.name = config["name"]
        self.digits = config["digits"]
        self.lattice._fromdict(config["lattice"])
        self.reflections._fromdict(config["reflections"], core=core)
        self.reflections.order = config["reflections_order"]
        self.U = config["U"]
        self.UB = config["UB"]

    def refine_lattice(self) -> None:
        """Refine the lattice parameters from 3 or more reflections."""
        self.lattice = self.core.refine_lattice()

    def remove_reflection(self, name: str) -> None:
        """Remove the named reflection."""
        if name not in self.reflections:
            raise KeyError(f"Reflection {name!r} is not found.")
        self.reflections.pop(name)
        if name in self.reflections.order:
            self.reflections.order.remove(name)

    # --------- get/set properties

    @property
    def digits(self) -> int:
        """Sample crystal lattice."""
        return self.lattice.digits

    @digits.setter
    def digits(self, value: int) -> None:
        self.lattice.digits = value

    @property
    def lattice(self) -> Lattice:
        """Sample crystal lattice."""
        return self._lattice

    @lattice.setter
    def lattice(self, value: Union[LatticeDictType, Lattice]):
        if isinstance(value, dict):
            value = Lattice(**value)
        if not isinstance(value, Lattice):
            raise TypeError(f"Must supply Lattice() object, received {value!r}")
        self._lattice = value

    @property
    def name(self) -> str:
        """Sample name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, (type(None), str)):
            raise TypeError(f"Must supply str, received {value!r}")
        self._name = value

    @property
    def reflections(self) -> ReflectionsDict:
        """Ordered dictionary of orientation reflections."""
        return self._reflections

    @reflections.setter
    def reflections(self, value: ReflectionsDict) -> None:
        if not isinstance(value, ReflectionsDict):
            raise TypeError(f"Must supply ReflectionsDict() object, received {value!r}")
        self._reflections = value

    def _validate_matrices(self, value: Matrix3x3, name: str) -> None:
        """(internal) Validate U & UB matrices."""
        arr = np.array(value)
        if not np.isreal(arr).all():
            raise TypeError(f"{name} matrix must be numerical.  Received {value}")
        if arr.shape != (3, 3):
            raise ValueError(f"{name} matrix must by 3x3. Received {value}")
        if name == "UB":
            return
        # Rows and columns of U matrix must have unit norms.
        if not np.allclose(norm(arr, axis=1), [1, 1, 1], atol=1e-6):
            raise ValueError(f"{name} matrix rows must be normalized. Received {value}")
        if not np.allclose(norm(arr.T, axis=1), [1, 1, 1], atol=1e-6):
            raise ValueError(
                f"{name} matrix columns must be normalized. Received {value}"
            )

    @property
    def U(self) -> Matrix3x3:
        """Return the matrix, U, crystal orientation on the diffractometer."""
        return self._U

    @U.setter
    def U(self, value: Matrix3x3) -> None:
        self._validate_matrices(value, "U")

        self._U = value
        self.core._solver_needs_update = True

    @property
    def UB(self) -> Matrix3x3:
        """
        Return the crystal orientation matrix, UB.

        * :math:`UB` - orientation matrix
        * :math:`B` - crystal lattice on the diffractometer
        * :math:`U` - rotation matrix, relative orientation of crystal on diffractometer
        """
        return self._UB

    @UB.setter
    def UB(self, value: Matrix3x3) -> None:
        self._validate_matrices(value, "UB")

        self._UB = value
        self.core._solver_needs_update = True
