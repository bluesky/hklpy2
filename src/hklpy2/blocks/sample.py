"""
A Crystalline Sample.

.. autosummary::

    ~Sample
"""

import logging
import math
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from deprecated.sphinx import versionadded
from numpy.linalg import norm

from ..utils import _SolverDirty
from ..utils import unique_name
from ..typing import Matrix3x3
from .lattice import Lattice
from .lattice import LatticeDictType
from .reflection import Reflection
from .reflection import ReflectionsDict

# Reflection fields that, when changed for an orienting reflection,
# invalidate the previously computed UB.  ``digits`` is intentionally
# omitted (presentation only); ``name`` is implicit in ``order[:2]``.
_UB_REFLECTION_FIELDS: Tuple[str, ...] = (
    "geometry",
    "pseudos",
    "reals",
    "reals_units",
    "wavelength",
    "wavelength_units",
)

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
        ~UB_is_stale
    """

    def __init__(
        self,
        core: object,
        name: str,
        lattice: Lattice,
    ) -> None:
        from ..utils import IDENTITY_MATRIX_3X3
        from ..ops import Core

        if not isinstance(core, Core):
            raise TypeError(f"Unexpected type {core=!r}, expected Core")
        self.name = name or unique_name()
        self.core = core
        # Snapshot of the orientation-reflection state at the end of the
        # last successful ``calc_UB``.  ``None`` means "no snapshot is
        # stored", which is the case until ``calc_UB`` runs and any time
        # the user takes ownership of U / UB by direct assignment.  See
        # :issue:`391`.
        self._ub_snapshot: Optional[tuple] = None
        self.lattice = lattice
        self.U = IDENTITY_MATRIX_3X3
        # Consider: UB = self.U @ self.lattice.B
        self.UB = ((2 * math.pi / self.lattice.a) * np.array(self.U)).tolist()
        # New sample requires a full solver re-sync (lattice + UB).
        self.core.request_solver_update(_SolverDirty.SAMPLE | _SolverDirty.UB)
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

        # Wire callback so direct parameter changes (e.g. lattice.a = 5)
        # also flag the solver for update (#240).
        value._on_change = self._lattice_changed

        # Flag the solver for update so it receives the new lattice (#240).
        # A lattice change requires a full re-push of the solver's
        # sample state.  Some backends discard U / UB as a side effect,
        # so mark both domains dirty.
        # Guard: during __init__, core is not yet fully wired.
        if hasattr(self, "_U"):
            self.core.request_solver_update(_SolverDirty.SAMPLE | _SolverDirty.UB)

    def _lattice_changed(self):
        """Called by Lattice.__setattr__ when a parameter changes (#240)."""
        # Lattice mutation invalidates the solver's sample state and,
        # as a side effect of re-pushing it, U / UB in some backends.
        self.core.request_solver_update(_SolverDirty.SAMPLE | _SolverDirty.UB)

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
            raise ValueError(f"{name} matrix must be 3x3.  Received {value}")
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
        # U changes require only a UB push; no full _sample rebuild.
        self.core.request_solver_update(_SolverDirty.UB)
        # Direct assignment: the user has taken ownership of U; clear
        # any prior calc_UB snapshot so ``UB_is_stale`` reports False
        # until the next ``calc_UB`` (:issue:`391`).
        self._ub_snapshot = None

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
        # UB changes require only a UB push; no full _sample rebuild.
        self.core.request_solver_update(_SolverDirty.UB)
        # Direct assignment: the user has taken ownership of UB; clear
        # any prior calc_UB snapshot so ``UB_is_stale`` reports False
        # until the next ``calc_UB`` (:issue:`391`).
        self._ub_snapshot = None

    def _compute_ub_snapshot(self) -> Optional[tuple]:
        """
        Return a comparable snapshot of the current orientation-reflection state.

        The snapshot captures the two orienting reflection names plus the
        physics-relevant fields of those reflections (``geometry``,
        ``pseudos``, ``reals``, ``reals_units``, ``wavelength``,
        ``wavelength_units``).  Any change to those values invalidates a
        previously computed UB.  Returns ``None`` when fewer than two
        reflections are available (UB is not computable) or when an
        ordered name no longer maps to a known reflection.
        """
        order = self.reflections.order
        if len(order) < 2:
            return None
        head = tuple(order[:2])
        try:
            contents = tuple(
                tuple(
                    (field, self.reflections[name]._asdict()[field])
                    for field in _UB_REFLECTION_FIELDS
                )
                for name in head
            )
        except KeyError:
            # ``order`` references a name that is no longer in the dict.
            return None
        return (head, contents)

    @property
    @versionadded(
        version="0.6.2",
        reason=(
            "Detect when the orientation reflections have changed since "
            "the last ``calc_UB``, indicating the stored UB no longer "
            "reflects the chosen orienting pair.  See :issue:`391`."
        ),
    )
    def UB_is_stale(self) -> bool:
        """
        ``True`` iff the orientation reflections have changed since the
        last successful ``calc_UB`` for this sample.

        Returns ``False`` in any of these cases:

        * ``calc_UB`` has not been called for this sample (no snapshot
          to compare against),
        * the user has assigned :attr:`U` or :attr:`UB` directly (they
          have explicitly taken ownership of the matrix; the snapshot
          is cleared),
        * the orientation reflections (``reflections.order[:2]`` and
          their physics-relevant contents) are unchanged since the last
          ``calc_UB``.

        Returns ``True`` when ``order[:2]`` itself changes (reorder,
        prepend a new orienting reflection, remove an orienting one) or
        when an in-place mutation of one of the orienting reflections
        changes its ``pseudos``, ``reals``, ``wavelength``, or unit
        fields.  Reflections outside ``order[:2]`` do not affect
        staleness.
        """
        if self._ub_snapshot is None:
            return False
        return self._compute_ub_snapshot() != self._ub_snapshot
