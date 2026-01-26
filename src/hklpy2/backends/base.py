"""
Abstract base class for all solvers.

.. autosummary::

    ~SolverBase
"""

import logging
from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from pyRestTable import Table

from ..misc import ANTIGRAVITY_DIRECTION
from ..misc import FORWARD_DIRECTION
from ..misc import IDENTITY_MATRIX_3X3
from ..misc import INTERNAL_ANGLE_UNITS
from ..misc import INTERNAL_LENGTH_UNITS
from ..misc import CoordinateSystem
from ..misc import KeyValueMap
from ..misc import Matrix3x3
from ..misc import NamedFloatDict
from ..misc import istype
from ..misc import validate_and_canonical_unit

logger = logging.getLogger(__name__)


class SolverBase(ABC):
    """
    Base class for all |hklpy2| |solver| classes.

    PARAMETERS

    geometry : str
        Name of geometry.
    mode: str
        Name of operating mode.  (default: current mode)

    Example::

        import hklpy2

        class MySolver(hklpy2.SolverBase):
            ...

    .. note:: :class:`~SolverBase`, an `abstract base
        class <https://docs.python.org/3/library/abc.html#abc.ABC>`_,
        cannot not be used directly by |hklpy2| users.

    As the parent class for all custom :index:`Solver` classes,
    :class:`~SolverBase` defines the methods and attributes to be written
    that will connect |hklpy2| with the support library that defines
    specific diffractometer geometries and the computations for
    using them.  Subclasses should implement each of these methods
    as best fits the underlying support library.

    .. seealso:: :mod:`~hklpy2.backends.hkl_soleil` & :mod:`~hklpy2.backends.no_op`

    .. rubric:: Python Abstract Methods

    Subclasses must override each of these methods.

    .. autosummary::

        ~addReflection
        ~calculate_UB
        ~extra_axis_names
        ~forward
        ~geometries
        ~inverse
        ~modes
        ~pseudo_axis_names
        ~real_axis_names
        ~refineLattice
        ~removeAllReflections

    .. rubric:: Python Properties

    .. autosummary::

        ~all_extra_axis_names
        ~extra_axis_names
        ~extras
        ~geometry
        ~lattice
        ~mode
        ~sample
        ~UB
    """

    from .. import __version__

    name: str = "base"
    """Name of this Solver."""

    version: str = __version__
    """Version of this Solver."""

    ANGLE_UNITS: str = "degrees"
    """
    Angle units used by this solver for unit cell and real axis rotations.

    Solver can override this **constant**.  Must be convertible to
    ``INTERNAL_ANGLE_UNITS``.
    """

    LENGTH_UNITS: str = "angstrom"
    """
    Length units used by this solver for unit cell and wavelength.

    Solver can override this **constant**.  Must be convertible to
    ``INTERNAL_LENGTH_UNITS``.
    """

    def __init__(
        self,
        geometry: str,
        *,
        mode: str = "",  # "": accept solver's default mode
        **kwargs: Any,
    ) -> None:
        self._gname: str = geometry
        self.mode = mode
        self._all_extra_axis_names: Optional[List[str]] = None
        self._sample: Optional[KeyValueMap] = None

        self._coordinate_system = self.define_coordinate_framework()

        validate_and_canonical_unit(self.ANGLE_UNITS, INTERNAL_ANGLE_UNITS)
        validate_and_canonical_unit(self.LENGTH_UNITS, INTERNAL_LENGTH_UNITS)

        logger.debug("geometry=%s, kwargs=%s", repr(geometry), repr(kwargs))

    def __repr__(self) -> str:
        # fmt: off
        args = [
            f"{s}={getattr(self, s)!r}"
            for s in "name version geometry".split()
        ]
        # fmt: on
        return f"{self.__class__.__name__}({', '.join(args)})"

    @property
    def _metadata(self) -> KeyValueMap:
        """Dictionary with this solver's summary metadata."""
        return {
            "name": self.name,
            "description": repr(self),
            "geometry": self.geometry,
            "real_axes": self.real_axis_names,
            "version": self.version,
        }

    @abstractmethod
    def addReflection(self, reflection: KeyValueMap) -> None:
        """Add coordinates of a diffraction condition (a reflection)."""

    @property
    def all_extra_axis_names(self) -> List[str]:
        """Unique, sorted list of extra axis names in all modes for chosen engine."""
        if self._all_extra_axis_names is None:
            # Only collect this once.
            original = self.mode
            names: List[str] = []
            for mode in self.modes:
                self.mode = mode
                names += self.extra_axis_names
            self.mode = original  # put it back
            self._all_extra_axis_names = sorted(list(set(names)))
        return self._all_extra_axis_names

    @abstractmethod
    def calculate_UB(
        self,
        r1: KeyValueMap,
        r2: KeyValueMap,
    ) -> Matrix3x3:
        """
        Calculate the UB (orientation) matrix with two reflections.

        The method of Busing & Levy, Acta Cryst 22 (1967) 457.
        """
        # return self.UB

    def define_coordinate_framework(self):
        """Define the coordinate framework."""
        return CoordinateSystem(
            # same as hklpy2
            vx=np.cross(ANTIGRAVITY_DIRECTION, FORWARD_DIRECTION),
            vy=ANTIGRAVITY_DIRECTION,
            vz=FORWARD_DIRECTION,
        )

    @property
    @abstractmethod
    def extra_axis_names(self) -> List[str]:
        """Ordered list of any extra axis names (such as x, y, z)."""
        # Do NOT sort.
        # return []

    @property
    def extras(self) -> KeyValueMap:
        """
        Ordered dictionary of any extra parameters.
        """
        return {}

    @abstractmethod
    def forward(self, pseudos: NamedFloatDict) -> List[NamedFloatDict]:
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        # based on geometry and mode
        # return [{}]

    @classmethod
    @abstractmethod
    def geometries(cls) -> List[str]:
        """
        Ordered list of the geometry names.

        EXAMPLES::

            >>> from hklpy2 import get_solver
            >>> Solver = get_solver("no_op")
            >>> Solver.geometries()
            []
            >>> solver = Solver("TH TTH Q")
            >>> solver.geometries()
            []
        """
        # return []

    @property
    def geometry(self) -> str:
        """
        Name of selected diffractometer geometry.

        Cannot be changed once solver is created.  Instead, make a new solver
        for each geometry.
        """
        return self._gname

    @abstractmethod
    def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
        """Compute dict of pseudos from reals (angles -> hkl)."""
        # return {}

    @property
    def lattice(self) -> NamedFloatDict:
        """
        Crystal lattice parameters.  (Not used by this |solver|.)
        """
        return self._lattice

    @lattice.setter
    def lattice(self, value: NamedFloatDict) -> None:
        if not istype(value, NamedFloatDict):
            raise TypeError(f"Must supply {NamedFloatDict} object, received {value!r}")
        self._lattice = value

    @property
    def mode(self) -> str:
        """
        Diffractometer geometry operation mode for :meth:`forward()`.

        A mode defines which axes will be modified by the
        :meth:`forward` computation.
        """
        try:
            self._mode
        except AttributeError:
            self._mode = ""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        from ..misc import check_value_in_list  # avoid circular import here

        check_value_in_list("Mode", value, self.modes, blank_ok=True)
        self._mode = value

    @property
    @abstractmethod
    def modes(self) -> List[str]:
        """List of the geometry operating modes."""
        # return []

    @property
    @abstractmethod
    def pseudo_axis_names(self) -> List[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        # Do NOT sort.
        # return []

    @property
    @abstractmethod
    def real_axis_names(self) -> List[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        # Do NOT sort.
        # return []

    @abstractmethod
    def refineLattice(self, reflections: List[KeyValueMap]) -> NamedFloatDict:
        """Refine the lattice parameters from a list of reflections."""

    @abstractmethod
    def removeAllReflections(self) -> None:
        """Remove all reflections."""

    @property
    def sample(self) -> Union[KeyValueMap, None]:
        """
        Crystalline sample.
        """
        return self._sample

    @sample.setter
    def sample(self, value: KeyValueMap) -> None:
        if not istype(value, KeyValueMap):
            raise TypeError(f"Must supply dictionary, received {value!r}")
        self._sample = value

    @property
    def _summary_dict(self) -> KeyValueMap:
        """Return a summary of the geometry (modes, axes)"""
        geometry_name = self.geometry
        description: KeyValueMap = {
            "name": geometry_name,
            "pseudos": self.pseudo_axis_names,
            "reals": self.real_axis_names,
            "modes": {},
        }

        for mode in self.modes:
            self.mode = mode
            desc: KeyValueMap = {
                "extras": [],
                # the reals to be written in this mode (solver should override)
                "reals": self.real_axis_names,
            }
            description["modes"][mode] = desc

        return description

    @property
    def summary(self) -> Table:
        """
        Table of this geometry (modes, axes).

        .. seealso:: :ref:`geometries.summary_tables`,
            :func:`hklpy2.user.solver_summary()`
        """
        table = Table()
        table.labels = "mode pseudo(s) real(s) writable(s) extra(s)".split()
        sdict = self._summary_dict
        for mode_name, mode in sdict["modes"].items():
            self.mode = mode_name
            row = [
                mode_name,
                ", ".join(sdict["pseudos"]),
                ", ".join(sdict["reals"]),
                ", ".join(mode["reals"]),
                ", ".join(mode["extras"]),
            ]
            table.addRow(row)
        return table

    @property
    def UB(self) -> Matrix3x3:
        """Orientation matrix (3x3)."""
        return IDENTITY_MATRIX_3X3
