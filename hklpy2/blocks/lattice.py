"""
Lattice parameters for a single crystal.

.. autosummary::

    ~DEFAULT_LATTICE_DIGITS
    ~Lattice
    ~SI_LATTICE_PARAMETER
    ~SI_LATTICE_PARAMETER_UNCERTAINTY
"""

import enum
import logging
import math
from typing import Optional

import pint

from ..misc import INTERNAL_ANGLE_UNITS, INTERNAL_LENGTH_UNITS, LatticeError
from ..misc import compare_float_dicts

logger = logging.getLogger(__name__)

DEFAULT_LATTICE_DIGITS = 4
"""Default number of digits to display for lattice parameters."""

SI_LATTICE_PARAMETER = 5.431020511
"""
2018 CODATA recommended lattice parameter of silicon, Angstrom.

:see: https://physics.nist.gov/cgi-bin/cuu/Value?asil
"""

SI_LATTICE_PARAMETER_UNCERTAINTY = 0.000000089
"""
2018 CODATA reported uncertainty of :data:`SI_LATTICE_PARAMETER`.
"""


CrystalSystem = enum.Enum(  # in order from lowest symmetry
    "CrystalSystem",
    """
        triclinic
        monoclinic
        orthorhombic
        tetragonal
        rhombohedral
        hexagonal
        cubic
    """.split(),
)


class Lattice:
    """
    Crystal lattice parameters.

    EXAMPLE::

        >>> from hklpy2.blocks.lattice import Lattice
        >>> Lattice(4.74, c=9.515, gamma=120)
        Lattice(a=4.74, c=9.515, gamma=120, system='hexagonal')

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~__eq__
        ~__repr__
        ~crystal_system
        ~system_parameter_names
    """

    def __init__(
        self,
        a: float,
        b: float = None,
        c: float = None,
        alpha: float = 90.0,  # degrees
        beta: float = None,  # degrees
        gamma: float = None,  # degrees
        *,
        # TODO: Review and update support for length_units throughout entire project.
        # TODO: Add getter/setter property methods here for length_units
        # TODO: Add pytests for length_units.
        length_units: Optional[str] = None,
        # FIXME 136  lattice should define its angle units
        angle_units: Optional[str] = None,
        digits: Optional[int] = None,
    ):
        """Initialize lattice parameters.

        Parameters
        ----------
        a, b, c : float
            Unit cell edge lengths (default units: Angstrom unless length_units specified)
        alpha, beta, gamma : float
            Unit cell angles (default units: degrees unless angle_units specified)
        length_units : str, optional
            Units for lattice lengths (e.g., 'angstrom', 'nm', 'pm')
        angle_units : str, optional
            Units for lattice angles (not yet implemented)
        digits : int, optional
            Number of digits to display.  (default: 4)
        """
        self.a = a
        self.b = b or a
        self.c = c or a
        self.alpha = alpha
        self.beta = beta or alpha
        self.gamma = gamma or alpha
        self.length_units = length_units or INTERNAL_LENGTH_UNITS
        self.angle_units = angle_units or INTERNAL_ANGLE_UNITS  # FIXME 136
        self.digits = digits or DEFAULT_LATTICE_DIGITS

    def __eq__(self, latt):
        """
        Compare two lattices for equality.

        EXAMPLE::

            lattice1 == lattice2
        """
        if not isinstance(latt, self.__class__):
            return False
        digits = min(self.digits, latt.digits)
        return compare_float_dicts(
            self._asdict(), latt._asdict(), min(self.digits, digits)
        )

    def __repr__(self):
        """
        Standard representation of lattice.
        """
        system = self.crystal_system
        parm_names = self.system_parameter_names(system)
        parameters = [
            f"{k}={round(v, self.digits)}"
            for k, v in self._asdict().items()
            if k in parm_names
        ]
        parameters.append(f"{system=!r}")
        return f"{self.__class__.__name__}({', '.join(parameters)})"

    def _asdict(self):
        """Return a new dict which maps lattice constant names and values."""
        # note: name is identical to namedtuple._asdict method
        return {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            # "digits": self.digits,
        }

    def _fromdict(self, config):
        """Redefine lattice from a (configuration) dictionary."""
        for k in "a b c alpha beta gamma".split():
            setattr(self, k, config[k])

    def system_parameter_names(self, system: str):
        """Return list of lattice parameter names for this crystal system."""
        all = "a b c alpha beta gamma".split()
        return {
            "cubic": ["a"],
            "hexagonal": "a c gamma".split(),
            "rhombohedral": "a alpha".split(),
            "tetragonal": "a c".split(),
            "orthorhombic": "a b c".split(),
            "monoclinic": "a b c beta".split(),
            "triclinic": all,
        }.get(system, all)

    # ---- get/set properties

    @property
    def crystal_system(self):
        """
        The crystal system of this lattice.  By inspection of the parameters.

        .. seealso:: https://dictionary.iucr.org/Crystal_system
        """

        def very_close(value, ref, tol=1e-7):
            return math.isclose(value, ref, abs_tol=tol)

        def angles(alpha, beta, gamma):
            return (
                very_close(self.alpha, alpha)
                and very_close(self.beta, beta)
                and very_close(self.gamma, gamma)
            )

        def edges(a, b, c):
            return (
                very_close(self.a, a)
                and very_close(self.b, b)
                and very_close(self.c, c)
            )

        def all_angles(ref):
            return angles(ref, ref, ref)

        def all_edges(ref):
            return edges(ref, ref, ref)

        # filter by testing symmetry elements from lowest system first
        if not very_close(self.alpha, 90) and not very_close(self.alpha, self.beta):
            # no need to compare alpha != gamma
            return CrystalSystem.triclinic.name

        if very_close(self.alpha, 90) and not very_close(self.alpha, self.beta):
            return CrystalSystem.monoclinic.name

        if all_angles(90) and not very_close(self.a, self.b):
            return CrystalSystem.orthorhombic.name

        if (
            all_angles(90)
            and very_close(self.a, self.b)
            and not very_close(self.a, self.c)
        ):
            return CrystalSystem.tetragonal.name

        if (
            not very_close(self.alpha, 90)
            and all_angles(self.alpha)
            and all_edges(self.a)
        ):
            return CrystalSystem.rhombohedral.name

        if (
            angles(90, 90, 120)
            and very_close(self.a, self.b)
            and not very_close(self.a, self.c)
        ):
            return CrystalSystem.hexagonal.name

        if all_angles(90) and all_edges(self.a):
            return CrystalSystem.cubic.name

        raise LatticeError(f"Unrecognized crystal system: {self._asdict()!r}")

    @property
    def digits(self) -> int:
        """Number of digits to display."""
        return self._digits

    @digits.setter
    def digits(self, value: int):
        self._digits = value

    @property
    def length_units(self) -> str:
        """Units for lattice lengths (e.g. 'angstrom')."""
        return self._length_units

    @length_units.setter
    def length_units(self, value: str) -> None:
        # Ensure that new value is convertible to the internal length units.
        # Use pint to validate units; an exception will be raised for invalid units.
        assert pint.UnitRegistry().convert(1, value, INTERNAL_LENGTH_UNITS)
        self._length_units = value
