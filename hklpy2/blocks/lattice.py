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

from ..misc import INTERNAL_ANGLE_UNITS
from ..misc import INTERNAL_LENGTH_UNITS
from ..misc import LatticeError
from ..misc import compare_float_dicts
from ..misc import convert_units
from ..misc import validate_and_canonical_unit

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

    If only the parameter ``a`` is given, the cell is treated as cubic: ``b``
    and ``c`` are set equal to ``a``, and α, β, γ are 90°. Supplying the
    nonredundant parameters for another crystal system (for example, the
    hexagonal case below) defines that lattice.

    EXAMPLE::

        >>> from hklpy2.blocks.lattice import Lattice
        >>> Lattice(4.74, c=9.515, gamma=120)
        Lattice(a=4.74, c=9.515, gamma=120, system='hexagonal')

    PARAMETERS

    a, b, c : float
        Unit cell edge lengths (default units: Angstrom unless length_units specified)
    alpha, beta, gamma : float
        Unit cell angles (default units: degrees unless angle_units specified)
    length_units : str, optional
        Units for unit cell lengths (e.g., 'angstrom', 'nm', 'pm')
    angle_units : str, optional
        Units for unit cell angles (e.g., 'degrees', 'radians')
    digits : int, optional
        Number of digits to display.  (default: 4)

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
        length_units: Optional[str] = None,
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
        self.angle_units = angle_units or INTERNAL_ANGLE_UNITS
        self.digits = digits or DEFAULT_LATTICE_DIGITS

    def __eq__(self, latt):
        """
        Compare two lattices for equality.

        Equality is defined by the six canonical lattice parameters
        (a, b, c, alpha, beta, gamma).  This method attempts to convert
        lengths and angles to the internal units before comparison.  If
        conversion fails for any reason, it falls back to a raw numeric
        comparison of the parameters.

        EXAMPLE::

            lattice1 == lattice2
        """
        if not isinstance(latt, self.__class__):
            return False
        digits = min(self.digits, latt.digits)

        keys = "a b c alpha beta gamma".split()
        # Prepare dicts of values for conversion
        vals_self = {k: getattr(self, k) for k in keys}
        vals_other = {k: getattr(latt, k) for k in keys}

        try:
            # Convert lengths to internal length units
            for k in ("a", "b", "c"):
                vals_self[k] = convert_units(
                    vals_self[k], self.length_units, INTERNAL_LENGTH_UNITS
                )
                vals_other[k] = convert_units(
                    vals_other[k], latt.length_units, INTERNAL_LENGTH_UNITS
                )
            # Convert angles to internal angle units
            for k in ("alpha", "beta", "gamma"):
                vals_self[k] = convert_units(
                    vals_self[k], self.angle_units, INTERNAL_ANGLE_UNITS
                )
                vals_other[k] = convert_units(
                    vals_other[k], latt.angle_units, INTERNAL_ANGLE_UNITS
                )
        except Exception:
            # Fallback: use raw attribute values (no unit conversion)
            vals_self = {k: getattr(self, k) for k in keys}
            vals_other = {k: getattr(latt, k) for k in keys}

        return compare_float_dicts(vals_self, vals_other, digits)

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
            "digits": self.digits,
            "angle_units": self.angle_units,
            "length_units": self.length_units,
        }

    def _fromdict(self, config):
        """Redefine lattice from a (configuration) dictionary."""
        for k in "a b c alpha beta gamma".split():
            setattr(self, k, config[k])

        # Restore optional properties if present
        if "digits" in config:
            self.digits = config["digits"]
        if "length_units" in config:
            self.length_units = config["length_units"]
        if "angle_units" in config:
            self.angle_units = config["angle_units"]

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
    def angle_units(self) -> str:
        """Units for lattice angles (e.g. 'degrees')."""
        return self._angle_units

    @angle_units.setter
    def angle_units(self, value: str) -> None:
        # Validate and canonicalize units using project's helper (raises ValueError)
        canon = validate_and_canonical_unit(value, INTERNAL_ANGLE_UNITS)
        self._angle_units = canon

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
        # Validate and canonicalize units using project's helper (raises ValueError)
        canon = validate_and_canonical_unit(value, INTERNAL_LENGTH_UNITS)
        self._length_units = canon
