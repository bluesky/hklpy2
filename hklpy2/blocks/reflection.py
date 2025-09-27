# pylint: disable=too-many-arguments
"""
Coordinates of a crystalline reflection.

Associates diffractometer angles (real-space) with crystalline reciprocal-space
(pseudo) coordinates.

.. autosummary::

    ~Reflection
    ~ReflectionError
    ~ReflectionsDict
    ~UNUSED_REFLECTION
"""

import logging
from typing import Optional

import pint

from ..misc import INTERNAL_WAVELENGTH_UNITS
from ..misc import ConfigurationError
from ..misc import ReflectionError
from ..misc import check_value_in_list
from ..misc import compare_float_dicts

logger = logging.getLogger(__name__)

DEFAULT_REFLECTION_DIGITS = 4

UNUSED_REFLECTION = "unused"
"""Identifies an unused reflection in the ReflectionsDict."""


class Reflection:
    """
    Coordinates real and pseudo axes.

    Two reflections can be added, subtracted, or compared for equality.

    .. note:: Internal use only.

       It is expected this class is called from a method of
       :class:`~hklpy2.ops.Core`, not directly by the user.

    .. rubric:: Parameters

    * ``name`` (str): Reference name for this reflection.
    * ``pseudos`` (dict): Unordered dictionary of pseudo-space axes and values.
    * ``reals`` (dict): Unordered dictionary of real-space axes and values.
    * ``wavelength`` (float): Wavelength of incident radiation.
    * ``geometry`` (str): Geometry name for this reflection.
    * ``pseudo_names`` ([str]): Ordered list of pseudo names for this geometry.
    * ``rnames`` ([str]): Ordered list of real names for this geometry.

    Optional items (such as 'azimuth', 'h1', 'h2', zones, ...) are not
    part of a "reflection".

    .. autosummary::

        ~__add__
        ~__eq__
        ~__sub__
        ~_asdict
        ~_fromdict
        ~_validate_pseudos
        ~_validate_reals
        ~_validate_wavelength
        ~pseudos
        ~reals
        ~wavelength
    """

    def __init__(
        self,
        name: str,
        pseudos: dict,
        reals: dict,
        wavelength: float,
        geometry: str,
        pseudo_axis_names: list,
        real_axis_names: list,
        *,
        core: Optional[object] = None,
        digits: Optional[int] = None,
        wavelength_units: str = None,
    ) -> None:
        from ..ops import Core

        if isinstance(core, Core):
            # What if axes names in wrong sequence?  Required order is assumed.
            # What if axes renamed?  All reflections must use the same real_axis_names.
            axes_local = core.diffractometer.real_axis_names
            axes_solver = core.solver.real_axis_names
            if real_axis_names not in (axes_local, axes_solver):
                raise ReflectionError(
                    f"{real_axis_names=}"
                    f" do not match diffractometer ({axes_local})"
                    f" or solver ({axes_solver})."
                )

        self.digits = DEFAULT_REFLECTION_DIGITS if digits is None else digits
        self.geometry = geometry
        self.name = name
        self.pseudo_axis_names = pseudo_axis_names
        self.real_axis_names = real_axis_names

        # property setters
        self.pseudos = pseudos
        self.reals = reals
        self.wavelength = wavelength
        self.wavelength_units = wavelength_units or INTERNAL_WAVELENGTH_UNITS

    def __add__(self, other):
        """
        Add a Reflection object to this one and return as a new Reflection.

        Combines the pseudos and reals of self and other.
        """
        if not isinstance(other, Reflection):
            raise TypeError(
                "Unsupported operand type(s) for +: 'Reflection'"
                #
                f" and '{type(other).__name__}'"
            )

        # Create a new Reflection with combined pseudo and real values.
        new_name = f"{self.name}_plus_{other.name}"
        new_pseudos = {
            key: self.pseudos[key] + other.pseudos[key]
            #
            for key in self.pseudos
        }
        new_reals = {key: self.reals[key] + other.reals[key] for key in self.reals}
        return Reflection(
            name=new_name,
            pseudos=new_pseudos,
            reals=new_reals,
            wavelength=self.wavelength,  # Preserve wavelength from self.
            geometry=self.geometry,
            pseudo_axis_names=self.pseudo_axis_names,
            real_axis_names=self.real_axis_names,
        )

    def __eq__(self, r2):
        """
        Compare this reflection with another for equality.

        Precision is controlled by rounding to smallest number of digits
        between the reflections.
        """
        from ..misc import convert_units

        digits = min(self.digits, r2.digits)
        pseudos_ok = compare_float_dicts(self.pseudos, r2.pseudos, digits)
        reals_ok = compare_float_dicts(self.reals, r2.reals, digits)
        # Convert r2 wavelength to this reflection's units before comparing
        try:
            r2_wl_in_self_units = convert_units(
                r2.wavelength, r2.wavelength_units, self.wavelength_units
            )
        except Exception:
            # If conversion fails, fall back to raw comparison (will likely fail)
            r2_wl_in_self_units = r2.wavelength
        wavelength_ok = round(self.wavelength, digits) == round(r2_wl_in_self_units, digits)
        return pseudos_ok and reals_ok and wavelength_ok

    def __repr__(self):
        """
        Standard brief representation of reflection.
        """
        pseudos = [
            f"{k}={round(v, self.digits)}"  # roundoff
            #
            for k, v in self.pseudos.items()
        ]
        guts = [f"name={self.name!r}"] + pseudos
        return f"{self.__class__.__name__}({', '.join(guts)})"

    def __sub__(self, other):
        """
        Subtract another Reflection from this one and return as a new Reflection.

        Subtracts the pseudos and reals of other from self.
        """
        if not isinstance(other, Reflection):
            raise TypeError(
                "Unsupported operand type(s) for -: 'Reflection' "
                #
                f"and '{type(other).__name__}'"
            )

        # Create a new Reflection with subtracted pseudo and real values.
        new_name = f"{self.name}_minus_{other.name}"
        new_pseudos = {
            key: self.pseudos[key] - other.pseudos[key]
            #
            for key in self.pseudos
        }
        new_reals = {key: self.reals[key] - other.reals[key] for key in self.reals}
        return Reflection(
            name=new_name,
            pseudos=new_pseudos,
            reals=new_reals,
            wavelength=self.wavelength,  # Preserve wavelength from self.
            geometry=self.geometry,
            pseudo_axis_names=self.pseudo_axis_names,
            real_axis_names=self.real_axis_names,
        )

    def _asdict(self):
        """Describe this reflection as a dictionary."""
        return {
            "name": self.name,
            "geometry": self.geometry,
            "pseudos": self.pseudos,
            "reals": self.reals,
            "wavelength": self.wavelength,
            "wavelength_units": self.wavelength_units,
            "digits": self.digits,
        }

    def _fromdict(self, config):
        """Redefine this reflection from a (configuration) dictionary."""
        if config.get("name") != self.name:
            raise ConfigurationError(
                f"Mismatched name for reflection {self.name!r}."
                #
                f" Received configuration: {config!r}"
            )
        if config.get("geometry") != self.geometry:
            raise ConfigurationError(
                f"Mismatched geometry for reflection {self.name!r}."
                f" Expected geometry: {self.geometry!r}."
                f" Received configuration: {config!r}"
            )
        if list(self.pseudos) != list(config["pseudos"]):
            raise ConfigurationError(
                f"Mismatched pseudo axis names for reflection {self.name!r}."
                f" Expected: {list(self.pseudos)!r}."
                f" Received: {list(config['pseudos'])!r}"
            )
        if list(self.reals) != list(config["reals"]):
            raise ConfigurationError(
                f"Mismatched real axis names for reflection {self.name!r}."
                f" Expected: {list(self.reals)!r}."
                f" Received: {list(config['reals'])!r}"
            )

        self.digits = config.get("digits", self.digits)
        # accept explicit wavelength_units if present
        if "wavelength_units" in config:
            self.wavelength_units = config.get("wavelength_units")
        self.wavelength = config.get("wavelength", self.wavelength)
        self.pseudos = config["pseudos"]
        self.reals = config["reals"]

    def _validate_pseudos(self, value):
        """Raise Exception if pseudos do not match expectations."""
        if not isinstance(value, dict):
            raise TypeError(f"Must supply dict, received pseudos={value!r}")
        for key in value:
            check_value_in_list("pseudo axis", key, self.pseudo_axis_names)
        for key in self.pseudo_axis_names:
            if key not in value:
                # fmt: off
                raise ReflectionError(
                    f"Missing pseudo axis {key!r}."
                    f" Required names: {self.pseudo_axis_names!r}"
                )
            # fmt: on

    def _validate_reals(self, value):
        """Raise Exception if reals do not match expectations."""
        if not isinstance(value, dict):
            raise TypeError(f"Must supply dict, received reals={value!r}")
        for key in value:
            check_value_in_list("real axis", key, self.real_axis_names)
        for key in self.real_axis_names:
            if key not in value:
                # fmt: off
                raise ReflectionError(
                    f"Missing real axis {key!r}."
                    f" Required names: {self.real_axis_names!r}"
                )
            # fmt: on

    def _validate_wavelength(self, value):
        """Raise exception if pseudos do not match expectations."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"Must supply number, received {value=!r}")
        if value <= 0:
            raise ValueError(f"Must be >=0, received {value=}")

    # --------- get/set properties

    @property
    def name(self):
        """Sample name."""
        return self._name

    @name.setter
    def name(self, value):
        if not isinstance(value, str):
            raise TypeError(f"Must supply str, received name={value!r}")
        self._name = value

    @property
    def pseudos(self):
        """
        Ordered dictionary of diffractometer's reciprocal-space axes.

        Dictionary keys are the axis names, as defined by the diffractometer.
        """
        return self._pseudos

    @pseudos.setter
    def pseudos(self, values):
        self._validate_pseudos(values)
        self._pseudos = values

    @property
    def reals(self):
        """
        Ordered dictionary of diffractometer's real-space axes.

        Dictionary keys are the axis names, as defined by the diffractometer.
        """
        return self._reals

    @reals.setter
    def reals(self, values):
        self._validate_reals(values)
        self._reals = values

    @property
    def wavelength(self):
        """Wavelength of reflection."""
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._validate_wavelength(value)
        self._wavelength = value

    @property
    def wavelength_units(self) -> str:
        """Engineering units of this reflection's wavelength."""
        return self._wavelength_units

    @wavelength_units.setter
    def wavelength_units(self, value: str) -> None:
        # Ensure that new value is convertible to the internal wavelength units.
        assert pint.UnitRegistry().convert(1, value, INTERNAL_WAVELENGTH_UNITS)
        self._wavelength_units = value


class ReflectionsDict(dict):
    """
    Dictionary of Reflections.

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~_validate_reflection
        ~add
        ~order
        ~prune
        ~set_orientation_reflections
        ~setor
        ~swap
    """

    # TODO 137 wavelength_units

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._order = []
        self.geometry = None

    def _asdict(self):
        """
        Describe the reflections list as an ordered dictionary.

        Order by reflections order.
        """
        self.prune()
        return {v.name: v._asdict() for v in self.values()}

    def _fromdict(self, config, core=None):
        """Add or redefine reflections from a (configuration) dictionary."""
        from ..ops import Core

        for refl_config in config.values():
            if isinstance(core, Core):
                # Remap the names of all the real axes to the current solver.
                # Real axes MUST be specified in the order specified by the solver.
                refl_config["reals"] = {
                    axis: value
                    for axis, value in zip(
                        # core.diffractometer.real_axis_names,
                        core.solver.real_axis_names,
                        refl_config["reals"].values(),
                    )
                }

            reflection = Reflection(
                refl_config["name"],
                refl_config["pseudos"],
                    refl_config["reals"],
                    wavelength=refl_config["wavelength"],
                    wavelength_units=refl_config.get("wavelength_units"),
                geometry=refl_config["geometry"],
                pseudo_axis_names=list(refl_config["pseudos"]),
                real_axis_names=list(refl_config["reals"]),
                digits=refl_config.get("digits"),
                core=core,
            )
            self.add(reflection, replace=True)

    def set_orientation_reflections(
        self,
        reflections: list[Reflection],
    ) -> None:
        """
        Designate the order of the reflections to be used.

        .. note:: Raises ``KeyError`` if any
           ``reflections`` are not already defined.

           This method does not *add* any new reflections.

        .. rubric:: Parameters

        * ``reflections`` ([Reflection]) : List of
          :class:`hklpy2.blocks.reflection.Reflection` objects.
        """
        self.order = [r.name for r in reflections]

    setor = set_orientation_reflections
    """Common alias for :meth:`~set_orientation_reflections`."""

    def add(self, reflection: Reflection, replace: bool = False) -> None:
        """Add a single orientation reflection."""
        self._validate_reflection(reflection, replace)

        self[reflection.name] = reflection
        if reflection.name not in self.order:
            self.order.append(reflection.name)
        self.prune()

    def prune(self):
        """Remove any undefined reflections from order list."""
        self.order = [refl for refl in self.order if refl in self]

    def swap(self):
        """Swap the first two orientation reflections."""
        if len(self.order) < 2:
            raise ReflectionError("Need at least two reflections to swap.")
        rname1, rname2 = self.order[:2]
        self._order[0] = rname2
        self._order[1] = rname1
        return self.order

    def _validate_reflection(self, reflection, replace):
        """Validate the new reflection."""
        if not isinstance(reflection, Reflection):
            raise TypeError(
                #
                f"Unexpected {reflection=!r}.  Must be a 'Reflection' type."
            )

        # matching content
        matching = [v.name for v in self.values() if v == reflection]
        if reflection.name in self:
            # matching name
            if reflection.name not in matching:
                matching.append(reflection.name)

        if replace:
            # remove ALL matches (name or content matches)
            for nm in matching:
                r = self.pop(nm)
                logger.debug("Replacing known reflection %r", r)
            matching = []
        if len(matching) > 0:  # still?
            if reflection.name in matching:
                raise ReflectionError(
                    f"Reflection name {reflection.name!r} is known."
                    #
                    "  Use 'replace=True' to overwrite."
                )
            else:
                raise ReflectionError(
                    f"Reflection {reflection!r} matches one or more"
                    " existing reflections.  Use 'replace=True' to overwrite."
                )

        if self.geometry is None or len(self) == 0:
            self.geometry = reflection.geometry

        if reflection.geometry != self.geometry:
            # fmt: off
            raise ValueError(
                "geometry does not match previous reflections:"
                f" received {reflection.geometry!r}"
                f" previous: {self.geometry!r}."
            )
            # fmt: on

    # ---- get/set properties

    @property
    def order(self):
        """Ordered list of reflection names used for orientation."""
        return self._order

    @order.setter
    def order(self, value):
        self._order = list(value)
