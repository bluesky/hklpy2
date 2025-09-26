"""
Wavelength of the monochromatic source radiation.

.. autosummary::

    ~_WavelengthBase
    ~Wavelength
    ~WavelengthXray
    ~EpicsWavelengthRO
    ~EpicsMonochromatorRO

.. rubric:: Which class to use?

:class:`EpicsMonochromatorRO`
    Such as synchrotron X-ray monochromator. Wavelength and Energy provided by
    EPICS PVs.  Use EPICS tools to change energy or wavelength.
:class:`EpicsWavelengthRO`
    Wavelength provided by an EPICS PV.  Such as X-ray source or reactor neutron
    source using helical velocity selector.  Use EPICS tools to change
    wavelength.
`Wavelength`
    Constant wavelength sources, such as X-ray tube or rotating anode.
`WavelengthXray`
    Changeable-wavelength X-ray sources, such as testing, simulation, or when no
    EPICS PVs are available.

The EPICS-related classes here have read-only support for wavelength (and
associated energy). Control of the EPICS PVs is beyond the scope of
diffractometer controls. Refer to the EPICS controls for the monochromator or
wavelength PV. Or, create a subclass of
:class:`~hklpy2.incident._WavelengthBase()`.

The :class:`hklpy2.ops.Core()` class is responsible for converting
the engineering units.

.. note:: While the *energy* of the incident beam may be interesting to
    diffractometer users at X-ray synchrotrons, *wavelength* is the general term
    used by both neutron and X-ray diffraction science.  Some classes provide
    for energy as it is in common use with diffractometers at X-ray synchrotrons.

    A similar handling could be made for monochromators with other types of
    radiation such as neutrons.
"""

import atexit
import logging
import weakref

import pint
from ophyd import Component
from ophyd import Device
from ophyd import EpicsSignalRO
from ophyd import FormattedComponent as FC
from ophyd import Signal
from ophyd import SignalRO
from ophyd.signal import AttributeSignal

from .misc import DEFAULT_ENERGY_UNITS
from .misc import DEFAULT_WAVELENGTH_UNITS

logger = logging.getLogger(__name__)
DEFAULT_SOURCE_TYPE = "Synchrotron X-ray Source"
DEFAULT_WAVELENGTH = 1.0
DEFAULT_WAVELENGTH_DEADBAND = 0.000_1

XRAY_ENERGY_EQUIVALENT_ = 8.065_543_937e5
"""
Energy equivalent factor :math:`1 / (h \\nu)`

Per NIST publication, of CODATA Fundamental Physical Constants, 2022 revision.

:see: https://physics.nist.gov/cuu/Constants/factors.html ("1 eV" *v*. "1/m")
"""

A_KEV = 1e7 / XRAY_ENERGY_EQUIVALENT_  # 1 Angstrom ~= 12.39842 keV
"""
X-ray voltage wavelength product (:math:`h \\nu`), per NIST standard.
"""


class _WavelengthBase(Device):
    """
    (internal) Base for any monochromatic wavelength (:math:`\\lambda`) classes.

    In this class, wavelength is a constant.

    .. autosummary::

        ~_asdict
        ~_fromdict
    """

    source_type = Component(SignalRO, value=DEFAULT_SOURCE_TYPE, kind="config")
    """
    Description of the incident radiation.

    Defined here as metadata for scientific analyses.  Not used by hklpy2.

    Suggest using one of the types enumerated by `NeXus
    <https://manual.nexusformat.org/classes/base_classes/NXsource.html#nxsource-type-field>`_.
    """
    wavelength = Component(SignalRO, value=DEFAULT_WAVELENGTH, kind="hinted")
    """Constant wavelength (:math:`\\lambda`) of incident monochromatic beam."""
    wavelength_units = Component(
        SignalRO, value=DEFAULT_WAVELENGTH_UNITS, kind="config"
    )
    """Constant engineering units of wavelength. (Same units as unit cell lengths.)"""

    wavelength_deadband = Component(
        Signal, value=DEFAULT_WAVELENGTH_DEADBAND, kind="config"
    )
    """Allowed variation in wavelength before signaling change to diffractometer."""

    _keyset: list[str] = "source_type wavelength wavelength_units".split()
    """List of Component names for '_asdict()' and '_fromdict()'."""

    def _asdict(self) -> dict[str, (float | str)]:
        """Returns dictionary with attributes named in '_keyset'."""
        info = {"class": self.__class__.__name__}
        for attr in self._keyset:
            info[attr] = getattr(self, attr).get()
        return info

    def _fromdict(self, info: dict[str, (float | str)]) -> None:
        """Set attributes from dictionary based on keys in '_keyset'."""
        if info.get("class") == self.__class__.__name__:
            for attr, value in info.items():
                # Check first for incompatible units, before any signal.put() operations.
                if attr == "energy_units" and value is not None:
                    pint.UnitRegistry().convert(1, value, DEFAULT_ENERGY_UNITS)
                elif attr == "wavelength_units" and value is not None:
                    pint.UnitRegistry().convert(1, value, DEFAULT_WAVELENGTH_UNITS)

            for attr in self._keyset:
                value = info.get(attr)
                if value is not None:
                    signal = getattr(self, attr)
                    if signal.write_access:
                        signal.put(value)

    def __init__(
        self,
        prefix: str = "",
        *,
        source_type: str = None,
        wavelength: float = None,
        wavelength_units: str = None,
        wavelength_deadband: float = DEFAULT_WAVELENGTH_DEADBAND,
        connection_timeout: float = None,
        **kwargs,
    ):
        """."""
        super().__init__(prefix, **kwargs)

        if source_type is not None:
            self.source_type._readback = source_type

        if wavelength_units is not None:
            # validate first
            pint.UnitRegistry().convert(1, wavelength_units, DEFAULT_WAVELENGTH_UNITS)
            self.wavelength_units.put(wavelength_units)

        if wavelength is not None:
            self.wavelength.wait_for_connection(timeout=connection_timeout)
            self.wavelength.put(wavelength)

        if wavelength_deadband is not None and self.wavelength_deadband.connected:
            self.wavelength_deadband.put(wavelength_deadband)
        self._wavelength_reference = None
        self.wavelength_updated_func = None

        self.wavelength.subscribe(self.cb_wavelength)

        # cancel subscriptions before object is garbage collected
        weakref.finalize(self.wavelength, self.wavelength.unsubscribe_all)
        atexit.register(self.cleanup_subscriptions)

    def cb_wavelength(self, value, **kwargs):
        """
        Called when wavelength changes (EPICS CA monitor event) or on-demand.

        When wavelength changes more than deadband from reference, call the
        supplied function with a value of ``True``.
        """
        if self.wavelength.connected and self.wavelength_updated_func is not None:
            if self._wavelength_reference is None:
                self._wavelength_reference = value
                self.wavelength_updated_func(True)
            if abs(value - self._wavelength_reference) > self.wavelength_deadband.get():
                self._wavelength_reference = value
                self.wavelength_updated_func(True)

    def cleanup_subscriptions(self):
        """Clear subscriptions on exit."""
        self.wavelength.unsubscribe_all()


class Wavelength(_WavelengthBase):
    """
    Adjustable monochromatic wavelength (:math:`\\lambda`).

    PARAMETERS

    source_type str:
        Description of the incident radiation.

        Suggest using one of the types enumerated by `NeXus
        <https://manual.nexusformat.org/classes/base_classes/NXsource.html#nxsource-type-field>`_.

    wavelength float:
        Monochromatic wavelength of the incident radiation.  It is expected that
        wavelength and unit cell dimensions have compatible units.

    wavelength_units str:
        Constant engineering units of wavelength.  It is required that
        wavelength and unit cell dimensions have compatible units.

        Will raise ``pint.DimensionalityError`` if not convertible to
        compatible wavelength units or ``pint.UndefinedUnitError`` if
        unit string is not recognized.
    """

    wavelength = Component(Signal, value=DEFAULT_WAVELENGTH, kind="hinted")
    """Wavelength (:math:`\\lambda`) of incident monochromatic beam."""
    wavelength_units = Component(Signal, value=DEFAULT_WAVELENGTH_UNITS, kind="config")
    """Engineering units of wavelength. (Same units as unit cell lengths.)"""


class WavelengthXray(Wavelength):
    """Monochromatic X-ray wavelength and photon energy."""

    energy = Component(
        AttributeSignal, attr="_energy", kind="hinted", write_access=True
    )
    """
    Monochromatic X-ray photon energy (:math:`E`).

        .. math::

            \\lambda = (h \\nu) / E

    """
    energy_units = Component(Signal, value=DEFAULT_ENERGY_UNITS, kind="config")
    """
    Engineering units of energy.

    Will raise ``pint.DimensionalityError`` if not convertible to
    compatible wavelength units or ``pint.UndefinedUnitError`` if
    unit string is not recognized.
    """

    _keyset: list[str] = """
        source_type
        energy wavelength
        energy_units wavelength_units
        """.split()
    """List of Component names for '_asdict()' and '_fromdict()'."""

    def __init__(
        self,
        prefix: str = "",
        *,
        energy: float = None,
        energy_units: str = None,
        **kwargs,
    ):
        """."""
        super().__init__(prefix, **kwargs)
        if energy_units is not None:
            # validate first
            pint.UnitRegistry().convert(1, energy_units, DEFAULT_ENERGY_UNITS)
            self.energy_units.put(energy_units)
        if energy is not None:
            self.energy.put(energy)

    @property
    def _energy(self) -> float:
        """Return the energy, computed from wavelength, in the current units."""
        from .misc import convert_units

        wavelength = convert_units(
            self.wavelength.get(),
            self.wavelength_units.get(),
            DEFAULT_WAVELENGTH_UNITS,
        )
        return convert_units(
            A_KEV / wavelength,
            DEFAULT_ENERGY_UNITS,
            self.energy_units.get(),
        )

    @_energy.setter
    def _energy(self, value: float):
        """Given energy, set the wavelength, in the current units."""
        from .misc import convert_units

        energy = convert_units(
            value,
            self.energy_units.get(),
            DEFAULT_ENERGY_UNITS,
        )
        self.wavelength.put(
            convert_units(
                A_KEV / energy,
                DEFAULT_WAVELENGTH_UNITS,
                self.wavelength_units.get(),
            )
        )


class EpicsWavelengthRO(_WavelengthBase):
    """Monochromatic wavelength (:math:`\\lambda`) from an EPICS PV."""

    wavelength = FC(EpicsSignalRO, "{prefix}{_pv_wavelength}", kind="hinted")
    wavelength_units = Component(
        SignalRO, value=DEFAULT_WAVELENGTH_UNITS, kind="config"
    )

    def __init__(
        self,
        prefix: str = "",
        *,
        pv_wavelength: str = "",
        wavelength_units: str = DEFAULT_WAVELENGTH_UNITS,
        **kwargs,
    ):
        """."""
        self._pv_wavelength = pv_wavelength
        super().__init__(prefix, **kwargs)
        self.wavelength_units._readback = wavelength_units

    def _asdict(self):
        """."""
        info = super()._asdict()
        info["wavelength_PV"] = self.wavelength.pvname
        return info


class EpicsMonochromatorRO(EpicsWavelengthRO):
    """
    Monochromatic X-ray wavelength (:math:`\\lambda`) & energy from EPICS PVs.

    The EPICS controls are responsible for conversions between wavelength &
    energy and for making changing these values.
    """

    energy = FC(EpicsSignalRO, "{prefix}{_pv_energy}", kind="hinted")
    energy_units = Component(SignalRO, value=DEFAULT_ENERGY_UNITS, kind="config")

    _keyset: list[str] = (
        "source_type energy wavelength energy_units wavelength_units".split()
    )
    """List of Component names for '_asdict()' and '_fromdict()'."""

    def __init__(
        self,
        prefix: str = "",
        *,
        pv_energy: str = "",
        energy_units: str = DEFAULT_ENERGY_UNITS,
        **kwargs,
    ):
        """."""
        self._pv_energy = pv_energy
        super().__init__(prefix, **kwargs)
        self.energy_units._readback = energy_units

    def _asdict(self):
        """."""
        info = super()._asdict()
        info["energy_PV"] = self.energy.pvname
        return info
