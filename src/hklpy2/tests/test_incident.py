"""Test the incident beam module."""

import logging
import math
import re
from contextlib import nullcontext as does_not_raise

import pint
import pytest
from ophyd.utils import ReadOnlyError

from ..diffract import creator
from ..incident import A_KEV
from ..incident import DEFAULT_SOURCE_TYPE
from ..incident import DEFAULT_WAVELENGTH
from ..incident import DEFAULT_WAVELENGTH_DEADBAND
from ..incident import EpicsMonochromatorRO
from ..incident import EpicsWavelengthRO
from ..incident import Wavelength
from ..incident import WavelengthXray
from ..incident import _WavelengthBase
from ..utils import INTERNAL_LENGTH_UNITS
from ..utils import INTERNAL_XRAY_ENERGY_UNITS
from .common import IOC_PREFIX

logger = logging.getLogger(__name__)


def check_keys(wl, ref, tol=0.001):
    info = wl._asdict()
    for key, value in ref.items():
        assert key in info
        if isinstance(value, (float, int)):
            assert math.isclose(info[key], value, abs_tol=tol)
        else:
            assert info[key] == value


@pytest.mark.parametrize(
    "Klass, parms, ref, context",
    [
        pytest.param(
            _WavelengthBase,
            {},
            dict(
                wavelength=DEFAULT_WAVELENGTH,
                wavelength_units=INTERNAL_LENGTH_UNITS,
                source_type=DEFAULT_SOURCE_TYPE,
            ),
            does_not_raise(),
            id="WavelengthBase-defaults",
        ),
        pytest.param(
            _WavelengthBase,
            dict(wavelength=0.5),
            {},
            pytest.raises(
                ReadOnlyError, match=re.escape("The signal wl_wavelength is readonly.")
            ),
            id="WavelengthBase-readonly-error",
        ),
        pytest.param(
            Wavelength,
            {},
            dict(
                wavelength=DEFAULT_WAVELENGTH,
                wavelength_units=INTERNAL_LENGTH_UNITS,
                source_type=DEFAULT_SOURCE_TYPE,
            ),
            does_not_raise(),
            id="Wavelength-defaults",
        ),
        pytest.param(
            Wavelength,
            dict(wavelength=2),
            dict(
                wavelength=2,
                wavelength_units=INTERNAL_LENGTH_UNITS,
                source_type=DEFAULT_SOURCE_TYPE,
            ),
            does_not_raise(),
            id="Wavelength-set-wavelength-2",
        ),
        pytest.param(
            Wavelength,
            dict(wavelength_units="Mfurlongs"),
            dict(wavelength_units="Mfurlongs"),
            does_not_raise(),
            id="Wavelength-units-Mfurlongs",
        ),
        pytest.param(
            Wavelength,
            dict(wavelength_units="banana"),
            {},
            pytest.raises(pint.UndefinedUnitError, match=re.escape("banana")),
            id="Wavelength-units-banana-error",
        ),
        pytest.param(
            Wavelength,
            dict(source_type="unit testing"),
            dict(source_type="unit testing"),
            does_not_raise(),
            id="Wavelength-source-type-custom",
        ),
        pytest.param(
            WavelengthXray,
            {},
            dict(
                energy=12.3984,
                energy_units=INTERNAL_XRAY_ENERGY_UNITS,
                wavelength=DEFAULT_WAVELENGTH,
                wavelength_units=INTERNAL_LENGTH_UNITS,
                source_type=DEFAULT_SOURCE_TYPE,
            ),
            does_not_raise(),
            id="WavelengthXray-defaults",
        ),
        pytest.param(
            WavelengthXray,
            dict(energy_units="banana"),
            {},
            pytest.raises(pint.UndefinedUnitError, match=re.escape("banana")),
            id="WavelengthXray-energy-units-banana-error",
        ),
        pytest.param(
            WavelengthXray,
            dict(energy_units="eV"),
            dict(energy_units="eV"),
            does_not_raise(),
            id="WavelengthXray-energy-units-eV",
        ),
        pytest.param(
            WavelengthXray,
            dict(energy=10),
            dict(energy=10),
            does_not_raise(),
            id="WavelengthXray-energy-10",
        ),
    ],
)
def test_constructors(Klass, parms, ref, context):
    with context:
        wl = Klass(**parms, name="wl")
        wl.wait_for_connection()
        check_keys(wl, ref)


@pytest.mark.parametrize(
    "Klass, input, context",
    [
        pytest.param(
            Wavelength,
            {},
            does_not_raise(),
            id="Wavelength-empty-dict",
        ),
        pytest.param(
            Wavelength,
            {"wavelength": 2},  # missing "class='Wavelength'" key.
            pytest.raises(AssertionError, match=re.escape("isclose")),
            id="Wavelength-missing-class-key-error",
        ),
        pytest.param(
            Wavelength,
            {"class": "Wavelength", "wavelength": 2},
            does_not_raise(),
            id="Wavelength-set-wavelength-2",
        ),
        pytest.param(
            Wavelength,
            {"class": "Wavelength", "wavelength_units": "kg"},  # incompatible
            pytest.raises(
                pint.DimensionalityError, match=re.escape(INTERNAL_LENGTH_UNITS)
            ),
            id="Wavelength-incompatible-units-kg-error",
        ),
        pytest.param(
            Wavelength,
            {"class": "Wavelength", "wavelength_units": "banana"},
            pytest.raises(pint.UndefinedUnitError, match=re.escape("banana")),
            id="Wavelength-undefined-units-banana-error",
        ),
        pytest.param(
            Wavelength,
            {"class": "Wavelength", "energy_units": "pg"},
            pytest.raises(
                pint.DimensionalityError, match=re.escape("kiloelectron_volt")
            ),
            id="Wavelength-energy-units-pg-error",
        ),
        pytest.param(
            WavelengthXray,
            {"class": "WavelengthXray", "energy": 20},
            does_not_raise(),
            id="WavelengthXray-energy-20",
        ),
        pytest.param(
            WavelengthXray,
            {"class": "WavelengthXray", "energy_units": "eV"},
            does_not_raise(),
            id="WavelengthXray-energy-units-eV",
        ),
        pytest.param(
            WavelengthXray,
            {"class": "WavelengthXray", "source_type": "unit testing"},
            pytest.raises(
                AssertionError, match=re.escape(DEFAULT_SOURCE_TYPE)
            ),  # Can't change after constructor.
            id="WavelengthXray-source-type-immutable-error",
        ),
    ],
)
def test__fromdict(Klass, input, context):
    with context:
        wl = Klass(name="wl")
        wl.wait_for_connection()
        wl._fromdict(input)
        check_keys(wl, input)


@pytest.mark.parametrize(
    "Klass, input, ref, context",
    [
        pytest.param(
            EpicsWavelengthRO,
            dict(prefix=IOC_PREFIX, pv_wavelength="wavelength"),
            {"class": "EpicsWavelengthRO", "wavelength": 1.0},
            does_not_raise(),
            id="EpicsWavelengthRO-valid",
        ),
        pytest.param(
            EpicsMonochromatorRO,
            dict(prefix=IOC_PREFIX, pv_energy="energy", pv_wavelength="wavelength"),
            {"class": "EpicsMonochromatorRO", "energy": 12.3984, "wavelength": 1.0},
            does_not_raise(),
            id="EpicsMonochromatorRO-valid",
        ),
        pytest.param(
            EpicsWavelengthRO,
            dict(prefix=IOC_PREFIX, pv_wavelength="wrong_pv"),
            {},
            pytest.raises(TimeoutError, match=re.escape(f"{IOC_PREFIX}wrong_pv")),
            id="EpicsWavelengthRO-wrong-pv-error",
        ),
        pytest.param(
            EpicsWavelengthRO,
            dict(prefix=IOC_PREFIX, pv_wavelength="force:pytest.skip"),
            {},
            does_not_raise(),
            id="EpicsWavelengthRO-force-skip",
        ),
    ],
)
def test_EpicsClasses(Klass, input, ref, context):
    with context:
        wl = Klass(name="wl", **input)
        try:
            wl.wait_for_connection(timeout=2)
        except TimeoutError as exinfo:
            # Don't fail if the IOC is not running.
            if input["pv_wavelength"] != "wrong_pv":
                pytest.skip(f"{exinfo}", allow_module_level=True)
        check_keys(wl, ref)


@pytest.mark.parametrize(
    "parms, moves, context",
    [
        pytest.param(
            {
                "class": WavelengthXray,
                "wavelength_deadband": DEFAULT_WAVELENGTH_DEADBAND,
            },
            [
                (1.1, True),
                (1.10001, False),
                (1.100111, True),
            ],
            does_not_raise(),
            id="WavelengthXray-default-deadband",
        ),
        pytest.param(
            {"class": Wavelength, "wavelength_deadband": 0.01},
            [
                (1.1, True),
                (1.10999, False),
                (1.111, True),
                (1.1111, False),
                (1.1011, False),
                (1.1001, True),
                (1.0, True),
            ],
            does_not_raise(),
            id="Wavelength-custom-deadband-0.01",
        ),
    ],
)
def test_wavelength_update(parms, moves, context):
    with context:
        sim = creator(beam_kwargs=parms)
        sim.wait_for_connection()

        for position, updated in moves:
            sim.core._solver_needs_update = False
            sim.beam.wavelength.put(position)
            assert sim.core._solver_needs_update == updated, f"{position=}"


@pytest.mark.parametrize(
    "parms, moves, context",
    [
        pytest.param(
            {
                "class": WavelengthXray,
                "wavelength_deadband": DEFAULT_WAVELENGTH_DEADBAND,
            },
            [
                (1.1, True),
                (1.10001, False),
                (1.100111, True),
            ],
            does_not_raise(),
            id="WavelengthXray-cleanup-subscriptions",
        ),
    ],
)
def test_cleanup(parms, moves, context):
    with context:
        sim = creator(beam_kwargs=parms)
        sim.wait_for_connection()
        sim.beam.cleanup_subscriptions()

        for position, _ in moves:
            sim.core._solver_needs_update = False
            sim.beam.wavelength.put(position)
            assert not sim.core._solver_needs_update, f"{position=}"


@pytest.mark.parametrize(
    "parms, target, tol, context",
    [
        pytest.param(
            dict(geometry="TH TTH Q", solver="th_tth"),
            10,
            0.001,
            does_not_raise(),
            id="10 keV",
        ),
        pytest.param(
            dict(geometry="TH TTH Q", solver="th_tth"),
            24,
            0.001,
            does_not_raise(),
            id="24 keV",
        ),
    ],
)
def test_issue_159(parms, target, tol, context):
    """Reported energy value should agree with wavelength at all times."""
    with context:
        sim = creator(**parms)
        energy = sim.beam.energy
        wavelength = sim.beam.wavelength

        # This is the initial problem, energy should not be zero
        assert math.isclose(energy._readback, A_KEV, abs_tol=tol)
        assert math.isclose(wavelength._readback, 1)

        energy.put(target)
        assert math.isclose(energy._readback, target, abs_tol=tol)
        assert math.isclose(wavelength._readback, A_KEV / target)

        wavelength.put(1)
        assert math.isclose(energy._readback, A_KEV, abs_tol=tol)
        assert math.isclose(wavelength._readback, 1)
