"""Test code in the 'misc' module."""

import math
import numbers
import pathlib
import re
import types
from collections import namedtuple
from contextlib import nullcontext as does_not_raise
from typing import Union

import numpy as np
import pint
import pytest
from bluesky import RunEngine
from bluesky import plans as bp
from ophyd import Component
from ophyd import Device
from ophyd import EpicsMotor
from ophyd import PVPositioner
from ophyd import Signal
from ophyd import SoftPositioner
from yaml.parser import ParserError

from ..diffract import creator
from ..diffract import diffractometer_class_factory
from ..misc import AnyAxesType
from ..misc import AxesArray
from ..misc import AxesDict
from ..misc import AxesList
from ..misc import AxesTuple
from ..misc import ConfigurationRunWrapper
from ..misc import NoForwardSolutions
from ..misc import SolverError
from ..misc import VirtualPositionerBase
from ..misc import axes_to_dict
from ..misc import compare_float_dicts
from ..misc import convert_units
from ..misc import dict_device_factory
from ..misc import distance_between_pos_tuples
from ..misc import dynamic_import
from ..misc import flatten_lists
from ..misc import get_run_orientation
from ..misc import get_solver
from ..misc import istype
from ..misc import list_orientation_runs
from ..misc import load_yaml_file
from ..misc import make_dynamic_instance
from ..misc import parse_factory_axes
from ..misc import pick_closest_solution
from ..misc import pick_first_solution
from ..misc import roundoff
from ..tests.common import HKLPY2_DIR

sim4c = creator(name="sim4c")
sim6c = creator(name="sim6c", geometry="E6C")
signal = Signal(name="signal", value=1.234)


class MyPVPositioner(PVPositioner):
    done = Component(Signal, value=1)
    limits = (-100, 100)
    readback = Component(Signal, value=0)
    setpoint = Component(Signal, value=0)


@pytest.fixture
def cat():
    from tiled.client import from_uri
    from tiled.server import SimpleTiledServer

    with SimpleTiledServer() as server:
        client = from_uri(server.uri)
        yield client


@pytest.fixture
def RE(cat):
    from bluesky_tiled_plugins import TiledWriter

    tw = TiledWriter(cat)
    engine = RunEngine({})
    engine.subscribe(tw)
    yield engine


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "input, names, context",
    [
        pytest.param(
            [0, 0, 0],
            "h k l",
            pytest.raises(TypeError, match=re.escape("Expected a list of names")),
            id="names-not-list",
        ),
        pytest.param(
            [0, 0, 0],
            [0, 0, 0],
            pytest.raises(TypeError, match=re.escape("Each name should be text,")),
            id="names-not-text",
        ),
        pytest.param(
            dict(h=0, k=0, l=0),
            "h k l".split(),
            does_not_raise(),
            id="dict-input",
        ),
        pytest.param(
            dict(a=0, k=0, l=0),
            "h k l".split(),
            pytest.raises(KeyError, match=re.escape("Missing axis 'h'")),
            id="missing-axis-h",
        ),
        pytest.param(
            namedtuple("PseudoTuple", "h k l".split())(0, 0, 0),
            "h k l".split(),
            does_not_raise(),
            id="namedtuple-input",
        ),
        pytest.param(
            np.array([0, 1, -1]),
            "h k l".split(),
            does_not_raise(),
            id="ndarray-input",
        ),
        pytest.param(
            "123",
            "h k l".split(),
            pytest.raises(TypeError, match=re.escape("Unexpected type")),
            id="string-input",
        ),
        pytest.param(
            (1, 2),
            "h k l".split(),
            pytest.raises(
                ValueError, match=re.escape("Expected at least 3 axes, received 2")
            ),
            id="too-few-axes",
        ),
        pytest.param(
            (1, 2, 3, 4),
            "h k l".split(),
            pytest.raises(
                UserWarning,
                match=re.escape(" Extra inputs will be ignored. Expected 3."),
            ),
            id="too-many-axes",
        ),
        pytest.param(
            [0, 1, -1],
            "aa bb cc".split(),
            does_not_raise(),
            id="custom-names",
        ),
        pytest.param(
            [1.1, 2.2, 3.3, 4, 5],
            "able baker charlie delta echo".split(),
            does_not_raise(),
            id="five-element-list",
        ),
        pytest.param(
            [1.1, 2.2, 3.3, 4, "text"],
            "able baker charlie delta echo".split(),
            pytest.raises(
                TypeError, match=re.escape("Expected a number. Received: 'text'")
            ),
            id="non-numeric-element",
        ),
        pytest.param(
            "1 2 3".split(),
            "h k l".split(),
            pytest.raises(TypeError, match=re.escape("Expected 'AnyAxesType'.")),
            id="list-of-strings",
        ),
    ],
)
def test_axes_to_dict(input, names, context):
    with context:
        axes = axes_to_dict(input, names)
        assert isinstance(axes, dict)
        for name in names:
            for name in names:
                assert isinstance(axes.get(name), numbers.Real)


@pytest.mark.parametrize(
    "a1, a2, tol, equal, context",
    [
        pytest.param({}, {}, 0.1, True, does_not_raise(), id="empty-dicts"),
        pytest.param(
            {"a": 0.1}, {"a": 0.1}, 0.1, True, does_not_raise(), id="equal-values"
        ),
        pytest.param(
            {"a": 0.1}, {"a": 1.1}, 0.1, False, does_not_raise(), id="unequal-values"
        ),
        pytest.param(
            {"a": 0.1}, {"b": 0.1}, 0.1, False, does_not_raise(), id="different-keys"
        ),
        pytest.param({"a": 0.1}, {}, 0.1, False, does_not_raise(), id="one-empty"),
        pytest.param(
            {},
            {},
            -0.1,
            False,
            pytest.raises(ValueError, match=re.escape("should be tol >0")),
            id="negative-tol",
        ),
        pytest.param(
            {"a": 0.11}, {"a": 0.12}, 1, True, does_not_raise(), id="within-tol"
        ),
        pytest.param(
            {"a": 0.11}, {"a": 0.12}, 2, False, does_not_raise(), id="outside-tol"
        ),
    ],
)
def test_compare_float_dicts(a1, a2, tol, equal, context):
    with context:
        assert compare_float_dicts(a1, a2, tol=tol) == equal


@pytest.mark.parametrize(
    "data, context",
    [
        pytest.param({"aa": 1, "bb": "two"}, does_not_raise(), id="valid-dict"),
        pytest.param(
            1,
            pytest.raises(
                AttributeError, match=re.escape("object has no attribute 'items'")
            ),
            id="int-input",
        ),
        pytest.param(
            [1],
            pytest.raises(
                AttributeError, match=re.escape("object has no attribute 'items'")
            ),
            id="list-input",
        ),
    ],
)
def test_dict_device_factory(data, context):
    with context:
        device_class = dict_device_factory(data)
        assert issubclass(device_class, Device)
        assert device_class.__class__.__name__ == "type"
        assert "DictionaryDevice" in str(device_class)
        for k, v in data.items():
            signal = getattr(device_class, k, None)
            assert signal is not None, f"{v=}"
            assert isinstance(signal, Component), f"{signal=}"
            # assert signal.get() == v

        device = device_class(name="device")
        assert isinstance(device, Device)
        assert device.__class__.__name__ == "DictionaryDevice"
        assert "DictionaryDevice" in str(device)
        for k, v in data.items():
            signal = getattr(device, k, None)
            assert signal is not None, f"{v=}"
            assert isinstance(signal, Signal), f"{signal=}"
            assert signal.get() == v


@pytest.mark.parametrize(
    "source, context, answer",
    [
        pytest.param(
            [[1], [2, 3, 4]], does_not_raise(), [1, 2, 3, 4], id="nested-sublists"
        ),
        pytest.param(
            [[1, 2], [3, 4]], does_not_raise(), [1, 2, 3, 4], id="equal-sublists"
        ),
        pytest.param([1, 2, 3, 4], does_not_raise(), [1, 2, 3, 4], id="flat-list"),
        pytest.param([], does_not_raise(), [], id="empty-list"),
        pytest.param(
            1,
            pytest.raises(TypeError, match=re.escape("object is not iterable")),
            1,
            id="not-iterable",
        ),
    ],
)
def test_flatten_lists(source, context, answer):
    with context:
        result = flatten_lists(source)
        assert isinstance(result, types.GeneratorType)

        result = list(result)
        assert result == answer, f"{source=} {answer=} {result=}"


@pytest.mark.parametrize(
    "solver_name, context, expected",
    [
        pytest.param("hkl_soleil", does_not_raise(), None, id="hkl-soleil"),
        pytest.param("no_op", does_not_raise(), None, id="no-op"),
        pytest.param("th_tth", does_not_raise(), None, id="th-tth"),
        pytest.param(
            "no_such_thing",
            pytest.raises(SolverError),
            "unknown.  Pick one of:",
            id="unknown-solver",
        ),
    ],
)
def test_get_solver(solver_name, context, expected):
    with context:
        solver = get_solver(solver_name)

    if expected is None:
        assert solver is not None


@pytest.mark.parametrize(
    "path, context, expected, keys",
    [
        pytest.param(
            # YAML file with expected content
            HKLPY2_DIR / "tests" / "e4cv_orient.yml",
            does_not_raise(),
            None,
            ["_header", "name"],
            id="valid-yaml-file",
        ),
        pytest.param(
            # file does not exist (wrong directory)
            HKLPY2_DIR / "e4cv_orient.yml",
            pytest.raises(FileExistsError),
            "YAML file ",
            None,
            id="file-not-found",
        ),
        pytest.param(
            # Not a YAML file, empty
            HKLPY2_DIR / "__init__.py",
            pytest.raises(ParserError),
            "<scalar>",
            None,
            id="not-yaml-empty",
        ),
        pytest.param(
            # Not a YAML file, not empty
            HKLPY2_DIR / "diffract.py",
            pytest.raises(ParserError),
            "expected '<document start>', but found",
            None,
            id="not-yaml-nonempty",
        ),
    ],
)
def test_load_yaml_file(path, context, expected, keys):
    assert isinstance(path, (pathlib.Path, str))
    with context:
        contents = load_yaml_file(path)

    if expected is None:
        # test keys
        not_found = object()
        for key in keys:
            assert contents.get(key, not_found) != not_found, f"{key=}"


@pytest.mark.parametrize(
    "value, digits, expected_text",
    [
        pytest.param(0, None, "0", id="zero-default-digits"),
        pytest.param(0.123456, None, "0", id="fractional-default-digits"),
        pytest.param(0.123456, 4, "0.1235", id="fractional-4-digits"),
        pytest.param(-0, 4, "0", id="neg-zero-4-digits"),
        pytest.param(123456, 4, "123456", id="integer-4-digits"),
        pytest.param(123456, -4, "120000", id="integer-neg4-digits"),
        pytest.param(1.23456e-10, 4, "0", id="tiny-4-digits"),
        pytest.param(1.23456e-10, 12, "1.23e-10", id="tiny-12-digits"),
    ],
)
def test_roundoff(value, digits, expected_text):
    result = roundoff(value, digits)
    assert str(result) == expected_text


@pytest.mark.parametrize(
    "devices, context",
    [
        pytest.param([sim4c], does_not_raise(), id="single-diffractometer"),
        pytest.param(
            [sim4c.chi],
            pytest.raises(TypeError, match=re.escape("SoftPositioner")),
            id="soft-positioner-rejected",
        ),
        pytest.param([sim4c, sim6c], does_not_raise(), id="two-diffractometers"),
        pytest.param(
            [sim4c, sim6c.h],
            pytest.raises(TypeError, match=re.escape("Hklpy2PseudoAxis")),
            id="pseudo-axis-rejected",
        ),
    ],
)
@pytest.mark.parametrize(
    "enabled",
    [
        pytest.param(True, id="enabled"),
        pytest.param(False, id="disabled"),
    ],
)
def test_ConfigurationRunWrapper(devices, context, enabled):
    with context:
        crw = ConfigurationRunWrapper(*devices)
        for dev in devices:
            assert dev in crw.devices

        crw.enable = enabled
        assert crw.enable == enabled

        documents = []

        def collector(key, doc):
            nonlocal documents
            documents.append((key, doc))

        assert len(documents) == 0

        RE = RunEngine()
        RE.preprocessors.append(crw.wrapper)
        RE(bp.count([signal]), collector)
        assert len(documents) >= 4

        for key, doc in documents:
            if key == "start":
                configs = doc.get(crw.start_key)
                if enabled:
                    assert configs is not None
                    assert signal.name not in configs
                    for name in crw.device_names:
                        assert name in configs
                    for dev in devices:
                        with does_not_raise() as message:
                            # Try to restore the configuration
                            dev.configuration = configs[dev.name]
                        assert message is None, f"{dev.name=!r} {configs[dev.name]=}"
                else:
                    assert configs is None


@pytest.mark.parametrize(
    "devices",
    [
        pytest.param([], id="no-devices"),
        pytest.param([sim4c], id="sim4c-only"),
        pytest.param([sim4c, sim6c], id="sim4c-and-sim6c"),
        pytest.param([sim6c], id="sim6c-only"),
    ],
)
def test_list_orientation_runs(devices, cat, RE):
    det = signal
    device_names = [d.name for d in devices]
    crw = ConfigurationRunWrapper(*devices)  # TODO: #34 decorator
    RE.preprocessors.append(crw.wrapper)

    def scans():
        yield from bp.count([det])
        yield from bp.count([sim4c])
        yield from bp.count([sim6c])
        yield from bp.count([sim4c, sim6c])

    uids = RE(scans())
    scan_ids = [cat[uid].metadata["start"]["scan_id"] for uid in uids]
    assert scan_ids == [1, 2, 3, 4]

    scan_id = scan_ids[0]
    assert scan_id == 1

    # test get_run_orientation() for specific diffractometer
    info = get_run_orientation(cat[uids[0]], name="sim4c")
    assert isinstance(info, dict)
    if sim4c in devices:
        assert len(info) > 0
        assert "_header" in info
    else:
        assert len(info) == 0

    runs = list_orientation_runs(cat)
    assert len(runs) == 4 * len(devices), f"{runs=!r}"
    if len(devices) > 0:
        assert scan_id in runs.scan_id.to_list(), f"{runs=!r}"

    runs = runs.T.to_dict()  # simpler to test as dict structure.
    assert len(runs) == 4 * len(devices), f"{runs=!r}"

    for row in runs.values():
        assert row["scan_id"] in scan_ids
        assert row["diffractometer"] in device_names


@pytest.mark.parametrize(
    "value, annotation, context",
    [
        pytest.param(
            {"h": 1.2, "k": 1, "l": -1}, AxesDict, does_not_raise(), id="dict-AxesDict"
        ),
        pytest.param(
            namedtuple("Position", "a b c d".split())(1, 2, 3, 4),
            AxesTuple,
            does_not_raise(),
            id="namedtuple-AxesTuple",
        ),
        pytest.param([1, 2, 3], AxesList, does_not_raise(), id="list-AxesList"),
        pytest.param((1, 2, 3), AxesTuple, does_not_raise(), id="tuple-AxesTuple"),
        pytest.param(
            np.array((1, 2, 3, 4, 5)),
            AxesArray,
            does_not_raise(),
            id="ndarray-AxesArray",
        ),
        pytest.param(
            {"h": 1.2, "k": 1, "l": -1},
            AnyAxesType,
            does_not_raise(),
            id="dict-AnyAxesType",
        ),
        pytest.param(
            namedtuple("Position", "a b c d".split())(1, 2, 3, 4),
            AnyAxesType,
            does_not_raise(),
            id="namedtuple-AnyAxesType",
        ),
        pytest.param([1, 2, 3], AnyAxesType, does_not_raise(), id="list-AnyAxesType"),
        pytest.param((1, 2, 3), AnyAxesType, does_not_raise(), id="tuple-AnyAxesType"),
        pytest.param(
            np.array((1, 2, 3, 4, 5)),
            AnyAxesType,
            does_not_raise(),
            id="ndarray-AnyAxesType",
        ),
        pytest.param(
            None, Union[AnyAxesType, None], does_not_raise(), id="None-Optional"
        ),
        pytest.param(
            None,
            AnyAxesType,
            pytest.raises(AssertionError, match=re.escape("False")),
            id="None-AnyAxesType",
        ),
        pytest.param(
            1.234,
            AnyAxesType,
            pytest.raises(AssertionError, match=re.escape("False")),
            id="float-AnyAxesType",
        ),
        pytest.param(
            "text",
            AnyAxesType,
            pytest.raises(AssertionError, match=re.escape("False")),
            id="str-AnyAxesType",
        ),
        pytest.param(
            sim4c,
            AnyAxesType,
            pytest.raises(AssertionError, match=re.escape("False")),
            id="device-AnyAxesType",
        ),
    ],
)
def test_axes_type_annotations(value, annotation, context):
    with context:
        assert istype(value, annotation)


@pytest.mark.parametrize(
    "name, context",
    [
        pytest.param("ophyd.EpicsMotor", does_not_raise(), id="EpicsMotor"),
        pytest.param("hklpy2.diffract.creator", does_not_raise(), id="creator"),
        pytest.param(
            "hklpy2.diffract.does_not_exist",
            pytest.raises(
                AttributeError, match=re.escape("has no attribute 'does_not_exist'")
            ),
            id="missing-attr",
        ),
        pytest.param(
            "does.not.exist",
            pytest.raises(
                ModuleNotFoundError, match=re.escape("No module named 'does'")
            ),
            id="missing-module",
        ),
        pytest.param(
            "LocalName",
            pytest.raises(ValueError, match=re.escape("Must use a dotted path")),
            id="no-dots",
        ),
        pytest.param(
            ".test_utils.CATALOG",
            pytest.raises(
                ValueError,
                match=re.escape("Must use absolute path, no relative imports"),
            ),
            id="relative-import",
        ),
    ],
)
def test_dynamic_import(name, context):
    with context:
        dynamic_import(name)


@pytest.mark.parametrize(
    "value, units1, units2, ref, context",
    [
        pytest.param(32, "fahrenheit", "celsius", 0, does_not_raise(), id="F-to-C"),
        pytest.param(100, "pm", "angstrom", 1, does_not_raise(), id="pm-to-angstrom"),
        pytest.param(0.1, "nm", "angstrom", 1, does_not_raise(), id="nm-to-angstrom"),
        pytest.param(12400, "eV", "keV", 12.4, does_not_raise(), id="eV-to-keV"),
        pytest.param(
            0.1,
            "nm",
            "banana",
            1,
            pytest.raises(pint.UndefinedUnitError, match=re.escape("'banana'")),
            id="undefined-unit",
        ),
    ],
)
def test_convert_units(value, units1, units2, ref, context):
    with context:
        assert math.isclose(convert_units(value, units1, units2), ref, abs_tol=0.01)


@pytest.mark.parametrize(
    "pos1, pos2, dist, tol, context",
    [
        pytest.param(
            namedtuple("Position", "a b c".split())(0, 0, 0),
            namedtuple("Position", "a b c".split())(1, 1, 1),
            1,
            1e-6,
            does_not_raise(),
            id="unit-distance",
        ),
        pytest.param(
            namedtuple("Position", "a b c".split())(0, 0, 0),
            namedtuple("NameIgnored", "a b c".split())(1, 0, 0),
            math.sqrt(1 / 3),
            1e-6,
            does_not_raise(),
            id="partial-offset",
        ),
        pytest.param(
            namedtuple("Position", "x y z".split())(0, 0, 0),
            namedtuple("Position", "a b c".split())(1, 1, 1),
            1,
            1e-6,
            pytest.raises(
                AttributeError,
                match=re.escape("'Position' object has no attribute 'x'"),
            ),
            id="mismatched-fields",
        ),
        pytest.param(
            namedtuple("Position", "d e".split())(0, 0),
            namedtuple("Position", "a b c".split())(1, 1, 1),
            1,
            1e-6,
            pytest.raises(AttributeError, match=re.escape("are not the same length.")),
            id="different-lengths",
        ),
        pytest.param(
            (),
            namedtuple("Ignored", "a b c".split())(1, 0, 0),
            0,
            1e-6,
            pytest.raises(AttributeError, match=re.escape("are not the same length.")),
            id="empty-vs-nonempty",
        ),
        pytest.param(
            (),
            (),
            0,
            1e-6,
            does_not_raise(),
            id="both-empty",
        ),
    ],
)
def test_distance_between_pos_tuples(pos1, pos2, dist, tol, context):
    with context:
        assert math.isclose(
            distance_between_pos_tuples(pos1, pos2),
            dist,
            abs_tol=tol,
        )


@pytest.mark.parametrize(
    "pos, possibilities, function, selected, context",
    [
        pytest.param(
            (),
            "a b c".split(),
            pick_first_solution,
            "a",
            does_not_raise(),
            id="pick-first",
        ),
        pytest.param(
            "a b c".split(),
            (),
            pick_first_solution,
            None,
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="pick-first-empty",
        ),
        pytest.param(
            namedtuple("Position", "a b c".split())(0, 0, 0),
            [
                namedtuple("Position", "a b c".split())(1, -1, 1),
                namedtuple("Position", "a b c".split())(1, 1, 1),
                namedtuple("Position", "a b c".split())(3, 2, 1),
            ],
            pick_closest_solution,
            namedtuple("Position", "a b c".split())(1, -1, 1),  # first, closest
            does_not_raise(),
            id="pick-closest",
        ),
        pytest.param(
            namedtuple("Position", "a b c".split())(0, 0, 0),
            [],
            pick_closest_solution,
            None,
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="pick-closest-empty",
        ),
    ],
)
def test_choice_function(pos, possibilities, function, selected, context):
    with context:
        choice = function(pos, possibilities)
        assert choice == selected


@pytest.mark.parametrize(
    "specs, context",
    [
        pytest.param(
            {},
            pytest.raises(
                ValueError, match=re.escape("Must provide a value for 'physical_name'.")
            ),
            id="missing-physical-name",
        ),
        pytest.param(
            {"physical_name": "guess"},
            pytest.raises(
                AttributeError,
                match=re.escape("'NoneType' object has no attribute 'guess'"),
            ),
            id="invalid-physical-name",
        ),
    ],
)
def test_VirtualPositionerBase(specs, context):
    with context:
        VirtualPositionerBase(name="gonio", **specs)


@pytest.mark.parametrize(
    "specs, context",
    [
        pytest.param(
            dict(init_pos=0, physical_name="linear", kind="hinted"),
            does_not_raise(),
            id="valid-specs",
        ),
        pytest.param(
            dict(),
            pytest.raises(
                ValueError, match=re.escape("Must provide a value for 'physical_name'.")
            ),
            id="missing-physical-name",
        ),
        pytest.param(
            # Compare with 'guess' test case above.
            dict(init_pos=0, physical_name="guess"),
            pytest.raises(
                RuntimeError,
                match=re.escape("AttributeError while instantiating component: tth"),
            ),
            id="invalid-physical-name",
        ),
    ],
)
def test_virtual_axis(specs, context):
    GoniometerBase = diffractometer_class_factory(
        solver="hkl_soleil",
        geometry="E4CV",
    )

    class Goniometer(GoniometerBase):
        tth = Component(VirtualPositionerBase, **specs)

        # Add the translation axis 'dy'.
        linear = Component(
            SoftPositioner,
            init_pos=0,
            limits=(-10, 200),
            kind="hinted",
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tth._finish_setup()

    with context:
        gonio = Goniometer(name="gonio")
        gonio.add_sample("vibranium", 2 * math.pi)
        gonio.wh()

        assert np.allclose(list(gonio.position), (0, 0, 0), atol=0.001)
        assert np.allclose(list(gonio.real_position), (0, 0, 0, 0), atol=0.001)
        assert gonio.linear.position == 0

        gonio.linear.move(1)
        assert gonio.linear.position == 1
        assert np.allclose(list(gonio.real_position), (0, 0, 0, 2), atol=0.001)
        assert math.isclose(gonio.l.position, 0.22, abs_tol=0.01)
        assert np.allclose(list(gonio.position), (0, 0, 0.22), atol=0.01)
        assert math.isclose(
            gonio.tth.forward(gonio.tth.inverse(math.pi)),
            math.pi,
            abs_tol=0.01,
        )


@pytest.mark.parametrize(
    "klass, specs, context",
    [
        pytest.param(
            EpicsMotor, dict(prefix="IOC:m1"), does_not_raise(), id="EpicsMotor"
        ),
        pytest.param(MyPVPositioner, dict(), does_not_raise(), id="PVPositioner"),
        pytest.param(SoftPositioner, dict(), does_not_raise(), id="SoftPositioner"),
        pytest.param(
            Signal,
            dict(),
            pytest.raises(
                TypeError, match=re.escape("Unknown 'readback' for 'gonio_linear'.")
            ),
            id="Signal-invalid",
        ),
    ],
)
def test_virtual_axis_physical(klass, specs, context):
    GoniometerBase = diffractometer_class_factory(
        solver="hkl_soleil",
        geometry="E4CV",
    )

    class Goniometer(GoniometerBase):
        tth = Component(
            VirtualPositionerBase,
            init_pos=0,
            physical_name="linear",
            kind="hinted",
        )

        # Add the translation axis 'dy'.
        linear = Component(klass, **specs)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.tth._finish_setup()

    with context:
        gonio = Goniometer(name="gonio")
        gonio.tth._finish_setup()
        gonio.tth.move(-2)
        if gonio.connected:
            assert math.isclose(gonio.linear.position or -1, -1, abs_tol=0.01)


def test_virtual_axis_finish_setup_trigger():
    class SpecialCase(Device):
        physical = Component(
            SoftPositioner,
            init_pos=0,
            limits=(-10, 10),
            kind="hinted",
        )
        virtual = Component(
            VirtualPositionerBase,
            init_pos=0,
            limits=(-10, 10),
            physical_name="physical",
            kind="hinted",
        )

    multi = SpecialCase("", name="multi")
    assert multi.connected
    assert not multi.virtual._setup_finished
    multi.physical.position
    assert not multi.virtual._setup_finished
    multi.virtual.position
    assert multi.virtual._setup_finished


def test_get_solver_raises_for_unknown():
    """Ensure get_solver raises SolverError for unknown solver name."""
    with pytest.raises(SolverError):
        get_solver("this_solver_does_not_exist_12345")


def test_get_run_orientation_basic():
    """Basic behavior of get_run_orientation for start metadata.

    - If name is None, returns the top-level dict for the start_key
    - If name is provided, returns the nested dict or empty dict
    """

    class DummyRun:
        def __init__(self, md):
            self.metadata = {"start": {"diffractometers": md}}

    md = {"devA": {"k": 1}, "devB": {"k": 2}}
    run = DummyRun(md)

    # no name -> full dict
    full = get_run_orientation(run)
    assert isinstance(full, dict)
    assert "devA" in full and "devB" in full

    # specific name -> nested dict
    a = get_run_orientation(run, name="devA")
    assert isinstance(a, dict)
    assert a == {"k": 1}

    # missing name returns empty dict
    missing = get_run_orientation(run, name="no_device_here")
    assert missing == {}


def test_istype_with_numpy_scalar_and_none():
    """Ensure istype handles numpy scalars and None appropriately."""
    # numpy scalar should not match AxesArray (ndarray) annotation
    assert not istype(np.int64(1), AxesArray)

    # numpy array should match AxesArray
    assert istype(np.array([1, 2, 3]), AxesArray)

    # None against Optional/Union types: already covered elsewhere, but sanity-check
    assert istype(None, Union[AxesArray, None])


@pytest.mark.parametrize(
    "params, context",
    [
        pytest.param(
            dict(
                space="pseudos",
                canonical="h k l".split(),
            ),
            does_not_raise(),
            id="Ok, default 'axes' & 'order'",
        ),
        pytest.param(
            dict(
                space="pseudos",
                order="aa bb cc dd".split(),
                canonical="h k l".split(),
            ),
            pytest.raises(ValueError, match=re.escape("Too many axes specified")),
            id="ValueError: \"Too many axes specified in order=['aa', 'bb', 'cc', 'dd']. Expected 3.\"",
        ),
        pytest.param(
            dict(
                space="pseudos",
                order="aa bb cc".split(),
                canonical="h k l".split(),
            ),
            pytest.raises(KeyError, match=re.escape("Unknown axis_name=")),
            id="KeyError: Unknown axis_name=",
        ),
        pytest.param(
            dict(
                space="pseudos",
                axes="h k l".split(),
                order="h h l".split(),
                canonical="h k l".split(),
            ),
            pytest.raises(ValueError, match=re.escape("Duplicates in order=")),
            id="ValueError: Duplicates in order=",
        ),
        pytest.param(
            dict(space="not recognized"),
            pytest.raises(KeyError, match=re.escape("Unknown space='not recognized'")),
            id="KeyError: Unknown space='not recognized'",
        ),
        pytest.param(
            dict(
                space="pseudos",
                axes="aa bb cc".split(),
                canonical="h k l".split(),
            ),
            does_not_raise(),
            id="Ok, default 'order'",
        ),
        pytest.param(
            dict(
                space="reals",
                canonical="aaa bbb ccc".split(),
            ),
            does_not_raise(),
            id="Ok, default 'axes' & 'order'",
        ),
        pytest.param(
            dict(
                space="reals",
                order="th tth".split(),
                canonical="aaa bbb ccc".split(),
            ),
            pytest.raises(
                ValueError, match=re.escape("len(order)=2 must be >= len(canonical)=3")
            ),
            id="ValueError: 'len(order)=2 must be >= len(canonical)=3'",
        ),
        pytest.param(
            dict(
                space="reals",
                axes="m1 m2 m3 m4".split(),
                canonical="aaa bbb ccc".split(),
            ),
            does_not_raise(),
            id="Ok, extra 'axes', custom names, default 'order'",
        ),
        pytest.param(
            dict(
                space="reals",
                axes="m1 m2 m3 m4 m1".split(),
                canonical="aaa bbb ccc".split(),
            ),
            pytest.raises(
                ValueError,
                match=re.escape(
                    #
                    "Duplicates in axes=['m1', 'm2', 'm3', 'm4', 'm1']"
                ),
            ),
            id="ValueError: Duplicates in axes=['m1', 'm2', 'm3', 'm4', 'm1']",
        ),
        pytest.param(
            dict(
                space="reals",
                axes="m1 m2 m3 m4".split(),
                order="m2 m4 m1".split(),
                canonical="aaa bbb ccc".split(),
            ),
            does_not_raise(),
            id="Ok: custom 'axes', custom 'order'.",
        ),
        pytest.param(
            dict(
                space="reals",
                axes=dict(
                    m1=None,  # SoftPositioner
                    m2="IOC:m2",  # EpicsMotor
                    m3={  # Custom
                        "class": "ophyd.EpicsMotor",
                        "prefix": "IOC:m3",
                        "labels": "motors reals".split(),
                    },
                ),
                canonical="aaa bbb ccc".split(),
            ),
            does_not_raise(),
            id="Ok: default, EPICS, and custom 'axes'",
        ),
        pytest.param(
            dict(
                space="reals",
                axes=dict(m1=None, m2=None, m3=np.array([1, 2])),
                canonical="aaa bbb ccc".split(),
            ),
            pytest.raises(
                TypeError,
                match=re.escape(
                    #
                    "Incorrect type 'ndarray' for specs=array([1, 2])."
                ),
            ),
            id="TypeError: Incorrect type 'ndarray' for specs=array([1, 2]).",
        ),
        pytest.param(
            dict(
                space="reals",
                # axes="m1 m2 m3 m4".split(),
                canonical="aaa bbb ccc".split(),
            ),
            does_not_raise(),
            id="Ok, minimum spec",
        ),
    ],
)
def test_parse_factory_axes(params, context):
    with context:
        parse_factory_axes(**params)


def test_make_dynamic_instance_raises():
    non_callable = "hklpy2.misc.DEFAULT_MOTOR_LABELS"
    with pytest.raises(
        TypeError,
        match=re.escape(f"{non_callable!r} is not callable"),
    ):
        make_dynamic_instance(non_callable)
