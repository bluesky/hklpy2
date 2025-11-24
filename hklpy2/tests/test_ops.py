"""Test the hklpy2.ops module."""

import math
import uuid
from collections import namedtuple
from contextlib import nullcontext as does_not_raise

import pyRestTable
import pytest

from ..diffract import DiffractometerBase
from ..diffract import creator
from ..misc import ConfigurationError
from ..misc import ReflectionError
from ..ops import DEFAULT_SAMPLE_NAME
from ..ops import Core
from ..ops import CoreError
from ..tests.models import add_oriented_vibranium_to_e4cv
from ..user import set_diffractometer
from ..user import setor
from .common import assert_context_result
from .models import AugmentedFourc
from .models import MultiAxis99
from .models import MultiAxis99NoSolver
from .models import TwoC

SKIP_EXACT_VALUE_TEST = str(uuid.uuid4())

fourc = creator()


@pytest.mark.parametrize(
    "geometry, solver, name, keypath, value",
    [
        ["E4CV", "hkl_soleil", "fourc", "_header", SKIP_EXACT_VALUE_TEST],
        ["E4CV", "hkl_soleil", "fourc", "_header.datetime", SKIP_EXACT_VALUE_TEST],
        ["E4CV", "hkl_soleil", "fourc", "beam.wavelength", SKIP_EXACT_VALUE_TEST],
        ["E4CV", "hkl_soleil", "fourc", "name", "fourc"],
        ["E4CV", "hkl_soleil", "fourc", "solver.geometry", "E4CV"],
        ["E4CV", "hkl_soleil", "fourc", "solver.name", "hkl_soleil"],
        ["E4CV", "hkl_soleil", "fourc", "samples", SKIP_EXACT_VALUE_TEST],
        ["E4CV", "hkl_soleil", "fourc", "solver.version", SKIP_EXACT_VALUE_TEST],
        #
        ["TH TTH Q", "th_tth", "t2t", "_header", SKIP_EXACT_VALUE_TEST],
        ["TH TTH Q", "th_tth", "t2t", "_header.datetime", SKIP_EXACT_VALUE_TEST],
        ["TH TTH Q", "th_tth", "t2t", "beam.wavelength", SKIP_EXACT_VALUE_TEST],
        ["TH TTH Q", "th_tth", "t2t", "name", "t2t"],
        [
            "TH TTH Q",
            "th_tth",
            "t2t",
            "axes.axes_xref",
            {"q": "q", "th": "th", "tth": "tth"},
        ],
        ["TH TTH Q", "th_tth", "t2t", "axes.pseudo_axes", ["q"]],
        ["TH TTH Q", "th_tth", "t2t", "axes.real_axes", ["th", "tth"]],
        ["TH TTH Q", "th_tth", "t2t", "solver.geometry", "TH TTH Q"],
        ["TH TTH Q", "th_tth", "t2t", "solver.name", "th_tth"],
    ],
)
def test_asdict(geometry, solver, name, keypath, value):
    """."""
    diffractometer = creator(name=name, geometry=geometry, solver=solver)
    assert isinstance(
        diffractometer, DiffractometerBase
    ), f"{geometry=} {solver=} {name=}"
    assert isinstance(diffractometer.core, Core), f"{geometry=} {solver=} {name=}"

    db = diffractometer.core._asdict()
    assert db["name"] == name

    # Walk through the keypath, revising the db object at each step
    for k in keypath.split("."):
        db = db.get(k)  # narrow the search
        assert db is not None, f"{k=!r}"  # Ensure the path exists, so far.

    if value == SKIP_EXACT_VALUE_TEST:
        assert value is not None  # Anything BUT 'None'
    else:
        assert value == db, f"{value=!r}  {db=!r}"


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "pseudos, context, expected",
    [
        # * dict: {"h": 0, "k": 1, "l": -1}
        # * namedtuple: (h=0.0, k=1.0, l=-1.0)
        # * ordered list: [0, 1, -1]  (for h, k, l)
        # * ordered tuple: (0, 1, -1)  (for h, k, l)
        [dict(h=1, k=2, l=3), does_not_raise(), None],
        [[1, 2, 3], does_not_raise(), None],
        [(1, 2, 3), does_not_raise(), None],
        [
            namedtuple("PseudoTuple", "h k l".split())(1, 2, 3),
            does_not_raise(),
            None,
        ],
        [
            [1, 2, 3, 4],
            pytest.raises(UserWarning),
            "Extra inputs will be ignored. Expected 3.",
        ],
        [
            dict(h=1, k=2, lll=3),
            pytest.raises(KeyError),
            "Missing axis 'l'",
        ],
        [
            "abc",
            pytest.raises(TypeError),
            "Expected 'AnyAxesType'.",
        ],
    ],
)
def test_standardize_pseudos(pseudos, context, expected):
    with context as reason:
        fourc.core.standardize_pseudos(pseudos)
    assert_context_result(expected, reason)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "reals, context, expected",
    [
        [
            dict(omega=1, chi=2, phi=3, tth=4),
            does_not_raise(),
            None,
        ],
        [[1, 2, 3, 4], does_not_raise(), None],
        [(1, 2, 3, 4), does_not_raise(), None],
        [None, does_not_raise(), None],
        [
            [1, 2, 3, 4, 5],
            pytest.raises(UserWarning),
            "Extra inputs will be ignored. Expected 4.",
        ],
        [
            dict(theta=1, chi=2, phi=3, ttheta=4),
            pytest.raises(KeyError),
            "Missing axis 'omega'.",
        ],
        [
            "abcd",
            pytest.raises(TypeError),
            "Expected 'AnyAxesType'.",
        ],
    ],
)
def test_standardize_reals(reals, context, expected):
    with context as reason:
        fourc.core.standardize_reals(reals)
    assert_context_result(expected, reason)


def test_unknown_reflection():
    sim = creator(name="sim")
    set_diffractometer(sim)
    r1 = setor(1, 0, 0, 10, 0, 0, 20)

    with pytest.raises(KeyError) as reason:
        sim.core.calc_UB(r1, "r_unknown")
    assert_context_result(" unknown.  Knowns: ", reason)


@pytest.mark.parametrize(
    "pseudos, reals, assign, context, expected",
    [
        [
            "h k l".split(),
            dict(a=1, b=2, c=3, d=4),  # cannot use number as value
            "h b c d".split(),  # duplicate name is pseudos and reals
            pytest.raises(TypeError),
            "Incorrect type 'int' for pv=",
        ],
        [
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "h b c d".split(),  # same name used in both pseudos and reals
            pytest.raises(ValueError),
            "Axis name cannot be in more than list.",
        ],
        [
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "x y z".split(),  # unknown names
            pytest.raises(KeyError),
            "Unknown real='x'.",
        ],
        [
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "b a d c".split(),  # known names, specify in different order
            does_not_raise(),
            None,
        ],
        [
            "h k l".split(),
            dict(
                a=None,
                b="m5",
                c=None,
                d={"prefix": "m6"},
            ),
            "a b c d".split(),  # standard order
            pytest.raises(KeyError),
            "Expected 'class' key, received None",
        ],
        [
            "h k l".split(),
            dict(
                a=None,
                b="m5",
                c={"class": "ophyd.SoftPositioner", "init_pos": 10, "limits": (0, 50)},
                d={"class": "ophyd.EpicsMotor", "prefix": "m6", "labels": ["test"]},
            ),
            "a b c d".split(),  # standard order
            does_not_raise(),
            None,
        ],
    ],
)
def test_assign_axes(pseudos, reals, assign, context, expected):
    with context as reason:
        geom = creator(name="flaky", pseudos=pseudos, reals=reals)
        assert geom.pseudo_axis_names == pseudos
        assert geom.real_axis_names == list(reals)
        geom.core.assign_axes("h k l".split(), assign)
    assert_context_result(expected, reason)


def test_repeat_sample():
    geom = creator(name="geom")
    with pytest.raises(CoreError) as reason:
        geom.add_sample("sample", 4.1)
    expected = "Sample name='sample' already defined."
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "gonio, axes, prop, context, expected",
    [
        [fourc, "h k l".split(), "local_pseudo_axes", does_not_raise(), None],
        [
            creator(name="k4cv", geometry="K4CV"),
            "h k l".split(),
            "local_pseudo_axes",
            does_not_raise(),
            None,
        ],
        [
            creator(name="sixc", geometry="E6C", solver_kwargs=dict(engine="psi")),
            ["psi"],
            "local_pseudo_axes",
            does_not_raise(),
            None,
        ],
        [
            AugmentedFourc(name="a4c"),
            "h k l".split(),
            "local_pseudo_axes",
            does_not_raise(),
            None,
        ],
        [
            TwoC(name="cc"),
            ["another"],
            "local_pseudo_axes",
            pytest.raises(AssertionError),
            "assert ['q'] == ['another']",
        ],
        [TwoC(name="cc"), ["q"], "local_pseudo_axes", does_not_raise(), None],
        [
            MultiAxis99NoSolver(name="ma99"),
            [],
            "local_pseudo_axes",
            does_not_raise(),
            None,
        ],
        # ------------------
        [fourc, "omega chi phi tth".split(), "local_real_axes", does_not_raise(), None],
        [
            creator(name="k4cv", geometry="K4CV"),
            "komega kappa kphi tth".split(),
            "local_real_axes",
            does_not_raise(),
            None,
        ],
        [
            creator(name="sixc", geometry="E6C", solver_kwargs=dict(engine="psi")),
            "mu omega chi phi gamma delta".split(),
            "local_real_axes",
            does_not_raise(),
            None,
        ],
        [
            AugmentedFourc(name="a4c_again"),
            "omega chi phi tth".split(),
            "local_real_axes",
            does_not_raise(),
            None,
        ],
        [
            TwoC(name="cc"),
            ["another"],
            "local_real_axes",
            pytest.raises(AssertionError),
            "assert ['theta', 'ttheta'] == ['another']",
        ],
        [
            TwoC(name="cc"),
            ["theta", "ttheta"],
            "local_real_axes",
            does_not_raise(),
            None,
        ],
        [
            MultiAxis99NoSolver(name="ma99"),
            [],
            "local_real_axes",
            does_not_raise(),
            None,
        ],
    ],
)
def test_local_pseudo_axes(gonio, axes, prop, context, expected):
    with context as reason:
        assert getattr(gonio.core, prop) == axes
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "gonio, context, expected",
    [
        [MultiAxis99(name="ma99"), does_not_raise(), None],
        [
            MultiAxis99NoSolver(name="ma99"),
            pytest.raises(CoreError),
            "Did you forget to call `assign_axes()`?",
        ],
    ],
)
def test_axes_xref_reversed(gonio, context, expected):
    with context as reason:
        xref = gonio.core.axes_xref_reversed
        assert isinstance(xref, dict)
    assert_context_result(expected, reason)


def test_reset_samples():
    gonio = creator(name="gonio", solver="hkl_soleil", geometry="SOLEIL SIXS MED1+2")
    assert isinstance(gonio, DiffractometerBase)
    assert len(gonio.samples) == 1
    assert gonio.sample.name == DEFAULT_SAMPLE_NAME

    gonio.add_sample("vibranium", 2 * math.pi)
    assert len(gonio.samples) == 2
    gonio.add_sample("kryptonite", 0.01)
    assert len(gonio.samples) == 3

    gonio.core.reset_samples()
    assert len(gonio.samples) == 1
    assert gonio.sample.name == DEFAULT_SAMPLE_NAME


@pytest.mark.parametrize(
    "solver, geometry",
    [
        ["hkl_soleil", "APS POLAR"],
        ["hkl_soleil", "E4CV"],
        ["th_tth", "TH TTH Q"],
    ],
)
def test_signature(solver: str, geometry: str):
    sim = creator(name="sim", solver=solver, geometry=geometry)
    assert isinstance(sim, DiffractometerBase)
    core = sim.core
    assert isinstance(core, Core)
    assert isinstance(core.solver_signature, str)
    assert solver in core.solver_signature
    assert geometry in core.solver_signature


@pytest.mark.parametrize(
    "solver, geometry, mode",
    [
        ["hkl_soleil", "E4CV", "bissector"],
        ["hkl_soleil", "E4CV", "double_diffraction"],
        ["th_tth", "TH TTH Q", "bissector"],
    ],
)
def test_modes(solver: str, geometry: str, mode: str):
    sim = creator(name="sim", solver=solver, geometry=geometry)
    assert isinstance(sim, DiffractometerBase)
    core = sim.core
    assert isinstance(core, Core)

    assert mode in core.modes  # Is it available?
    core.mode = mode  # Set it.
    assert core.mode == mode  # Check it.


@pytest.mark.parametrize(
    "solver, geometry",
    [
        ["hkl_soleil", "E4CV"],
        ["hkl_soleil", "APS POLAR"],
        ["th_tth", "TH TTH Q"],
    ],
)
def test_solver_summary(solver: str, geometry: str):
    sim = creator(name="sim", solver=solver, geometry=geometry)
    assert isinstance(sim, DiffractometerBase)
    summary = sim.core.solver_summary
    assert isinstance(summary, pyRestTable.Table)


@pytest.mark.parametrize(
    "solver, geometry, solver_kwargs, expected",
    [
        ["hkl_soleil", "E4CV", {}, "h2 k2 l2 psi".split()],
        [
            "hkl_soleil",
            "K6C",
            {},
            "azimuth chi h2 incidence k2 l2 omega phi psi x y z".split(),
        ],
        ["hkl_soleil", "APS POLAR", {}, "h2 k2 l2 psi".split()],
        ["hkl_soleil", "APS POLAR", {"engine": "psi"}, "h2 k2 l2".split()],
        ["th_tth", "TH TTH Q", {}, []],
    ],
)
def test_all_extras(solver, geometry, solver_kwargs, expected):
    sim = creator(
        name="sim",
        solver=solver,
        geometry=geometry,
        solver_kwargs=solver_kwargs,
    )
    assert isinstance(sim.core.all_extras, dict)
    assert list(sim.core.all_extras) == list(sorted(expected))


@pytest.mark.parametrize(
    "solver, geometry, solver_kwargs, mode, expected",
    [
        ["hkl_soleil", "E4CV", {}, "bissector", []],
        [
            "hkl_soleil",
            "K6C",
            {},
            "constant_incidence",
            "x y z incidence azimuth".split(),
        ],
        [
            "hkl_soleil",
            "K6C",
            {"engine": "eulerians"},
            "eulerians",
            ["solutions"],
        ],
        ["hkl_soleil", "APS POLAR", {}, "lifting detector tau", []],
        [
            "hkl_soleil",
            "APS POLAR",
            {"engine": "psi"},
            "psi_vertical",
            "h2 k2 l2".split(),
        ],
        ["th_tth", "TH TTH Q", {}, "bissector", []],
    ],
)
def test_extras_getter(solver, geometry, solver_kwargs, mode, expected):
    sim = creator(
        name="sim",
        solver=solver,
        geometry=geometry,
        solver_kwargs=solver_kwargs,
    )
    sim.core.mode = mode
    assert isinstance(sim.core.extras, dict)
    assert list(sim.core.extras) == expected


@pytest.mark.parametrize(
    "solver, geometry, solver_kwargs, mode, values, context, expected",
    [
        ["hkl_soleil", "E4CV", {}, "bissector", dict(), does_not_raise(), None],
        [
            "hkl_soleil",
            "E4CV",
            {},
            "bissector",
            dict(h2=1),  # Parameter 'h2' not defined in the current mode.
            pytest.raises(KeyError),
            "Unexpected extra axis name(s)",
        ],
        [
            "hkl_soleil",
            "ZAXIS",
            dict(engine="qper_qpar"),
            "qper, qpar",
            dict(x=1, y=2, z=3),
            does_not_raise(),
            None,
        ],
        [
            "hkl_soleil",
            "SOLEIL SIXS MED2+3 v2",
            dict(engine="hkl"),
            "emergence_fixed",
            dict(z=3),  # incomplete dictionary is OK
            does_not_raise(),
            None,
        ],
    ],
)
def test_extras_setter(
    solver,
    geometry,
    solver_kwargs,
    mode,
    values,
    context,
    expected,
):
    with context as reason:
        sim = creator(
            name="sim",
            solver=solver,
            geometry=geometry,
            solver_kwargs=solver_kwargs,
        )
        sim.core.mode = mode
        sim.core.extras = values
        for key, value in values.items():
            assert key in sim.core.all_extras
            assert sim.core.extras.get(key) in (None, value)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "setup, config, context, expected",
    [
        [{}, {}, pytest.raises(KeyError), "'axes'"],
        [{}, {"axes": {}}, pytest.raises(KeyError), "'extra_axes'"],
        [  # 2
            {},
            {
                "axes": {"extra_axes": 0},
            },
            pytest.raises(AttributeError),
            "'items'",
        ],
        [  # 3
            {},
            {
                "axes": {"extra_axes": {}},
            },
            pytest.raises(KeyError),
            "'samples'",
        ],
        [  # 4
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": 0,
            },
            pytest.raises(AttributeError),
            "'items'",
        ],
        [  # 5
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
            },
            pytest.raises(KeyError),
            "'constraints'",
        ],
        [  # 6
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": 0,
            },
            pytest.raises(KeyError),
            "0",
        ],
        [  # 7
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": "wrong_name",
            },
            pytest.raises(KeyError),
            "'wrong_name'",
        ],
        [  # 8
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": "sample",
            },
            pytest.raises(KeyError),
            "'constraints'",
        ],
        [  # 9
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": 0,
            },
            pytest.raises(AttributeError),
            "'items'",
        ],
        [  # 10
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {},
            },
            does_not_raise(),
            None,
        ],
        [  # 11
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"not_dict": 0},
            },
            pytest.raises(TypeError),
            "'int' object is not subscriptable",
        ],
        [  # 12
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"needs_class": {}},
            },
            pytest.raises(KeyError),
            "'class'",
        ],
        [  # 13
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"abc": {"class": 0}},
            },
            pytest.raises(KeyError),
            "'abc'",
        ],
        [  # 14
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": 0}},
            },
            pytest.raises(ConfigurationError),
            "Wrong configuration class",
        ],
        [  # 15
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint"}},
            },
            pytest.raises(KeyError),
            "'label'",
        ],
        [  # 16
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(KeyError),
            "'real_axes'",
        ],
        [  # 17
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": 0},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(TypeError),
            "argument of type 'int' is not iterable",
        ],
        [  # 18
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": []},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(KeyError),
            "Constraint label axis=0 not found",
        ],
        [  # 19
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": []},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": "tth"}},
            },
            pytest.raises(ConfigurationError),
            "'low_limit'",
        ],
        [  # 20
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": []},
                "samples": {},
                "constraints": {
                    "tth": {
                        "class": "LimitsConstraint",
                        "label": "tth",
                        "high_limit": 125,
                        "low_limit": -5,
                    }
                },
            },
            does_not_raise(),
            None,
        ],
        [  # 21
            {},
            {
                "axes": {
                    "extra_axes": {},
                    "real_axes": [
                        "omega",
                        "chi",
                        "phi",
                        "tth",
                    ],
                },
                "samples": {},
                "constraints": {
                    "tth": {
                        "class": "LimitsConstraint",
                        "label": "tth",
                        "high_limit": 125,
                        "low_limit": -5,
                    }
                },
            },
            pytest.raises(KeyError),
            "'axes_xref'",
        ],
        [  # 22
            {},
            {
                "axes": {
                    "extra_axes": {},
                    "real_axes": [
                        "omega",
                        "chi",
                        "phi",
                        "tth",
                    ],
                    "axes_xref": {},
                },
                "samples": {},
                "constraints": {
                    "tth": {
                        "class": "LimitsConstraint",
                        "label": "tth",
                        "high_limit": 125,
                        "low_limit": -5,
                    }
                },
            },
            pytest.raises(KeyError),
            "'tth'",
        ],
        [  # 23
            {},
            {
                "axes": {
                    "extra_axes": {},
                    "real_axes": [
                        "omega",
                        "chi",
                        "phi",
                        "tth",
                    ],
                    "axes_xref": {"tth": "tth"},
                },
                "samples": {},
                "constraints": {
                    "tth": {
                        "class": "LimitsConstraint",
                        "label": "tth",
                        "high_limit": 125,
                        "low_limit": -5,
                    }
                },
            },
            does_not_raise(),
            None,
        ],
    ],  # does_not_raise(), None],
)
def test__fromdict(setup, config, context, expected):
    with context as reason:
        sim = creator(**setup)
        sim.core._fromdict(config)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "parms, context, expected",
    [
        [
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=1.2,
                name="r2",
                replace=False,
            ),
            does_not_raise(),
            None,
        ],
        [  # issue 97
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=None,  # in #97, this raised exception
                name="r2",
                replace=False,
            ),
            does_not_raise(),
            None,
        ],
        [
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=1.2,
                name="r1",
                replace=False,
            ),
            pytest.raises(ReflectionError),
            " known.  Use 'replace=True' to overwrite.",
        ],
        [
            dict(
                pseudos=(0, 0, 1),  # same as "first"
                reals=(1, 2, 3, 4),  # same as "first"
                name="r2",
            ),
            pytest.raises(ReflectionError),
            "matches one or more existing reflections.",
        ],
    ],
)
def test_add_reflection(parms, context, expected):
    with context as reason:
        sim = creator()
        # add first reflection
        sim.core.add_reflection((0, 0, 1), (1, 2, 3, 4), name="r1")
        # add supplied reflection
        sim.core.add_reflection(**parms)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "rnames, context, expected",
    [
        [[], does_not_raise(), None],
        [["r400", "r040", "r004"], does_not_raise(), None],
        [[], does_not_raise(), None],
        [["r400", "r040"], pytest.raises(CoreError), ""],
        [["r400", "r040", "wrong"], pytest.raises(KeyError), ""],
    ],
)
def test_refine_lattice(rnames, context, expected):
    with context as reason:
        e4cv = creator()
        add_oriented_vibranium_to_e4cv(e4cv)
        reflections = [e4cv.sample.reflections[key] for key in rnames]
        e4cv.core.refine_lattice(*reflections)

    assert_context_result(expected, reason)
