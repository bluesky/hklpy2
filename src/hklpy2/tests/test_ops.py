"""Test the hklpy2.ops module."""

import math
import re
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
from .models import AugmentedFourc
from .models import MultiAxis99
from .models import MultiAxis99NoSolver
from .models import TwoC

SKIP_EXACT_VALUE_TEST = str(uuid.uuid4())

fourc = creator()


@pytest.mark.parametrize(
    "geometry, solver, name, keypath, value",
    [
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "_header",
            SKIP_EXACT_VALUE_TEST,
            id="e4cv-header",
        ),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "_header.datetime",
            SKIP_EXACT_VALUE_TEST,
            id="e4cv-header-datetime",
        ),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "beam.wavelength",
            SKIP_EXACT_VALUE_TEST,
            id="e4cv-beam-wavelength",
        ),
        pytest.param("E4CV", "hkl_soleil", "fourc", "name", "fourc", id="e4cv-name"),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "solver.geometry",
            "E4CV",
            id="e4cv-solver-geometry",
        ),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "solver.name",
            "hkl_soleil",
            id="e4cv-solver-name",
        ),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "samples",
            SKIP_EXACT_VALUE_TEST,
            id="e4cv-samples",
        ),
        pytest.param(
            "E4CV",
            "hkl_soleil",
            "fourc",
            "solver.version",
            SKIP_EXACT_VALUE_TEST,
            id="e4cv-solver-version",
        ),
        #
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "_header",
            SKIP_EXACT_VALUE_TEST,
            id="thtthq-header",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "_header.datetime",
            SKIP_EXACT_VALUE_TEST,
            id="thtthq-header-datetime",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "beam.wavelength",
            SKIP_EXACT_VALUE_TEST,
            id="thtthq-beam-wavelength",
        ),
        pytest.param("TH TTH Q", "th_tth", "t2t", "name", "t2t", id="thtthq-name"),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "axes.axes_xref",
            {"q": "q", "th": "th", "tth": "tth"},
            id="thtthq-axes-xref",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "axes.pseudo_axes",
            ["q"],
            id="thtthq-pseudo-axes",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "axes.real_axes",
            ["th", "tth"],
            id="thtthq-real-axes",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "solver.geometry",
            "TH TTH Q",
            id="thtthq-solver-geometry",
        ),
        pytest.param(
            "TH TTH Q",
            "th_tth",
            "t2t",
            "solver.name",
            "th_tth",
            id="thtthq-solver-name",
        ),
    ],
)
def test_asdict(geometry, solver, name, keypath, value):
    """."""
    diffractometer = creator(name=name, geometry=geometry, solver=solver)
    assert isinstance(diffractometer, DiffractometerBase), (
        f"{geometry=} {solver=} {name=}"
    )
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
    "pseudos, context",
    [
        # * dict: {"h": 0, "k": 1, "l": -1}
        # * namedtuple: (h=0.0, k=1.0, l=-1.0)
        # * ordered list: [0, 1, -1]  (for h, k, l)
        # * ordered tuple: (0, 1, -1)  (for h, k, l)
        pytest.param(dict(h=1, k=2, l=3), does_not_raise(), id="dict"),
        pytest.param([1, 2, 3], does_not_raise(), id="list"),
        pytest.param((1, 2, 3), does_not_raise(), id="tuple"),
        pytest.param(
            namedtuple("PseudoTuple", "h k l".split())(1, 2, 3),
            does_not_raise(),
            id="namedtuple",
        ),
        pytest.param(
            [1, 2, 3, 4],
            pytest.raises(
                UserWarning,
                match=re.escape("Extra inputs will be ignored. Expected 3."),
            ),
            id="extra-inputs",
        ),
        pytest.param(
            dict(h=1, k=2, lll=3),
            pytest.raises(KeyError, match=re.escape("Missing axis 'l'")),
            id="missing-axis",
        ),
        pytest.param(
            "abc",
            pytest.raises(TypeError, match=re.escape("Expected 'AnyAxesType'.")),
            id="wrong-type",
        ),
    ],
)
def test_standardize_pseudos(pseudos, context):
    with context:
        fourc.core.standardize_pseudos(pseudos)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(
    "reals, context",
    [
        pytest.param(dict(omega=1, chi=2, phi=3, tth=4), does_not_raise(), id="dict"),
        pytest.param([1, 2, 3, 4], does_not_raise(), id="list"),
        pytest.param((1, 2, 3, 4), does_not_raise(), id="tuple"),
        pytest.param(None, does_not_raise(), id="none"),
        pytest.param(
            [1, 2, 3, 4, 5],
            pytest.raises(
                UserWarning,
                match=re.escape("Extra inputs will be ignored. Expected 4."),
            ),
            id="extra-inputs",
        ),
        pytest.param(
            dict(theta=1, chi=2, phi=3, ttheta=4),
            pytest.raises(KeyError, match=re.escape("Missing axis 'omega'.")),
            id="missing-axis",
        ),
        pytest.param(
            "abcd",
            pytest.raises(TypeError, match=re.escape("Expected 'AnyAxesType'.")),
            id="wrong-type",
        ),
    ],
)
def test_standardize_reals(reals, context):
    with context:
        fourc.core.standardize_reals(reals)


def test_unknown_reflection():
    sim = creator(name="sim")
    set_diffractometer(sim)
    r1 = setor(1, 0, 0, 10, 0, 0, 20)

    with pytest.raises(KeyError, match=re.escape(" unknown.  Knowns: ")):
        sim.core.calc_UB(r1, "r_unknown")


@pytest.mark.parametrize(
    "pseudos, reals, assign, context",
    [
        pytest.param(
            "h k l".split(),
            dict(a=1, b=2, c=3, d=4),  # cannot use number as value
            "h b c d".split(),  # duplicate name is pseudos and reals
            pytest.raises(
                TypeError, match=re.escape("Incorrect type 'int' for specs=")
            ),
            id="Incorrect type 'int' for specs=",
        ),
        pytest.param(
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "h b c d".split(),  # same name used in both pseudos and reals
            pytest.raises(
                ValueError, match=re.escape("Axis name cannot be in more than list.")
            ),
            id="Axis name cannot be in more than list.",
        ),
        pytest.param(
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "x y z".split(),  # unknown names
            pytest.raises(KeyError, match=re.escape("Unknown real='x'.")),
            id="unknown-real",
        ),
        pytest.param(
            "h k l".split(),
            dict(a=None, b="m5", c=None, d=None),
            "b a d c".split(),  # known names, specify in different order
            does_not_raise(),
            id="reorder-reals",
        ),
        pytest.param(
            "h k l".split(),
            dict(
                a=None,
                b="m5",
                c=None,
                d={"prefix": "m6"},
            ),
            "a b c d".split(),  # standard order
            pytest.raises(
                KeyError,
                match=re.escape("Expected 'class' key, received None"),
            ),
            id="Expected 'class' key, received None",
        ),
        pytest.param(
            "h k l".split(),
            dict(
                a=None,
                b="m5",
                c={"class": "ophyd.SoftPositioner", "init_pos": 10, "limits": (0, 50)},
                d={"class": "ophyd.EpicsMotor", "prefix": "m6", "labels": ["test"]},
            ),
            "a b c d".split(),  # standard order
            does_not_raise(),
            id="Ok: mixed reals",
        ),
    ],
)
def test_assign_axes(pseudos, reals, assign, context):
    with context:
        geom = creator(name="flaky", pseudos=pseudos, reals=reals)
        assert geom.pseudo_axis_names == pseudos
        assert geom.real_axis_names == list(reals)
        geom.core.assign_axes("h k l".split(), assign)


def test_repeat_sample():
    geom = creator(name="geom")
    with pytest.raises(
        CoreError, match=re.escape("Sample name='sample' already defined.")
    ):
        geom.add_sample("sample", 4.1)


@pytest.mark.parametrize(
    "gonio, axes, prop, context",
    [
        pytest.param(
            fourc,
            "h k l".split(),
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-fourc",
        ),
        pytest.param(
            creator(name="k4cv", geometry="K4CV"),
            "h k l".split(),
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-k4cv",
        ),
        pytest.param(
            creator(name="sixc", geometry="E6C", solver_kwargs=dict(engine="psi")),
            ["psi"],
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-e6c-psi",
        ),
        pytest.param(
            AugmentedFourc(name="a4c"),
            "h k l".split(),
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-augmented",
        ),
        pytest.param(
            TwoC(name="cc"),
            ["another"],
            "local_pseudo_axes",
            pytest.raises(
                AssertionError, match=re.escape("assert ['q'] == ['another']")
            ),
            id="pseudo-twoc-wrong",
        ),
        pytest.param(
            TwoC(name="cc"),
            ["q"],
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-twoc",
        ),
        pytest.param(
            MultiAxis99NoSolver(name="ma99"),
            [],
            "local_pseudo_axes",
            does_not_raise(),
            id="pseudo-ma99",
        ),
        # ------------------
        pytest.param(
            fourc,
            "omega chi phi tth".split(),
            "local_real_axes",
            does_not_raise(),
            id="real-fourc",
        ),
        pytest.param(
            creator(name="k4cv", geometry="K4CV"),
            "komega kappa kphi tth".split(),
            "local_real_axes",
            does_not_raise(),
            id="real-k4cv",
        ),
        pytest.param(
            creator(name="sixc", geometry="E6C", solver_kwargs=dict(engine="psi")),
            "mu omega chi phi gamma delta".split(),
            "local_real_axes",
            does_not_raise(),
            id="real-e6c-psi",
        ),
        pytest.param(
            AugmentedFourc(name="a4c_again"),
            "omega chi phi tth".split(),
            "local_real_axes",
            does_not_raise(),
            id="real-augmented",
        ),
        pytest.param(
            TwoC(name="cc"),
            ["another"],
            "local_real_axes",
            pytest.raises(
                AssertionError,
                match=re.escape("assert ['theta', 'ttheta'] == ['another']"),
            ),
            id="real-twoc-wrong",
        ),
        pytest.param(
            TwoC(name="cc"),
            ["theta", "ttheta"],
            "local_real_axes",
            does_not_raise(),
            id="real-twoc",
        ),
        pytest.param(
            MultiAxis99NoSolver(name="ma99"),
            [],
            "local_real_axes",
            does_not_raise(),
            id="real-ma99",
        ),
    ],
)
def test_local_pseudo_axes(gonio, axes, prop, context):
    with context:
        assert getattr(gonio.core, prop) == axes


@pytest.mark.parametrize(
    "gonio, context",
    [
        pytest.param(MultiAxis99(name="ma99"), does_not_raise(), id="with-solver"),
        pytest.param(
            MultiAxis99NoSolver(name="ma99"),
            pytest.raises(
                CoreError, match=re.escape("Did you forget to call `assign_axes()`?")
            ),
            id="no-solver",
        ),
    ],
)
def test_axes_xref_reversed(gonio, context):
    with context:
        xref = gonio.core.axes_xref_reversed
        assert isinstance(xref, dict)


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
        pytest.param("hkl_soleil", "APS POLAR", id="aps-polar"),
        pytest.param("hkl_soleil", "E4CV", id="e4cv"),
        pytest.param("th_tth", "TH TTH Q", id="th-tth-q"),
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
        pytest.param("hkl_soleil", "E4CV", "bissector", id="e4cv-bissector"),
        pytest.param(
            "hkl_soleil", "E4CV", "double_diffraction", id="e4cv-double-diffraction"
        ),
        pytest.param("th_tth", "TH TTH Q", "bissector", id="thtthq-bissector"),
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
        pytest.param("hkl_soleil", "E4CV", id="e4cv"),
        pytest.param("hkl_soleil", "APS POLAR", id="aps-polar"),
        pytest.param("th_tth", "TH TTH Q", id="th-tth-q"),
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
        pytest.param("hkl_soleil", "E4CV", {}, "h2 k2 l2 psi".split(), id="e4cv"),
        pytest.param(
            "hkl_soleil",
            "K6C",
            {},
            "azimuth chi h2 incidence k2 l2 omega phi psi x y z".split(),
            id="k6c",
        ),
        pytest.param(
            "hkl_soleil", "APS POLAR", {}, "h2 k2 l2 psi".split(), id="aps-polar"
        ),
        pytest.param(
            "hkl_soleil",
            "APS POLAR",
            {"engine": "psi"},
            "h2 k2 l2".split(),
            id="aps-polar-psi",
        ),
        pytest.param("th_tth", "TH TTH Q", {}, [], id="th-tth-q"),
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
        pytest.param("hkl_soleil", "E4CV", {}, "bissector", [], id="e4cv-bissector"),
        pytest.param(
            "hkl_soleil",
            "K6C",
            {},
            "constant_incidence",
            "x y z incidence azimuth".split(),
            id="k6c-constant-incidence",
        ),
        pytest.param(
            "hkl_soleil",
            "K6C",
            {"engine": "eulerians"},
            "eulerians",
            ["solutions"],
            id="k6c-eulerians",
        ),
        pytest.param(
            "hkl_soleil",
            "APS POLAR",
            {},
            "lifting detector tau",
            [],
            id="aps-polar-lifting-detector",
        ),
        pytest.param(
            "hkl_soleil",
            "APS POLAR",
            {"engine": "psi"},
            "psi_vertical",
            "h2 k2 l2".split(),
            id="aps-polar-psi-vertical",
        ),
        pytest.param(
            "th_tth", "TH TTH Q", {}, "bissector", [], id="th-tth-q-bissector"
        ),
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
    "solver, geometry, solver_kwargs, mode, values, context",
    [
        pytest.param(
            "hkl_soleil",
            "E4CV",
            {},
            "bissector",
            dict(),
            does_not_raise(),
            id="empty-extras",
        ),
        pytest.param(
            "hkl_soleil",
            "E4CV",
            {},
            "bissector",
            dict(h2=1),  # Parameter 'h2' not defined in the current mode.
            pytest.raises(KeyError, match=re.escape("Unexpected extra axis name(s)")),
            id="unexpected-extra",
        ),
        pytest.param(
            "hkl_soleil",
            "ZAXIS",
            dict(engine="qper_qpar"),
            "qper, qpar",
            dict(x=1, y=2, z=3),
            does_not_raise(),
            id="zaxis-qper-qpar",
        ),
        pytest.param(
            "hkl_soleil",
            "SOLEIL SIXS MED2+3 v2",
            dict(engine="hkl"),
            "emergence_fixed",
            dict(z=3),  # incomplete dictionary is OK
            does_not_raise(),
            id="partial-extras",
        ),
    ],
)
def test_extras_setter(
    solver,
    geometry,
    solver_kwargs,
    mode,
    values,
    context,
):
    with context:
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


@pytest.mark.parametrize(
    "setup, config, context",
    [
        pytest.param(
            {},
            {},
            pytest.raises(KeyError, match=re.escape("'axes'")),
            id="missing-axes",
        ),
        pytest.param(
            {},
            {"axes": {}},
            pytest.raises(KeyError, match=re.escape("'extra_axes'")),
            id="missing-extra_axes",
        ),
        pytest.param(  # 2
            {},
            {
                "axes": {"extra_axes": 0},
            },
            pytest.raises(AttributeError, match=re.escape("'items'")),
            id="extra_axes-not-dict",
        ),
        pytest.param(  # 3
            {},
            {
                "axes": {"extra_axes": {}},
            },
            pytest.raises(KeyError, match=re.escape("'samples'")),
            id="missing-samples",
        ),
        pytest.param(  # 4
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": 0,
            },
            pytest.raises(AttributeError, match=re.escape("'items'")),
            id="samples-not-dict",
        ),
        pytest.param(  # 5
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
            },
            pytest.raises(KeyError, match=re.escape("'constraints'")),
            id="missing-constraints",
        ),
        pytest.param(  # 6
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": 0,
            },
            pytest.raises(KeyError, match=re.escape("0")),
            id="sample_name-int",
        ),
        pytest.param(  # 7
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": "wrong_name",
            },
            pytest.raises(KeyError, match=re.escape("'wrong_name'")),
            id="sample_name-wrong",
        ),
        pytest.param(  # 8
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "sample_name": "sample",
            },
            pytest.raises(KeyError, match=re.escape("'constraints'")),
            id="sample-no-constraints",
        ),
        pytest.param(  # 9
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": 0,
            },
            pytest.raises(AttributeError, match=re.escape("'items'")),
            id="constraints-not-dict",
        ),
        pytest.param(  # 10
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {},
            },
            does_not_raise(),
            id="empty-constraints-ok",
        ),
        pytest.param(  # 11
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"not_dict": 0},
            },
            pytest.raises(
                TypeError, match=re.escape("'int' object is not subscriptable")
            ),
            id="constraint-not-subscriptable",
        ),
        pytest.param(  # 12
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"needs_class": {}},
            },
            pytest.raises(KeyError, match=re.escape("'class'")),
            id="constraint-no-class",
        ),
        pytest.param(  # 13
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"abc": {"class": 0}},
            },
            pytest.raises(KeyError, match=re.escape("'abc'")),
            id="constraint-unknown-axis",
        ),
        pytest.param(  # 14
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": 0}},
            },
            pytest.raises(
                ConfigurationError, match=re.escape("Wrong configuration class")
            ),
            id="constraint-wrong-class",
        ),
        pytest.param(  # 15
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint"}},
            },
            pytest.raises(KeyError, match=re.escape("'label'")),
            id="constraint-no-label",
        ),
        pytest.param(  # 16
            {},
            {
                "axes": {"extra_axes": {}},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(KeyError, match=re.escape("'real_axes'")),
            id="constraint-no-real_axes",
        ),
        pytest.param(  # 17
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": 0},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(
                TypeError, match=re.escape("argument of type 'int' is not iterable")
            ),
            id="real_axes-not-iterable",
        ),
        pytest.param(  # 18
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": []},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": 0}},
            },
            pytest.raises(
                KeyError, match=re.escape("Constraint label axis=0 not found")
            ),
            id="constraint-label-not-found",
        ),
        pytest.param(  # 19
            {},
            {
                "axes": {"extra_axes": {}, "real_axes": []},
                "samples": {},
                "constraints": {"tth": {"class": "LimitsConstraint", "label": "tth"}},
            },
            pytest.raises(ConfigurationError, match=re.escape("'low_limit'")),
            id="constraint-no-low_limit",
        ),
        pytest.param(  # 20
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
            id="limits-ok",
        ),
        pytest.param(  # 21
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
            pytest.raises(KeyError, match=re.escape("'axes_xref'")),
            id="missing-axes_xref",
        ),
        pytest.param(  # 22
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
            pytest.raises(KeyError, match=re.escape("'tth'")),
            id="axes_xref-missing-tth",
        ),
        pytest.param(  # 23
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
            id="full-config-ok",
        ),
    ],
)
def test__fromdict(setup, config, context):
    with context:
        sim = creator(**setup)
        sim.core._fromdict(config)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=1.2,
                name="r2",
                replace=False,
            ),
            does_not_raise(),
            id="add-r2",
        ),
        pytest.param(  # issue 97
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=None,  # in #97, this raised exception
                name="r2",
                replace=False,
            ),
            does_not_raise(),
            id="add-r2-no-wavelength",
        ),
        pytest.param(
            dict(
                pseudos=(0, 1, 0),
                reals=None,
                wavelength=1.2,
                name="r1",
                replace=False,
            ),
            pytest.raises(
                ReflectionError,
                match=re.escape(" known.  Use 'replace=True' to overwrite."),
            ),
            id="duplicate-name",
        ),
        pytest.param(
            dict(
                pseudos=(0, 0, 1),  # same as "first"
                reals=(1, 2, 3, 4),  # same as "first"
                name="r2",
            ),
            pytest.raises(
                ReflectionError,
                match=re.escape("matches one or more existing reflections."),
            ),
            id="duplicate-reflection",
        ),
    ],
)
def test_add_reflection(parms, context):
    with context:
        sim = creator()
        # add first reflection
        sim.core.add_reflection((0, 0, 1), (1, 2, 3, 4), name="r1")
        # add supplied reflection
        sim.core.add_reflection(**parms)


@pytest.mark.parametrize(
    "rnames, context",
    [
        pytest.param([], does_not_raise(), id="no-reflections"),
        pytest.param(
            ["r400", "r040", "r004"], does_not_raise(), id="three-reflections"
        ),
        pytest.param([], does_not_raise(), id="no-reflections-again"),
        pytest.param(
            ["r400", "r040"], pytest.raises(CoreError), id="too-few-reflections"
        ),
        pytest.param(
            ["r400", "r040", "wrong"], pytest.raises(KeyError), id="unknown-reflection"
        ),
    ],
)
def test_refine_lattice(rnames, context):
    with context:
        e4cv = creator()
        add_oriented_vibranium_to_e4cv(e4cv)
        reflections = [e4cv.sample.reflections[key] for key in rnames]
        e4cv.core.refine_lattice(*reflections)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            pytest.raises(
                CoreError,
                match=re.escape("does not support lattice refinement"),
            ),
            id="th_tth solver raises CoreError for unsupported refinement",
        ),
    ],
)
def test_refine_lattice_unsupported_solver(parms, context):
    with context:
        sim = creator(solver="th_tth", geometry="TH TTH Q")
        sim.wavelength = 1.0
        # Add 3 reflections to pass the minimum-count guard.
        for i, tth in enumerate([30, 60, 90]):
            sim.add_reflection(
                (tth / 2,), reals=dict(th=tth / 2, tth=tth), name=f"r{i}"
            )
        sim.core.refine_lattice()
