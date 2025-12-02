from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...misc import ConfigurationError
from ...tests.common import assert_context_result
from ..constraints import ConstraintBase
from ..constraints import ConstraintsError
from ..constraints import LimitsConstraint
from ..constraints import RealAxisConstraints


class PlainConstraint(ConstraintBase):
    def valid(self, **values):
        return True


def test_raises():
    with pytest.raises(TypeError) as excuse:
        ConstraintBase()
    assert "Can't instantiate abstract class" in str(excuse)

    with pytest.raises(ConstraintsError) as excuse:
        LimitsConstraint(0, 1)
    assert "Must provide a value" in str(excuse)

    c = LimitsConstraint(0, 1, label="test")
    with pytest.raises(ConstraintsError) as excuse:
        c.valid()
    assert "did not include this constraint" in str(excuse)


@pytest.mark.parametrize(
    "lo, hi, value, result",
    [
        [None, None, 0, True],
        [None, None, 2000, False],
        [0, None, 0, True],
        [0, None, -1, False],
        [10, 20, 0, False],
        [10, 20, 15, True],
        [20, 10, 10, True],
        [20, 10, 15, True],
        [20, 10, 20, True],
    ],
)
def test_LimitsConstraint(lo, hi, value, result):
    c = LimitsConstraint(lo, hi, label="axis")
    assert len(c._asdict()) == 4

    text = str(c)
    assert " <= " in text

    assert c.low_limit == lo or -180, f"{c!r}"
    assert c.high_limit == hi or 180, f"{c!r}"
    assert c.valid(axis=value) == result, f"{c!r}"


@pytest.mark.parametrize(
    "reals, result",
    [
        [{"aa": 0, "bb": 0, "cc": 0}, True],
        [{"aa": 0, "bb": 200, "cc": 0}, False],
    ],
)
def test_RealAxisConstraints(reals, result):
    ac = RealAxisConstraints(list(reals))
    assert len(ac) == len(reals)
    assert len(ac._asdict()) == len(reals), f"{ac._asdict()!r}"
    assert ac.valid(**reals) == result


@pytest.mark.parametrize(
    "supplied, kwargs, context, expected",
    [
        ["you me".split(), dict(you=0, me=0), does_not_raise(), None],
        [
            "tinker evers chance".split(),
            dict(you=0, me=0),
            pytest.raises(ConstraintsError),
            "did not include this constraint",
        ],
    ],
)
def test_RealAxisConstraintsKeys(supplied, kwargs, context, expected):
    ac = RealAxisConstraints(supplied)
    with context as reason:
        ac.valid(**kwargs)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "config, context, expected",
    [
        [
            {
                "th": {
                    "class": "LimitsConstraint",
                    "high_limit": 120.0,
                    "label": "th",
                    "low_limit": -5.0,
                },
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                },
            },
            does_not_raise(),
            None,
        ],
        [
            {
                "omega": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "omega",
                    "low_limit": 30.0,
                },
            },
            pytest.raises(KeyError),
            "omega",
        ],
        [
            {
                "tth": {
                    "class": "LimitsConstraint",
                    # "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                },
            },
            pytest.raises(ConfigurationError),
            "Missing key for LimitsConstraint",
        ],
        [
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    # "low_limit": 30.0,
                },
            },
            pytest.raises(ConfigurationError),
            "Missing key for LimitsConstraint",
        ],
        [
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    # "label": "tth",
                    "low_limit": 30.0,
                },
            },
            pytest.raises(ConfigurationError),
            " Expected key: 'label'.",
        ],
        [
            {
                "tth": {
                    # "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                },
            },
            pytest.raises(KeyError),
            "class",
        ],
        [
            {
                "tth": {
                    "class": "WrongClassLimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                },
            },
            pytest.raises(ConfigurationError),
            "class",
        ],
        [
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "wrong label",
                    "low_limit": 30.0,
                },
            },
            does_not_raise(),
            None,
        ],
    ],
)
def test_fromdict(config, context, expected):
    with context as reason:
        assert isinstance(config, dict)
        sim2c = creator(name="sim2c", solver="th_tth", geometry="TH TTH Q")
        ac = sim2c.core.constraints
        ac._fromdict(config)
        for axis in config:
            assert axis in ac
            assert ac[axis].low_limit == config[axis]["low_limit"]
            assert ac[axis].high_limit == config[axis]["high_limit"]

    assert_context_result(expected, reason)


def test_fromdict_KeyError():
    """Edge case: restore custom real which differs from local custom real."""
    config = {
        "class": "LimitsConstraint",
        "high_limit": 85.0,
        "label": "incoming",
        "low_limit": 30.0,
    }
    context = pytest.raises(KeyError)
    expected = " not found in diffractometer reals: "
    with context as reason:
        e4cv = creator(
            name="e4cv",
            reals=dict(aaa=None, bbb=None, ccc=None, ddd=None),
        )
        constraint = e4cv.core.constraints["aaa"]
        constraint._fromdict(config, core=e4cv.core)
    assert_context_result(expected, reason)


def test_repr():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    rep = repr(sim.core.constraints)
    assert rep.startswith("[")
    assert "-180.0 <= th <= 180.0" in rep
    assert "-180.0 <= tth <= 180.0" in rep
    assert rep.endswith("]")


def test_limits_property():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    constraint = sim.core.constraints["th"]
    assert constraint.limits == (-180, 180)
    constraint.limits = 0, 20.1
    assert constraint.limits == (0, 20.1)

    expected = "Use exactly two values"
    with pytest.raises(ConstraintsError) as reason:
        constraint.limits = 0, 20.1, 3
    assert_context_result(expected, reason)


def test_ConstraintsBase():
    expected = None
    with does_not_raise() as reason:
        constraint = PlainConstraint()
        assert constraint.valid(key="ignored", also="ignored")

        rep = repr(constraint)
        assert rep.startswith("PlainConstraint(")
        assert "class=" in rep
        assert rep.endswith(")")

    assert_context_result(expected, reason)
