import math
import re
from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...exceptions import ConfigurationError
from ..constraints import DEFAULT_CUT_POINT
from ..constraints import ENDPOINT_TOLERANCE
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
        pytest.param(None, None, 0, True, id="default-limits-zero"),
        pytest.param(None, None, 2000, False, id="default-limits-2000"),
        pytest.param(0, None, 0, True, id="lo-0-hi-none-val-0"),
        pytest.param(0, None, -1, False, id="lo-0-hi-none-val-neg1"),
        pytest.param(10, 20, 0, False, id="below-range"),
        pytest.param(10, 20, 15, True, id="within-range"),
        pytest.param(20, 10, 10, True, id="reversed-at-lo"),
        pytest.param(20, 10, 15, True, id="reversed-within"),
        pytest.param(20, 10, 20, True, id="reversed-at-hi"),
        # floating-point boundary cases (issue #242): solver may return
        # values like 180.0000001 which must still be accepted at limit=180
        pytest.param(
            -180, 180, 180 + ENDPOINT_TOLERANCE * 0.5, True, id="fp-at-hi-within-tol"
        ),
        pytest.param(
            -180, 180, -180 - ENDPOINT_TOLERANCE * 0.5, True, id="fp-at-lo-within-tol"
        ),
        pytest.param(
            -180, 180, 180 + ENDPOINT_TOLERANCE * 2, False, id="fp-beyond-hi-tol"
        ),
        pytest.param(
            -180, 180, -180 - ENDPOINT_TOLERANCE * 2, False, id="fp-beyond-lo-tol"
        ),
    ],
)
def test_LimitsConstraint(lo, hi, value, result):
    c = LimitsConstraint(lo, hi, label="axis")
    assert len(c._asdict()) == 5  # label, low_limit, high_limit, cut_point, class

    text = str(c)
    assert " <= " in text
    assert "[cut=" in text

    assert c.low_limit == lo or -180, f"{c!r}"
    assert c.high_limit == hi or 180, f"{c!r}"
    assert c.valid(axis=value) == result, f"{c!r}"


@pytest.mark.parametrize(
    "reals, result",
    [
        pytest.param({"aa": 0, "bb": 0, "cc": 0}, True, id="all-within-limits"),
        pytest.param({"aa": 0, "bb": 200, "cc": 0}, False, id="bb-out-of-limits"),
    ],
)
def test_RealAxisConstraints(reals, result):
    ac = RealAxisConstraints(list(reals))
    assert len(ac) == len(reals)
    assert len(ac._asdict()) == len(reals), f"{ac._asdict()!r}"
    assert ac.valid(**reals) == result


@pytest.mark.parametrize(
    "supplied, kwargs, context",
    [
        pytest.param(
            "you me".split(), dict(you=0, me=0), does_not_raise(), id="matching-keys"
        ),
        pytest.param(
            "tinker evers chance".split(),
            dict(you=0, me=0),
            pytest.raises(
                ConstraintsError, match=re.escape("did not include this constraint")
            ),
            id="mismatched-keys",
        ),
    ],
)
def test_RealAxisConstraintsKeys(supplied, kwargs, context):
    ac = RealAxisConstraints(supplied)
    with context:
        ac.valid(**kwargs)


@pytest.mark.parametrize(
    "config, context",
    [
        pytest.param(
            {
                "th": {
                    "class": "LimitsConstraint",
                    "high_limit": 120.0,
                    "label": "th",
                    "low_limit": -5.0,
                    "cut_point": -5.0,
                },
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            does_not_raise(),
            id="valid-config",
        ),
        pytest.param(
            {
                "omega": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "omega",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(KeyError, match=re.escape("omega")),
            id="unknown-axis-omega",
        ),
        pytest.param(
            {
                "tth": {
                    "class": "LimitsConstraint",
                    # "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(
                ConfigurationError, match=re.escape("Missing key for LimitsConstraint")
            ),
            id="missing-high-limit",
        ),
        pytest.param(
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    # "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(
                ConfigurationError, match=re.escape("Missing key for LimitsConstraint")
            ),
            id="missing-low-limit",
        ),
        pytest.param(
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    # "label": "tth",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(
                ConfigurationError, match=re.escape(" Expected key: 'label'.")
            ),
            id="missing-label",
        ),
        pytest.param(
            {
                "tth": {
                    # "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(KeyError, match=re.escape("class")),
            id="missing-class",
        ),
        pytest.param(
            {
                "tth": {
                    "class": "WrongClassLimitsConstraint",
                    "high_limit": 85.0,
                    "label": "tth",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            pytest.raises(ConfigurationError, match=re.escape("class")),
            id="wrong-class",
        ),
        pytest.param(
            {
                "tth": {
                    "class": "LimitsConstraint",
                    "high_limit": 85.0,
                    "label": "wrong label",
                    "low_limit": 30.0,
                    "cut_point": -180.0,
                },
            },
            does_not_raise(),
            id="wrong-label-no-error",
        ),
    ],
)
def test_fromdict(config, context):
    with context:
        assert isinstance(config, dict)
        sim2c = creator(name="sim2c", solver="th_tth", geometry="TH TTH Q")
        ac = sim2c.core.constraints
        ac._fromdict(config)
        for axis in config:
            assert axis in ac
            assert ac[axis].low_limit == config[axis]["low_limit"]
            assert ac[axis].high_limit == config[axis]["high_limit"]
            assert ac[axis].cut_point == config[axis]["cut_point"]


def test_fromdict_KeyError():
    """Edge case: restore custom real which differs from local custom real."""
    config = {
        "class": "LimitsConstraint",
        "high_limit": 85.0,
        "label": "incoming",
        "low_limit": 30.0,
        "cut_point": -180.0,
    }
    with pytest.raises(
        KeyError, match=re.escape(" not found in diffractometer reals: ")
    ):
        e4cv = creator(
            name="e4cv",
            reals=dict(aaa=None, bbb=None, ccc=None, ddd=None),
        )
        constraint = e4cv.core.constraints["aaa"]
        constraint._fromdict(config, core=e4cv.core)


def test_repr():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    rep = repr(sim.core.constraints)
    assert rep.startswith("[")
    assert "-180.0 <= th <= 180.0 [cut=-180.0]" in rep
    assert "-180.0 <= tth <= 180.0 [cut=-180.0]" in rep
    assert rep.endswith("]")


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cut_point=0.0),
            does_not_raise(),
            id="zero is valid",
        ),
        pytest.param(
            dict(cut_point=-180.0),
            does_not_raise(),
            id="default is valid",
        ),
        pytest.param(
            dict(cut_point=720.0),
            does_not_raise(),
            id="large finite value is valid",
        ),
        pytest.param(
            dict(cut_point=float("inf")),
            pytest.raises(
                ConstraintsError, match=re.escape("cut_point must be a finite number")
            ),
            id="inf rejected",
        ),
        pytest.param(
            dict(cut_point=float("-inf")),
            pytest.raises(
                ConstraintsError, match=re.escape("cut_point must be a finite number")
            ),
            id="-inf rejected",
        ),
        pytest.param(
            dict(cut_point=float("nan")),
            pytest.raises(
                ConstraintsError, match=re.escape("cut_point must be a finite number")
            ),
            id="nan rejected",
        ),
    ],
)
def test_cut_point_validation(parms, context):
    with context:
        c = LimitsConstraint(label="axis", cut_point=parms["cut_point"])
        assert math.isfinite(c.cut_point)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(attr="cut_point", value=0.0),
            does_not_raise(),
            id="cut_point finite accepted",
        ),
        pytest.param(
            dict(attr="cut_point", value=float("nan")),
            pytest.raises(
                ConstraintsError, match=re.escape("cut_point must be a finite number")
            ),
            id="cut_point nan rejected post-construction",
        ),
        pytest.param(
            dict(attr="cut_point", value=float("inf")),
            pytest.raises(
                ConstraintsError, match=re.escape("cut_point must be a finite number")
            ),
            id="cut_point inf rejected post-construction",
        ),
        pytest.param(
            dict(attr="low_limit", value=50.0),
            does_not_raise(),
            id="low_limit set via property sorted with high",
        ),
        pytest.param(
            dict(attr="high_limit", value=-50.0),
            does_not_raise(),
            id="high_limit below low_limit is sorted",
        ),
    ],
)
def test_post_construction_validation(parms, context):
    c = LimitsConstraint(-180, 180, label="axis")
    with context:
        setattr(c, parms["attr"], parms["value"])
        # after setting low or high, sorted order is maintained
        if parms["attr"] in ("low_limit", "high_limit"):
            assert c.low_limit <= c.high_limit


def test_limits_property():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    constraint = sim.core.constraints["th"]
    assert constraint.limits == (-180, 180)
    constraint.limits = 0, 20.1
    assert constraint.limits == (0, 20.1)

    with pytest.raises(ConstraintsError, match=re.escape("Use exactly two values")):
        constraint.limits = 0, 20.1, 3


def test_ConstraintsBase():
    with does_not_raise():
        constraint = PlainConstraint()
        assert constraint.valid(key="ignored", also="ignored")

        rep = repr(constraint)
        assert rep.startswith("PlainConstraint(")
        assert "class=" in rep
        assert rep.endswith(")")


# ---------------------------------------------------------------------------
# cut-point tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cut=-180.0, value=45.0, expected=45.0),
            does_not_raise(),
            id="default-cut-in-window",
        ),
        pytest.param(
            dict(cut=-180.0, value=200.0, expected=-160.0),
            does_not_raise(),
            id="default-cut-above-window",
        ),
        pytest.param(
            dict(cut=-180.0, value=-200.0, expected=160.0),
            does_not_raise(),
            id="default-cut-below-window",
        ),
        pytest.param(
            dict(cut=-180.0, value=-180.0, expected=-180.0),
            does_not_raise(),
            id="default-cut-at-cut-exactly",
        ),
        pytest.param(
            dict(cut=-180.0, value=180.0, expected=-180.0),
            does_not_raise(),
            id="default-cut-at-open-end-wraps-to-cut",
        ),
        pytest.param(
            dict(cut=0.0, value=90.0, expected=90.0),
            does_not_raise(),
            id="cut-zero-in-window",
        ),
        pytest.param(
            dict(cut=0.0, value=-90.0, expected=270.0),
            does_not_raise(),
            id="cut-zero-below-window",
        ),
        pytest.param(
            dict(cut=0.0, value=370.0, expected=10.0),
            does_not_raise(),
            id="cut-zero-above-window",
        ),
        pytest.param(
            dict(cut=0.0, value=360.0, expected=0.0),
            does_not_raise(),
            id="cut-zero-at-open-end-wraps-to-cut",
        ),
        pytest.param(
            dict(cut=0.0, value=0.0, expected=0.0),
            does_not_raise(),
            id="cut-zero-at-cut-exactly",
        ),
        pytest.param(
            dict(cut=90.0, value=90.0, expected=90.0),
            does_not_raise(),
            id="cut-90-at-cut-exactly",
        ),
        pytest.param(
            dict(cut=90.0, value=89.9, expected=449.9),
            does_not_raise(),
            id="cut-90-just-below-wraps-up",
        ),
    ],
)
def test_apply_cut(parms, context):
    c = LimitsConstraint(label="axis", cut_point=parms["cut"])
    with context:
        result = c.apply_cut(parms["value"])
        assert abs(result - parms["expected"]) < ENDPOINT_TOLERANCE, (
            f"apply_cut({parms['value']!r}, cut={parms['cut']!r})"
            f" expected {parms['expected']!r}, got {result!r}"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cut_point=-180.0),
            does_not_raise(),
            id="default-cut-round-trips",
        ),
        pytest.param(
            dict(cut_point=0.0),
            does_not_raise(),
            id="cut-zero-round-trips",
        ),
        pytest.param(
            dict(cut_point=90.0),
            does_not_raise(),
            id="cut-90-round-trips",
        ),
    ],
)
def test_cut_point_round_trip(parms, context):
    with context:
        c = LimitsConstraint(label="axis", cut_point=parms["cut_point"])
        d = c._asdict()
        assert "cut_point" in d
        assert d["cut_point"] == parms["cut_point"]

        # Restore from dict.
        c2 = LimitsConstraint(label="axis")
        config = {**d, "class": "LimitsConstraint"}
        c2._fromdict(config)
        assert c2.cut_point == parms["cut_point"]


def test_cut_point_fromdict_backward_compat():
    """Old configs without cut_point key silently default to DEFAULT_CUT_POINT."""
    old_config = {
        "class": "LimitsConstraint",
        "label": "th",
        "low_limit": -180.0,
        "high_limit": 180.0,
        # No "cut_point" key — simulates a pre-296 configuration file.
    }
    c = LimitsConstraint(label="th")
    c._fromdict(old_config)
    assert c.cut_point == DEFAULT_CUT_POINT


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                cut_point=0.0,
                limits=(0.0, 360.0),
                raw_angle=-90.0,  # wraps to 270 with cut=0
                expect_valid=True,
            ),
            does_not_raise(),
            id="cut-0-wraps-neg90-to-270-within-limits",
        ),
        pytest.param(
            dict(
                cut_point=-180.0,
                limits=(-180.0, 180.0),
                raw_angle=-90.0,  # already in window with default cut
                expect_valid=True,
            ),
            does_not_raise(),
            id="default-cut-neg90-in-window",
        ),
        pytest.param(
            dict(
                cut_point=0.0,
                limits=(0.0, 180.0),
                raw_angle=-90.0,  # wraps to 270, outside (0, 180)
                expect_valid=False,
            ),
            does_not_raise(),
            id="cut-0-wraps-neg90-to-270-outside-limits",
        ),
    ],
)
def test_cut_point_and_valid(parms, context):
    """Cut-point wrapping then limit check gives expected validity."""
    with context:
        c = LimitsConstraint(
            low_limit=parms["limits"][0],
            high_limit=parms["limits"][1],
            label="axis",
            cut_point=parms["cut_point"],
        )
        wrapped = c.apply_cut(parms["raw_angle"])
        result = c.valid(axis=wrapped)
        assert result == parms["expect_valid"], (
            f"raw={parms['raw_angle']}, wrapped={wrapped}, "
            f"limits={parms['limits']}, cut={parms['cut_point']}"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cut_point=-180.0),
            does_not_raise(),
            id="default-cut-point-attribute",
        ),
        pytest.param(
            dict(cut_point=0.0),
            does_not_raise(),
            id="set-cut-point-zero",
        ),
    ],
)
def test_cut_point_attribute(parms, context):
    """cut_point attribute is readable and settable on LimitsConstraint."""
    with context:
        sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
        constraint = sim.core.constraints["th"]
        assert constraint.cut_point == DEFAULT_CUT_POINT
        constraint.cut_point = parms["cut_point"]
        assert constraint.cut_point == parms["cut_point"]
