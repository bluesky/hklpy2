"""Unit tests of the zone module."""

import logging
import re
from contextlib import nullcontext as does_not_raise

import bluesky
import numpy as np
import pytest
from ophyd.sim import noisy_det

from ...diffract import creator
from ...plans import move_zone
from ...plans import scan_zone
from ..zone import OrthonormalZone
from ..zone import zone_series
from ..zone import zonespace


def sim4c2():
    """For the zonespace test."""
    sim = creator()
    sim.add_sample("test", 4, 5, 6, 75, 85, 95, replace=True)
    r1 = sim.add_reflection((4, 0, 0), (30.345, 10, 10, 60.69))
    r2 = sim.add_reflection((0, 4, 0), (-24.63, -9.265, -85.08, -49.27))
    sim.core.calc_UB(r1, r2)
    sim.core.mode = "psi_constant"
    return sim


@pytest.mark.parametrize(
    "parms, vector, in_zone, context",
    [
        pytest.param(
            {},
            None,
            True,
            pytest.raises(ValueError, match=re.escape("zone axis is undefined")),
            id="ValueError: No kwargs, zone axis is undefined",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            None,
            True,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Received None with shape ()",
                ),
            ),
            id="b1 is None",
        ),
        pytest.param(
            dict(axis=np.array([1, 2, 3])),
            [1, 0, 0],
            False,
            does_not_raise(),
            id="b1 not in zone",
        ),
        pytest.param(
            dict(b1=(1, 0, 0)),
            (1, 0, 0),
            True,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Both b1 and b2 must be provided",
                ),
            ),
            id="b1 but not b2",
        ),
        pytest.param(
            dict(b2=(0, 1, 0)),
            (1, 0, 0),
            True,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Both b1 and b2 must be provided",
                ),
            ),
            id="b2 but not b1",
        ),
        pytest.param(
            dict(axis=(0, 0, 1), b1=(1, 0, 0), b2=(0, 1, 0)),
            (1, 0, 0),
            True,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Cannot specify both 'axis' and 'b1/b2' parameters",
                ),
            ),
            id="axis & b1 & b2 provided",
        ),
        pytest.param(
            dict(b1=(1, 0, 0), b2=(0, 1, 0)),
            (1, 0, 0),
            True,
            does_not_raise(),
            id="b1 & b2 provided",
        ),
        pytest.param(
            dict(b1=(0, 2, 0), b2=(0, 1, 0)),
            (1, 0, 0),
            True,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "cross product is zero and cannot define a zone axis.",
                ),
            ),
            id="b1 x b2 = 0",
        ),
    ],
)
def test_OrthonormalZone_constructor(parms, vector, in_zone, context):
    with context:
        zone = OrthonormalZone(**parms)
        assert zone.in_zone(vector) == in_zone


@pytest.mark.parametrize(
    "parms, result, context",
    [
        pytest.param(
            dict(b1=(1, 0, 0), b2=(0, 1, 0)),
            "OrthonormalZone(axis=array([0., 0., 1.]))",
            does_not_raise(),
            id="with zone axis",
        ),
        pytest.param(
            dict(),
            "OrthonormalZone(axis='undefined')",
            does_not_raise(),
            id="without zone axis",
        ),
    ],
)
def test_OrthonormalZone_repr(parms, result, context):
    with context:
        zone = OrthonormalZone(**parms)
        assert result in repr(zone)


@pytest.mark.parametrize(
    "parms, vector, context",
    [
        pytest.param(
            dict(axis=(0, 0, 1)),
            (1, 0, 0),
            does_not_raise(),
            id="tuple, shape=(3,)",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            1,
            pytest.raises(ValueError, match=re.escape("vector must be 1-D array-like")),
            id="vector shape !=(3,)",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            dict(),
            pytest.raises(
                ValueError,
                match=re.escape("Cannot create vector from empty dictionary"),
            ),
            id="empty dict",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            [1, 0, 0],
            does_not_raise(),
            id="axis is list",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            dict(h=1, k=0, l=1),
            does_not_raise(),
            id="axis is dict",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            np.array([1, 0, 1]),
            does_not_raise(),
            id="axis is np.array",
        ),
    ],
)
def test_OrthonormalZone_standardize_vector(parms, vector, context):
    with context:
        zone = OrthonormalZone(**parms)
        assert isinstance(zone._standardize_vector(vector), np.ndarray)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(axis=(0, 0, 0)),
            pytest.raises(
                ValueError,
                match=re.escape("Zero |axis| not allowed."),
            ),
            id="|axis| = 0",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            does_not_raise(),
            id="|axis| = 1",
        ),
    ],
)
def test_OrthonormalZone_set_zone_axis(parms, context):
    with context:
        OrthonormalZone(**parms)


@pytest.mark.parametrize(
    "parms, b1, b2, n, context",
    [
        pytest.param(
            dict(axis=(0, 0, 1)),
            (1, 0, 0),
            (0, 1, 0),
            5,
            does_not_raise(),
            id="Ok with axis, n=5",
        ),
        pytest.param(
            dict(),
            (1, 0, 0),
            (0, 1, 0),
            5,
            does_not_raise(),
            id="Ok, defined zone with b1/b2",
        ),
        pytest.param(
            dict(),
            (1, 0, 0),
            (0, 1, 0),
            1,
            pytest.raises(
                ValueError,
                match=re.escape(
                    "Cannot check zone membership: zone axis is undefined",
                ),
            ),
            id="n=1, no zone",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            (1, 0, 0),
            (0, 1, 0),
            1,
            does_not_raise(),
            id="Ok, n=1, only return b1",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            (1, 0, 1),
            (0, 1, 0),
            5,
            pytest.raises(
                ValueError,
                match=re.escape("b1=(1, 0, 1) not in zone"),
            ),
            id="b1 not in zone",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            (1, 0, 0),
            (0, 1, 1),
            5,
            pytest.raises(
                ValueError,
                match=re.escape("b2=(0, 1, 1) not in zone"),
            ),
            id="b2 not in zone",
        ),
    ],
)
def test_OrthonormalZone_vecspace(parms, b1, b2, n, context):
    with context:
        zone = OrthonormalZone(**parms)
        for b_vector in zone.vecspace(b1, b2, n):
            assert zone.in_zone(b_vector)


@pytest.mark.parametrize(
    "diff, hkl_1, hkl_2, n, log, context",
    [
        pytest.param(
            creator(),
            (1, 0, 0),
            (0, 1, 0),
            3,
            None,
            does_not_raise(),
            id="Ok",
        ),
        pytest.param(
            sim4c2(),
            (1, 0, 0),
            (0, 1, 0),
            3,
            "no solution for forward",
            does_not_raise(),
            id="NoForwardSolutions",
        ),
    ],
)
def test_zonespace_and_series(
    diff,
    hkl_1,
    hkl_2,
    n,
    log,
    context,
    capsys,
    caplog,
):
    with context:
        caplog.set_level(logging.DEBUG, logger="hklpy2.blocks.zone")
        count = 0
        for pseudos, reals in zonespace(diff, hkl_1, hkl_2, n):
            count += 1
        if log is None:
            assert count == n
        else:
            assert log in caplog.text

        zone_series(diff, hkl_1, hkl_2, n)
        out, err = capsys.readouterr()
        assert err == ""
        assert isinstance(out, str)
        assert len(out.splitlines()) > n


@pytest.mark.parametrize(
    "dets, diff, v1, v2, n, context",
    [
        pytest.param(
            [noisy_det],
            creator(),
            (1, 0, 0),
            (0, 1, 0),
            5,
            does_not_raise(),
            id="Ok",
        ),
    ],
)
def test_scan_zone(dets, diff, v1, v2, n, context):
    with context:
        RE = bluesky.RunEngine()
        (uid,) = RE(scan_zone([noisy_det], diff, v1, v2, n))
        assert isinstance(uid, str)
        assert len(uid) > 0


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(hkl=(1, 0, 0)),
            does_not_raise(),
            id="move to (1,0,0) tuple",
        ),
        pytest.param(
            dict(hkl=(0, 1, 0)),
            does_not_raise(),
            id="move to (0,1,0) tuple",
        ),
    ],
)
def test_move_zone(parms, context):
    with context:
        fourc = creator()
        RE = bluesky.RunEngine()
        RE(move_zone(fourc, parms["hkl"]))
        # After moving, the diffractometer pseudo position should reflect the target.
        for name, val in zip(fourc.pseudo_axis_names, parms["hkl"]):
            assert getattr(fourc, name).get().readback == pytest.approx(val, abs=1e-3)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(md={"user": "test"}),
            does_not_raise(),
            id="scan_zone metadata is passed through",
        ),
        pytest.param(
            dict(md=None),
            does_not_raise(),
            id="scan_zone metadata defaults to plan_name only",
        ),
    ],
)
def test_scan_zone_metadata(parms, context):
    """Verify the _md fix: metadata dict is built correctly."""
    with context:
        fourc = creator()
        RE = bluesky.RunEngine()
        docs = []
        RE.subscribe(lambda name, doc: docs.append((name, doc)))
        RE(scan_zone([noisy_det], fourc, (1, 0, 0), (0, 1, 0), 3, md=parms["md"]))
        starts = [doc for name, doc in docs if name == "start"]
        assert len(starts) == 1
        assert starts[0]["plan_name"] == "scan_zone"
        if parms["md"]:
            for k, v in parms["md"].items():
                assert starts[0][k] == v


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="OrthonormalZone"),
            does_not_raise(),
            id="OrthonormalZone exported from hklpy2",
        ),
        pytest.param(
            dict(name="move_zone"),
            does_not_raise(),
            id="move_zone exported from hklpy2",
        ),
        pytest.param(
            dict(name="scan_zone"),
            does_not_raise(),
            id="scan_zone exported from hklpy2",
        ),
    ],
)
def test_zone_package_exports(parms, context):
    with context:
        import hklpy2

        assert hasattr(hklpy2, parms["name"])


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                func_name="move_zone",
                msg="move_zone from hklpy2.blocks.zone is deprecated",
            ),
            does_not_raise(),
            id="move_zone from blocks.zone emits DeprecationWarning",
        ),
        pytest.param(
            dict(
                func_name="scan_zone",
                msg="scan_zone from hklpy2.blocks.zone is deprecated",
            ),
            does_not_raise(),
            id="scan_zone from blocks.zone emits DeprecationWarning",
        ),
    ],
)
def test_zone_deprecated_shims(parms, context):
    """Importing move_zone/scan_zone from blocks.zone should emit DeprecationWarning."""
    with context:
        import importlib

        zone_mod = importlib.import_module("hklpy2.blocks.zone")
        func = getattr(zone_mod, parms["func_name"])
        # Trigger only the warnings.warn() in the shim body, without calling
        # the underlying plan (which would create a bluesky Plan generator that
        # crashes in __del__ if never iterated).  We do this by patching the
        # imported plan symbol so the shim's delegation never reaches @plan.
        from unittest.mock import patch

        target = f"hklpy2.plans.{parms['func_name']}"
        with patch(target, return_value=None):
            with pytest.warns(DeprecationWarning, match=re.escape(parms["msg"])):
                func()
