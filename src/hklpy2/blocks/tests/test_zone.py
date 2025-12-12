"""Unit tests of the zone module."""

import logging
import re
from contextlib import nullcontext as does_not_raise

import bluesky
import numpy as np
import pytest
from ophyd.sim import noisy_det

from ...diffract import creator
from ..zone import OrthonormalZone
from ..zone import scan_zone
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
            pytest.raises(ValueError, match="zone axis is undefined"),
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
            pytest.raises(ValueError, match="vector must be 1-D array-like"),
            id="vector shape !=(3,)",
        ),
        pytest.param(
            dict(axis=(0, 0, 1)),
            dict(),
            pytest.raises(
                ValueError,
                match="Cannot create vector from empty dictionary",
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
