"""Unit tests of the zone module."""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ..zone import OrthonormalZone


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

# TODO: test zonespace, zone_series, scan_zone
