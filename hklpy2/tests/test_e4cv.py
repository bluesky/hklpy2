"""Various configurations of the E4CV geometry."""

from contextlib import nullcontext as does_not_raise

import numpy as np
import ophyd.utils.errors
import pytest
from bluesky import RunEngine
from bluesky import plans as bp
from numpy.testing import assert_almost_equal

from ..diffract import creator
from .common import assert_context_result

sim4c = creator(name="sim4c")


@pytest.mark.parametrize(
    "start",
    [
        dict(h=1.2, k=1.2, l=0.001),
        dict(h=1, k=0, l=0),
        dict(h=1, k=1, l=1),
    ],
)
@pytest.mark.parametrize("h", np.arange(0.9, 1.1, 0.1))
@pytest.mark.parametrize("k", np.arange(0.0, 1.2, 0.6))
@pytest.mark.parametrize("l", np.arange(0, 1, 0.5))
@pytest.mark.parametrize(
    "digits, context, expected",
    [
        [3, does_not_raise(), None],
    ],
)
def test_pseudos_move(start, h, k, l, digits, context, expected):  # noqa: E741
    with context as reason:
        assert len(start) == 3
        e4cv = creator(name="e4cv")
        e4cv.move(start)
        ppos = e4cv.position._asdict()
        assert isinstance(ppos, dict)
        for axis in "h k l".split():
            (
                assert_almost_equal(
                    ppos[axis],
                    start[axis],
                    decimal=digits,
                ),
                f"{ppos=}",
            )

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "ppos, rpos, context, expected",
    [
        [
            dict(h=0.3473, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            does_not_raise(),
            None,
        ],
        [
            dict(h=1.5694, k=-0.6260, l=0.3808),
            dict(omega=10, chi=20, phi=30, tth=120),
            does_not_raise(),
            None,
        ],
    ],
)
def test_inverse(ppos, rpos, context, expected):
    with context as reason:
        e4cv = creator(name="e4cv")
        pseudos = e4cv.inverse(rpos)._asdict()
        assert isinstance(pseudos, dict)
        assert np.allclose(
            list(pseudos.values()),
            list(ppos.values()),
            atol=0.001,
        )
        for axis, value in ppos.items():
            assert_almost_equal(pseudos[axis], value, decimal=3), f"{axis=}"
        # assert pseudos == ppos

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "parms, context, expected",
    [
        [[sim4c.tth, 10, 20, 3], does_not_raise(), None],
        [[sim4c.k, 1, 0, 1], does_not_raise(), None],
        [
            [sim4c.tth, 10, 20, sim4c.k, 0, 0, 3],
            pytest.raises(ValueError),
            "mix of real and pseudo axis",
        ],
        [
            [sim4c.tth, 10_000, 10_002, 3],
            pytest.raises(ophyd.utils.errors.LimitError),
            "not within limits",
        ],
    ],
)
def test_scan(parms, context, expected):
    with context as reason:
        RE = RunEngine()
        # axis, first, last, npts = parms
        RE(bp.scan([sim4c], *parms))

    assert_context_result(expected, reason)
