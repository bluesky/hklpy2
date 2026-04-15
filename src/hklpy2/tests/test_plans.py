"""Unit tests for the plans module."""

import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import bluesky
import pytest
from ophyd.sim import noisy_det

from ..misc import creator_from_config
from ..misc import validate_not_parallel
from ..plans import _find_psi_axis
from ..plans import _find_psi_mode
from ..plans import scan_psi

HKLPY2_DIR = Path(__file__).parent.parent


def sim4c():
    """Oriented E4CV with silicon, ready for psi scans."""
    return creator_from_config(HKLPY2_DIR / "tests" / "e4cv_orient.yml")


# ---------------------------------------------------------------------------
# validate_not_parallel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(hkl=(1, 0, 0), hkl2=(0, 1, 0)),
            does_not_raise(),
            id="orthogonal vectors ok",
        ),
        pytest.param(
            dict(hkl=(1, 1, 0), hkl2=(1, -1, 0)),
            does_not_raise(),
            id="non-parallel vectors ok",
        ),
        pytest.param(
            dict(hkl=(2, 2, 0), hkl2=(1, 1, 0)),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="parallel vectors raise",
        ),
        pytest.param(
            dict(hkl=(1, 0, 0), hkl2=(-1, 0, 0)),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="anti-parallel vectors raise",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), hkl2=(0, 0, 5)),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="scaled parallel vectors raise",
        ),
    ],
)
def test_validate_not_parallel(parms, context):
    with context:
        validate_not_parallel(**parms)


# ---------------------------------------------------------------------------
# _find_psi_mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode_override=None),
            does_not_raise(),
            id="auto-detect psi_constant on E4CV",
        ),
        pytest.param(
            dict(mode_override="psi_constant"),
            does_not_raise(),
            id="explicit valid mode",
        ),
        pytest.param(
            dict(mode_override="bissector"),
            does_not_raise(),
            id="explicit non-psi mode is accepted",
        ),
        pytest.param(
            dict(mode_override="no_such_mode"),
            pytest.raises(ValueError, match=re.escape("not in available modes")),
            id="explicit unknown mode raises",
        ),
    ],
)
def test_find_psi_mode(parms, context):
    with context:
        fourc = sim4c()
        result = _find_psi_mode(fourc, **parms)
        if parms["mode_override"] is None:
            assert "psi" in result.lower()
        elif parms["mode_override"] != "no_such_mode":
            assert result == parms["mode_override"]


# ---------------------------------------------------------------------------
# _find_psi_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(psi_axis_override=None),
            does_not_raise(),
            id="auto-detect psi axis in psi_constant mode",
        ),
        pytest.param(
            dict(psi_axis_override="psi"),
            does_not_raise(),
            id="explicit valid psi axis",
        ),
        pytest.param(
            dict(psi_axis_override="no_such_axis"),
            pytest.raises(ValueError, match=re.escape("not in extra axes")),
            id="explicit unknown axis raises",
        ),
    ],
)
def test_find_psi_axis(parms, context):
    with context:
        fourc = sim4c()
        fourc.core.mode = "psi_constant"
        result = _find_psi_axis(fourc, **parms)
        if parms["psi_axis_override"] is None:
            assert "psi" in result.lower()
        elif parms["psi_axis_override"] != "no_such_axis":
            assert result == parms["psi_axis_override"]


# ---------------------------------------------------------------------------
# scan_psi — main plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
            ),
            does_not_raise(),
            id="basic psi scan succeeds",
        ),
        pytest.param(
            dict(
                h=1,
                k=1,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=180,
                num=7,
                fail_on_exception=False,
            ),
            does_not_raise(),
            id="psi scan with fail_on_exception=False",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
                mode="psi_constant",
            ),
            does_not_raise(),
            id="explicit mode override",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
                psi_axis="psi",
            ),
            does_not_raise(),
            id="explicit psi_axis override",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(2, 2, 0),  # parallel to hkl
                psi_start=0,
                psi_stop=90,
                num=4,
            ),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="parallel hkl and hkl2 raises",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
                mode="no_such_mode",
            ),
            pytest.raises(ValueError, match=re.escape("not in available modes")),
            id="invalid explicit mode raises",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
                psi_axis="no_such_axis",
            ),
            pytest.raises(ValueError, match=re.escape("not in extra axes")),
            id="invalid explicit psi_axis raises",
        ),
    ],
)
def test_scan_psi(parms, context):
    with context:
        fourc = sim4c()
        RE = bluesky.RunEngine()
        RE(scan_psi([noisy_det, fourc], fourc, **parms))


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=4,
            ),
            does_not_raise(),
            id="mode restored after successful scan",
        ),
    ],
)
def test_scan_psi_restores_mode(parms, context):
    """Verify the prior mode is restored after scan_psi completes."""
    with context:
        fourc = sim4c()
        fourc.core.mode = "bissector"
        prior_mode = fourc.core.mode

        RE = bluesky.RunEngine()
        RE(scan_psi([noisy_det, fourc], fourc, **parms))

        assert fourc.core.mode == prior_mode


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(2, 2, 0),  # parallel — will raise before scan
                psi_start=0,
                psi_stop=90,
                num=4,
            ),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="mode restored after failed scan",
        ),
    ],
)
def test_scan_psi_restores_mode_on_error(parms, context):
    """Verify the prior mode is restored even when scan_psi raises."""
    fourc = sim4c()
    fourc.core.mode = "bissector"
    prior_mode = fourc.core.mode

    RE = bluesky.RunEngine()
    with context:
        RE(scan_psi([noisy_det, fourc], fourc, **parms))

    assert fourc.core.mode == prior_mode


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=3,
                md={"user": "test", "sample": "silicon"},
            ),
            does_not_raise(),
            id="metadata passed through to run document",
        ),
    ],
)
def test_scan_psi_metadata(parms, context):
    """Verify user metadata appears in the Bluesky start document."""
    with context:
        fourc = sim4c()
        RE = bluesky.RunEngine()
        docs = []
        RE.subscribe(lambda name, doc: docs.append((name, doc)))
        RE(scan_psi([noisy_det, fourc], fourc, **parms))
        starts = [doc for name, doc in docs if name == "start"]
        assert len(starts) == 1
        if parms.get("md"):
            for key, val in parms["md"].items():
                assert starts[0][key] == val


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="scan_psi"),
            does_not_raise(),
            id="scan_psi exported from hklpy2",
        ),
    ],
)
def test_plans_package_exports(parms, context):
    with context:
        import hklpy2

        assert hasattr(hklpy2, parms["name"])


# ---------------------------------------------------------------------------
# user.scan_psi wrapper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(1, -1, 0),
                psi_start=0,
                psi_stop=90,
                num=3,
            ),
            does_not_raise(),
            id="user.scan_psi succeeds with selected diffractometer",
        ),
        pytest.param(
            dict(
                h=2,
                k=2,
                l=0,
                hkl2=(2, 2, 0),  # parallel
                psi_start=0,
                psi_stop=90,
                num=3,
            ),
            pytest.raises(ValueError, match=re.escape("are parallel")),
            id="user.scan_psi parallel hkl2 raises",
        ),
    ],
)
def test_user_scan_psi(parms, context):
    from hklpy2.user import scan_psi as user_scan_psi
    from hklpy2.user import set_diffractometer

    with context:
        fourc = sim4c()
        set_diffractometer(fourc)
        RE = bluesky.RunEngine()
        RE(user_scan_psi([noisy_det, fourc], **parms))
