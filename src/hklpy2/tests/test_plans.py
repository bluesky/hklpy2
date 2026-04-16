"""Unit tests for the plans module."""

import re
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import patch

import bluesky
import pytest
from ophyd.sim import noisy_det

from ..diffract import creator
from ..run_utils import creator_from_config
from ..misc import validate_not_parallel
from ..plans import _find_psi_axis
from ..plans import _find_psi_mode
from ..plans import move_zone
from ..plans import scan_psi
from ..plans import scan_zone

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


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode_override=None),
            pytest.raises(NotImplementedError, match=re.escape("No psi-capable mode")),
            id="no psi mode on th_tth raises NotImplementedError",
        ),
    ],
)
def test_find_psi_mode_no_psi_geometry(parms, context):
    with context:
        # th_tth solver has no psi-capable mode
        tth = creator(solver="th_tth", geometry="TH TTH Q")
        _find_psi_mode(tth, **parms)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode_override=None),
            pytest.raises(ValueError, match=re.escape("Multiple psi-capable modes")),
            id="multiple psi modes on E6C raises ValueError",
        ),
    ],
)
def test_find_psi_mode_multiple_psi_modes(parms, context):
    with context:
        # E6C has both psi_constant_vertical and psi_constant_horizontal
        e6c = creator(solver="hkl_soleil", geometry="E6C")
        _find_psi_mode(e6c, **parms)


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


@pytest.mark.parametrize(
    "fake_extras, context",
    [
        pytest.param(
            [],
            pytest.raises(ValueError, match=re.escape("No psi extra axis found")),
            id="no psi axis in extras raises",
        ),
        pytest.param(
            ["psi", "psi_vertical"],
            pytest.raises(ValueError, match=re.escape("Multiple psi-like extra axes")),
            id="multiple psi axes in extras raises",
        ),
    ],
)
def test_find_psi_axis_edge_cases(fake_extras, context):
    with context:
        fourc = sim4c()
        fourc.core.mode = "psi_constant"
        with patch.object(
            type(fourc.core),
            "solver_extra_axis_names",
            new_callable=lambda: property(lambda self: fake_extras),
        ):
            _find_psi_axis(fourc, None)


@pytest.mark.parametrize(
    "fake_extras, context",
    [
        pytest.param(
            ["psi"],  # only psi, no ref axes — ref_axes count will be 0
            pytest.raises(ValueError, match=re.escape("Expected exactly 3 reference")),
            id="wrong ref axis count raises in scan_psi",
        ),
    ],
)
def test_scan_psi_wrong_ref_axis_count(fake_extras, context):
    with context:
        fourc = sim4c()
        RE = bluesky.RunEngine()
        with patch.object(
            type(fourc.core),
            "solver_extra_axis_names",
            new_callable=lambda: property(lambda self: fake_extras),
        ):
            RE(
                scan_psi(
                    [noisy_det, fourc],
                    fourc,
                    h=2,
                    k=2,
                    l=0,
                    hkl2=(1, -1, 0),
                    psi_start=0,
                    psi_stop=90,
                    num=4,
                    mode="psi_constant",
                    psi_axis="psi",
                )
            )


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
# move_zone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(hkl=(1, 0, 0)),
            does_not_raise(),
            id="move_zone to (1,0,0) tuple",
        ),
        pytest.param(
            dict(hkl=(0, 1, 0)),
            does_not_raise(),
            id="move_zone to (0,1,0) tuple",
        ),
    ],
)
def test_move_zone(parms, context):
    with context:
        fourc = creator()
        RE = bluesky.RunEngine()
        RE(move_zone(fourc, parms["hkl"]))
        for name, val in zip(fourc.pseudo_axis_names, parms["hkl"]):
            assert getattr(fourc, name).get().readback == pytest.approx(val, abs=1e-3)


# ---------------------------------------------------------------------------
# scan_zone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(start=(1, 0, 0), finish=(0, 1, 0), num=5, md=None),
            does_not_raise(),
            id="scan_zone basic",
        ),
        pytest.param(
            dict(start=(1, 0, 0), finish=(0, 1, 0), num=3, md={"user": "test"}),
            does_not_raise(),
            id="scan_zone with metadata",
        ),
        pytest.param(
            dict(start=(1, 0, 0), finish=(0, 1, 0), num=3, md=None),
            does_not_raise(),
            id="scan_zone metadata defaults to plan_name only",
        ),
    ],
)
def test_scan_zone(parms, context):
    with context:
        fourc = creator()
        RE = bluesky.RunEngine()
        docs = []
        RE.subscribe(lambda name, doc: docs.append((name, doc)))
        (uid,) = RE(
            scan_zone(
                [noisy_det],
                fourc,
                parms["start"],
                parms["finish"],
                parms["num"],
                md=parms["md"],
            )
        )
        assert isinstance(uid, str)
        assert len(uid) > 0
        starts = [doc for name, doc in docs if name == "start"]
        assert len(starts) == 1
        assert starts[0]["plan_name"] == "scan_zone"
        if parms["md"]:
            for k, v in parms["md"].items():
                assert starts[0][k] == v


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="move_zone"),
            does_not_raise(),
            id="move_zone exported from hklpy2",
        ),
        pytest.param(
            dict(name="scan_psi"),
            does_not_raise(),
            id="scan_psi exported from hklpy2",
        ),
        pytest.param(
            dict(name="scan_zone"),
            does_not_raise(),
            id="scan_zone exported from hklpy2",
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
