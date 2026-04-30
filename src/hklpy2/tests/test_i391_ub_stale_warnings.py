"""
Tests for the stale-UB ``UserWarning`` in :class:`hklpy2.ops.Core`
and the ``pa()`` annotation (:issue:`391`).

When ``Sample.UB_is_stale`` is ``True``, ``Core.forward`` and
``Core.inverse`` emit a single ``UserWarning`` per call and
``DiffractometerBase.wh(full=True)`` (i.e. ``pa()``) prints a
``UB stale: True`` annotation.  When ``UB_is_stale`` is ``False``, no
warning is emitted and the annotation is absent.
"""

import re
import warnings
from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import creator
from .models import add_oriented_vibranium_to_e4cv

_STALE_PATTERN = re.compile(
    r"UB for sample .* is stale .*orientation reflections changed"
)


def _oriented_e4cv():
    e4cv = creator(name="e4cv")
    add_oriented_vibranium_to_e4cv(e4cv)
    return e4cv


def _make_stale(e4cv):
    """Force the diffractometer's UB to be stale."""
    e4cv.sample.reflections.order = list(reversed(e4cv.sample.reflections.order))
    assert e4cv.sample.UB_is_stale is True


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(method="forward", make_stale=True, expect_warning=True),
            does_not_raise(),
            id="forward-stale-warns",
        ),
        pytest.param(
            dict(method="forward", make_stale=False, expect_warning=False),
            does_not_raise(),
            id="forward-fresh-no-warning",
        ),
        pytest.param(
            dict(method="inverse", make_stale=True, expect_warning=True),
            does_not_raise(),
            id="inverse-stale-warns",
        ),
        pytest.param(
            dict(method="inverse", make_stale=False, expect_warning=False),
            does_not_raise(),
            id="inverse-fresh-no-warning",
        ),
    ],
)
def test_stale_ub_warning(parms, context):
    with context:
        e4cv = _oriented_e4cv()
        if parms["make_stale"]:
            _make_stale(e4cv)
        else:
            assert e4cv.sample.UB_is_stale is False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if parms["method"] == "forward":
                e4cv.core.forward(dict(h=1, k=0, l=0))
            else:
                e4cv.core.inverse(dict(omega=-145.451, chi=0, phi=0, tth=69.066))

        stale_warnings = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and _STALE_PATTERN.search(str(w.message))
        ]
        if parms["expect_warning"]:
            assert len(stale_warnings) == 1, [str(w.message) for w in caught]
        else:
            assert stale_warnings == []


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(make_stale=True, expect_annotation=True),
            does_not_raise(),
            id="pa-shows-stale-annotation",
        ),
        pytest.param(
            dict(make_stale=False, expect_annotation=False),
            does_not_raise(),
            id="pa-no-annotation-when-fresh",
        ),
    ],
)
def test_pa_surface_stale_annotation(parms, context, capsys):
    with context:
        e4cv = _oriented_e4cv()
        if parms["make_stale"]:
            _make_stale(e4cv)

        # ``pa()`` is ``wh(full=True)``.  Suppress the stale-UB warning
        # that ``wh()`` may itself indirectly trigger so it does not
        # interfere with capsys output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            e4cv.wh(full=True)

        captured = capsys.readouterr().out
        if parms["expect_annotation"]:
            assert "UB stale: True" in captured
            assert "orientation reflections changed since last calc_UB" in captured
        else:
            assert "UB stale" not in captured


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="recalc_UB-clears-warning",
        ),
    ],
)
def test_recalc_clears_warning(parms, context):
    """After re-running ``calc_UB`` the warning must stop firing."""
    with context:
        e4cv = _oriented_e4cv()
        _make_stale(e4cv)
        # Re-orient with the new pair.
        new_pair = list(e4cv.sample.reflections.order)
        e4cv.core.calc_UB(*new_pair)
        assert e4cv.sample.UB_is_stale is False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            e4cv.core.forward(dict(h=1, k=0, l=0))

        stale = [
            w
            for w in caught
            if issubclass(w.category, UserWarning)
            and _STALE_PATTERN.search(str(w.message))
        ]
        assert stale == []
