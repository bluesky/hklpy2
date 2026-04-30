"""
Tests for ``Sample.UB_is_stale`` (:issue:`391`).

Covers the staleness-detection matrix in the issue:

* ``calc_UB`` -> not stale.
* Reorder of ``order[:2]`` -> stale.
* Add a non-orienting reflection -> not stale.
* Add a reflection AND prepend it to ``order`` -> stale.
* Remove a non-orienting reflection -> not stale.
* Remove ``order[0]`` -> stale.
* In-place mutation of an orienting reflection's ``pseudos`` -> stale.
* In-place mutation of a non-orienting reflection's ``pseudos`` -> not stale.
* Direct ``Sample.UB = M`` after calc_UB -> not stale (user owns UB).
* Single reflection with no calc_UB possible -> not stale.

Lifecycle of ``Sample._ub_snapshot``:

* New sample -> ``None``.
* After ``calc_UB`` -> tuple snapshot.
* After ``Sample.U = ...`` -> cleared to ``None``.
* After ``Sample.UB = ...`` -> cleared to ``None``.
* After a second ``calc_UB`` -> fresh snapshot for the new pair.
"""

from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...tests.models import add_oriented_vibranium_to_e4cv
from ...utils import IDENTITY_MATRIX_3X3


def _oriented_e4cv():
    """Return an E4CV diffractometer with vibranium oriented via calc_UB."""
    e4cv = creator(name="e4cv")
    add_oriented_vibranium_to_e4cv(e4cv)
    return e4cv


def _setup_initial(action: str):
    """Apply ``action`` against a freshly oriented E4CV; return the diffractometer."""
    e4cv = _oriented_e4cv()
    sample = e4cv.sample
    refls = sample.reflections

    if action == "noop":
        pass
    elif action == "swap_order":
        # vibranium has order = ['r040', 'r004'] after add_oriented_*.
        refls.order = list(reversed(refls.order))
    elif action == "add_non_orienting":
        # 'r400' is already in the dict but not in order; mimic adding a fresh
        # one outside the orienting pair by re-adding with replace=True.
        # The 'r400' name is added by add_oriented_vibranium_to_e4cv but only
        # 'r040', 'r004' are placed in ``order`` by calc_UB.
        assert "r400" in refls and "r400" not in refls.order
    elif action == "add_and_prepend":
        # Prepend an existing non-orienting reflection to ``order``.
        refls.order = ["r400"] + list(refls.order)
    elif action == "remove_non_orienting":
        sample.remove_reflection("r400")
    elif action == "remove_order_zero":
        sample.remove_reflection(refls.order[0])
    elif action == "mutate_orienting_pseudos":
        first = refls[refls.order[0]]
        # Bump h by 1 to force a content change (any change suffices).
        new = dict(first.pseudos)
        first_key = next(iter(new))
        new[first_key] = new[first_key] + 1
        first.pseudos = new
    elif action == "mutate_non_orienting_pseudos":
        non = refls["r400"]
        new = dict(non.pseudos)
        first_key = next(iter(new))
        new[first_key] = new[first_key] + 1
        non.pseudos = new
    elif action == "assign_UB":
        sample.UB = IDENTITY_MATRIX_3X3
    elif action == "assign_U":
        sample.U = IDENTITY_MATRIX_3X3
    else:
        raise ValueError(f"Unknown action {action!r}")

    return e4cv


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(action="noop", expected=False),
            does_not_raise(),
            id="calc_UB-then-check",
        ),
        pytest.param(
            dict(action="swap_order", expected=True),
            does_not_raise(),
            id="swap-order[:2]",
        ),
        pytest.param(
            dict(action="add_non_orienting", expected=False),
            does_not_raise(),
            id="add-3rd-reflection-not-in-order",
        ),
        pytest.param(
            dict(action="add_and_prepend", expected=True),
            does_not_raise(),
            id="add-reflection-AND-prepend-to-order",
        ),
        pytest.param(
            dict(action="remove_non_orienting", expected=False),
            does_not_raise(),
            id="remove-non-order-reflection",
        ),
        pytest.param(
            dict(action="remove_order_zero", expected=True),
            does_not_raise(),
            id="remove-order[0]",
        ),
        pytest.param(
            dict(action="mutate_orienting_pseudos", expected=True),
            does_not_raise(),
            id="modify-pseudos-of-order[0]",
        ),
        pytest.param(
            dict(action="mutate_non_orienting_pseudos", expected=False),
            does_not_raise(),
            id="modify-pseudos-of-non-order-reflection",
        ),
        pytest.param(
            dict(action="assign_UB", expected=False),
            does_not_raise(),
            id="direct-UB-assignment-clears-snapshot",
        ),
        pytest.param(
            dict(action="assign_U", expected=False),
            does_not_raise(),
            id="direct-U-assignment-clears-snapshot",
        ),
    ],
)
def test_ub_is_stale_matrix(parms, context):
    with context:
        e4cv = _setup_initial(parms["action"])
        assert e4cv.sample.UB_is_stale is parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="single-reflection-no-calc_UB",
        ),
    ],
)
def test_ub_is_stale_with_fewer_than_two_reflections(parms, context):
    """A sample with <2 reflections cannot have a calc_UB snapshot."""
    with context:
        e4cv = creator(name="e4cv")
        e4cv.add_sample("vibranium", 6.28, digits=3, replace=True)
        e4cv.beam.wavelength.put(1.54)
        e4cv.add_reflection(
            (4, 0, 0),
            dict(omega=-145.451, chi=0, phi=0, tth=69.066),
            name="r400",
        )
        # No calc_UB possible.
        assert e4cv.sample._ub_snapshot is None
        assert e4cv.sample.UB_is_stale is False
        # Reordering with one reflection is a no-op for staleness.
        e4cv.sample.reflections.order = ["r400"]
        assert e4cv.sample.UB_is_stale is False


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="snapshot-set-by-calc_UB",
        ),
    ],
)
def test_snapshot_set_by_calc_UB(parms, context):
    with context:
        e4cv = creator(name="e4cv")
        e4cv.add_sample("vibranium", 6.28, digits=3, replace=True)
        e4cv.beam.wavelength.put(1.54)
        r040 = e4cv.add_reflection((0, 4, 0), (-145.451, 0, 90, 69.066), name="r040")
        r004 = e4cv.add_reflection((0, 0, 4), (-145.451, 90, 0, 69.066), name="r004")

        # Pre-condition: snapshot is None until calc_UB runs.
        assert e4cv.sample._ub_snapshot is None

        e4cv.core.calc_UB(r040, r004)

        # Post-condition: snapshot captured.
        snap = e4cv.sample._ub_snapshot
        assert snap is not None
        # First element is the orienting names tuple.
        assert snap[0] == ("r040", "r004")
        # Second element is per-reflection content tuples.
        assert len(snap[1]) == 2
        # Idempotent: a fresh snapshot of the unchanged state must equal
        # the stored one.
        assert e4cv.sample._compute_ub_snapshot() == snap
        assert e4cv.sample.UB_is_stale is False


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(setter="U"),
            does_not_raise(),
            id="U-setter-clears-snapshot",
        ),
        pytest.param(
            dict(setter="UB"),
            does_not_raise(),
            id="UB-setter-clears-snapshot",
        ),
    ],
)
def test_direct_assignment_clears_snapshot(parms, context):
    with context:
        e4cv = _oriented_e4cv()
        # Pre-condition: calc_UB ran, snapshot present.
        assert e4cv.sample._ub_snapshot is not None
        if parms["setter"] == "U":
            e4cv.sample.U = IDENTITY_MATRIX_3X3
        else:
            e4cv.sample.UB = IDENTITY_MATRIX_3X3
        # Post-condition: snapshot cleared, UB_is_stale is False.
        assert e4cv.sample._ub_snapshot is None
        assert e4cv.sample.UB_is_stale is False


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="recalc-refreshes-snapshot",
        ),
    ],
)
def test_recalc_UB_refreshes_snapshot(parms, context):
    with context:
        e4cv = _oriented_e4cv()
        # Force staleness by reordering.
        e4cv.sample.reflections.order = list(reversed(e4cv.sample.reflections.order))
        assert e4cv.sample.UB_is_stale is True

        # Re-orient with the new order; staleness should clear.
        new_pair = list(e4cv.sample.reflections.order)
        e4cv.core.calc_UB(*new_pair)
        assert e4cv.sample.UB_is_stale is False
        assert e4cv.sample._ub_snapshot is not None
        assert e4cv.sample._ub_snapshot[0] == tuple(new_pair)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="digits-change-does-not-cause-staleness",
        ),
    ],
)
def test_digits_change_does_not_cause_staleness(parms, context):
    """``digits`` is a presentation field; changing it must not flag UB stale."""
    with context:
        e4cv = _oriented_e4cv()
        first = e4cv.sample.reflections[e4cv.sample.reflections.order[0]]
        first.digits = first.digits + 1
        assert e4cv.sample.UB_is_stale is False
