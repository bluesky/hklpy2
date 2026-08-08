# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Regression tests for issue #397.

``Core.add_reflection`` and ``ReflectionsDict.set_orientation_reflections``
mutated the python-side reflection state without flagging
``_SolverDirty.SAMPLE``.  As a result, ``Core.update_solver()`` was a
no-op for the reflection list and the underlying solver never received
the reflections the user had just added.  ``HklSoleilSolver`` happens
to mask the gap because its ``calculate_UB`` rebuilds its reflection
table internally; any solver that trusts the upstream sync contract --
i.e. that ``update_solver`` has actually pushed the current sample
state -- would see an empty / stale list.

These tests assert the contract directly: every public mutation of the
reflection collection that affects the orienting list (``order``) must
flag ``_SolverDirty.SAMPLE | _SolverDirty.UB`` so the next
``update_solver()`` re-pushes the collection to the solver.  Mutations
that do not affect ``order`` -- e.g. adding a brand-new reflection that
is not yet in ``order``, or replacing/removing a reflection that is not
in ``order`` -- are not flagged because the solver does not see them.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from .. import creator
from ..blocks.reflection import Reflection
from ..blocks.reflection import ReflectionsDict
from ..utils import _SolverDirty

SAMPLE_DEF = {
    "name": "sapphire",
    "lattice": {"a": 4.785, "c": 12.991, "gamma": 120},
    "reflections": [
        {
            "name": "r006",
            "pseudos": (0, 0, 6),
            "reals": {"omega": 20.97, "chi": 90, "phi": 0, "tth": 41.9419},
        },
        {
            "name": "r100",
            "pseudos": (1, 0, 0),
            "reals": {"omega": 30, "chi": 0, "phi": 0, "tth": 60},
        },
        {
            "name": "r006b",
            "pseudos": (0, 0, 6),
            "reals": {"omega": 20.3654, "chi": 89.32, "phi": 0, "tth": 41.9394},
        },
    ],
}


def _build_sim_with_reflections(n_reflections: int = 2):
    """Create an E4CV sim with the named sample and ``n_reflections`` reflections."""
    sim = creator()
    sim.beam.wavelength.put(1.5498)
    sim.add_sample(
        SAMPLE_DEF["name"],
        SAMPLE_DEF["lattice"]["a"],
        c=SAMPLE_DEF["lattice"]["c"],
        gamma=SAMPLE_DEF["lattice"]["gamma"],
    )
    for refl in SAMPLE_DEF["reflections"][:n_reflections]:
        sim.add_reflection(
            pseudos=refl["pseudos"], reals=refl["reals"], name=refl["name"]
        )
    return sim


def _new_like(refl: Reflection, name: str) -> Reflection:
    """Return a fresh ``Reflection`` with the same content under ``name``."""
    return Reflection(
        name,
        refl.pseudos,
        refl.reals,
        refl.wavelength,
        refl.geometry,
        refl.pseudo_axis_names,
        refl.real_axis_names,
        wavelength_units=refl.wavelength_units,
    )


def _clean(sim) -> None:
    """Push everything to the solver and clear the dirty bitfield."""
    sim.core.update_solver()
    sim.core._solver_dirty = _SolverDirty(0)


# -- High-level public API: Core.add_reflection / Sample.remove_reflection /
#    set_orientation_reflections / setor / order setter / swap.
#
# These all touch ``order`` by construction, so they always flag.


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"action": "add_reflection"},
            does_not_raise(),
            id="Core.add_reflection adds to order and flags",
        ),
        pytest.param(
            {"action": "set_orientation_reflections"},
            does_not_raise(),
            id="set_orientation_reflections rewrites order and flags",
        ),
        pytest.param(
            {"action": "setor_alias"},
            does_not_raise(),
            id="setor (alias) rewrites order and flags",
        ),
        pytest.param(
            {"action": "set_order"}, does_not_raise(), id="order setter flags"
        ),
        pytest.param({"action": "swap"}, does_not_raise(), id="swap flags"),
        pytest.param(
            {"action": "add_via_dict_then_promote"},
            does_not_raise(),
            id="add via add() always flags (order is updated)",
        ),
    ],
)
def test_order_affecting_high_level_api_flags(parms, context):
    """High-level public mutations always touch ``order`` and must flag."""
    with context:
        sim = _build_sim_with_reflections(n_reflections=2)
        _clean(sim)
        refls = sim.sample.reflections
        action = parms["action"]

        if action == "add_reflection":
            sim.add_reflection(
                pseudos=SAMPLE_DEF["reflections"][2]["pseudos"],
                reals=SAMPLE_DEF["reflections"][2]["reals"],
                name=SAMPLE_DEF["reflections"][2]["name"],
            )
        elif action == "set_orientation_reflections":
            refls.set_orientation_reflections([refls["r100"], refls["r006"]])
        elif action == "setor_alias":
            refls.setor([refls["r100"], refls["r006"]])
        elif action == "set_order":
            refls.order = ["r100", "r006"]
        elif action == "swap":
            refls.swap()
        elif action == "add_via_dict_then_promote":
            # Direct ``add()`` call (a separate code path from
            # Core.add_reflection): always touches order.  Use a
            # distinct pseudos vector so content-validation does not
            # treat this as a duplicate.
            template = refls["r006"]
            new = Reflection(
                "r110",
                {"h": 1.0, "k": 1.0, "l": 0.0},
                template.reals,
                template.wavelength,
                template.geometry,
                template.pseudo_axis_names,
                template.real_axis_names,
                wavelength_units=template.wavelength_units,
            )
            refls.add(new)
            assert "r110" in refls.order
        else:
            raise AssertionError(f"unknown action {action!r}")

        assert sim.core._solver_dirty == _SolverDirty.SAMPLE | _SolverDirty.UB, (
            f"Expected SAMPLE|UB, got {sim.core._solver_dirty!r} "
            f"after action {action!r}"
        )


# -- Dict-API mutations: only flag when the touched key is in ``order``.


def _setup_for_action(action):
    """
    Build a sim with a setup tailored to the test action.

    The strict orientation-health check (#399) refuses transitions
    into a half-defined state (``len(reflections) >= 2`` while
    ``order`` shrinks below two effective entries).  Each test action
    therefore needs a setup whose post-mutation state stays
    well-defined so the original flag-behavior contract from #397 can
    be observed in isolation.

    Layout:

    * removal of an orienting key (``del``/``pop``/``remove_reflection``)
      starts with two reflections and a two-entry ``order``; the
      removal drops the dict to one entry, which is below the
      orientation threshold and therefore strict-allowed.
    * ``popitem`` of an orienting key starts with a single reflection.
    * non-orienting actions start with three reflections and a
      two-entry ``order``; the third reflection is the unambiguous
      non-orienting target.
    * ``clear`` and ``update`` actions (which never shrink ``order``
      below two effective entries) reuse the three-reflection setup
      for variety.
    """
    one_orienting_actions = {"popitem_in_order"}
    drop_orienting_actions = {
        "setitem_replace_in_order",
        "del_in_order",
        "pop_in_order",
        "update_overwrites_in_order",
        "remove_reflection_in_order",
    }

    if action in one_orienting_actions:
        sim = _build_sim_with_reflections(n_reflections=1)
        return sim
    if action in drop_orienting_actions:
        sim = _build_sim_with_reflections(n_reflections=2)
        sim.sample.reflections.order = ["r006", "r100"]
        return sim
    # Default: three reflections, order = [r006, r100], r006b is the
    # non-orienting reflection.
    sim = _build_sim_with_reflections(n_reflections=2)
    third = SAMPLE_DEF["reflections"][2]
    sim.add_reflection(
        pseudos=third["pseudos"], reals=third["reals"], name=third["name"]
    )
    sim.sample.reflections.order = ["r006", "r100"]
    assert "r006b" in sim.sample.reflections
    assert "r006b" not in sim.sample.reflections.order
    return sim


@pytest.mark.parametrize(
    "parms, context",
    [
        # Order-affecting variants: should flag.
        pytest.param(
            {"action": "setitem_replace_in_order", "flags": True},
            does_not_raise(),
            id="__setitem__ replacing an orienting reflection flags",
        ),
        pytest.param(
            {"action": "del_in_order", "flags": True},
            does_not_raise(),
            id="__delitem__ removing an orienting reflection flags",
        ),
        pytest.param(
            {"action": "pop_in_order", "flags": True},
            does_not_raise(),
            id="pop removing an orienting reflection flags",
        ),
        pytest.param(
            {"action": "popitem_in_order", "flags": True},
            does_not_raise(),
            id="popitem removing the only orienting reflection flags",
        ),
        pytest.param(
            {"action": "clear_with_order", "flags": True},
            does_not_raise(),
            id="clear with non-empty order flags",
        ),
        pytest.param(
            {"action": "update_overwrites_in_order", "flags": True},
            does_not_raise(),
            id="update overwriting an orienting reflection flags",
        ),
        pytest.param(
            {"action": "remove_reflection_in_order", "flags": True},
            does_not_raise(),
            id="Sample.remove_reflection on orienting reflection flags",
        ),
        # Non-order-affecting variants: must NOT flag.
        pytest.param(
            {"action": "setitem_new_key", "flags": False},
            does_not_raise(),
            id="__setitem__ adding a new (un-ordered) key does not flag",
        ),
        pytest.param(
            {"action": "setitem_replace_not_in_order", "flags": False},
            does_not_raise(),
            id="__setitem__ replacing a non-orienting key does not flag",
        ),
        pytest.param(
            {"action": "del_not_in_order", "flags": False},
            does_not_raise(),
            id="__delitem__ removing a non-orienting key does not flag",
        ),
        pytest.param(
            {"action": "pop_not_in_order", "flags": False},
            does_not_raise(),
            id="pop removing a non-orienting key does not flag",
        ),
        pytest.param(
            {"action": "pop_missing_with_default", "flags": False},
            does_not_raise(),
            id="pop on missing key with default does not flag",
        ),
        pytest.param(
            {"action": "clear_empty_order", "flags": False},
            does_not_raise(),
            id="clear with empty order does not flag",
        ),
        pytest.param(
            {"action": "update_new_keys_only", "flags": False},
            does_not_raise(),
            id="update introducing only new (un-ordered) keys does not flag",
        ),
        pytest.param(
            {"action": "update_empty", "flags": False},
            does_not_raise(),
            id="update with no incoming keys does not flag",
        ),
        pytest.param(
            {"action": "update_iterable_of_pairs", "flags": False},
            does_not_raise(),
            id="update from iterable of (key, value) pairs does not flag",
        ),
        pytest.param(
            {"action": "update_kwargs_only", "flags": False},
            does_not_raise(),
            id="update with kwargs only (no positional arg) does not flag",
        ),
        pytest.param(
            {"action": "setdefault_new", "flags": False},
            does_not_raise(),
            id="setdefault inserting a new key does not flag",
        ),
        pytest.param(
            {"action": "setdefault_existing", "flags": False},
            does_not_raise(),
            id="setdefault on existing key does not flag",
        ),
        pytest.param(
            {"action": "remove_reflection_not_in_order", "flags": False},
            does_not_raise(),
            id="Sample.remove_reflection on non-orienting reflection does not flag",
        ),
    ],
)
def test_dict_api_mutations_flag_only_when_order_affected(parms, context):
    """
    Dict-API mutations on ``ReflectionsDict`` flag the solver-dirty
    bitfield iff the touched key is in :attr:`order`.  Reflections
    outside ``order`` are not visible to the solver, so removing or
    replacing them does not require a solver re-push.
    """
    with context:
        action = parms["action"]
        sim = _setup_for_action(action)
        refls = sim.sample.reflections

        _clean(sim)

        if action == "setitem_replace_in_order":
            refls["r006"] = _new_like(refls["r006"], "r006")
        elif action == "del_in_order":
            del refls["r006"]
        elif action == "pop_in_order":
            refls.pop("r006")
        elif action == "popitem_in_order":
            # Single-reflection setup; popitem removes the only entry
            # (which is also the only orienting reflection).
            refls.popitem()
        elif action == "clear_with_order":
            refls.clear()
        elif action == "update_overwrites_in_order":
            refls.update({"r006": _new_like(refls["r006"], "r006")})
        elif action == "remove_reflection_in_order":
            sim.sample.remove_reflection("r006")
        elif action == "setitem_new_key":
            refls["brand_new"] = _new_like(refls["r006"], "brand_new")
        elif action == "setitem_replace_not_in_order":
            refls["r006b"] = _new_like(refls["r006b"], "r006b")
        elif action == "del_not_in_order":
            del refls["r006b"]
        elif action == "pop_not_in_order":
            refls.pop("r006b")
        elif action == "pop_missing_with_default":
            result = refls.pop("nonexistent", "fallback")
            assert result == "fallback"
        elif action == "clear_empty_order":
            # Drop ``order`` to two-then-empty in two well-defined
            # steps before clearing the dict.
            with refls._suspend_strict_check():
                refls.order = []
            _clean(sim)
            refls.clear()
        elif action == "update_new_keys_only":
            refls.update({"brand_new": _new_like(refls["r006"], "brand_new")})
        elif action == "update_empty":
            refls.update({})
        elif action == "update_iterable_of_pairs":
            refls.update([("brand_new", _new_like(refls["r006"], "brand_new"))])
        elif action == "update_kwargs_only":
            refls.update(brand_new=_new_like(refls["r006"], "brand_new"))
        elif action == "setdefault_new":
            refls.setdefault("default_new", _new_like(refls["r006"], "default_new"))
        elif action == "setdefault_existing":
            refls.setdefault("r006", refls["r006"])
        elif action == "remove_reflection_not_in_order":
            sim.sample.remove_reflection("r006b")
        else:
            raise AssertionError(f"unknown action {action!r}")

        if parms["flags"]:
            assert sim.core._solver_dirty == _SolverDirty.SAMPLE | _SolverDirty.UB, (
                f"Expected SAMPLE|UB, got {sim.core._solver_dirty!r} "
                f"after action {action!r}"
            )
        else:
            assert sim.core._solver_dirty == _SolverDirty(0), (
                f"Expected no flag, got {sim.core._solver_dirty!r} "
                f"after action {action!r}"
            )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"action": "add_reflection"},
            does_not_raise(),
            id="solver receives reflection added via Core.add_reflection",
        ),
        pytest.param(
            {"action": "set_orientation_reflections"},
            does_not_raise(),
            id="solver receives reordered reflections",
        ),
    ],
)
def test_solver_receives_pushed_reflections(parms, context):
    """
    End-to-end: after a public mutation followed by ``update_solver()``
    the solver-side ``sample["reflections"]`` reflects the mutation.

    This is the contract that issue #397 broke for solvers that do not
    self-heal in ``calculate_UB`` (such as ``ad_hoc``).
    """
    with context:
        sim = _build_sim_with_reflections(n_reflections=2)
        _clean(sim)
        sim.core.update_solver()
        assert len(sim.core.solver.sample["reflections"]) == 2

        action = parms["action"]
        if action == "add_reflection":
            sim.add_reflection(
                pseudos=SAMPLE_DEF["reflections"][2]["pseudos"],
                reals=SAMPLE_DEF["reflections"][2]["reals"],
                name=SAMPLE_DEF["reflections"][2]["name"],
            )
            expected_count = 3
        elif action == "set_orientation_reflections":
            sim.sample.reflections.set_orientation_reflections(
                [sim.sample.reflections["r100"], sim.sample.reflections["r006"]]
            )
            expected_count = 2
        else:
            raise AssertionError(f"unknown action {action!r}")

        # Without the issue #397 fix, the next call would be a no-op
        # because ``_solver_dirty`` would still be 0 and the solver
        # would still hold the pre-mutation reflection set.
        sim.core.update_solver()
        assert sim.core._solver_dirty == _SolverDirty(0)
        assert len(sim.core.solver.sample["reflections"]) == expected_count


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {}, does_not_raise(), id="standalone ReflectionsDict has no _core (no-op)"
        )
    ],
)
def test_standalone_reflectionsdict_no_core(parms, context):
    """
    A ``ReflectionsDict`` constructed without a ``Sample`` must remain
    usable: ``_core`` defaults to ``None`` and ``_request_solver_update``
    is a no-op.
    """
    with context:
        rd = ReflectionsDict()
        assert rd._core is None
        rd.order = []
        rd._request_solver_update()


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"method": "set_orientation_reflections"},
            pytest.raises(KeyError, match=re.escape("'unknown'")),
            id="set_orientation_reflections raises on unknown name",
        ),
        pytest.param(
            {"method": "remove_reflection"},
            pytest.raises(
                KeyError, match=re.escape("Reflection 'unknown' is not found.")
            ),
            id="remove_reflection raises on unknown name",
        ),
    ],
)
def test_known_failure_modes_unchanged(parms, context):
    """
    The dirty-flag fix must not change error semantics of the public
    mutating API.  These failure modes are documented; they continue to
    raise the same exceptions.
    """
    with context:
        sim = _build_sim_with_reflections(n_reflections=2)
        if parms["method"] == "set_orientation_reflections":
            sim.sample.reflections.set_orientation_reflections(
                [sim.sample.reflections["unknown"]]
            )
        elif parms["method"] == "remove_reflection":
            sim.sample.remove_reflection("unknown")
