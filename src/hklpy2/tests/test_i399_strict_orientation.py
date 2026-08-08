# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Regression tests for issue #399.

Strict orientation-health policy (option A): refuse mutations that
would leave the sample *half-defined* -- two or more reflections still
in the dict, but fewer than two named in
:attr:`hklpy2.blocks.reflection.ReflectionsDict.order` (restricted to
names that actually exist in the dict).

The user retains complete control: they must explicitly designate the
new orienting pair (via
:meth:`~hklpy2.blocks.reflection.ReflectionsDict.set_orientation_reflections`
or by assigning :attr:`order`) **before** removing the current one,
or remove the non-orienting reflections first so the sample falls
below the orientation threshold.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from .. import creator
from ..blocks.reflection import Reflection
from ..blocks.reflection import ReflectionsDict
from ..exceptions import ReflectionError

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


def _build_three_oriented():
    """Sim with three reflections; ``order = ['r006', 'r100']``."""
    sim = creator()
    sim.beam.wavelength.put(1.5498)
    sim.add_sample(
        SAMPLE_DEF["name"],
        SAMPLE_DEF["lattice"]["a"],
        c=SAMPLE_DEF["lattice"]["c"],
        gamma=SAMPLE_DEF["lattice"]["gamma"],
    )
    for refl in SAMPLE_DEF["reflections"]:
        sim.add_reflection(
            pseudos=refl["pseudos"], reals=refl["reals"], name=refl["name"]
        )
    sim.sample.reflections.order = ["r006", "r100"]
    return sim


def _new_like(refl: Reflection, name: str) -> Reflection:
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


HALF_DEFINED_MSG = re.escape("Refusing to leave the sample half-defined")


# -- All nine mutation paths from the issue: half-defined transitions
#    must raise; benign mutations must not.


@pytest.mark.parametrize(
    "parms, context",
    [
        # Path 1: Sample.remove_reflection on an orienting reflection
        # while two or more remain.
        pytest.param(
            {"action": "sample_remove_orienting"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-1-Sample.remove_reflection-on-orienting",
        ),
        # Path 2: __delitem__ on an orienting key.
        pytest.param(
            {"action": "del_orienting"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-2-__delitem__-on-orienting",
        ),
        # Path 3: pop on an orienting key.
        pytest.param(
            {"action": "pop_orienting"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-3-pop-on-orienting",
        ),
        # Path 4: popitem when it would remove an orienting key.
        pytest.param(
            {"action": "popitem_orienting"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-4-popitem-on-orienting",
        ),
        # Path 8: order setter assigned a list of length < 2 while
        # two or more reflections remain.
        pytest.param(
            {"action": "order_setter_length_zero"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-8a-order-setter-empty",
        ),
        pytest.param(
            {"action": "order_setter_length_one"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-8b-order-setter-single-entry",
        ),
        pytest.param(
            {"action": "order_setter_two_but_one_unknown"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-8c-order-setter-two-but-one-unknown-name",
        ),
        # Path 9: set_orientation_reflections / setor with < 2
        # reflections from a multi-reflection dict.
        pytest.param(
            {"action": "setor_single_reflection"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-9a-set_orientation_reflections-single-entry",
        ),
        pytest.param(
            {"action": "setor_alias_single_reflection"},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="path-9b-setor-alias-single-entry",
        ),
    ],
)
def test_strict_check_raises_on_half_defined_transition(parms, context):
    """Every documented half-defined transition raises ``ReflectionError``."""
    with context:
        sim = _build_three_oriented()
        refls = sim.sample.reflections
        action = parms["action"]

        if action == "sample_remove_orienting":
            sim.sample.remove_reflection("r006")
        elif action == "del_orienting":
            del refls["r006"]
        elif action == "pop_orienting":
            refls.pop("r006")
        elif action == "popitem_orienting":
            # Reduce the dict so popitem would remove the last (and
            # ordered) entry while two remain in the dict.  Insertion
            # order at this point is r006, r100, r006b.  Pop r006b
            # (non-orienting) first, then assign order to keep two
            # entries; popitem will then remove r100 (orienting).
            del refls["r006b"]
            # Now dict = {r006, r100}, order = [r006, r100]; popitem
            # would remove r100, leaving dict={r006} and effective
            # order=[r006] -- below threshold but dict<2 after.  That
            # case is *not* half-defined.  To force the half-defined
            # transition, add a non-orienting third entry and popitem
            # it... but popitem removes last-inserted.  Instead, drop
            # one orienting entry by reassigning order to keep both,
            # then add a third reflection so popitem removes that --
            # but again, that third would not be orienting.
            #
            # The cleanest way to provoke a half-defined ``popitem``
            # is to leave the ordered key at the end of the dict.
            # Re-add r006 last so it becomes the popitem target.
            r006_template = SAMPLE_DEF["reflections"][0]
            sim.add_reflection(
                pseudos=r006_template["pseudos"],
                reals=r006_template["reals"],
                name="r006",
                replace=True,
            )
            # Now dict = {r100, r006} (r006 is last), order should
            # still include both.  popitem removes r006: dict goes
            # 2 -> 1.  Effective order = [r100], dict size = 1
            # post-pop -- not half-defined.  Need 3 entries with the
            # ordered one at the end.
            extra = SAMPLE_DEF["reflections"][2]
            sim.add_reflection(
                pseudos=extra["pseudos"],
                reals=extra["reals"],
                name="r006b",
                replace=True,
            )
            # popitem removes r006b (last inserted) but r006b is not
            # in order.  Adjust order to include r006b instead of one
            # of the others, but keep two entries.
            refls.order = ["r006b", "r100"]
            # Now order's last (r006b) is the popitem target and dict
            # has 3 entries; post-popitem dict = 2, effective order =
            # [r100] (only). Half-defined.
            refls.popitem()
        elif action == "order_setter_length_zero":
            refls.order = []
        elif action == "order_setter_length_one":
            refls.order = ["r006"]
        elif action == "order_setter_two_but_one_unknown":
            refls.order = ["r006", "no_such_name"]
        elif action == "setor_single_reflection":
            refls.set_orientation_reflections([refls["r006"]])
        elif action == "setor_alias_single_reflection":
            refls.setor([refls["r006"]])
        else:
            raise AssertionError(f"unknown action {action!r}")


@pytest.mark.parametrize(
    "parms, context",
    [
        # Removing the only orienting reflection while the dict drops
        # below two entries: not half-defined.
        pytest.param(
            {"action": "remove_orienting_drops_below_two"},
            does_not_raise(),
            id="remove-orienting-when-dict-drops-below-two",
        ),
        # Removing a non-orienting reflection: order is untouched.
        pytest.param(
            {"action": "remove_non_orienting"},
            does_not_raise(),
            id="remove-non-orienting",
        ),
        # Order setter assigned a valid two-reflection list.
        pytest.param(
            {"action": "order_setter_two_valid"},
            does_not_raise(),
            id="order-setter-two-valid-entries",
        ),
        # Atomic rewrite to a valid pair (issue acceptance criterion).
        pytest.param(
            {"action": "atomic_swap_via_order_setter"},
            does_not_raise(),
            id="atomic-swap-via-order-setter",
        ),
        # set_orientation_reflections with two reflections.
        pytest.param(
            {"action": "setor_two_reflections"},
            does_not_raise(),
            id="setor-two-reflections",
        ),
        # __setitem__ does not shrink dict or order; never half-defined.
        pytest.param(
            {"action": "setitem_replace_orienting"},
            does_not_raise(),
            id="setitem-replace-orienting-key",
        ),
        # update overwriting an orienting key: same as setitem.
        pytest.param(
            {"action": "update_overwrites_orienting"},
            does_not_raise(),
            id="update-overwrites-orienting-key",
        ),
        # update inserting only new keys: dict grows, order unchanged.
        pytest.param(
            {"action": "update_new_keys"},
            does_not_raise(),
            id="update-inserts-new-keys",
        ),
        # clear: post-state is always empty (not half-defined).
        pytest.param(
            {"action": "clear_three_reflections"},
            does_not_raise(),
            id="clear-three-reflections",
        ),
        # Sample with a single reflection: removing it is allowed
        # (dict drops to 0; not half-defined).
        pytest.param(
            {"action": "single_reflection_removal"},
            does_not_raise(),
            id="single-reflection-removal",
        ),
        # The downstream ``calc_UB(r1, r2)`` path must continue to
        # work because it explicitly calls
        # ``set_orientation_reflections([r1, r2])`` first -- the
        # strict check on the order setter sees a length-2 list and
        # accepts it.
        pytest.param(
            {"action": "calc_UB_round_trip"}, does_not_raise(), id="calc_UB-still-works"
        ),
    ],
)
def test_strict_check_does_not_fire_on_well_defined_transitions(parms, context):
    """Mutations whose post-state is well-defined must not raise."""
    with context:
        action = parms["action"]
        if action == "single_reflection_removal":
            sim = creator()
            sim.beam.wavelength.put(1.5498)
            sim.add_sample(
                SAMPLE_DEF["name"],
                SAMPLE_DEF["lattice"]["a"],
                c=SAMPLE_DEF["lattice"]["c"],
                gamma=SAMPLE_DEF["lattice"]["gamma"],
            )
            sim.add_reflection(
                pseudos=SAMPLE_DEF["reflections"][0]["pseudos"],
                reals=SAMPLE_DEF["reflections"][0]["reals"],
                name="r006",
            )
            assert len(sim.sample.reflections) == 1
            sim.sample.remove_reflection("r006")
            assert len(sim.sample.reflections) == 0
            return

        sim = _build_three_oriented()
        refls = sim.sample.reflections

        if action == "remove_orienting_drops_below_two":
            # Remove non-orienting first so the dict is at 2; then
            # remove an orienting reflection -- post-state has 1
            # entry, so no half-defined raise.
            sim.sample.remove_reflection("r006b")
            sim.sample.remove_reflection("r006")
            assert len(refls) == 1
        elif action == "remove_non_orienting":
            sim.sample.remove_reflection("r006b")
            assert "r006b" not in refls
            assert refls.order == ["r006", "r100"]
        elif action == "order_setter_two_valid":
            refls.order = ["r100", "r006"]
            assert refls.order == ["r100", "r006"]
        elif action == "atomic_swap_via_order_setter":
            refls.order = ["r006b", "r100"]
            assert refls.order == ["r006b", "r100"]
        elif action == "setor_two_reflections":
            refls.set_orientation_reflections([refls["r006b"], refls["r100"]])
            assert refls.order == ["r006b", "r100"]
        elif action == "setitem_replace_orienting":
            refls["r006"] = _new_like(refls["r006"], "r006")
        elif action == "update_overwrites_orienting":
            refls.update({"r006": _new_like(refls["r006"], "r006")})
        elif action == "update_new_keys":
            refls.update({"brand_new": _new_like(refls["r006"], "brand_new")})
            assert "brand_new" in refls
        elif action == "clear_three_reflections":
            refls.clear()
            assert len(refls) == 0
        elif action == "calc_UB_round_trip":
            sim.core.calc_UB("r006b", "r100")
            assert refls.order == ["r006b", "r100"]
        else:
            raise AssertionError(f"unknown action {action!r}")


# -- The dict and ``order`` must be unchanged after the exception fires.


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"action": "del_orienting"},
            does_not_raise(),
            id="state-preserved-after-del-raises",
        ),
        pytest.param(
            {"action": "pop_orienting"},
            does_not_raise(),
            id="state-preserved-after-pop-raises",
        ),
        pytest.param(
            {"action": "sample_remove_orienting"},
            does_not_raise(),
            id="state-preserved-after-Sample.remove_reflection-raises",
        ),
        pytest.param(
            {"action": "order_setter"},
            does_not_raise(),
            id="state-preserved-after-order-setter-raises",
        ),
        pytest.param(
            {"action": "setor"},
            does_not_raise(),
            id="state-preserved-after-setor-raises",
        ),
    ],
)
def test_state_unchanged_when_strict_check_raises(parms, context):
    """The dict and ``order`` are unchanged when the exception fires."""
    with context:
        sim = _build_three_oriented()
        refls = sim.sample.reflections
        # Snapshot the pre-mutation state.
        keys_before = list(refls.keys())
        order_before = list(refls.order)

        action = parms["action"]
        if action == "del_orienting":
            with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
                del refls["r006"]
        elif action == "pop_orienting":
            with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
                refls.pop("r006")
        elif action == "sample_remove_orienting":
            with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
                sim.sample.remove_reflection("r006")
        elif action == "order_setter":
            with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
                refls.order = ["r006"]
        elif action == "setor":
            with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
                refls.set_orientation_reflections([refls["r006"]])
        else:
            raise AssertionError(f"unknown action {action!r}")

        assert list(refls.keys()) == keys_before, (
            "Dict keys changed despite the exception"
        )
        assert list(refls.order) == order_before, "Order changed despite the exception"


# -- Standalone (no Core) ReflectionsDict still enforces the policy
#    because it does not depend on the Core back-reference.


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {},
            pytest.raises(ReflectionError, match=HALF_DEFINED_MSG),
            id="standalone-ReflectionsDict-still-enforces",
        )
    ],
)
def test_strict_check_works_standalone(parms, context):
    """The strict check is independent of the Core back-reference."""
    with context:
        rd = ReflectionsDict()
        # Two minimally valid reflections, same geometry.
        r1 = Reflection(
            "r1",
            {"h": 1, "k": 0, "l": 0},
            {"omega": 0},
            1.0,
            "g",
            ["h", "k", "l"],
            ["omega"],
        )
        r2 = Reflection(
            "r2",
            {"h": 0, "k": 1, "l": 0},
            {"omega": 1},
            1.0,
            "g",
            ["h", "k", "l"],
            ["omega"],
        )
        rd.add(r1)
        rd.add(r2)
        # Both are orienting after add().  Drop order to length 1
        # while two reflections remain in the dict -- raise.
        rd.order = ["r1"]


# -- The suspend mechanism deliberately bypasses the strict check
#    for internal multi-step operations (and for tests that need to
#    construct half-defined states intentionally).


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {}, does_not_raise(), id="suspend-mechanism-allows-half-defined-setup"
        )
    ],
)
def test_suspend_mechanism_bypasses_strict_check(parms, context):
    """``_suspend_strict_check`` allows internal/setup code to traverse the half-defined state."""
    with context:
        sim = _build_three_oriented()
        refls = sim.sample.reflections

        with refls._suspend_strict_check():
            # Same mutations that would normally raise.
            refls.order = ["r006"]
            assert refls.order == ["r006"]

        # Outside the context, the strict check is active again.
        with pytest.raises(ReflectionError, match=HALF_DEFINED_MSG):
            del refls["r006"]
