"""
Regression tests for issues #384 and #386.

Issue #384
----------
``wh()`` (and ``inverse()``) returned values computed against a stale
solver state immediately after a ``calc_UB()`` call.  A workaround was
to call ``forward()`` once before reading pseudos.

Issue #386
----------
After switching the active sample (``core.sample = "name"``), ``wh()``
continued to return values consistent with the previous sample's
lattice.  Even calling ``calc_UB()`` on the new sample did not refresh
the displayed pseudos; only re-applying the lattice parameters did.

Both bugs share a root cause: the previous single-boolean dirty flag
in :class:`~hklpy2.ops.Core` did not faithfully track which solver
state domains were out of sync.  The fix is a fine-grained
:class:`~hklpy2.utils._SolverDirty` bitfield and corresponding
corrections to the ``Core.sample`` setter and ``Core.calc_UB``.
"""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from .. import creator
from ..utils import _SolverDirty


# Two distinct cubic samples with one obvious orientation difference
# each, so a sample switch produces a clearly different inverse() result
# at the same real-space position.
SAMPLE_A = dict(
    name="cubic_a",
    lattice=dict(a=5.0),
    reflections=[
        dict(name="A100", pseudos=(1, 0, 0), reals=(-145, 0, 0, 70)),
        dict(name="A010", pseudos=(0, 1, 0), reals=(-145, 0, 90, 70)),
    ],
)
SAMPLE_B = dict(
    name="cubic_b",
    lattice=dict(a=4.0),
    reflections=[
        dict(name="B100", pseudos=(1, 0, 0), reals=(-145, 0, 0, 70)),
        dict(name="B010", pseudos=(0, 1, 0), reals=(-145, 0, 90, 70)),
    ],
)
TOL = 1e-3
PROBE_REALS = dict(omega=-145, chi=0, phi=0, tth=70)


def _build_two_samples():
    """Create an E4CV simulator with two oriented cubic samples."""
    sim = creator()
    sim.beam.wavelength.put(1.54)

    for sample_def in (SAMPLE_A, SAMPLE_B):
        sim.add_sample(sample_def["name"], sample_def["lattice"]["a"], replace=True)
        for refl in sample_def["reflections"]:
            sim.add_reflection(
                pseudos=refl["pseudos"],
                reals=refl["reals"],
                name=refl["name"],
            )
        sim.core.calc_UB(*[r["name"] for r in sample_def["reflections"]])

    return sim


def _pseudos_at_probe(sim):
    """Compute (h, k, l) at the canonical probe real-space position."""
    return sim.inverse(**PROBE_REALS)


@pytest.mark.parametrize(
    "parms, context",
    [
        # ------------------------------------------------------------------
        # #386 -- sample switch must re-sync the solver's lattice + UB.
        # ------------------------------------------------------------------
        pytest.param(
            dict(action="switch", target="cubic_a"),
            does_not_raise(),
            id="switch-to-A-uses-A-lattice",
        ),
        pytest.param(
            dict(action="switch", target="cubic_b"),
            does_not_raise(),
            id="switch-to-B-uses-B-lattice",
        ),
        pytest.param(
            dict(action="round_trip", target="cubic_a"),
            does_not_raise(),
            id="round-trip-back-to-A-uses-A-lattice",
        ),
        # ------------------------------------------------------------------
        # #384 -- inverse() / wh() right after calc_UB must not be stale.
        # ------------------------------------------------------------------
        pytest.param(
            dict(action="calc_UB_then_inverse", target="cubic_a"),
            does_not_raise(),
            id="calc_UB-then-inverse-on-A",
        ),
        pytest.param(
            dict(action="switch_then_calc_UB", target="cubic_b"),
            does_not_raise(),
            id="switch-then-calc_UB-then-inverse-on-B",
        ),
        # ------------------------------------------------------------------
        # #386 -- unknown sample name must not silently corrupt state.
        # ------------------------------------------------------------------
        pytest.param(
            dict(action="switch", target="does_not_exist"),
            pytest.raises(
                KeyError,
                match=re.escape(
                    "'does_not_exist' not in sample list: "
                    "['sample', 'cubic_a', 'cubic_b']"
                ),
            ),
            id="switch-to-unknown-sample-raises-KeyError",
        ),
    ],
)
def test_solver_resyncs_after_state_change(parms, context):
    """Solver must be re-synced after sample switch and after calc_UB."""
    with context:
        sim = _build_two_samples()
        action = parms["action"]
        target = parms["target"]

        if action == "switch":
            sim.core.sample = target
            actual = _pseudos_at_probe(sim)
            assert sim.sample.name == target
        elif action == "round_trip":
            sim.core.sample = "cubic_b"
            _pseudos_at_probe(sim)  # ensure intermediate state is real
            sim.core.sample = target
            actual = _pseudos_at_probe(sim)
            assert sim.sample.name == target
        elif action == "calc_UB_then_inverse":
            sim.core.sample = target
            sim.core.calc_UB(*[r["name"] for r in SAMPLE_A["reflections"]])
            actual = _pseudos_at_probe(sim)
        elif action == "switch_then_calc_UB":
            sim.core.sample = "cubic_a"
            _pseudos_at_probe(sim)
            sim.core.sample = target
            sim.core.calc_UB(*[r["name"] for r in SAMPLE_B["reflections"]])
            actual = _pseudos_at_probe(sim)
        else:
            raise AssertionError(f"unknown action {action!r}")

        # Build a freshly-constructed simulator with only the target
        # sample so we have an independent reference for the expected
        # pseudos at the probe position.
        ref = creator()
        ref.beam.wavelength.put(1.54)
        sample_def = SAMPLE_A if target == "cubic_a" else SAMPLE_B
        ref.add_sample(target, sample_def["lattice"]["a"], replace=True)
        for refl in sample_def["reflections"]:
            ref.add_reflection(
                pseudos=refl["pseudos"],
                reals=refl["reals"],
                name=refl["name"],
            )
        ref.core.calc_UB(*[r["name"] for r in sample_def["reflections"]])
        expected = _pseudos_at_probe(ref)

        for axis in ("h", "k", "l"):
            assert np.isclose(
                getattr(actual, axis), getattr(expected, axis), atol=TOL
            ), (
                f"axis {axis}: actual={actual} expected={expected} "
                f"(action={action}, target={target})"
            )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(action="set_lattice_param"),
            does_not_raise(),
            id="lattice-edit-flags-SAMPLE-and-UB",
        ),
        pytest.param(
            dict(action="set_lattice_object"),
            does_not_raise(),
            id="lattice-replace-flags-SAMPLE-and-UB",
        ),
        pytest.param(
            dict(action="set_U"),
            does_not_raise(),
            id="U-edit-flags-UB-only",
        ),
        pytest.param(
            dict(action="set_UB"),
            does_not_raise(),
            id="UB-edit-flags-UB-only",
        ),
        pytest.param(
            dict(action="switch_sample"),
            does_not_raise(),
            id="sample-switch-flags-SAMPLE-and-UB",
        ),
        pytest.param(
            dict(action="set_mode"),
            does_not_raise(),
            id="mode-change-flags-MODE-and-EXTRAS",
        ),
        pytest.param(
            dict(action="bool_True"),
            does_not_raise(),
            id="back-compat-True-flags-ALL",
        ),
        pytest.param(
            dict(action="bool_False"),
            does_not_raise(),
            id="back-compat-False-clears-all",
        ),
        pytest.param(
            dict(action="needs_update_setter_True"),
            does_not_raise(),
            id="back-compat-_solver_needs_update-setter-True",
        ),
        pytest.param(
            dict(action="needs_update_setter_False"),
            does_not_raise(),
            id="back-compat-_solver_needs_update-setter-False",
        ),
    ],
)
def test_solver_dirty_flag_granularity(parms, context):
    """Per-domain flagging is correct for each documented mutation."""
    with context:
        sim = _build_two_samples()
        # Force a clean state so we can observe what the next mutation
        # flags.  Mutations performed by the harness above leave the
        # bitfield in an arbitrary state.
        sim.core.update_solver()
        sim.core._solver_dirty = _SolverDirty(0)
        assert sim.core._solver_dirty == _SolverDirty(0)

        action = parms["action"]
        if action == "set_lattice_param":
            sim.sample.lattice.a = 5.5
            assert sim.core._solver_dirty == _SolverDirty.SAMPLE | _SolverDirty.UB
        elif action == "set_lattice_object":
            sim.sample.lattice = dict(a=5.5)
            assert sim.core._solver_dirty == _SolverDirty.SAMPLE | _SolverDirty.UB
        elif action == "set_U":
            sim.sample.U = sim.sample.U
            assert sim.core._solver_dirty == _SolverDirty.UB
        elif action == "set_UB":
            sim.sample.UB = sim.sample.UB
            assert sim.core._solver_dirty == _SolverDirty.UB
        elif action == "switch_sample":
            sim.core.sample = "cubic_a"
            assert sim.core._solver_dirty == _SolverDirty.SAMPLE | _SolverDirty.UB
        elif action == "set_mode":
            other_modes = [m for m in sim.core.modes if m != sim.core.mode]
            assert other_modes, "test fixture has only one mode"
            sim.core.mode = other_modes[0]
            assert sim.core._solver_dirty == _SolverDirty.MODE | _SolverDirty.EXTRAS
        elif action == "bool_True":
            sim.core.request_solver_update(True)
            assert sim.core._solver_dirty == _SolverDirty.ALL
        elif action == "bool_False":
            sim.core.request_solver_update(_SolverDirty.ALL)
            sim.core.request_solver_update(False)
            assert sim.core._solver_dirty == _SolverDirty(0)
        elif action == "needs_update_setter_True":
            sim.core._solver_needs_update = True
            assert sim.core._solver_dirty == _SolverDirty.ALL
            assert sim.core._solver_needs_update is True
        elif action == "needs_update_setter_False":
            sim.core.request_solver_update(_SolverDirty.ALL)
            sim.core._solver_needs_update = False
            assert sim.core._solver_dirty == _SolverDirty(0)
            assert sim.core._solver_needs_update is False
        else:
            raise AssertionError(f"unknown action {action!r}")


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(domains=_SolverDirty.SAMPLE),
            does_not_raise(),
            id="additive-SAMPLE",
        ),
        pytest.param(
            dict(domains=_SolverDirty.UB),
            does_not_raise(),
            id="additive-UB",
        ),
        pytest.param(
            dict(domains=_SolverDirty.MODE | _SolverDirty.EXTRAS),
            does_not_raise(),
            id="additive-MODE-and-EXTRAS",
        ),
    ],
)
def test_request_solver_update_is_additive_for_flag_argument(parms, context):
    """Flag-typed argument adds to existing dirty state; bool replaces it."""
    with context:
        sim = _build_two_samples()
        sim.core.update_solver()
        sim.core._solver_dirty = _SolverDirty.WAVELENGTH

        sim.core.request_solver_update(parms["domains"])
        assert sim.core._solver_dirty == _SolverDirty.WAVELENGTH | parms["domains"]
