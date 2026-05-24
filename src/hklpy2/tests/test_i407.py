"""
Regression test for issue #407.

``Diffractometer.export()`` / ``core.configuration`` must reflect the
*current* value of ``core.mode`` after assignment, even if no
``forward()`` / ``inverse()`` / ``update_solver()`` call has intervened
to flush the dirty bitfield to the underlying solver.

The fix flushes pending solver writes at the top of
``Core._asdict()`` (gated on the dirty bitfield) so the snapshot is
self-consistent.  Covered here:

* The real ``hkl_soleil`` (``HklSolver``) backend, whose ``_metadata``
  reads ``self.engine.current_mode_get()``.
* A minimal ``SolverBase``-style stand-in whose ``_metadata`` reads
  ``self.mode`` (the same pattern out-of-tree solvers use).
"""

from contextlib import nullcontext as does_not_raise
from typing import List

import pytest

from hklpy2 import creator
from hklpy2 import solver_utils
from hklpy2.backends.no_op import NoOpSolver
from hklpy2.backends.typing import GeometryDescriptor


class _StandInSolver407(NoOpSolver):
    """
    Minimal stand-in solver that emits ``mode`` through ``_metadata``.

    Mirrors the pattern used by ``HklSolver`` (and by out-of-tree
    solvers) where ``_metadata`` surfaces the live solver-side mode.
    Used to confirm the staleness fix is solver-agnostic.
    """

    name = "stand_in_407"
    _geometry_registry: dict = {}

    @property
    def _metadata(self):
        meta = dict(super()._metadata)
        meta["mode"] = self.mode
        return meta

    @property
    def real_axis_names(self) -> List[str]:
        return ["omega"]

    @property
    def pseudo_axis_names(self) -> List[str]:
        return ["h"]

    @property
    def modes(self) -> List[str]:
        return ["alpha", "beta", "gamma"]


_StandInSolver407.register_geometry(
    GeometryDescriptor(
        name="STANDIN407",
        pseudo_axis_names=["h"],
        real_axis_names=["omega"],
        modes=["alpha", "beta", "gamma"],
        extra_axis_names={"alpha": [], "beta": [], "gamma": []},
    )
)


@pytest.fixture
def _patched_solvers(monkeypatch):
    """Make the stand-in solver discoverable via ``get_solver``."""
    real_get_solver = solver_utils.get_solver
    real_solvers = solver_utils.solvers

    def fake_get_solver(name):
        if name == _StandInSolver407.name:
            return _StandInSolver407
        return real_get_solver(name)

    def fake_solvers():
        mapping = dict(real_solvers())
        mapping[_StandInSolver407.name] = (
            f"{_StandInSolver407.__module__}:{_StandInSolver407.__name__}"
        )
        return mapping

    monkeypatch.setattr(solver_utils, "get_solver", fake_get_solver)
    monkeypatch.setattr(solver_utils, "solvers", fake_solvers)
    yield


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(new_mode="constant_omega"),
            does_not_raise(),
            id="hkl_soleil E4CV constant_omega flushed on export",
        ),
        pytest.param(
            dict(new_mode="constant_chi"),
            does_not_raise(),
            id="hkl_soleil E4CV constant_chi flushed on export",
        ),
        pytest.param(
            dict(new_mode="constant_phi"),
            does_not_raise(),
            id="hkl_soleil E4CV constant_phi flushed on export",
        ),
    ],
)
def test_hkl_soleil_mode_flushed_on_export(parms, context):
    """A bare ``core.mode = X`` is reflected by ``core.configuration``."""
    pytest.importorskip("hkl")
    with context:
        sim = creator(solver="hkl_soleil", geometry="E4CV", name="e4cv_407")
        # Sanity: the new mode differs from the default.
        assert sim.core.mode != parms["new_mode"]

        sim.core.mode = parms["new_mode"]
        # No forward() / inverse() / update_solver() between the
        # assignment and the snapshot.
        cfg = sim.configuration

        assert cfg["solver"]["mode"] == parms["new_mode"]
        # And the cached Python-side mode also matches.
        assert sim.core.mode == parms["new_mode"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(new_mode="beta"),
            does_not_raise(),
            id="standin solver mode beta flushed on export",
        ),
        pytest.param(
            dict(new_mode="gamma"),
            does_not_raise(),
            id="standin solver mode gamma flushed on export",
        ),
    ],
)
def test_standin_solver_mode_flushed_on_export(parms, context, _patched_solvers):
    """Solver-agnostic check: any ``_metadata`` that reads ``self.mode``."""
    with context:
        sim = creator(
            name="standin407",
            solver=_StandInSolver407.name,
            geometry="STANDIN407",
        )
        assert sim.core.mode != parms["new_mode"]

        sim.core.mode = parms["new_mode"]
        cfg = sim.configuration

        assert cfg["solver"]["mode"] == parms["new_mode"]
        assert sim.core.solver.mode == parms["new_mode"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="clean bitfield: snapshot does not re-push to solver",
        ),
    ],
)
def test_clean_bitfield_skips_flush(parms, context, _patched_solvers, monkeypatch):
    """The fast path must not invoke ``update_solver()`` when clean."""
    with context:
        sim = creator(
            name="standin407_clean",
            solver=_StandInSolver407.name,
            geometry="STANDIN407",
        )
        # Force a clean state.
        sim.core.update_solver()
        assert not sim.core._solver_dirty

        # Spy on update_solver: it must NOT be called by _asdict()
        # when the bitfield is clean.
        calls: list = []
        real_update = sim.core.update_solver

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return real_update(*args, **kwargs)

        monkeypatch.setattr(sim.core, "update_solver", spy)
        _ = sim.configuration

        assert calls == [], (
            f"expected no update_solver() calls on clean snapshot, got {calls}"
        )
