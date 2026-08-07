# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Regression test for issue #405.

``simulator_from_config()`` must forward every non-reserved key in the
persisted ``solver:`` block to the solver constructor via
``solver_kwargs``.  This lets out-of-tree solvers persist construction
state by overriding ``_metadata`` and accepting the matching keyword
in their ``__init__``.
"""

import logging
from contextlib import nullcontext as does_not_raise

import pytest

from hklpy2 import creator, solver_utils
from hklpy2.backends.no_op import NoOpSolver
from hklpy2.backends.typing import GeometryDescriptor
from hklpy2.run_utils import _RESERVED_SOLVER_KEYS, simulator_from_config

# Module-level mutable cell so the stand-in solver's ``__init__`` can
# report what kwargs it received to the active test function.
_LAST_INIT_KWARGS: dict = {}


class _StandInSolver(NoOpSolver):
    """
    Minimal stand-in solver used only by the issue-405 round-trip test.

    Persists an arbitrary ``marker`` field through ``_metadata`` and
    accepts it back via ``__init__``.  Recording the received kwargs
    in a module-level cell lets the test verify that the value made
    it through the full export / restore round-trip.
    """

    name = "stand_in_405"
    _geometry_registry: dict = {}

    def __init__(
        self, geometry: str, *, marker: str = "", another: object = None, **kwargs
    ) -> None:
        super().__init__(geometry, **kwargs)
        self._marker = marker
        self._another = another
        _LAST_INIT_KWARGS.clear()
        _LAST_INIT_KWARGS.update(
            {"geometry": geometry, "marker": marker, "another": another}
        )

    @property
    def _metadata(self):
        meta = dict(super()._metadata)
        meta["marker"] = self._marker
        return meta

    @property
    def real_axis_names(self) -> list[str]:
        return ["omega"]

    @property
    def pseudo_axis_names(self) -> list[str]:
        return ["h"]

    @property
    def modes(self) -> list[str]:
        return ["default"]


# Register a single fake geometry on the stand-in solver.
_StandInSolver.register_geometry(
    GeometryDescriptor(
        name="STANDIN",
        pseudo_axis_names=["h"],
        real_axis_names=["omega"],
        modes=["default"],
        extra_axis_names={"default": []},
    )
)


@pytest.fixture
def _patched_solvers(monkeypatch):
    """Make the stand-in solver discoverable via ``get_solver``."""

    real_get_solver = solver_utils.get_solver
    real_solvers = solver_utils.solvers

    def fake_get_solver(name):
        if name == _StandInSolver.name:
            return _StandInSolver
        return real_get_solver(name)

    def fake_solvers():
        mapping = dict(real_solvers())
        mapping[_StandInSolver.name] = (
            f"{_StandInSolver.__module__}:{_StandInSolver.__name__}"
        )
        return mapping

    monkeypatch.setattr(solver_utils, "get_solver", fake_get_solver)
    monkeypatch.setattr(solver_utils, "solvers", fake_solvers)
    yield


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"marker": "hello-405"},
            does_not_raise(),
            id="non-reserved marker key survives round-trip",
        ),
        pytest.param(
            {"marker": ""}, does_not_raise(), id="empty marker still forwarded"
        ),
        pytest.param(
            {"marker": {"nested": [1, 2, 3]}},
            does_not_raise(),
            id="structured marker value survives round-trip",
        ),
    ],
)
def test_solver_kwargs_roundtrip(parms, context, _patched_solvers, caplog):
    """Non-reserved ``solver:`` keys flow through ``solver_kwargs``."""
    with context:
        original = creator(
            name="standin",
            solver=_StandInSolver.name,
            geometry="STANDIN",
            solver_kwargs={"marker": parms["marker"]},
        )

        # The solver received the marker on the first construction.
        assert _LAST_INIT_KWARGS["marker"] == parms["marker"]

        # Snapshot, then rebuild from the config dict.
        config = original.configuration
        assert config["solver"]["marker"] == parms["marker"]

        # Capture the debug-log entry that lists the forwarded kwargs.
        _LAST_INIT_KWARGS.clear()
        with caplog.at_level(logging.DEBUG, logger="hklpy2.run_utils"):
            sim = simulator_from_config(config)

        # Stand-in solver's ``__init__`` was called with the marker.
        assert _LAST_INIT_KWARGS["marker"] == parms["marker"]
        # The new simulator carries the same marker.
        assert sim.core.solver._marker == parms["marker"]

        # Debug logging announces forwarded kwargs (issue #405).
        forwarded_msgs = [
            r.message for r in caplog.records if "forwarding solver_kwargs" in r.message
        ]
        assert forwarded_msgs, "expected a 'forwarding solver_kwargs' debug line"
        assert "marker" in forwarded_msgs[0]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"extra_keys": {"marker": "m", "another": 42}},
            does_not_raise(),
            id="multiple non-reserved keys all forwarded",
        )
    ],
)
def test_multiple_non_reserved_keys_forwarded(parms, context, _patched_solvers):
    """Every non-reserved key in ``solver:`` flows through, not just one."""
    with context:
        # Build a minimal config dict by hand so we exercise the
        # forwarding path on arbitrary extra keys without depending on
        # the stand-in's ``_metadata`` shape.
        original = creator(
            name="standin",
            solver=_StandInSolver.name,
            geometry="STANDIN",
            solver_kwargs={"marker": "m"},
        )
        config = original.configuration
        # Inject additional non-reserved keys after export.
        config["solver"].update(parms["extra_keys"])

        _LAST_INIT_KWARGS.clear()
        simulator_from_config(config)

        # Both non-reserved keys reached the stand-in constructor.
        for key, value in parms["extra_keys"].items():
            assert _LAST_INIT_KWARGS[key] == value


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {
                "reserved": {
                    "name",
                    "description",
                    "geometry",
                    "real_axes",
                    "version",
                    "mode",
                }
            },
            does_not_raise(),
            id="reserved key set matches the documented contract",
        )
    ],
)
def test_reserved_solver_keys_contract(parms, context):
    """The reserved-key set is stable and matches its documentation."""
    with context:
        assert set(_RESERVED_SOLVER_KEYS) == parms["reserved"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {}, does_not_raise(), id="HklSolver engine field still survives round-trip"
        )
    ],
)
def test_hkl_soleil_engine_still_forwarded(parms, context):
    """
    The pre-existing ``engine`` special case is subsumed by the generic
    forwarding; HklSolver configs must continue to round-trip unchanged.
    """
    pytest.importorskip("hkl")
    with context:
        original = creator(name="sixc", geometry="E6C", solver_kwargs={"engine": "psi"})
        config = original.configuration
        assert config["solver"]["engine"] == "psi"

        sim = simulator_from_config(config)
        assert sim.core.solver.engine_name == "psi"
