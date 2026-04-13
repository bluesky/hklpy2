"""Tests for GeometryDescriptor and SolverBase geometry registry."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ..base import SolverBase
from ..th_tth_q import TH_TTH_Q_GEOMETRY
from ..th_tth_q import ThTthSolver
from ..typing import GeometryDescriptor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_descriptor(name="MY GEO", pseudo=None, real=None, modes=None, **kwargs):
    """Return a minimal GeometryDescriptor for testing."""
    return GeometryDescriptor(
        name=name,
        pseudo_axis_names=pseudo or ["h", "k", "l"],
        real_axis_names=real or ["mu", "delta"],
        modes=modes or ["mode_a"],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# GeometryDescriptor construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                name="TH TTH Q",
                pseudo_axis_names=["q"],
                real_axis_names=["th", "tth"],
                modes=["bissector"],
                default_mode="bissector",
                description="theta/two-theta",
            ),
            does_not_raise(),
            id="full specification",
        ),
        pytest.param(
            dict(
                name="MINIMAL",
                pseudo_axis_names=["h"],
                real_axis_names=["omega"],
                modes=[],
            ),
            does_not_raise(),
            id="minimal - no modes, no description",
        ),
        pytest.param(
            dict(
                name="WITH EXTRAS",
                pseudo_axis_names=["h", "k", "l"],
                real_axis_names=["omega", "chi", "phi", "tth"],
                modes=["mode_a", "mode_b"],
                extra_axis_names={"mode_a": ["psi"], "mode_b": []},
            ),
            does_not_raise(),
            id="with per-mode extra axis names",
        ),
    ],
)
def test_geometry_descriptor_construction(parms, context):
    with context:
        desc = GeometryDescriptor(**parms)
        assert desc.name == parms["name"]
        assert desc.pseudo_axis_names == parms["pseudo_axis_names"]
        assert desc.real_axis_names == parms["real_axis_names"]
        assert desc.modes == parms["modes"]
        # defaults
        assert isinstance(desc.default_mode, str)
        assert isinstance(desc.description, str)
        assert isinstance(desc.extra_axis_names, dict)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="A", pseudo_axis_names=["q"], real_axis_names=["th"], modes=[]),
            does_not_raise(),
            id="extra_axis_names defaults to empty dict",
        ),
    ],
)
def test_geometry_descriptor_defaults(parms, context):
    with context:
        desc = GeometryDescriptor(**parms)
        assert desc.extra_axis_names == {}
        assert desc.default_mode == ""
        assert desc.description == ""


# ---------------------------------------------------------------------------
# GeometryDescriptor instances are independent (mutable default field safety)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(), does_not_raise(), id="extra_axis_names not shared between instances"
        ),
    ],
)
def test_geometry_descriptor_field_independence(parms, context):
    with context:
        desc_a = GeometryDescriptor(
            name="A", pseudo_axis_names=[], real_axis_names=[], modes=[]
        )
        desc_b = GeometryDescriptor(
            name="B", pseudo_axis_names=[], real_axis_names=[], modes=[]
        )
        desc_a.extra_axis_names["mode_x"] = ["psi"]
        assert "mode_x" not in desc_b.extra_axis_names


# ---------------------------------------------------------------------------
# SolverBase.register_geometry() — using ThTthSolver as the concrete class
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(descriptor=_make_descriptor("DYNAMIC GEO")),
            does_not_raise(),
            id="register a valid descriptor",
        ),
        pytest.param(
            dict(descriptor="not a descriptor"),
            pytest.raises(TypeError, match=re.escape("Expected GeometryDescriptor")),
            id="non-descriptor raises TypeError",
        ),
        pytest.param(
            dict(descriptor=42),
            pytest.raises(TypeError, match=re.escape("Expected GeometryDescriptor")),
            id="integer raises TypeError",
        ),
    ],
)
def test_register_geometry(parms, context):
    # Use a fresh isolated subclass so tests don't pollute ThTthSolver's registry.
    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    with context:
        _IsolatedSolver.register_geometry(parms["descriptor"])
        assert parms["descriptor"].name in _IsolatedSolver._geometry_registry
        assert parms["descriptor"].name in _IsolatedSolver.geometries()


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(names=["GEO A", "GEO B", "GEO C"]),
            does_not_raise(),
            id="geometries() returns sorted list of registered names",
        ),
        pytest.param(
            dict(names=[]),
            does_not_raise(),
            id="empty registry returns empty list",
        ),
    ],
)
def test_geometries_sorted(parms, context):
    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    with context:
        for name in parms["names"]:
            _IsolatedSolver.register_geometry(_make_descriptor(name))
        result = _IsolatedSolver.geometries()
        assert result == sorted(parms["names"])


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="OVERWRITE"),
            does_not_raise(),
            id="registering same name overwrites previous entry",
        ),
    ],
)
def test_register_geometry_overwrite(parms, context):
    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    with context:
        desc_v1 = _make_descriptor(parms["name"], real=["a"])
        desc_v2 = _make_descriptor(parms["name"], real=["x", "y"])
        _IsolatedSolver.register_geometry(desc_v1)
        _IsolatedSolver.register_geometry(desc_v2)
        stored = _IsolatedSolver._geometry_registry[parms["name"]]
        assert stored.real_axis_names == ["x", "y"]


# ---------------------------------------------------------------------------
# Registry isolation: ThTthSolver registry must not pollute SolverBase
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(dict(), does_not_raise(), id="base class registry stays empty"),
    ],
)
def test_registry_isolation(parms, context):
    with context:
        # SolverBase._geometry_registry should be empty (no geometries registered
        # directly on the abstract base).
        assert SolverBase._geometry_registry == {}
        # ThTthSolver has its own registry with the built-in geometry.
        assert TH_TTH_Q_GEOMETRY in ThTthSolver._geometry_registry
        # They must be different objects.
        assert ThTthSolver._geometry_registry is not SolverBase._geometry_registry


# ---------------------------------------------------------------------------
# ThTthSolver property dispatch via registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(geometry=TH_TTH_Q_GEOMETRY, expected_extras=[]),
            does_not_raise(),
            id="built-in geometry: correct axes and modes from descriptor",
        ),
        pytest.param(
            dict(
                geometry="GEO WITH EXTRAS",
                extra_axis_names={"mode_a": ["psi", "eta"], "mode_b": ["phi"]},
                expected_extras=["eta", "phi", "psi"],
            ),
            does_not_raise(),
            id="geometry with per-mode extra axis names: union returned sorted",
        ),
    ],
)
def test_th_tth_solver_uses_registry(parms, context):
    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    # Pre-populate with the built-in geometry so the TH_TTH_Q case works.
    _IsolatedSolver.register_geometry(ThTthSolver._geometry_registry[TH_TTH_Q_GEOMETRY])

    if parms["geometry"] != TH_TTH_Q_GEOMETRY:
        _IsolatedSolver.register_geometry(
            GeometryDescriptor(
                name=parms["geometry"],
                pseudo_axis_names=["h"],
                real_axis_names=["omega"],
                modes=list(parms.get("extra_axis_names", {}).keys()),
                extra_axis_names=parms.get("extra_axis_names", {}),
            )
        )

    with context:
        solver = _IsolatedSolver(parms["geometry"])
        desc = _IsolatedSolver._geometry_registry[parms["geometry"]]
        assert solver.pseudo_axis_names == desc.pseudo_axis_names
        assert solver.real_axis_names == desc.real_axis_names
        assert solver.modes == desc.modes
        assert solver.extra_axis_names == parms["expected_extras"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(geometry="UNREGISTERED"),
            does_not_raise(),
            id="unregistered geometry: empty axes and modes",
        ),
    ],
)
def test_th_tth_solver_unregistered_geometry(parms, context):
    with context:
        solver = ThTthSolver(parms["geometry"])
        assert solver.pseudo_axis_names == []
        assert solver.real_axis_names == []
        assert solver.modes == []
        assert solver.extra_axis_names == []


# ---------------------------------------------------------------------------
# Dynamic runtime registration and immediate use
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                name="RUNTIME GEO",
                pseudo=["h", "k", "l"],
                real=["mu", "delta", "nu"],
                modes=["mode_a", "mode_b"],
                default_mode="mode_a",
            ),
            does_not_raise(),
            id="geometry registered at runtime is immediately usable",
        ),
    ],
)
def test_dynamic_registration(parms, context):
    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    with context:
        desc = GeometryDescriptor(
            name=parms["name"],
            pseudo_axis_names=parms["pseudo"],
            real_axis_names=parms["real"],
            modes=parms["modes"],
            default_mode=parms["default_mode"],
        )
        _IsolatedSolver.register_geometry(desc)
        assert parms["name"] in _IsolatedSolver.geometries()

        solver = _IsolatedSolver(parms["name"])
        assert solver.pseudo_axis_names == parms["pseudo"]
        assert solver.real_axis_names == parms["real"]
        assert solver.modes == parms["modes"]
