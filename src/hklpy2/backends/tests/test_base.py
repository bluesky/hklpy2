"""Test the solver base class."""

# Many features are tested, albeit indrectly, in specific solvers.

import re
from contextlib import nullcontext as does_not_raise

import pyRestTable
import pytest

from ...blocks.lattice import Lattice
from ...blocks.reflection import Reflection
from ...exceptions import SolverError
from ...utils import IDENTITY_MATRIX_3X3
from ..base import SolverBase
from ..no_op import NoOpSolver
from ..th_tth_q import BISECTOR_MODE
from ..th_tth_q import TH_TTH_Q_GEOMETRY
from ..th_tth_q import ThTthSolver
from ..typing import GeometryDescriptor


class TrivialSolver(SolverBase):
    """Trivial implementation for testing."""

    def addReflection(self, reflection: Reflection):
        """."""

    def calculate_UB(
        self,
        r1: Reflection,
        r2: Reflection,
    ) -> list[list[float]]:
        """."""
        return IDENTITY_MATRIX_3X3

    @property
    def extra_axis_names(self) -> list[str]:
        """."""
        return []

    def forward(self, pseudos: dict) -> list[dict[str, float]]:
        """."""
        return [{}]

    @classmethod
    def geometries(cls) -> list[str]:
        """."""
        return []

    def inverse(self, reals: dict) -> dict[str, float]:
        """."""
        return {}

    @property
    def modes(self) -> list[str]:
        """."""
        return []

    @property
    def pseudo_axis_names(self) -> list[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        # Do NOT sort.
        return []

    @property
    def real_axis_names(self) -> list[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        # Do NOT sort.
        return []

    def refineLattice(self, reflections: list[Reflection]) -> Lattice:
        """Refine the lattice parameters from a list of reflections."""
        return Lattice(1.0)  # always cubic, for testing

    def removeAllReflections(self) -> None:
        """Remove all reflections."""


def test_SolverBase():
    assert TrivialSolver.geometries() == []

    solver = TrivialSolver("test_geo")
    assert isinstance(solver, SolverBase)
    assert solver.name == "base"
    assert solver.sample is None
    assert solver.UB == IDENTITY_MATRIX_3X3
    assert solver.calculate_UB(None, None) == IDENTITY_MATRIX_3X3
    assert solver.extra_axis_names == [], f"{solver.extra_axis_names=}"
    assert solver.extras == {}, f"{solver.extras=}"
    assert solver.forward({}) == [{}]
    assert (
        list(solver._metadata) == "name description geometry real_axes version".split()
    )
    assert solver.mode == ""
    assert solver.inverse({}) == {}
    assert solver.inverse({}) == {}
    assert solver.pseudo_axis_names == [], f"{solver.pseudo_axis_names=}"
    assert solver.real_axis_names == [], f"{solver.real_axis_names=}"
    assert solver.refineLattice([]) == Lattice(a=1.0)

    delattr(solver, "_mode")
    assert solver.mode == ""

    md = solver._metadata
    assert isinstance(md, dict)
    expected = {
        "name": solver.name,
        "description": repr(solver),
        "geometry": solver.geometry,
        "real_axes": solver.real_axis_names,
        "version": solver.version,
    }
    assert md == expected

    expected = "\n".join(
        [
            "==== ========= ======= =========== ========",
            "mode pseudo(s) real(s) writable(s) extra(s)",
            "==== ========= ======= =========== ========",
            "==== ========= ======= =========== ========",
        ]
    )
    summary = solver.summary
    assert isinstance(summary, pyRestTable.Table)
    assert str(summary).strip() == expected

    with pytest.raises(TypeError, match=re.escape("Must supply")):
        solver.lattice = 1.0
    solver.lattice = dict(a=1, b=2, c=3, alpha=90, beta=90, gamma=90)
    assert solver.lattice == dict(a=1, b=2, c=3, alpha=90, beta=90, gamma=90)

    with pytest.raises(TypeError, match=re.escape("Must supply")):
        solver.sample = 1.0
    solver.sample = {"name": "si", "lattice": {}, "reflections": {}, "order": []}
    assert solver.sample is not None


def test_SolverBase_abstractmethods():
    # Need to test certain abstract methods of base class code
    # that require values not in the base class.
    solver = ThTthSolver(TH_TTH_Q_GEOMETRY)
    expected = "\n".join(
        [
            "========= ========= ======= =========== ========",
            "mode      pseudo(s) real(s) writable(s) extra(s)",
            "========= ========= ======= =========== ========",
            "bissector q         th, tth th, tth             ",
            "========= ========= ======= =========== ========",
        ]
    )
    summary = solver.summary
    assert isinstance(summary, pyRestTable.Table)
    assert str(summary).strip() == expected

    with pytest.raises(
        AttributeError,
        match=re.escape("Cannot change 'geometry' after solver is created."),
    ):
        solver.geometry = TH_TTH_Q_GEOMETRY


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reals={}),
            does_not_raise(),
            id="set_reals() on TrivialSolver is a no-op (empty dict)",
        ),
        pytest.param(
            dict(reals={"th": 10.0, "tth": 20.0}),
            does_not_raise(),
            id="set_reals() on TrivialSolver is a no-op (with values)",
        ),
    ],
)
def test_SolverBase_set_reals_noop(parms, context):
    """SolverBase.set_reals() is a no-op; any solver inheriting it must not raise."""
    with context:
        solver = TrivialSolver("test_geo")
        result = solver.set_reals(**parms)
        assert result is None


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reals={"th": 5.0, "tth": 10.0}),
            does_not_raise(),
            id="set_reals() on ThTthSolver (non-hkl_soleil) succeeds via inherited no-op",
        ),
    ],
)
def test_SolverBase_set_reals_inherited(parms, context):
    """Non-hkl_soleil solvers inheriting set_reals() must not raise AttributeError."""
    with context:
        solver = ThTthSolver(TH_TTH_Q_GEOMETRY)
        result = solver.set_reals(**parms)
        assert result is None


# A non-identity 3x3 matrix for testing U/UB round-trips.
_SAMPLE_MATRIX = [
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(attr="U", value=IDENTITY_MATRIX_3X3),
            does_not_raise(),
            id="U getter returns identity by default",
        ),
        pytest.param(
            dict(attr="UB", value=IDENTITY_MATRIX_3X3),
            does_not_raise(),
            id="UB getter returns identity by default",
        ),
        pytest.param(
            dict(attr="U", value=_SAMPLE_MATRIX),
            does_not_raise(),
            id="U setter stores value; getter returns it",
        ),
        pytest.param(
            dict(attr="UB", value=_SAMPLE_MATRIX),
            does_not_raise(),
            id="UB setter stores value; getter returns it",
        ),
    ],
)
def test_SolverBase_U_UB(parms, context):
    """SolverBase U and UB properties store and return orientation matrices."""
    with context:
        solver = TrivialSolver("test_geo")
        attr = parms["attr"]
        value = parms["value"]
        # Verify default is identity before any assignment.
        assert getattr(solver, attr) == IDENTITY_MATRIX_3X3
        # Set and round-trip.
        setattr(solver, attr, value)
        assert getattr(solver, attr) == value


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reals={"th": 5.0, "tth": 10.0}),
            does_not_raise(),
            id="ThTthSolver U/UB inherit base store-and-return behaviour",
        ),
    ],
)
def test_SolverBase_U_UB_inherited(parms, context):
    """Non-hkl_soleil solvers inherit U/UB store-and-return without AttributeError."""
    with context:
        solver = ThTthSolver(TH_TTH_Q_GEOMETRY)
        # Default is identity.
        assert solver.U == IDENTITY_MATRIX_3X3
        assert solver.UB == IDENTITY_MATRIX_3X3
        # Setter stores; getter returns — no AttributeError.
        solver.U = _SAMPLE_MATRIX
        assert solver.U == _SAMPLE_MATRIX
        solver.UB = _SAMPLE_MATRIX
        assert solver.UB == _SAMPLE_MATRIX


# ---------------------------------------------------------------------------
# Issue #372: Solver default geometry & default mode contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cls=ThTthSolver, expected=TH_TTH_Q_GEOMETRY),
            does_not_raise(),
            id="ThTthSolver: first registered geometry",
        ),
        pytest.param(
            dict(cls=NoOpSolver, expected=None),
            pytest.raises(
                SolverError,
                match=re.escape("NoOpSolver has no registered geometries"),
            ),
            id="NoOpSolver: no geometries -> SolverError",
        ),
    ],
)
def test_default_geometry(parms, context):
    """Solver.default_geometry() returns geometries()[0] or raises."""
    with context:
        result = parms["cls"].default_geometry()
        assert result == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cls=ThTthSolver, geometry=TH_TTH_Q_GEOMETRY, expected=BISECTOR_MODE),
            does_not_raise(),
            id="ThTthSolver: descriptor.default_mode wins",
        ),
        pytest.param(
            dict(cls=ThTthSolver, geometry="UNKNOWN", expected=None),
            pytest.raises(
                SolverError,
                match=re.escape("ThTthSolver has no registered geometry 'UNKNOWN'"),
            ),
            id="ThTthSolver: unregistered geometry -> SolverError",
        ),
        pytest.param(
            dict(cls=NoOpSolver, geometry="anything", expected=None),
            pytest.raises(
                SolverError,
                match=re.escape("NoOpSolver has no registered geometry 'anything'"),
            ),
            id="NoOpSolver: any geometry -> SolverError",
        ),
    ],
)
def test_default_mode(parms, context):
    """Solver.default_mode(geometry) honours descriptor.default_mode or raises."""
    with context:
        result = parms["cls"].default_mode(parms["geometry"])
        assert result == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(cls=ThTthSolver, geometry=TH_TTH_Q_GEOMETRY, expected=BISECTOR_MODE),
            does_not_raise(),
            id="ThTthSolver: __init__ applies default_mode",
        ),
        pytest.param(
            dict(cls=NoOpSolver, geometry="anything", expected=""),
            does_not_raise(),
            id="NoOpSolver: __init__ leaves mode blank when no default exists",
        ),
    ],
)
def test_init_applies_default_mode(parms, context):
    """Solver(geometry, mode='') resolves to default_mode when available."""
    with context:
        solver = parms["cls"](parms["geometry"])
        assert solver.mode == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(modes=["a", "b", "c"], default_mode="b", expected="b"),
            does_not_raise(),
            id="explicit default_mode wins over modes[0]",
        ),
        pytest.param(
            dict(modes=["a", "b", "c"], default_mode="", expected="a"),
            does_not_raise(),
            id="blank default_mode falls back to modes[0]",
        ),
    ],
)
def test_descriptor_default_mode_resolution(parms, context):
    """Verify descriptor's default_mode field takes precedence over modes[0]."""

    class _IsolatedSolver(ThTthSolver):
        _geometry_registry = {}

    with context:
        desc = GeometryDescriptor(
            name="ISOLATED",
            pseudo_axis_names=["q"],
            real_axis_names=["th", "tth"],
            modes=parms["modes"],
            default_mode=parms["default_mode"],
        )
        _IsolatedSolver.register_geometry(desc)
        assert _IsolatedSolver.default_mode("ISOLATED") == parms["expected"]
        # __init__ should also apply this default.
        solver = _IsolatedSolver("ISOLATED")
        assert solver.mode == parms["expected"]
