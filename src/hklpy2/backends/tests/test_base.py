"""Test the solver base class."""

# Many features are tested, albeit indrectly, in specific solvers.

import re
from contextlib import nullcontext as does_not_raise

import pyRestTable
import pytest

from ...blocks.lattice import Lattice
from ...blocks.reflection import Reflection
from ...misc import IDENTITY_MATRIX_3X3
from ..base import SolverBase
from ..th_tth_q import TH_TTH_Q_GEOMETRY
from ..th_tth_q import ThTthSolver


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
