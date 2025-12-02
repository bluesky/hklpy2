"""Test the solver base class."""

# Many features are tested, albeit indrectly, in specific solvers.

import pyRestTable
import pytest

from ...blocks.lattice import Lattice
from ...blocks.reflection import Reflection
from ...misc import IDENTITY_MATRIX_3X3
from ...tests.common import assert_context_result
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

    with pytest.raises(TypeError) as reason:
        solver.lattice = 1.0
    assert_context_result("Must supply", reason)
    solver.lattice = dict(a=1, b=2, c=3, alpha=90, beta=90, gamma=90)
    assert solver.lattice == dict(a=1, b=2, c=3, alpha=90, beta=90, gamma=90)

    with pytest.raises(TypeError) as reason:
        solver.sample = 1.0
    assert_context_result("Must supply", reason)


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

    with pytest.raises(AttributeError) as reason:
        solver.geometry = TH_TTH_Q_GEOMETRY
    assert_context_result("geometry", reason)
