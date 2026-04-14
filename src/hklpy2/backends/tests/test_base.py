"""Test the solver base class."""

# Many features are tested, albeit indrectly, in specific solvers.

import re
from contextlib import nullcontext as does_not_raise

import pyRestTable
import pytest

from ...blocks.reflection import Reflection
from ...misc import IDENTITY_MATRIX_3X3
from ..base import SolverBase
from ..th_tth_q import TH_TTH_Q_GEOMETRY
from ..th_tth_q import ThTthSolver


class TrivialSolver(SolverBase):
    """Trivial implementation for testing.

    Uses inherited default reflection management from SolverBase.
    """

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
    assert solver.refineLattice([]) is None

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


SAMPLE_REFLECTION = dict(
    name="r1",
    pseudos={"h": 1.0, "k": 0.0, "l": 0.0},
    reals={"omega": 10.0, "chi": 0.0, "phi": 0.0, "tth": 20.0},
    wavelength=1.54,
)

SAMPLE_REFLECTION_2 = dict(
    name="r2",
    pseudos={"h": 0.0, "k": 1.0, "l": 0.0},
    reals={"omega": 15.0, "chi": 0.0, "phi": 90.0, "tth": 30.0},
    wavelength=1.54,
)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reflection=SAMPLE_REFLECTION),
            does_not_raise(),
            id="valid reflection dict",
        ),
        pytest.param(
            dict(reflection="not a dict"),
            pytest.raises(TypeError, match=re.escape("Must supply ReflectionDict")),
            id="string raises TypeError",
        ),
        pytest.param(
            dict(reflection=42),
            pytest.raises(TypeError, match=re.escape("Must supply ReflectionDict")),
            id="int raises TypeError",
        ),
    ],
)
def test_addReflection_default(parms, context):
    solver = TrivialSolver("test_geo")
    with context:
        solver.addReflection(**parms)
        assert len(solver.reflections) == 1
        assert solver.reflections[0] == parms["reflection"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reflections=[SAMPLE_REFLECTION, SAMPLE_REFLECTION_2]),
            does_not_raise(),
            id="add two then remove all",
        ),
        pytest.param(
            dict(reflections=[]),
            does_not_raise(),
            id="remove from empty list",
        ),
    ],
)
def test_removeAllReflections_default(parms, context):
    solver = TrivialSolver("test_geo")
    with context:
        for r in parms["reflections"]:
            solver.addReflection(r)
        assert len(solver.reflections) == len(parms["reflections"])
        solver.removeAllReflections()
        assert len(solver.reflections) == 0


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reflections=[]),
            does_not_raise(),
            id="empty list returns None",
        ),
        pytest.param(
            dict(reflections=[SAMPLE_REFLECTION]),
            does_not_raise(),
            id="one reflection returns None",
        ),
    ],
)
def test_refineLattice_default(parms, context):
    solver = TrivialSolver("test_geo")
    with context:
        result = solver.refineLattice(**parms)
        assert result is None


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                reflections=[SAMPLE_REFLECTION, SAMPLE_REFLECTION_2],
                expected_count=2,
            ),
            does_not_raise(),
            id="two reflections stored",
        ),
        pytest.param(
            dict(
                reflections=[],
                expected_count=0,
            ),
            does_not_raise(),
            id="empty reflections list",
        ),
    ],
)
def test_reflections_property(parms, context):
    solver = TrivialSolver("test_geo")
    with context:
        for r in parms["reflections"]:
            solver.addReflection(r)
        result = solver.reflections
        assert len(result) == parms["expected_count"]
        # Verify it returns a copy, not the internal list.
        result.append({"dummy": True})
        assert len(solver.reflections) == parms["expected_count"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(ub=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            does_not_raise(),
            id="set identity UB",
        ),
        pytest.param(
            dict(ub=[[2, 0, 0], [0, 2, 0], [0, 0, 2]]),
            does_not_raise(),
            id="set non-identity UB",
        ),
    ],
)
def test_UB_setter(parms, context):
    solver = TrivialSolver("test_geo")
    assert solver.UB == IDENTITY_MATRIX_3X3
    with context:
        solver.UB = parms["ub"]
        assert solver.UB == parms["ub"]
