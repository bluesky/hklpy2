"""Test the no_op solver class."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ...misc import IDENTITY_MATRIX_3X3
from ..no_op import NoOpSolver

SAMPLE_REFLECTION = dict(
    name="r1",
    pseudos={"h": 1.0, "k": 0.0, "l": 0.0},
    reals={"omega": 10.0, "chi": 0.0, "phi": 0.0, "tth": 20.0},
    wavelength=1.54,
)


def test_NoOpSolver():
    assert NoOpSolver.geometries() == []

    solver = NoOpSolver("no_geometry")
    assert solver.removeAllReflections() is None
    assert solver.refineLattice([]) is None
    assert solver.extra_axis_names == []
    assert solver.forward({}) == [{}]
    assert solver.inverse({}) == {}
    assert solver.calculate_UB(None, None) == IDENTITY_MATRIX_3X3
    assert solver.reflections == []


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reflection=SAMPLE_REFLECTION),
            does_not_raise(),
            id="valid reflection dict",
        ),
        pytest.param(
            dict(reflection=None),
            pytest.raises(TypeError, match=re.escape("Must supply ReflectionDict")),
            id="None raises TypeError",
        ),
    ],
)
def test_NoOpSolver_addReflection(parms, context):
    solver = NoOpSolver("no_geometry")
    with context:
        solver.addReflection(**parms)
        assert len(solver.reflections) == 1
