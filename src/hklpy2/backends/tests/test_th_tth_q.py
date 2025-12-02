from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ...misc import IDENTITY_MATRIX_3X3
from ...misc import SolverError
from ...misc import get_solver
from ...misc import solver_factory
from ...tests.common import assert_context_result
from ..base import SolverBase
from ..th_tth_q import BISECTOR_MODE
from ..th_tth_q import TH_TTH_Q_GEOMETRY
from ..th_tth_q import ThTthSolver


def test_solver():
    assert issubclass(ThTthSolver, SolverBase)
    assert get_solver("th_tth") == ThTthSolver

    gname = "TH TTH Q"
    solver = ThTthSolver(gname)
    assert isinstance(solver, SolverBase)
    assert solver.name == "th_tth"
    assert solver.geometries() == [gname]

    assert solver.geometry == gname
    assert solver.pseudo_axis_names == ["q"]
    assert solver.real_axis_names == "th tth".split()
    assert solver.extra_axis_names == []
    assert solver.refineLattice([]) is None
    assert solver.calculate_UB(None, None) == IDENTITY_MATRIX_3X3

    assert solver.mode == "", f"{solver.mode=!r}"
    solver.mode = BISECTOR_MODE
    assert solver.mode == BISECTOR_MODE

    with pytest.raises(TypeError) as reason:
        solver.forward([0])
    assert_context_result("Must supply dict", reason)

    with pytest.raises(SolverError) as reason:
        solver.forward({})
    assert_context_result("'q' not defined.", reason)

    with pytest.raises(SolverError) as reason:
        solver.forward(dict(q=0.1))
    assert_context_result("Wavelength is not set.", reason)

    with pytest.raises(TypeError) as reason:
        solver.inverse([0, 20])
    assert_context_result("Must supply dict", reason)

    with pytest.raises(SolverError) as reason:
        solver.inverse(dict(th=0))
    assert_context_result("'tth' not defined.", reason)

    with pytest.raises(SolverError) as reason:
        solver.inverse(dict(th=0, tth=20))
    assert_context_result("Wavelength is not set. Add a reflection", reason)

    with pytest.raises(TypeError) as reason:
        solver.wavelength = "-1"
    assert_context_result("Must supply number", reason)

    with pytest.raises(ValueError) as reason:
        solver.wavelength = -1
    assert_context_result("Must supply positive number", reason)

    with pytest.raises(NotImplementedError) as reason:
        solver.removeAllReflections()


@pytest.mark.parametrize(
    "transform, wavelength, inputs, outputs, tol",
    [
        ["inverse", 0.5, {"th": 60, "tth": 120}, {"q": 21.7655924}, 0.0001],
        ["forward", 0.5, {"q": 21.7655924}, {"th": 60, "tth": 120}, 0.0001],
        ["inverse", 1.0, {"th": 5, "tth": 10}, {"q": 1.095231}, 0.0001],
        ["forward", 1.0, {"q": 1.095231}, {"th": 5, "tth": 10}, 0.0001],
        ["inverse", 1.54, {"th": 14.9131, "tth": 29.8262}, {"q": 2.1}, 0.0001],
        ["forward", 1.54, {"q": 2.1}, {"th": 14.9131, "tth": 29.8262}, 0.0001],
        ["inverse", 2.1, {"th": 1, "tth": 2}, {"q": 0.104435}, 0.0001],
        ["forward", 2.1, {"q": 0.104435}, {"th": 1, "tth": 2}, 0.0001],
        ["inverse", 2.1, {"th": 0.1, "tth": 0.2}, {"q": 0.0104440}, 0.0001],
        ["forward", 2.1, {"q": 0.0104440}, {"th": 0.1, "tth": 0.2}, 0.0001],
        ["inverse", 2.1, {"th": 0.01, "tth": 0.02}, {"q": 0.00104440}, 0.0001],
        ["forward", 2.1, {"q": 0.00104440}, {"th": 0.01, "tth": 0.02}, 0.0001],
    ],
)
def test_transforms(transform, wavelength, inputs, outputs, tol):
    solver = solver_factory("th_tth", "TH TTH Q")
    solver.mode = BISECTOR_MODE
    solver.wavelength = wavelength
    if transform == "forward":
        result = solver.forward(inputs)
        assert isinstance(result, list)
        assert len(result) == 1
        result = result[0]
        assert "th" in result
        assert "tth" in result
    elif transform == "inverse":
        result = solver.inverse(inputs)
        assert "q" in result
    assert isinstance(result, dict)
    assert list(result.keys()) == list(outputs.keys())
    arr_r = np.array(list(result.values()))
    arr_o = np.array(list(outputs.values()))
    assert np.allclose(arr_r, arr_o, atol=tol), f"{result=}  {outputs=}"


@pytest.mark.parametrize(
    "value, context, expected",
    [
        pytest.param(
            dict(
                name="r1",
                pseudos=dict(q=0),
                reals=dict(th=0, tth=0),
                wavelength=1.0,
                geometry=TH_TTH_Q_GEOMETRY,
                pseudo_axis_names=["q"],
                real_axis_names=["th", "tth"],
            ),
            does_not_raise(),
            None,
            id="standard case",
        ),
        pytest.param(
            "wrong object",
            pytest.raises(TypeError),
            "Must supply SolverReflection (dict)",
            id="Wrong object type",
        ),
        pytest.param(
            dict(
                name="r1",
                pseudos=dict(q=0),
                reals=dict(th=0, tth=0),
                wavelength=10.0,
                geometry=TH_TTH_Q_GEOMETRY,
                pseudo_axis_names=["q"],
                real_axis_names=["th", "tth"],
            ),
            pytest.raises(SolverError),
            "All reflections must have same wavelength",
            id="Different reflection wavelengths",
        ),
    ],
)
def test_reflections(value, context, expected):
    with context as reason:
        solver = solver_factory("th_tth", "TH TTH Q")
        r0 = dict(
            name="r0",
            pseudos=dict(q=0.1),
            reals=dict(th=0, tth=1),
            wavelength=1.0,
            geometry=TH_TTH_Q_GEOMETRY,
            pseudo_axis_names=["q"],
            real_axis_names=["th", "tth"],
        )
        solver.addReflection(r0)  # pre-existing
        solver.addReflection(value)
    assert_context_result(expected, reason)
