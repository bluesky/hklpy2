import inspect

import pytest

from ...misc import get_solver
from ...tests.common import assert_context_result

'''  FIXME: unit test failure
________________________________ test_HklSolver ________________________________
    def test_HklSolver():
        Solver = get_solver("hkl_soleil")
        assert Solver is not None

        solver = Solver("E4CV")
        assert solver is not None
        assert isinstance(solver.version, str)

        gname = "ESRF ID01 PSIC"
        assert solver.geometry != gname

        with pytest.raises(AttributeError) as reason:
            solver.geometry = "E4CV"
>       assert_context_result("has no setter", reason)
hklpy2/blocks/tests/test_solver.py:43:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
expected = 'has no setter'
reason = <ExceptionInfo AttributeError("can't set attribute 'geometry'") tblen=1>
    def assert_context_result(expected, reason):
        """Common handling for tests below."""
        if expected is None:
            assert reason is None
        else:
>           assert expected in str(reason), f"{expected=!r} {reason=}"
E           AssertionError: expected='has no setter' reason=<ExceptionInfo AttributeError("can't set attribute 'geometry'") tblen=1>
hklpy2/tests/common.py:16: AssertionError
'''


@pytest.mark.parametrize(
    "solver_name, geometry", [["hkl_soleil", "E4CV"], ["no_op", "anything"]]
)
def test_solvers(solver_name, geometry):
    from importlib.metadata import entry_points

    solvers = entry_points(group="hklpy2.solver")
    assert len(solvers) > 0
    assert solver_name in solvers.names, f"{solver_name=}"

    entrypoint = solvers[solver_name]
    assert entrypoint is not None

    solver_class = entrypoint.load()
    assert inspect.isclass(solver_class)

    solver = solver_class(geometry)
    assert isinstance(solver.version, str)
    assert len(solver.version) > 0, f"{solver.version=}"


def test_HklSolver():
    Solver = get_solver("hkl_soleil")
    assert Solver is not None

    solver = Solver("E4CV")
    assert solver is not None
    assert isinstance(solver.version, str)

    gname = "ESRF ID01 PSIC"
    assert solver.geometry != gname

    with pytest.raises(AttributeError) as reason:
        solver.geometry = "E4CV"
    assert_context_result("has no setter", reason)

    assert solver.engine_name == "hkl"

    reals = solver.real_axis_names
    assert reals == "omega chi phi tth".split()

    pseudos = solver.pseudo_axis_names
    assert pseudos == "h k l".split()

    solver = Solver(gname)  # new geometry
    assert solver.geometry == gname  # did not change
    assert solver.engine_name == "hkl"

    reals = solver.real_axis_names
    assert reals == "mu eta phi nu delta".split()

    pseudos = solver.pseudo_axis_names
    assert pseudos == "h k l".split()
