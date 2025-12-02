import pytest

from .. import SolverBase as AppBase
from ..backends.base import SolverBase as BackendBase
from ..backends.no_op import NoOpSolver as BackendSolver
from ..misc import get_solver
from . import NO_OP_SOLVER_TYPE_STR


@pytest.mark.parametrize(
    "solver_class, base_class, geometry",
    [
        [get_solver("no_op"), AppBase, "any"],  # as user
        [BackendSolver, BackendBase, "any"],  # as unit tester
    ],
)
def test_solvers(solver_class, base_class, geometry):
    """Test import of the Solver classes, as user and as unit tester."""
    # test the classes, themselves
    assert type(solver_class) is type(base_class)
    assert issubclass(solver_class, base_class)

    # confirm that users cannot use the base class directly
    with pytest.raises(TypeError) as reason:
        solver = base_class()
    assert "Can't instantiate abstract class" in str(reason)

    # test an object created from the Solver class
    solver = solver_class(geometry)
    assert solver is not None
    assert isinstance(solver, solver_class)
    assert isinstance(solver, base_class)

    # confirm that "arg 1 must be a class"
    for klass in (base_class, solver_class):
        with pytest.raises(TypeError) as reason:
            assert not issubclass(solver, klass)
        assert "issubclass() arg 1 must be a class" in str(reason)
    # Here's the __right__ way to check an object with issubclass
    assert issubclass(type(solver), klass)
    assert str(type(solver)) == NO_OP_SOLVER_TYPE_STR
