"""
Solver discovery and instantiation for |hklpy2|.

These utilities locate |solver| backend classes via Python entry points and
create solver instances for use by :class:`~hklpy2.ops.Core`.

.. autosummary::

    ~get_solver
    ~solver_factory
    ~solvers
    ~SOLVER_ENTRYPOINT_GROUP
"""

import logging
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Mapping

from .exceptions import SolverError

if TYPE_CHECKING:
    from .backends.base import SolverBase  # noqa: F401

logger = logging.getLogger(__name__)

__all__ = [
    "SOLVER_ENTRYPOINT_GROUP",
    "get_solver",
    "solver_factory",
    "solvers",
]

SOLVER_ENTRYPOINT_GROUP: str = "hklpy2.solver"
"""Name by which |hklpy2| |solver| classes are grouped."""


def get_solver(solver_name: str) -> "SolverBase":
    """
    Load a Solver class from a named entry point.

    ::

        import hklpy2
        SolverClass = hklpy2.get_solver("hkl_soleil")
        libhkl_solver = SolverClass()
    """
    if solver_name not in solvers():
        raise SolverError(f"{solver_name=!r} unknown.  Pick one of: {solvers()!r}")
    logger.debug("Loading solver %r from entry points", solver_name)
    entries = entry_points(group=SOLVER_ENTRYPOINT_GROUP)
    return entries[solver_name].load()


def solver_factory(
    solver_name: str,
    geometry: str,
    **kwargs: Mapping,
) -> "SolverBase":
    """
    Create a |solver| object with geometry and axes.
    """
    logger.debug(
        "Creating solver %r geometry=%r kwargs=%r", solver_name, geometry, kwargs
    )
    solver_class = get_solver(solver_name)
    return solver_class(geometry, **kwargs)


def solvers() -> Mapping[str, "SolverBase"]:
    """
    Dictionary of available Solver classes, mapped by entry point name.

    ::

        import hklpy2
        print(hklpy2.solvers())
    """
    return {ep.name: ep.value for ep in entry_points(group=SOLVER_ENTRYPOINT_GROUP)}
