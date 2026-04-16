"""
Regression test for issue #193.

cahkl shows no solution at specific motor positions in constant_phi mode.
The root cause is the user's narrow constraint limits silently discarding
all solver solutions at certain phi values where the computed axes cross
constraint boundaries.

A phi sweep with wide constraints (-180, 180) confirms that libhkl's solver
finds solutions at ALL phi positions.  The breakpoints are entirely due to
the user's narrow constraints (omega: [-1, 180], chi: [-1, 110],
tth: [-1, 180]).

This test verifies that with wide constraints, forward() returns solutions
at every phi position, including those that fail with narrow constraints.
It also verifies that narrow constraints correctly exclude solutions where
a computed axis falls outside the allowed range.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..diffract import creator
from ..exceptions import NoForwardSolutions


class _allows_no_solutions:
    """Context manager that accepts either no exception or NoForwardSolutions.

    Used for test cases where cut-point wrapping may reveal or hide a
    solution depending on solver floating-point behavior (Python-version
    specific).
    """

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is NoForwardSolutions:
            return True  # suppress the exception
        return False  # re-raise anything else


def _make_e4cv(constraints=None):
    """Reproduce the user's exact E4CV setup from issue #193."""
    e4cv = creator(geometry="E4CV", solver="hkl_soleil")
    e4cv.add_sample(
        "s2",
        a=3.8995,
        b=3.8995,
        c=3.8995,
        alpha=90,
        beta=90,
        gamma=90,
    )
    e4cv.beam.energy.put(2.5)  # keV, wavelength ~4.96 angstrom

    or1 = e4cv.add_reflection(
        (0, 0, 1),
        dict(omega=39.177, chi=88.826, phi=-2.465, tth=78.971),
        name="r_43bd",
    )
    or2 = e4cv.add_reflection(
        (1, 0, 1),
        dict(omega=108.15, chi=88.734, phi=-2.465, tth=128.05),
        name="r_d2c1",
    )
    e4cv.core.calc_UB(or1, or2)
    e4cv.core.mode = "constant_phi"

    if constraints is not None:
        for axis, limits in constraints.items():
            e4cv.core.constraints[axis].limits = limits

    return e4cv


WIDE = dict(
    omega=(-180, 180),
    chi=(-180, 180),
    phi=(-180, 180),
    tth=(-180, 180),
)

USER = dict(
    omega=(-1.0, 180.0),
    chi=(-1.0, 110.0),
    phi=(-120.0, 120.0),
    tth=(-1.0, 180.0),
)

_NO_SOLUTIONS = re.escape("No solutions.")


@pytest.mark.parametrize(
    "parms, context",
    [
        # --- Wide constraints: solutions at EVERY phi ---
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-2.465, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=-2.465 (user orientation)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=0, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=0",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=30, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=30",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-120, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=-120 (user breakpoint)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-98, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=-98 (user breakpoint)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-62, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=-62 (user breakpoint)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=12, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=12 (user breakpoint)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=96, constraints=WIDE),
            does_not_raise(),
            id="wide: (001) phi=96 (user breakpoint)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-2.465, constraints=WIDE),
            does_not_raise(),
            id="wide: (101) phi=-2.465 (user orientation)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=0, constraints=WIDE),
            does_not_raise(),
            id="wide: (101) phi=0",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=45, constraints=WIDE),
            does_not_raise(),
            id="wide: (101) phi=45",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-100, constraints=WIDE),
            does_not_raise(),
            id="wide: (101) phi=-100 (user has no solution)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=80, constraints=WIDE),
            does_not_raise(),
            id="wide: (101) phi=80 (user has no solution)",
        ),
        # --- User constraints: solutions exist at some phi values ---
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-2.465, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-2.465 (has solution)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-50, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-50 (has solution)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=80, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=80 (has solution)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-25, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=-25 (has solution)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-15, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=-15 (has solution)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-98, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-98 (has solution after #240 fix)",
        ),
        # --- User constraints: narrow limits reject all solutions ---
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-120, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (001) phi=-120 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=12, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (001) phi=12 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=96, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (001) phi=96 (no solution, constraints)",
        ),
        pytest.param(
            # After #296, cut-point wrapping may map omega ~=-180 to ~=+180,
            # creating a solution that passes the (-1, 180) omega constraint.
            # Whether this occurs depends on solver floating-point (Python-
            # version specific); accept either outcome (0 or 1 solutions).
            dict(hkl=(1, 0, 1), phi=-6, constraints=USER),
            _allows_no_solutions(),
            id="user: (101) phi=-6 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=0, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (101) phi=0 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-100, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (101) phi=-100 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=80, constraints=USER),
            pytest.raises(NoForwardSolutions, match=_NO_SOLUTIONS),
            id="user: (101) phi=80 (no solution, constraints)",
        ),
    ],
)
def test_issue_193(parms, context):
    """
    forward() in constant_phi mode: with wide constraints, solutions exist
    at every phi.  With narrow user constraints, some phi values have no
    solutions because the computed axes fall outside the allowed range;
    NoForwardSolutions is raised to signal this to the caller.
    """
    with context:
        e4cv = _make_e4cv(constraints=parms["constraints"])
        e4cv.phi.move(parms["phi"])

        solutions = e4cv.core.forward(
            dict(h=parms["hkl"][0], k=parms["hkl"][1], l=parms["hkl"][2])
        )

        if not solutions:
            raise NoForwardSolutions("No solutions.")

        # Every solution must hold phi constant at the requested value.
        for sol in solutions:
            assert_almost_equal(sol.phi, parms["phi"], decimal=4)
