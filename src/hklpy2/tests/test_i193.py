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

from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..diffract import creator


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


@pytest.mark.parametrize(
    "parms, context",
    [
        # Wide constraints: solutions at EVERY phi, including the user's
        # breakpoint phi values.  These confirm libhkl solver succeeds.
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
        # User constraints: solutions exist at some phi values.
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-2.465, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-2.465 (orientation, has solution)",
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
            dict(hkl=(1, 0, 1), phi=-6, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=-6 (has solution)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-19, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=-19 (has solution)",
        ),
    ],
)
def test_issue_193(parms, context):
    """
    forward() must return solutions in constant_phi mode across a range of
    phi positions.  With wide constraints, solutions exist at every phi.
    """
    with context:
        e4cv = _make_e4cv(constraints=parms["constraints"])
        e4cv.phi.move(parms["phi"])

        solutions = e4cv.core.forward(
            dict(h=parms["hkl"][0], k=parms["hkl"][1], l=parms["hkl"][2])
        )

        assert len(solutions) > 0, (
            f"No solutions for hkl={parms['hkl']}"
            f" at phi={parms['phi']}"
            f" with constraints={parms['constraints']}"
        )

        # Every solution must hold phi constant at the requested value.
        for sol in solutions:
            assert_almost_equal(sol.phi, parms["phi"], decimal=4)


@pytest.mark.parametrize(
    "parms, context",
    [
        # User constraints: phi values where solutions SHOULD be rejected.
        # Verified by sweep: solver returns solutions but constraints
        # exclude them (omega or tth goes negative).
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-120, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-120 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=-98, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=-98 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=12, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=12 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(0, 0, 1), phi=96, constraints=USER),
            does_not_raise(),
            id="user: (001) phi=96 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=0, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=0 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=-100, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=-100 (no solution, constraints)",
        ),
        pytest.param(
            dict(hkl=(1, 0, 1), phi=80, constraints=USER),
            does_not_raise(),
            id="user: (101) phi=80 (no solution, constraints)",
        ),
    ],
)
def test_issue_193_no_solution_with_narrow_constraints(parms, context):
    """
    forward() returns an empty list when user's narrow constraints reject
    all solver solutions.  With wide constraints, these same phi values
    DO have solutions (tested above), confirming the issue is constraint
    filtering, not solver failure.
    """
    with context:
        e4cv = _make_e4cv(constraints=parms["constraints"])
        e4cv.phi.move(parms["phi"])

        solutions = e4cv.core.forward(
            dict(h=parms["hkl"][0], k=parms["hkl"][1], l=parms["hkl"][2])
        )

        assert len(solutions) == 0, (
            f"Expected no solutions for hkl={parms['hkl']}"
            f" at phi={parms['phi']} with narrow constraints,"
            f" but got {len(solutions)}"
        )
