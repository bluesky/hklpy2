"""
Regression test for issue #207.

BUG: calc_UB raises a confusing "U matrix rows must be normalized" error
when a misconfigured axes_xref causes the solver to receive degenerate
reflection geometry (detector at the direct-beam position, Q=0).

The fix raises a clear ValueError from HklSolver.calculate_UB() when
libhkl silently returns an all-zero U matrix.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import creator


@pytest.fixture()
def polar_swapped():
    """
    APS POLAR diffractometer with the misconfigured axes_xref from issue #207.

    The user's axes_xref swaps detector arm angles (solver: gamma, delta)
    with sample rotation angles (solver: chi, phi):
        user gamma -> solver chi   (detector angle sent as sample rotation)
        user chi   -> solver gamma (sample rotation sent as detector angle)
        user delta -> solver phi
        user phi   -> solver delta

    With this mapping, reflections where the user's gamma=40 (detector arm)
    are passed to the solver as chi=40, while solver.gamma receives the
    user's chi value (0 or -90).  Since solver.gamma=0 and solver.delta=0
    for both reflections, the detector sits at the direct-beam position
    (Q=0), causing libhkl to return an all-zero U matrix.
    """
    sim = creator(
        name="polar",
        solver="hkl_soleil",
        geometry="APS POLAR",
        reals=dict(tau=None, mu=None, gamma=None, delta=None, chi=None, phi=None),
    )
    # solver real order: tau, mu, chi, phi, gamma, delta
    # user  real order:  tau, mu, gamma, delta, chi, phi  (as in the issue)
    sim.core.assign_axes(
        "h k l".split(),
        "tau mu gamma delta chi phi".split(),
    )
    sim.add_sample("test", 5.0)  # cubic, a=5 Å
    sim.beam.wavelength.put(1.7225)
    return sim


@pytest.fixture()
def polar_correct():
    """
    APS POLAR diffractometer with correct identity axes_xref.

    Solver real order: tau, mu, chi, phi, gamma, delta.
    User axis names match solver names exactly.
    """
    sim = creator(
        name="polar",
        solver="hkl_soleil",
        geometry="APS POLAR",
        reals=dict(tau=None, mu=None, chi=None, phi=None, gamma=None, delta=None),
    )
    sim.add_sample("test", 5.0)  # cubic, a=5 Å
    sim.beam.wavelength.put(1.7225)
    return sim


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                # The exact reflections from the user's issue #207 report.
                # With the swapped axes_xref, solver receives gamma=0,delta=0
                # for both reflections → Q=0 → all-zero U matrix.
                r1_hkl=(1, 0, 0),
                r1_reals=dict(tau=0, mu=20, gamma=40, delta=0, chi=0, phi=0),
                r2_hkl=(0, 0, 3),
                r2_reals=dict(tau=0, mu=20, gamma=40, delta=0, chi=-90, phi=0),
            ),
            pytest.raises(
                ValueError,
                match=re.escape("UB calculation produced a degenerate U matrix"),
            ),
            id="issue-207 swapped axes_xref yields degenerate UB",
        ),
    ],
)
def test_calc_UB_swapped_axes_xref(polar_swapped, parms, context):
    """
    Reproduces issue #207: calc_UB raises a clear ValueError (not a cryptic
    'rows must be normalized' error) when a misconfigured axes_xref causes
    the solver to receive degenerate reflection geometry (Q=0).
    """
    r1 = polar_swapped.add_reflection(
        parms["r1_hkl"], parms["r1_reals"], wavelength=1.7225, name="r1"
    )
    r2 = polar_swapped.add_reflection(
        parms["r2_hkl"], parms["r2_reals"], wavelength=1.7225, name="r2"
    )
    with context:
        polar_swapped.core.calc_UB(r1, r2)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                # Valid reflections with correct axes_xref — solver receives
                # non-zero gamma/delta, so Q != 0 and UB succeeds.
                r1_hkl=(1, 0, 0),
                r1_reals=dict(tau=0, mu=20, chi=0, phi=0, gamma=40, delta=0),
                r2_hkl=(0, 0, 3),
                r2_reals=dict(tau=0, mu=20, chi=-90, phi=0, gamma=40, delta=0),
            ),
            does_not_raise(),
            id="correct axes_xref with non-degenerate reflections succeeds",
        ),
    ],
)
def test_calc_UB_correct_axes_xref(polar_correct, parms, context):
    """
    With a correct axes_xref, calc_UB succeeds for valid reflections.
    """
    r1 = polar_correct.add_reflection(
        parms["r1_hkl"], parms["r1_reals"], wavelength=1.7225, name="r1"
    )
    r2 = polar_correct.add_reflection(
        parms["r2_hkl"], parms["r2_reals"], wavelength=1.7225, name="r2"
    )
    with context:
        ub = polar_correct.core.calc_UB(r1, r2)
        assert ub is not None
