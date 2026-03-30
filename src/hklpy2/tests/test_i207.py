"""
Regression test for issue #207.

BUG: calc_UB raises a confusing "U matrix rows must be normalized" error
when the ``reals`` dict is passed to ``creator()`` in hardware naming order
that differs from the solver's expected axis order, without using ``_real``
to declare the correct mapping.

Root cause: ``creator()`` maps ``reals`` keys to solver axes by position.
If the user's hardware axes are named ``tau, mu, gamma, delta, chi, phi``
(in that order) but the solver expects ``tau, mu, chi, phi, gamma, delta``,
the positional zip produces a swapped ``axes_xref``:
    user gamma -> solver chi
    user chi   -> solver gamma
    user delta -> solver phi
    user phi   -> solver delta

Fix: pass ``_real="tau mu chi phi gamma delta".split()`` to ``creator()``
to declare which local axis name corresponds to each solver axis slot.

The fix in hklpy2 raises a clear ValueError from HklSolver.calculate_UB()
when libhkl returns a degenerate U matrix, instead of the cryptic downstream
"U matrix rows must be normalized" error.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import creator


@pytest.fixture()
def polar_swapped():
    """
    APS POLAR diffractometer reproducing the user's misconfiguration.

    The user's hardware axes are named tau, mu, gamma, delta, chi, phi
    (in their motor order), but the APS POLAR solver expects them in the
    order tau, mu, chi, phi, gamma, delta.

    Without ``_real``, ``creator()`` zips the reals dict keys against the
    solver's axis order by position, producing the swapped axes_xref:
        user gamma -> solver chi   (detector angle sent as sample rotation)
        user chi   -> solver gamma (sample rotation sent as detector angle)
        user delta -> solver phi
        user phi   -> solver delta
    """
    sim = creator(
        name="polar",
        solver="hkl_soleil",
        geometry="APS POLAR",
        # Hardware axis names in user's order — NOT the solver's order.
        # This is the exact pattern from the issue report.
        reals=dict(tau=None, mu=None, gamma=None, delta=None, chi=None, phi=None),
        # _real is NOT provided — this is the bug.
    )
    sim.add_sample("test", 5.0)  # cubic, a=5 Å
    sim.beam.wavelength.put(1.7225)
    return sim


@pytest.fixture()
def polar_fixed():
    """
    APS POLAR diffractometer with the corrected configuration.

    Same hardware axis names as polar_swapped, but ``_real`` declares
    the solver's expected axis order so axes_xref is built correctly.
    """
    sim = creator(
        name="polar",
        solver="hkl_soleil",
        geometry="APS POLAR",
        reals=dict(tau=None, mu=None, gamma=None, delta=None, chi=None, phi=None),
        # _real declares which local name maps to each solver axis slot:
        # solver order: tau, mu, chi, phi, gamma, delta
        _real="tau mu chi phi gamma delta".split(),
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
            id="issue-207 missing _real yields degenerate UB",
        ),
    ],
)
def test_calc_UB_swapped_axes_xref(polar_swapped, parms, context):
    """
    Reproduces issue #207: calc_UB raises a clear ValueError (not a cryptic
    'rows must be normalized' error) when reals dict order does not match
    solver axis order and _real is not supplied to creator().
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
                # Same hardware axis names and reflections as the swapped case,
                # but with _real supplied to creator() — UB succeeds.
                r1_hkl=(1, 0, 0),
                r1_reals=dict(tau=0, mu=20, gamma=40, delta=0, chi=0, phi=0),
                r2_hkl=(0, 0, 3),
                r2_reals=dict(tau=0, mu=20, gamma=40, delta=0, chi=-90, phi=0),
            ),
            does_not_raise(),
            id="issue-207 with _real supplied succeeds",
        ),
    ],
)
def test_calc_UB_fixed_real_order(polar_fixed, parms, context):
    """
    With _real supplied to creator(), the same hardware axes and reflections
    produce a valid UB matrix.
    """
    r1 = polar_fixed.add_reflection(
        parms["r1_hkl"], parms["r1_reals"], wavelength=1.7225, name="r1"
    )
    r2 = polar_fixed.add_reflection(
        parms["r2_hkl"], parms["r2_reals"], wavelength=1.7225, name="r2"
    )
    with context:
        ub = polar_fixed.core.calc_UB(r1, r2)
        assert ub is not None
