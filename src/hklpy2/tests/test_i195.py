"""
Regression test for issue #195.

In v0.1.5, forward() used real-axis values from the last-added orientation
reflection (via libhkl's internal geometry state) instead of the current
diffractometer position when computing solutions for constant-axis modes.

Root cause: ``update_solver()`` called ``solver.sample = ...``, which called
``addReflection()`` for each orientation reflection.  Each ``addReflection``
call modified the libhkl geometry via ``set_reals(reflection["reals"])``,
leaving the geometry with the last reflection's angles.  The subsequent
``solver.forward()`` then used those stale geometry values as the fixed-axis
values for the current mode.

Fix: before calling ``solver.forward()``, ``ops.Core.forward()`` now
explicitly calls ``solver.set_reals(presets)`` where ``presets`` are derived
from the current diffractometer real-axis positions.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..blocks.lattice import SI_LATTICE_PARAMETER
from ..diffract import creator
from ..exceptions import NoForwardSolutions


@pytest.mark.parametrize(
    "phi_current, phi_limits, context",
    [
        pytest.param(
            0.0,
            (-180, 180),
            does_not_raise(),
            id="phi=0 (matches or1)",
        ),
        pytest.param(
            45.0,
            (-180, 180),
            does_not_raise(),
            id="phi=45 (between or1 and or2 — primary bug case)",
        ),
        pytest.param(
            90.0,
            (-180, 180),
            does_not_raise(),
            id="phi=90 (matches or2)",
        ),
        pytest.param(
            -30.0,
            (-180, 180),
            does_not_raise(),
            id="phi=-30 (outside or1/or2 range)",
        ),
        pytest.param(
            45.0,
            (80, 100),  # phi=45 is excluded by this constraint
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="NoForwardSolutions: phi excluded by constraints",
        ),
    ],
)
def test_issue_195(phi_current, phi_limits, context):
    """
    forward() must use the current diffractometer phi in constant_phi mode,
    not the phi value of either orientation reflection (issue #195).

    Two orientation reflections are set with phi=0 (or1) and phi=90 (or2).
    After computing the UB matrix, libhkl's internal geometry retains phi=90
    (from or2, the last reflection processed).  Without the fix in
    ``ops.Core.forward()``, this stale phi=90 is used as the fixed-axis
    value instead of the current diffractometer position.

    Covers:
    - phi equal to or1's value, or2's value, in-between, and outside range.
    - ``core.forward()`` (all constraint-filtered solutions).
    - ``diffractometer.forward()`` (single picked solution).
    - ``NoForwardSolutions`` when the phi constraint excludes all solutions.
    """
    with context:
        e4cv = creator()
        e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
        e4cv.beam.wavelength.put(1.54)

        # Two orientation reflections with DIFFERENT phi values.
        # or1: phi=0  (4,0,0) planes in the scattering plane.
        # or2: phi=90 (0,4,0) planes after a 90° phi rotation; same d-spacing.
        # After update_solver() adds both, the libhkl geometry retains
        # phi=90 (from or2).  The fix ensures forward() uses phi_current.
        or1 = e4cv.add_reflection(
            (4, 0, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
            wavelength=1.54,
            name="r400",
        )
        or2 = e4cv.add_reflection(
            (0, 4, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
            wavelength=1.54,
            name="r040",
        )
        e4cv.core.calc_UB(or1, or2)

        e4cv.core.mode = "constant_phi"
        e4cv.core.constraints["phi"].limits = phi_limits
        e4cv.phi.move(phi_current)

        # All solutions from core.forward() must use the current phi.
        # (Empty for the failure case; the loop body is skipped.)
        for solution in e4cv.core.forward(dict(h=1, k=1, l=1)):
            assert_almost_equal(solution.phi, phi_current, decimal=4)

        # diffractometer.forward() raises NoForwardSolutions when the
        # constraint filters out every solution (the failure case).
        position = e4cv.forward(dict(h=1, k=1, l=1))
        assert_almost_equal(position.phi, phi_current, decimal=4)
