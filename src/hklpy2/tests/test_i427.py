# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Regression test for issue #427."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import creator
from ..exceptions import NoForwardSolutions


@pytest.fixture()
def aps_polar():
    """Create the APS POLAR configuration reported in issue #427."""
    sim = creator(
        name="probe",
        geometry="APS POLAR",
        reals={
            "tau": None,
            "mu": None,
            "chi": None,
            "phi": None,
            "gamma": None,
            "delta": None,
        },
    )
    sim.add_sample("pr4310-exafs-cube1", 5.38, b=5.467, c=14.066, beta=101.0)
    wavelength = 0.6199478461064903
    sim.beam.wavelength.put(wavelength)
    r110 = sim.add_reflection(
        (1, 1, 0),
        {
            "tau": 0.0,
            "mu": 9.89806867221,
            "chi": 113.105327243248,
            "phi": 0.00173136532,
            "gamma": 9.42401385126,
            "delta": -0.00013200263,
        },
        wavelength=wavelength,
        name="543d0a3",
    )
    r004 = sim.add_reflection(
        (0, 0, 4),
        {
            "tau": 0.0,
            "mu": 10.848157453245,
            "chi": 30.583556860225,
            "phi": 0.00038517158,
            "gamma": 10.38248391045,
            "delta": -0.00011434739,
        },
        wavelength=wavelength,
        name="88e3c3d",
    )
    sim.core.calc_UB(r110, r004)
    sim.core.mode = "4-circles constant phi horizontal"
    sim.core.presets = {"delta": 0.0}
    return sim


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"chi": 0.0, "expected": (-66.8937, -9.3620)},
            does_not_raise(),
            id="current chi zero selects negative branch",
        ),
        pytest.param(
            {"chi": 30.5462, "expected": (113.1063, 9.3620)},
            does_not_raise(),
            id="current chi near 004 selects positive branch",
        ),
    ],
)
def test_forward_branch_changes_with_current_chi(aps_polar, parms, context):
    """The reported APS POLAR setup returns different first branches."""
    with context:
        aps_polar.chi.move(parms["chi"])
        solutions = aps_polar.core.forward({"h": 1, "k": 1, "l": 0})
        selected = aps_polar.forward({"h": 1, "k": 1, "l": 0})

        assert len(solutions) == 2
        assert (selected.chi, selected.gamma) == pytest.approx(
            parms["expected"], abs=0.001
        )
        assert selected == solutions[0]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"chi_limits": (112.0, 114.0), "expected": (113.1063, 9.3620)},
            does_not_raise(),
            id="tight chi limits retain positive branch",
        ),
        pytest.param(
            {"chi_limits": (-68.0, -66.0), "expected": (-66.8937, -9.3620)},
            does_not_raise(),
            id="tight chi limits retain negative branch",
        ),
        pytest.param(
            {"chi_limits": (113.0, 113.0)},
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="value above high limit is rejected before rounding",
        ),
        pytest.param(
            {"chi_limits": (-66.893, -66.8)},
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="value below low limit is rejected before rounding",
        ),
        pytest.param(
            {"digits": 2, "chi": 30.5462, "expected": (113.11, 9.36)},
            does_not_raise(),
            id="accepted result uses configured digits",
        ),
    ],
)
def test_forward_tight_chi_limits_select_expected_branch(aps_polar, parms, context):
    """Tight chi limits retain the corresponding reported solution."""
    with context:
        if "digits" in parms:
            aps_polar.digits = parms["digits"]
        aps_polar.core.constraints["chi"].limits = parms.get("chi_limits", (-180, 180))
        if "chi" in parms:
            aps_polar.chi.move(parms["chi"])
        solutions = aps_polar.core.forward({"h": 1, "k": 1, "l": 0})
        selected = aps_polar.forward({"h": 1, "k": 1, "l": 0})

        if "expected" in parms:
            assert len(solutions) == (2 if "digits" in parms else 1)
            assert (selected.chi, selected.gamma) == pytest.approx(
                parms["expected"], abs=0.001
            )
            assert selected == solutions[0]
