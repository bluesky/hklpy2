# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Regression tests for issue #428."""

from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import creator


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"selector": "hklpy2.utils.pick_first_solution"},
            does_not_raise(),
            id="first solution selector round trip",
        ),
        pytest.param(
            {"selector": "hklpy2.utils.pick_closest_solution"},
            does_not_raise(),
            id="closest solution selector round trip",
        ),
    ],
)
def test_configuration_preserves_forward_solution_selector(parms, context):
    """Configuration snapshots identify and restore the active selector."""
    with context:
        original = creator(
            name="original_428", forward_solution_function=parms["selector"]
        )
        config = original.configuration

        assert config["forward_solution_function"] == parms["selector"]

        restored = creator(name="restored_428")
        restored.restore(config)

        assert restored.forward_solution_function == parms["selector"]
