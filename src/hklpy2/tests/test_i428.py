# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Regression tests for issue #428."""

import re
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
        pytest.param(
            {
                "selector": type(
                    "Selector",
                    (),
                    {"__call__": lambda self, position, solutions: solutions[0]},
                )()
            },
            pytest.raises(
                TypeError,
                match=re.escape(
                    "The forward-solution selector must be an importable function "
                    "to be included in a configuration."
                ),
            ),
            id="non-importable selector cannot be serialized",
        ),
    ],
)
def test_configuration_preserves_forward_solution_selector(parms, context):
    """Configuration snapshots identify and restore the active selector."""
    with context:
        selector = parms["selector"]
        original = creator(
            name="original_428",
            forward_solution_function=selector if isinstance(selector, str) else None,
        )
        if callable(selector):
            original._forward_solution = selector
        config = original.configuration

        if isinstance(selector, str):
            assert config["forward_solution_function"] == selector

            restored = creator(name="restored_428")
            restored.restore(config)

            assert restored.forward_solution_function == selector
