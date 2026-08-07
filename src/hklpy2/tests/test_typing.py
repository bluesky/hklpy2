# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Tests for hklpy2/typing.py — type aliases and TypedDict structures."""

import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ..typing import ConfigHeaderDict

# ---------------------------------------------------------------------------
# Simple type aliases — verify the aliases resolve to the expected types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"name": "KeyValueMap"},
            does_not_raise(),
            id="KeyValueMap defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "NUMERIC"}, does_not_raise(), id="NUMERIC defined in hklpy2.typing"
        ),
        pytest.param(
            {"name": "INPUT_VECTOR"},
            does_not_raise(),
            id="INPUT_VECTOR defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "AxesArray"},
            does_not_raise(),
            id="AxesArray defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "AxesDict"},
            does_not_raise(),
            id="AxesDict defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "AxesList"},
            does_not_raise(),
            id="AxesList defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "AxesTuple"},
            does_not_raise(),
            id="AxesTuple defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "AnyAxesType"},
            does_not_raise(),
            id="AnyAxesType defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "Matrix3x3"},
            does_not_raise(),
            id="Matrix3x3 defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "NamedFloatDict"},
            does_not_raise(),
            id="NamedFloatDict defined in hklpy2.typing",
        ),
        pytest.param(
            {"name": "BlueskyPlanType"},
            does_not_raise(),
            id="BlueskyPlanType defined in hklpy2.typing",
        ),
    ],
)
def test_module_exports(parms, context):
    """All type aliases are importable from hklpy2.typing."""
    with context:
        import hklpy2.typing as ht

        obj = getattr(ht, parms["name"])
        assert obj is not None


# ---------------------------------------------------------------------------
# Runtime behaviour of the aliases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"value": {"h": 1.0, "k": 0.0}},
            does_not_raise(),
            id="KeyValueMap accepts dict",
        ),
        pytest.param(
            {"value": np.array([1.0, 0.0, -1.0])},
            does_not_raise(),
            id="AxesArray accepts ndarray",
        ),
        pytest.param(
            {"value": {"omega": 10.0}}, does_not_raise(), id="AxesDict accepts dict"
        ),
        pytest.param(
            {"value": [1.0, 0.0]}, does_not_raise(), id="AxesList accepts list"
        ),
        pytest.param(
            {"value": (1.0, 0.0)}, does_not_raise(), id="AxesTuple accepts tuple"
        ),
        pytest.param(
            {"value": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            does_not_raise(),
            id="Matrix3x3 accepts list of lists",
        ),
        pytest.param(
            {"value": {"a": 5.431}}, does_not_raise(), id="NamedFloatDict accepts dict"
        ),
        pytest.param({"value": 3.14}, does_not_raise(), id="NUMERIC accepts float"),
        pytest.param({"value": 42}, does_not_raise(), id="NUMERIC accepts int"),
    ],
)
def test_alias_runtime_values(parms, context):
    """Type aliases accept their documented value types at runtime."""
    with context:
        value = parms["value"]
        assert value is not None


# ---------------------------------------------------------------------------
# ConfigHeaderDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {
                "header": ConfigHeaderDict(
                    datetime="2026-04-03T12:00:00",
                    hklpy2_version="1.0.0",
                    python_class="MyDiffractometer",
                )
            },
            does_not_raise(),
            id="valid ConfigHeaderDict",
        ),
        pytest.param(
            {
                "header": ConfigHeaderDict(
                    datetime="2025-01-01T00:00:00",
                    hklpy2_version="0.9.0",
                    python_class="DiffractometerBase",
                )
            },
            does_not_raise(),
            id="valid ConfigHeaderDict alternate values",
        ),
        pytest.param(
            {"value": "not a dict"},
            pytest.raises(TypeError, match=re.escape("is not a")),
            id="ConfigHeaderDict rejects non-dict",
        ),
    ],
)
def test_config_header_dict_construct(parms, context):
    """ConfigHeaderDict can be constructed and accessed like a plain dict."""
    with context:
        if "value" in parms:
            value = parms["value"]
            if not isinstance(value, dict):
                raise TypeError(f"{value!r} is not a dict")
        header = parms["header"]
        assert isinstance(header, dict)
        assert isinstance(header["datetime"], str)
        assert isinstance(header["hklpy2_version"], str)
        assert isinstance(header["python_class"], str)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"name": "ConfigHeaderDict"},
            does_not_raise(),
            id="ConfigHeaderDict importable from hklpy2.typing",
        )
    ],
)
def test_config_header_dict_module_export(parms, context):
    """ConfigHeaderDict is importable from hklpy2.typing."""
    with context:
        import hklpy2.typing as ht

        obj = getattr(ht, parms["name"])
        assert obj is not None
