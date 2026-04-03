"""Tests for hklpy2/typing.py TypedDict structures."""

from contextlib import nullcontext as does_not_raise

import pytest

from ..typing import ConfigHeaderDict


# ---------------------------------------------------------------------------
# ConfigHeaderDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                header=ConfigHeaderDict(
                    datetime="2026-04-03T12:00:00",
                    hklpy2_version="1.0.0",
                    python_class="MyDiffractometer",
                )
            ),
            does_not_raise(),
            id="valid ConfigHeaderDict",
        ),
        pytest.param(
            dict(
                header=ConfigHeaderDict(
                    datetime="2025-01-01T00:00:00",
                    hklpy2_version="0.9.0",
                    python_class="DiffractometerBase",
                )
            ),
            does_not_raise(),
            id="valid ConfigHeaderDict alternate values",
        ),
    ],
)
def test_config_header_dict_construct(parms, context):
    """ConfigHeaderDict can be constructed and accessed like a plain dict."""
    with context:
        header = parms["header"]
        assert isinstance(header, dict)
        assert isinstance(header["datetime"], str)
        assert isinstance(header["hklpy2_version"], str)
        assert isinstance(header["python_class"], str)


# ---------------------------------------------------------------------------
# Package-level import
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="ConfigHeaderDict"),
            does_not_raise(),
            id="ConfigHeaderDict importable from hklpy2",
        ),
    ],
)
def test_public_exports(parms, context):
    """ConfigHeaderDict is available from the top-level hklpy2 package."""
    with context:
        import hklpy2

        obj = getattr(hklpy2, parms["name"])
        assert obj is not None
