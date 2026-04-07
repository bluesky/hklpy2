"""Test the hkl_soleil_utils module."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ...misc import SolverError


@pytest.mark.parametrize(
    "system, library, version, context",
    [
        pytest.param(
            "Darwin",
            "Hkl",
            "5.0",
            pytest.raises(
                SolverError,
                match=re.escape("'hkl_soleil' only available for linux 64-bit"),
            ),
            id="Darwin-unsupported",
        ),
        pytest.param("Linux", "Hkl", "5.0", does_not_raise(), id="Linux-valid"),
        pytest.param(
            "Windows",
            "Hkl",
            "5.0",
            pytest.raises(
                SolverError,
                match=re.escape("'hkl_soleil' only available for linux 64-bit"),
            ),
            id="Windows-unsupported",
        ),
        pytest.param(
            "Linux",
            "NOT FOUND",
            "5.0",
            pytest.raises(SolverError, match=re.escape("Cannot load 'gi' library:")),
            id="library-not-found",
        ),
        pytest.param(
            "Linux",
            "Hkl",
            "5.00",
            pytest.raises(SolverError, match=re.escape("Cannot load 'gi' library:")),
            id="wrong-version",
        ),
    ],
)
def test_gi_require_library(system, library, version, context):
    """Exercise the gi_require_library() function."""
    from ..hkl_soleil_utils import setup_libhkl

    with context:
        setup_libhkl(system, library, version)


def test_import_gi_failure():
    """Special case when 'gi' package is not installed."""
    import importlib.util
    import sys

    # Is the "gi" module available?
    gi_module = None
    if importlib.util.find_spec("gi") is not None:
        import gi  # noqa

        # Remove it from the dictionary.
        gi_module = sys.modules.pop("gi")

    # Import the function after manipulating 'sys.modules'.
    from ..hkl_soleil_utils import setup_libhkl

    # Proceed with testing, as above.
    with pytest.raises(
        SolverError,
        match=re.escape("Cannot import 'gi' (gobject-introspection) library."),
    ):
        setup_libhkl("Linux", "Hkl", "5.0")

    # Restore the 'gi' package to the dictionary.
    if gi_module is not None:
        sys.modules["gi"] = gi_module
