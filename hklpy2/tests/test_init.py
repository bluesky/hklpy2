"""Test the package constructor."""

import importlib
import runpy

import pytest

from .. import _get_version


@pytest.mark.parametrize(
    "version_module, expected_version",
    [
        ({"version": "1.0.0"}, "1.0.0"),  # Test with a valid version module
        (None, "2.0.0"),  # Test with importlib.metadata returning a version
    ],
)
def test_get_version_with_version_module(mocker, version_module, expected_version):
    if version_module is not None:
        mock_version_module = mocker.Mock()
        mock_version_module.version = version_module["version"]
        version = _get_version(version_module=mock_version_module)
    else:
        # Mock importlib.metadata.version to return the expected version
        mocker.patch("importlib.metadata.version", return_value=expected_version)
        version = _get_version()

    assert version == expected_version


@pytest.mark.parametrize(
    "side_effect, expected_version",
    [
        (Exception("Not found"), "0+unknown"),  # Test fallback scenario
    ],
)
def test_get_version_fallback(mocker, side_effect, expected_version):
    # Mock importlib.metadata.version to raise an exception
    mocker.patch("importlib.metadata.version", side_effect=side_effect)

    version = _get_version()
    assert version == expected_version


def test_prints_version_when_run_as_main(capsys):
    """
    Running the package as a script (so __name__ == "__main__") should
    execute the print(...) in hklpy2.__init__ and emit the package version.
    """
    pkg = importlib.import_module("hklpy2")

    # Execute the package __init__ as __main__ to trigger the guarded print.
    # To avoid RuntimeWarning from runpy about a package already being in
    # sys.modules, temporarily remove the package entries before running the
    # module as __main__, then restore them afterward.
    import sys

    saved_pkg = sys.modules.pop("hklpy2", None)
    saved_init = sys.modules.pop("hklpy2.__init__", None)
    try:
        runpy.run_module("hklpy2.__init__", run_name="__main__")
    finally:
        if saved_pkg is not None:
            sys.modules["hklpy2"] = saved_pkg
        if saved_init is not None:
            sys.modules["hklpy2.__init__"] = saved_init

    captured = capsys.readouterr()
    assert f"Package version: {pkg.__version__}" in captured.out
