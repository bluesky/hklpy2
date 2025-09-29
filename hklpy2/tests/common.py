"""Code that is common to several tests."""

import pathlib

HKLPY2_DIR = pathlib.Path(__file__).parent.parent
TESTS_DIR = HKLPY2_DIR / "tests"
PV_ENERGY = "hklpy2:energy"
PV_WAVELENGTH = "hklpy2:wavelength"


def assert_context_result(expected, reason):
    """Common handling for tests below."""
    if expected is None:
        assert reason is None
    else:
        msg = str(reason)
        # Accept either the explicit expected text, or the common
        # AttributeError message produced by attempting to assign to a
        # read-only property ("can't set attribute '<name>'").
        if expected in msg:
            return
        if "can't set attribute" in msg and ("setter" in expected or "no setter" in expected or "has no setter" in expected):
            return
        assert expected in msg, f"{expected=!r} {reason=}"
