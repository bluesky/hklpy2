"""Code that is common to several tests."""

import pathlib

HKLPY2_DIR = pathlib.Path(__file__).parent.parent
IOC_PREFIX = "hklpy2:"
PV_ENERGY = f"{IOC_PREFIX}energy"
PV_WAVELENGTH = f"{IOC_PREFIX}wavelength"
TESTS_DIR = HKLPY2_DIR / "tests"


def assert_context_result(expected, reason):
    """Common handling for tests below.

    The tests pass the value yielded by the context manager used in a
    ``with`` block as ``reason``. For ``does_not_raise()`` (``nullcontext``)
    this is ``None`` (no exception). For ``pytest.raises(...)`` the context
    manager yields an ExceptionInfo-like object when an exception was raised.

    Rules implemented here:
    - If ``reason`` is None -> no exception occurred: treat as success and
      return (tests may still assert on state after the with-block).
    - If ``reason`` is truthy (exception captured):
        * If ``expected`` is None, any exception is acceptable -> return.
        * If ``expected`` is a string, assert it appears in the exception
          message (or in the stringified ExceptionInfo).
    """
    # No exception was raised inside the context manager -> success.
    if reason is None:
        return

    # An exception was captured by the context manager (e.g., pytest.raises).
    if expected is None:
        # Any exception is acceptable here.
        return

    msg = str(reason)
    if expected in msg:
        return

    # Handle some legacy tests that look for text about missing setters.
    if "can't set attribute" in msg and any(
        k in expected for k in ("setter", "no setter", "has no setter")
    ):
        return

    raise AssertionError(f"{expected=!r} {reason=}")
