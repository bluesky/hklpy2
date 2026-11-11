# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Code that is common to several tests."""

import pathlib
import threading
import time

HKLPY2_DIR = pathlib.Path(__file__).parent.parent
IOC_PREFIX = "hklpy2:"
PV_ENERGY = f"{IOC_PREFIX}energy"
PV_WAVELENGTH = f"{IOC_PREFIX}wavelength"
TESTS_DIR = HKLPY2_DIR / "tests"

INTERLEAVE_TIMEOUT = 0.5  # s, how long main waits for an intruder that may be blocked


class Interleave:
    """
    Force a specific ordering between a main call and an intruder thread.

    A hook inside the main call starts the intruder and waits until it pauses
    (:meth:`intruder_point`) or finishes, or until :data:`INTERLEAVE_TIMEOUT`
    (the intruder is blocked, e.g. on a lock).  Without serialization the
    intruder runs inside the main call's window; with it, the intruder runs
    after.  A slow machine can only make an unserialized test pass falsely,
    never a serialized one fail.
    """

    def __init__(self):
        self.thread = None
        self.error = None
        self.paused = threading.Event()
        self.main_done = threading.Event()

    def start_intruder(self, fn):
        def run():
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001  re-raised by finish()
                self.error = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def is_intruder(self):
        return threading.current_thread() is self.thread

    def wait_for_intruder(self):
        """Return when the intruder pauses or finishes, or after the timeout."""
        deadline = time.monotonic() + INTERLEAVE_TIMEOUT
        while time.monotonic() < deadline:
            if self.paused.is_set() or not self.thread.is_alive():
                return
            time.sleep(0.001)

    def intruder_point(self):
        """Called on the intruder thread: pause until main is done."""
        self.paused.set()
        self.main_done.wait(INTERLEAVE_TIMEOUT)

    def finish(self):
        """Release the intruder, wait for it, and re-raise its exception."""
        self.main_done.set()
        if self.thread is None:
            return
        self.thread.join(5)
        assert not self.thread.is_alive(), "intruder still blocked (lock not released?)"
        if self.error is not None:
            raise self.error
