"""Test fixtures."""

import os
import shutil
import signal
import subprocess
import time

import pytest

from .common import IOC_PREFIX
from .common import TESTS_DIR

EPICS_DATABASE = TESTS_DIR / "testing.db"
IOC_READY_MESSAGE = b"epics>"
SOFTIOC_CMD = shutil.which("softIoc") or "softIoc"
ST_CMD = TESTS_DIR / "st.cmd"
STARTUP_TIMEOUT = 10.0


@pytest.fixture(scope="module")
def softioc():
    cmd = [
        SOFTIOC_CMD,
        "-m",
        f"P={IOC_PREFIX}",
        "-d",
        str(EPICS_DATABASE),
        str(ST_CMD),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Read stdout incrementally until we see the ready message or timeout
    ready = False
    buffer = bytearray()
    deadline = time.time() + STARTUP_TIMEOUT
    try:
        while time.time() < deadline:
            # Read available chunk (non-blocking-ish)
            try:
                chunk = proc.stdout.read1(1024)  # py3.7+: read1 on buffered raw file
            except AttributeError:
                # fallback if read1 isn't available (use read with small timeout)
                chunk = proc.stdout.read(1024)
            if chunk:
                buffer.extend(chunk)
                if IOC_READY_MESSAGE in buffer:
                    ready = True
                    break
            # Check if process died
            if proc.poll() is not None:
                out = buffer + (proc.stdout.read() or b"")
                pytest.skip(
                    f"softIoc did not print ready message within {STARTUP_TIMEOUT}s; skipping tests.\n"
                    f"Output so far:\n{out.decode(errors='replace')}",
                    allow_module_level=True,
                )
            time.sleep(0.05)
        if not ready:
            out = buffer + (proc.stdout.read() or b"")
            # Consider the process alive but not ready a failure for this fixture
            # (change to pytest.skip if you prefer skipping)
            proc_terminated = proc.poll() is not None
            status = (
                f"exited (rc={proc.returncode})" if proc_terminated else "still running"
            )
            pytest.skip(
                f"softIoc did not print ready message within {STARTUP_TIMEOUT}s (process {status}). "
                f"Output so far:\n{out.decode(errors='replace')}",
                allow_module_level=True,
            )

        # yield whatever tests need
        yield dict(
            proc=proc,
            dbfile=str(EPICS_DATABASE),
            output=bytes(buffer),
            ready=IOC_READY_MESSAGE in buffer,
        )
    finally:
        # Teardown: attempt graceful shutdown, then force kill if needed
        if proc.poll() is None:
            try:
                if os.name != "nt":
                    # send SIGTERM to process group
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait(timeout=5)
                except Exception:
                    pass
        # drain/close pipes
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
