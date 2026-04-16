"""
Benchmark test for issue #221.

Performance target: minimum 2,000 ``forward()`` and ``inverse()`` operations
per second.

Establishes the baseline measurement and verifies the target is met after
the performance fixes described in issue #221.
"""

import time
from contextlib import nullcontext as does_not_raise

import pytest

from .. import creator_from_config
from .common import TESTS_DIR

# Number of calls used to measure throughput.
N_CALLS = 500

# Minimum required operations per second (issue #221 target).
MIN_OPS_PER_SEC = 2_000


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                config="e4cv_orient.yml",
                pseudos=dict(h=1, k=0, l=0),
                mode="bissector",
            ),
            does_not_raise(),
            id="E4CV vibranium forward (1,0,0) bissector meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="e4cv_orient.yml",
                pseudos=dict(h=1, k=0, l=0),
                mode="constant_phi",
            ),
            does_not_raise(),
            id="E4CV vibranium forward (1,0,0) constant_phi meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="e4cv-silicon-example.yml",
                pseudos=dict(h=1, k=0, l=0),
                mode="bissector",
            ),
            does_not_raise(),
            id="E4CV silicon forward (1,0,0) bissector meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="configuration_i240.yml",
                pseudos=dict(h=1, k=0, l=0),
                mode="4-circles constant phi horizontal",
            ),
            does_not_raise(),
            id="APS POLAR forward (1,0,0) 4-circles const-phi meets 2000 ops/sec target",
        ),
    ],
)
def test_forward_throughput(parms, context):
    """
    Measure ``forward()`` throughput and assert it meets the 2,000 ops/sec
    target required by issue #221.
    """
    with context:
        sim = creator_from_config(TESTS_DIR / parms["config"])
        sim.core.solver.mode = parms["mode"]

        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            sim.forward(**parms["pseudos"])
        elapsed = time.perf_counter() - t0

        ops_per_sec = N_CALLS / elapsed
        print(
            f"\nforward() [{parms['mode']}] throughput: {ops_per_sec:.0f} ops/sec"
            f" ({elapsed * 1000 / N_CALLS:.2f} ms/call, {N_CALLS} calls)"
        )
        assert ops_per_sec >= MIN_OPS_PER_SEC, (
            f"forward() [{parms['mode']}] too slow: {ops_per_sec:.0f} ops/sec"
            f" (target: {MIN_OPS_PER_SEC} ops/sec)"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                config="e4cv_orient.yml",
                reals=dict(omega=-145, chi=0, phi=0, tth=69),
                mode="bissector",
            ),
            does_not_raise(),
            id="E4CV vibranium inverse (-145,0,0,69) bissector meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="e4cv_orient.yml",
                reals=dict(omega=-145, chi=0, phi=0, tth=69),
                mode="constant_phi",
            ),
            does_not_raise(),
            id="E4CV vibranium inverse (-145,0,0,69) constant_phi meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="e4cv-silicon-example.yml",
                reals=dict(omega=-8, chi=0, phi=0, tth=16),
                mode="bissector",
            ),
            does_not_raise(),
            id="E4CV silicon inverse (-8,0,0,16) bissector meets 2000 ops/sec target",
        ),
        pytest.param(
            dict(
                config="configuration_i240.yml",
                reals=dict(tau=0, mu=-114, chi=0, phi=0, gamma=-28, delta=0),
                mode="4-circles constant phi horizontal",
            ),
            does_not_raise(),
            id="APS POLAR inverse (0,-114,0,0,-28,0) 4-circles const-phi meets 2000 ops/sec target",
        ),
    ],
)
def test_inverse_throughput(parms, context):
    """
    Measure ``inverse()`` throughput and assert it meets the 2,000 ops/sec
    target required by issue #221 (and #223).
    """
    with context:
        sim = creator_from_config(TESTS_DIR / parms["config"])
        sim.core.solver.mode = parms["mode"]

        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            sim.inverse(**parms["reals"])
        elapsed = time.perf_counter() - t0

        ops_per_sec = N_CALLS / elapsed
        print(
            f"\ninverse() [{parms['mode']}] throughput: {ops_per_sec:.0f} ops/sec"
            f" ({elapsed * 1000 / N_CALLS:.2f} ms/call, {N_CALLS} calls)"
        )
        assert ops_per_sec >= MIN_OPS_PER_SEC, (
            f"inverse() [{parms['mode']}] too slow: {ops_per_sec:.0f} ops/sec"
            f" (target: {MIN_OPS_PER_SEC} ops/sec)"
        )
