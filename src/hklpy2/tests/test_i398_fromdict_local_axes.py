# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Regression tests for issue #398.

``ReflectionsDict._fromdict`` previously rewrote each restored
reflection's ``reals`` dict so its keys were the solver's canonical
real-axis names.  Every other consumer of ``Reflection.reals`` --
notably ``Core._reflections_to_solver`` -- treats the dict as
local-keyed and translates to canonical via ``axes_xref_reversed`` at
the point of use.  The asymmetry was invisible whenever local and
canonical names happened to coincide (the typical ``E4CV`` case where
``omega`` and ``tth`` are both canonical and local), but raised
``KeyError`` on the next ``forward()`` after any restore round-trip
(``Diffractometer.restore``, ``simulator_from_config``,
``benchmark(snapshot=True)``) whenever the user had renamed any real
axis.
"""

from contextlib import nullcontext as does_not_raise

import pytest

from .. import creator
from ..run_utils import simulator_from_config

SI_LATTICE = {"a": 5.431}
WAVELENGTH = 1.0
RENAMED_REAL_AXES = ["theta", "chi", "phi", "two_theta"]
CANONICAL_REAL_AXES = ["omega", "chi", "phi", "tth"]
PROBE = (1, 0, 0)


def _build_oriented(reals_names):
    """Create an ``E4CV`` diffractometer with ``reals_names`` and a UB."""
    d = creator(name="d", solver="hkl_soleil", geometry="E4CV", reals=reals_names)
    d.beam.wavelength.put(WAVELENGTH)
    d.add_sample("si", **SI_LATTICE)
    r1_reals = dict(zip(reals_names, (10.0, 0.0, 0.0, 20.0)))
    r2_reals = dict(zip(reals_names, (10.0, 0.0, 90.0, 20.0)))
    r1 = d.add_reflection(PROBE, reals=r1_reals, name="r1", replace=True)
    r2 = d.add_reflection((0, 1, 0), reals=r2_reals, name="r2", replace=True)
    d.core.calc_UB(r1, r2)
    return d


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"reals": RENAMED_REAL_AXES},
            does_not_raise(),
            id="renamed-real-axes round-trip via simulator_from_config",
        ),
        pytest.param(
            {"reals": CANONICAL_REAL_AXES},
            does_not_raise(),
            id="canonical-real-axes round-trip via simulator_from_config",
        ),
    ],
)
def test_simulator_from_config_preserves_reflection_reals_keys(parms, context):
    """
    After a snapshot round-trip via ``simulator_from_config``, each
    restored reflection's ``reals`` dict must be keyed by the
    diffractometer's **local** axis names (matching the original) and
    ``forward()`` on the restored copy must succeed and return the
    same solution as the original.
    """
    with context:
        original = _build_oriented(parms["reals"])
        original_forward = original.forward(*PROBE)

        sim = simulator_from_config(original)

        # The restored reflection's ``reals`` dict must be local-keyed.
        for name in ("r1", "r2"):
            restored_keys = list(sim.sample.reflections[name]._asdict()["reals"])
            assert restored_keys == parms["reals"], (
                f"Reflection {name!r}: expected local keys "
                f"{parms['reals']!r}, got {restored_keys!r}"
            )

        # And ``forward()`` on the restored copy must succeed and match.
        restored_forward = sim.forward(*PROBE)
        assert tuple(restored_forward) == tuple(original_forward), (
            f"forward({PROBE}) differs after round-trip: "
            f"original={original_forward!r}, restored={restored_forward!r}"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"reals": RENAMED_REAL_AXES},
            does_not_raise(),
            id="renamed-real-axes round-trip via export/restore",
        ),
        pytest.param(
            {"reals": CANONICAL_REAL_AXES},
            does_not_raise(),
            id="canonical-real-axes round-trip via export/restore",
        ),
    ],
)
def test_export_restore_preserves_reflection_reals_keys(parms, context, tmp_path):
    """
    The same contract applies to ``export()`` / ``restore()``: after
    saving and reloading a diffractometer's configuration, each
    restored reflection's ``reals`` is local-keyed and ``forward()``
    succeeds.
    """
    with context:
        original = _build_oriented(parms["reals"])
        original_forward = original.forward(*PROBE)

        config_file = tmp_path / "fourc.yml"
        original.export(str(config_file))

        # Build a fresh diffractometer with the same axis names and
        # restore from file.
        restored = creator(
            name="d", solver="hkl_soleil", geometry="E4CV", reals=parms["reals"]
        )
        restored.restore(
            str(config_file), restore_samples=True, restore_wavelength=True
        )

        for name in ("r1", "r2"):
            restored_keys = list(restored.sample.reflections[name]._asdict()["reals"])
            assert restored_keys == parms["reals"], (
                f"Reflection {name!r}: expected local keys "
                f"{parms['reals']!r}, got {restored_keys!r}"
            )

        restored_forward = restored.forward(*PROBE)
        assert tuple(restored_forward) == tuple(original_forward), (
            f"forward({PROBE}) differs after export/restore: "
            f"original={original_forward!r}, restored={restored_forward!r}"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {},
            does_not_raise(),
            id="snapshot benchmark does not raise on renamed real axes",
        )
    ],
)
def test_benchmark_snapshot_does_not_raise_on_renamed_axes(parms, context):
    """
    ``benchmark(diff, snapshot=True)`` (the default since 0.6.x) builds
    a simulator from the diffractometer's configuration and runs the
    timing loop on it.  Before the fix, this raised ``KeyError`` from
    ``_reflections_to_solver`` whenever the user had renamed any real
    axis.  This test asserts that the timing loop completes.
    """
    with context:
        from ..utils import benchmark

        d = _build_oriented(RENAMED_REAL_AXES)
        # n=2 keeps the test fast; the goal is to exercise the
        # snapshot-build + forward()/inverse() loop, not measure.
        # ``print=False`` returns the results dict instead of writing
        # to stdout.
        result = benchmark(d, n=2, print=False)
        assert result is not None
        assert "forward_ops_per_sec" in result
        assert "inverse_ops_per_sec" in result


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"reals": RENAMED_REAL_AXES},
            does_not_raise(),
            id="round-trip preserves UB matrix for renamed axes",
        ),
        pytest.param(
            {"reals": CANONICAL_REAL_AXES},
            does_not_raise(),
            id="round-trip preserves UB matrix for canonical axes",
        ),
    ],
)
def test_round_trip_preserves_UB(parms, context):
    """
    The restored diffractometer's ``UB`` matrix must match the
    original within numerical tolerance.  This is a stronger end-to-end
    check that nothing in the reflection-restore path silently swaps
    axes.
    """
    with context:
        import numpy as np

        original = _build_oriented(parms["reals"])
        sim = simulator_from_config(original)

        ub_original = np.array(original.sample.UB)
        ub_restored = np.array(sim.sample.UB)
        assert np.allclose(ub_original, ub_restored, atol=1e-9), (
            f"UB drift after round-trip: original={ub_original!r}, "
            f"restored={ub_restored!r}"
        )
