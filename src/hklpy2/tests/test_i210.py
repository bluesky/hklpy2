"""
Regression test for issue #210.

Test creator_from_config(): create a simulated diffractometer from a
saved configuration file or dict, with no hardware connections.
"""

import pathlib
import re
from contextlib import nullcontext as does_not_raise

import pytest

import hklpy2
from hklpy2.misc import creator_from_config

TESTS_DIR = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=TESTS_DIR / "e4cv-silicon-example.yml"),
            does_not_raise(),
            id="e4cv silicon from path",
        ),
        pytest.param(
            dict(config=TESTS_DIR / "fourc-configuration.yml"),
            does_not_raise(),
            id="fourc custom axes from path",
        ),
        pytest.param(
            dict(config=TESTS_DIR / "tardis.yml"),
            does_not_raise(),
            id="tardis E6C from path",
        ),
        pytest.param(
            dict(config=str(TESTS_DIR / "e4cv-silicon-example.yml")),
            does_not_raise(),
            id="e4cv silicon from str path",
        ),
        pytest.param(
            dict(config=42),
            pytest.raises(
                TypeError, match=re.escape("Expected a dict or path to a YAML file.")
            ),
            id="invalid config type raises TypeError",
        ),
        pytest.param(
            dict(config={"solver": {}, "axes": {}}),
            pytest.raises(
                KeyError, match=re.escape("Configuration is missing '_header' key.")
            ),
            id="missing _header raises KeyError",
        ),
    ],
)
def test_creator_from_config(parms, context):
    """Test that creator_from_config() returns a working diffractometer."""
    with context:
        sim = creator_from_config(parms["config"])
        assert sim is not None
        assert hasattr(sim, "core")
        assert hasattr(sim, "forward")
        assert hasattr(sim, "inverse")


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                config=TESTS_DIR / "e4cv-silicon-example.yml",
                expected_solver="hkl_soleil",
                expected_geometry="E4CV",
                expected_sample="silicon",
            ),
            does_not_raise(),
            id="e4cv silicon: solver, geometry, sample restored",
        ),
        pytest.param(
            dict(
                config=TESTS_DIR / "fourc-configuration.yml",
                expected_solver="hkl_soleil",
                expected_geometry="E4CV",
                expected_sample="vibranium",
                expected_real_axes=["theta", "chi", "phi", "ttheta"],
            ),
            does_not_raise(),
            id="fourc: custom axis names preserved",
        ),
        pytest.param(
            dict(
                config=TESTS_DIR / "tardis.yml",
                expected_solver="hkl_soleil",
                expected_geometry="E6C",
                expected_sample="KCF",
            ),
            does_not_raise(),
            id="tardis E6C: geometry and sample restored",
        ),
    ],
)
def test_creator_from_config_orientation(parms, context):
    """Test that orientation data is restored correctly."""
    with context:
        sim = creator_from_config(parms["config"])

        assert sim.core.solver.name == parms["expected_solver"]
        assert sim.core.solver.geometry == parms["expected_geometry"]
        assert parms["expected_sample"] in sim.core.samples

        if "expected_real_axes" in parms:
            assert sim.real_axis_names == parms["expected_real_axes"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=TESTS_DIR / "e4cv-silicon-example.yml"),
            does_not_raise(),
            id="simulator has soft positioners (no hardware)",
        ),
    ],
)
def test_simulator_is_simulated(parms, context):
    """Test that the simulator uses soft positioners, not EPICS."""
    with context:
        sim = creator_from_config(parms["config"])
        # Soft positioners are connected immediately without an IOC
        assert sim.connected


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=TESTS_DIR / "e4cv-silicon-example.yml"),
            does_not_raise(),
            id="creator_from_config accessible from hklpy2 namespace",
        ),
    ],
)
def test_creator_from_config_in_namespace(parms, context):
    """Test that creator_from_config is accessible from hklpy2 namespace."""
    with context:
        sim = hklpy2.creator_from_config(parms["config"])
        assert sim is not None


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=TESTS_DIR / "e4cv-silicon-example.yml"),
            does_not_raise(),
            id="roundtrip: config from simulator matches original",
        ),
    ],
)
def test_simulator_roundtrip(parms, context):
    """Test that a simulator's exported config can recreate another simulator."""
    with context:
        sim1 = creator_from_config(parms["config"])
        config1 = sim1.configuration

        sim2 = creator_from_config(config1)

        assert sim2.core.solver.geometry == sim1.core.solver.geometry
        assert set(sim2.core.samples.keys()) == set(sim1.core.samples.keys())
        assert sim2.real_axis_names == sim1.real_axis_names


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                config=TESTS_DIR / "tardis.yml",
                # tardis axes_xref: theta->mu, mu->omega, chi->chi,
                # phi->phi, delta->gamma, gamma->delta
                # solver E6C canonical order: [mu, omega, chi, phi, gamma, delta]
                # diffractometer order must follow: theta, mu, chi, phi, delta, gamma
                expected_real_axes=["theta", "mu", "chi", "phi", "delta", "gamma"],
            ),
            does_not_raise(),
            id="tardis: out-of-order axes_xref correctly reordered",
        ),
        pytest.param(
            dict(
                config=TESTS_DIR / "fourc-configuration.yml",
                # fourc axes_xref: theta->omega, chi->chi, phi->phi, ttheta->tth
                # solver E4CV canonical order: [omega, chi, phi, tth]
                expected_real_axes=["theta", "chi", "phi", "ttheta"],
            ),
            does_not_raise(),
            id="fourc: custom axis names reordered to match solver",
        ),
    ],
)
def test_simulator_axis_order(parms, context):
    """Test that real axes are in solver-expected order even with custom names."""
    with context:
        sim = creator_from_config(parms["config"])
        assert sim.real_axis_names == parms["expected_real_axes"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                config=TESTS_DIR / "e4cv-silicon-example.yml",
                expected_pseudo_axes=["h", "k", "l"],
            ),
            does_not_raise(),
            id="e4cv: pseudo axes in solver-expected order",
        ),
        pytest.param(
            dict(
                config=TESTS_DIR / "tardis.yml",
                expected_pseudo_axes=["h", "k", "l"],
            ),
            does_not_raise(),
            id="tardis: pseudo axes in solver-expected order",
        ),
    ],
)
def test_simulator_pseudo_order(parms, context):
    """Test that pseudo axes are in solver-expected order."""
    with context:
        sim = creator_from_config(parms["config"])
        assert sim.pseudo_axis_names == parms["expected_pseudo_axes"]
