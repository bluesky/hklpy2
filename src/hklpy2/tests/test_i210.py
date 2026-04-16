"""
Regression test for issue #210.

Test simulator_from_config(): create a simulated diffractometer from a
saved configuration file or dict, with no hardware connections.
"""

import copy
import pathlib
import re
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import yaml

import hklpy2
from hklpy2.run_utils import simulator_from_config

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
def test_simulator_from_config(parms, context):
    """Test that simulator_from_config() returns a working diffractometer."""
    with context:
        sim = simulator_from_config(parms["config"])
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
def test_simulator_from_config_orientation(parms, context):
    """Test that orientation data is restored correctly."""
    with context:
        sim = simulator_from_config(parms["config"])

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
        sim = simulator_from_config(parms["config"])
        # Soft positioners are connected immediately without an IOC
        assert sim.connected


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=TESTS_DIR / "e4cv-silicon-example.yml"),
            does_not_raise(),
            id="simulator_from_config accessible from hklpy2 namespace",
        ),
    ],
)
def test_simulator_from_config_in_namespace(parms, context):
    """Test that simulator_from_config is accessible from hklpy2 namespace."""
    with context:
        sim = hklpy2.simulator_from_config(parms["config"])
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
        sim1 = simulator_from_config(parms["config"])
        config1 = sim1.configuration

        sim2 = simulator_from_config(config1)

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
        sim = simulator_from_config(parms["config"])
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
        sim = simulator_from_config(parms["config"])
        assert sim.pseudo_axis_names == parms["expected_pseudo_axes"]


# ---------------------------------------------------------------------------
# Issue #243: simulator_from_config restores reflections with wrong axis values
# when YAML serialises reals dict keys in alphabetical order.
# ---------------------------------------------------------------------------

_I243_CONFIG_FILE = TESTS_DIR / "configuration_i240.yml"
with open(_I243_CONFIG_FILE) as _f:
    _I243_CONFIG = yaml.safe_load(_f)

_I243_SAMPLE = _I243_CONFIG["samples"][_I243_CONFIG["sample_name"]]
_I243_LATTICE_A = _I243_SAMPLE["lattice"]["a"]

# Expected ||UB|| for a cubic crystal: 2*pi*sqrt(3)/a
_I243_UB_NORM_EXPECTED = 2 * np.pi * np.sqrt(3) / _I243_LATTICE_A
_I243_TOL = 0.001


def _config_with_positionally_wrong_reals():
    """Return a copy of the i243 config with reals mis-assigned as the old
    (broken) code did: values taken positionally from the alphabetically-sorted
    YAML dict and zipped onto the solver's physical-order axis names.

    This simulates the pre-#243 bug: YAML loads keys alphabetically
    (chi, delta, gamma, mu, phi, tau) but the solver expects physical order
    (tau, mu, chi, phi, gamma, delta).  Positional zip swaps the values onto
    the wrong axes.
    """
    config = copy.deepcopy(_I243_CONFIG)
    real_axes = config["axes"]["real_axes"]  # physical order
    axes_xref = config["axes"]["axes_xref"]
    solver_names = [axes_xref[a] for a in real_axes]  # solver names, physical order
    for sample in config["samples"].values():
        for refl in sample["reflections"].values():
            # alphabetical key order — what YAML loads
            alpha_values = list(dict(sorted(refl["reals"].items())).values())
            # positional zip: old broken behaviour
            refl["reals"] = dict(zip(solver_names, alpha_values))
    return config


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config=_I243_CONFIG_FILE),
            does_not_raise(),
            id="from file: correct UB norm",
        ),
        pytest.param(
            dict(config=_I243_CONFIG),
            does_not_raise(),
            id="from dict: correct UB norm",
        ),
        pytest.param(
            dict(config=_config_with_positionally_wrong_reals()),
            pytest.raises(ValueError, match=re.escape("degenerate U matrix")),
            id="pre-#243 positional mis-assignment: degenerate U matrix",
        ),
    ],
)
def test_simulator_from_config_reflection_axis_order(parms, context):
    """Regression test for #243: reals must be assigned by key, not position.

    YAML serialises dict keys alphabetically.  Before the fix, restoring a
    config caused calc_UB() to receive wrong axis values (positionally
    assigned from the alphabetically-sorted YAML dict), producing a degenerate
    U matrix.  The bad-case parameter supplies a pre-mangled config that
    directly injects those wrong values so the test does not depend on the
    internal implementation path.
    """
    with context:
        sim = simulator_from_config(parms["config"])
        r1, r2 = list(sim.core.sample.reflections)[:2]
        ub = sim.core.calc_UB(r1, r2)
        norm = np.linalg.norm(ub)
        assert np.isclose(norm, _I243_UB_NORM_EXPECTED, atol=_I243_TOL), (
            f"Expected ||UB||≈{_I243_UB_NORM_EXPECTED:.4f}, got {norm:.4f}"
        )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(config="e4cv_orient.yml"),
            does_not_raise(),
            id="E4CV vibranium: backward compatible, no auxiliary_axes in old config",
        ),
        pytest.param(
            dict(config="e4cv-silicon-example.yml"),
            does_not_raise(),
            id="E4CV silicon: backward compatible",
        ),
    ],
)
def test_simulator_from_config_no_auxiliary_axes(parms, context):
    """
    Test simulator_from_config() backward compatibility (issue #361).

    Old config files without auxiliary_axes restore without error.
    """
    with context:
        sim = simulator_from_config(TESTS_DIR / parms["config"])
        for ax in ("omega", "chi", "phi", "tth"):
            assert ax in sim.component_names
        result = sim.forward(h=1, k=0, l=0)
        assert len(result) > 0


def test_simulator_from_config_auxiliary_axes_roundtrip(tmp_path):
    """
    Test that auxiliary axes are saved by export() and restored automatically
    by simulator_from_config() without requiring reals= (issue #361).
    """
    import yaml

    # Build diffractometer with auxiliary axes and orient it.
    sim = hklpy2.creator(
        name="e4cv",
        reals=dict(omega=None, chi=None, phi=None, tth=None, atheta=None, attheta=None),
    )
    sim.core.add_sample("vibranium", 4.04)
    r1 = sim.core.add_reflection(
        dict(h=4, k=0, l=0), dict(omega=-145.451, chi=0, phi=0, tth=69.066)
    )
    r2 = sim.core.add_reflection(
        dict(h=0, k=4, l=0), dict(omega=-145.451, chi=0, phi=90, tth=69.066)
    )
    sim.core.calc_UB(r1, r2)

    # Export — auxiliary_axes must appear in the saved config.
    config_file = tmp_path / "e4cv-analyzer.yml"
    sim.export(str(config_file))
    cfg = yaml.safe_load(config_file.read_text())
    assert cfg["axes"].get("auxiliary_axes") == ["atheta", "attheta"]

    # Restore without reals= — auxiliary axes come from the config automatically.
    sim2 = simulator_from_config(config_file)
    assert "atheta" in sim2.component_names
    assert "attheta" in sim2.component_names

    # Orientation is preserved — forward() gives consistent results.
    assert np.isclose(
        sim.forward(h=1, k=0, l=0)[0],
        sim2.forward(h=1, k=0, l=0)[0],
    )
