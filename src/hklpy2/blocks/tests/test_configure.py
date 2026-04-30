import math
import pathlib
import re
from contextlib import nullcontext as does_not_raise

import pytest

from ... import __version__
from ...diffract import DiffractometerBase
from ...diffract import creator
from ...exceptions import ConfigurationError
from ...utils import load_yaml_file
from ...tests.models import E4CV_CONFIG_FILE
from ...tests.models import add_oriented_vibranium_to_e4cv
from ...tests.models import e4cv_config
from ..configure import Configuration

e4cv = creator(name="e4cv")
add_oriented_vibranium_to_e4cv(e4cv)

sim2c = creator(name="sim2c", solver="th_tth", geometry="TH TTH Q")
twopi = 2 * math.pi


@pytest.mark.parametrize(
    "keypath, value",
    [
        pytest.param("_header.datetime", None, id="header-datetime"),
        pytest.param("_header.hklpy2_version", __version__, id="header-hklpy2-version"),
        pytest.param(
            "_header.python_class", e4cv.__class__.__name__, id="header-python-class"
        ),
        pytest.param("axes.axes_xref", e4cv.core.axes_xref, id="axes-xref"),
        pytest.param("axes.extra_axes", e4cv.core.all_extras, id="axes-extra"),
        pytest.param("axes.pseudo_axes", e4cv.pseudo_axis_names, id="axes-pseudo"),
        pytest.param("axes.real_axes", e4cv.real_axis_names, id="axes-real"),
        pytest.param(
            "beam.energy_units", e4cv.beam.energy_units.get(), id="beam-energy-units"
        ),
        pytest.param("beam.energy", e4cv.beam.energy.get(), id="beam-energy"),
        pytest.param(
            "beam.source_type", e4cv.beam.source_type.get(), id="beam-source-type"
        ),
        pytest.param(
            "beam.wavelength_units",
            e4cv.beam.wavelength_units.get(),
            id="beam-wavelength-units",
        ),
        pytest.param(
            "beam.wavelength", e4cv.beam.wavelength.get(), id="beam-wavelength"
        ),
        pytest.param(
            "constraints.chi.high_limit", 180.2, id="constraint-chi-high-limit"
        ),
        pytest.param("constraints.omega.label", "omega", id="constraint-omega-label"),
        pytest.param(
            "constraints.tth.low_limit", -180.2, id="constraint-tth-low-limit"
        ),
        pytest.param("name", e4cv.name, id="diffractometer-name"),
        pytest.param("sample_name", e4cv.sample.name, id="sample-name"),
        pytest.param("samples.sample.lattice.a", 1, id="sample-lattice-a"),
        pytest.param("samples.sample.lattice.alpha", 90, id="sample-lattice-alpha"),
        pytest.param("samples.sample.name", "sample", id="sample-default-name"),
        pytest.param(
            "samples.sample.reflections_order", [], id="sample-reflections-order-empty"
        ),
        pytest.param("samples.sample.reflections", {}, id="sample-reflections-empty"),
        pytest.param(
            "samples.sample.U",
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            id="sample-U-identity",
        ),
        pytest.param(
            "samples.sample.UB",
            [[twopi, 0, 0], [0, twopi, 0], [0, 0, twopi]],
            id="sample-UB-default",
        ),
        pytest.param("samples.vibranium.name", "vibranium", id="vibranium-name"),
        pytest.param(
            "samples.vibranium.reflections_order",
            "r040 r004".split(),
            id="vibranium-reflections-order-r040-r400",
        ),
        pytest.param(
            "samples.vibranium.reflections.r004.name", "r004", id="vibranium-r004-name"
        ),
        pytest.param(
            "samples.vibranium.reflections.r004.pseudos.h",
            0,
            id="vibranium-r004-pseudo-h",
        ),
        pytest.param(
            "samples.vibranium.reflections.r004.pseudos.k",
            0,
            id="vibranium-r004-pseudo-k",
        ),
        pytest.param(
            "samples.vibranium.reflections.r004.pseudos.l",
            4,
            id="vibranium-r004-pseudo-l",
        ),
        pytest.param(
            "samples.vibranium.reflections.r004.reals.chi",
            90,
            id="vibranium-r004-real-chi",
        ),
        pytest.param("samples.vibranium.U", e4cv.sample.U, id="vibranium-U-matrix"),
        pytest.param("samples.vibranium.UB", e4cv.sample.UB, id="vibranium-UB-matrix"),
        pytest.param("solver.engine", e4cv.core.solver.engine_name, id="solver-engine"),
        pytest.param("solver.geometry", e4cv.core.geometry, id="solver-geometry"),
        pytest.param("solver.name", e4cv.core.solver_name, id="solver-name"),
        pytest.param(
            "solver.real_axes", e4cv.core.solver_real_axis_names, id="solver-real-axes"
        ),
    ],
)
def test_Configuration(keypath, value):
    agent = Configuration(e4cv).diffractometer.configuration
    assert "_header" in agent, f"{agent=!r}"
    assert "file" not in agent["_header"], f"{agent=!r}"

    for k in keypath.split("."):
        agent = agent.get(k)  # narrow the search
        assert agent is not None, f"{k=!r}  {keypath=!r}"

    if value is not None:
        assert value == agent, f"{k=!r}  {value=!r}  {agent=!r}"


def test_Configuration_export(tmp_path):
    assert isinstance(tmp_path, pathlib.Path)
    assert tmp_path.exists()

    config_file = tmp_path / "config.yml"
    assert not config_file.exists()

    # write the YAML file
    agent = Configuration(e4cv)
    agent.diffractometer.export(config_file, comment="testing")
    assert config_file.exists()

    # read the YAML file, check for _header.file key
    config = load_yaml_file(config_file)
    assert "_header" in config, f"{config=!r}"
    assert "file" in config["_header"], f"{config=!r}"
    assert "comment" in config["_header"], f"{config=!r}"
    assert config["_header"]["comment"] == "testing"


def test_asdict():
    fourc = creator(name="fourc")
    add_oriented_vibranium_to_e4cv(fourc)

    cfg = Configuration(e4cv)._asdict()
    cfg["_header"].pop("datetime", None)
    for module in (e4cv, e4cv.core):
        module_config = module.configuration
        if not isinstance(module_config, dict):
            module_config = module_config._asdict()
        module_config["_header"].pop("datetime", None)
        for section in "axes samples constraints solver".split():
            assert cfg[section] == module_config[section]


def test_fromdict():
    fourc = creator(name="fourc")
    add_oriented_vibranium_to_e4cv(fourc)

    config = e4cv_config()
    assert config.get("name") == "e4cv"

    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")

    with pytest.raises(ConfigurationError) as reason:
        sim.core.configuration._fromdict(config)
    assert "solver mismatch" in str(reason)

    fourc = creator(name="fourc")
    add_oriented_vibranium_to_e4cv(fourc)

    assert fourc.name != config["name"]
    assert len(fourc.samples) == 2
    assert len(fourc.core.constraints) == 4

    fourc.core.reset_constraints()
    fourc.core.reset_samples()
    assert len(fourc.samples) == 1
    assert len(fourc.core.constraints) == 4
    assert len(fourc.sample.reflections) == 0

    for key, constraint in fourc.core.constraints.items():
        assert key in config["constraints"]
        cfg = config["constraints"][key]
        assert cfg["class"] == constraint.__class__.__name__
        for field in constraint._fields:
            assert field in cfg, f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
            if field == "label":
                assert cfg[field] == getattr(constraint, field)
            elif field == "cut_point":
                # cut_point in the fixture equals the reset default; only
                # verify the field is present (done above) and readable.
                assert isinstance(cfg[field], float)
            else:
                assert cfg[field] != getattr(constraint, field), (
                    f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
                )
    # A few pre-checks
    assert "geometry" not in config
    assert "solver" in config
    assert "geometry" in config["solver"]

    ###
    ### apply the configuration
    ###
    fourc.core.configuration._fromdict(config), f"{fourc=!r}"

    sample = config["sample_name"]
    assert sample == fourc.sample.name, f"{sample=!r}  {fourc.sample.name=!r}"
    assert len(fourc.samples) == len(config["samples"]), f"{config['samples']=!r}"
    assert (
        fourc.sample.reflections.order == config["samples"][sample]["reflections_order"]
    )

    assert len(fourc.sample.reflections) == 3
    sample_cfg = config["samples"][config["sample_name"]]
    for refl_name in fourc.sample.reflections.order:
        assert refl_name in fourc.sample.reflections
        refl = fourc.sample.reflections[refl_name]
        cfg_refl = sample_cfg["reflections"][refl_name]
        # Compare pseudo positions.
        for axis, value in cfg_refl["pseudos"].items():
            assert refl.pseudos[axis] == pytest.approx(value), (
                f"{refl_name=!r} pseudo {axis=!r}: {refl.pseudos[axis]=} != {value=}"
            )
        # Compare real positions.
        for axis, value in cfg_refl["reals"].items():
            assert refl.reals[axis] == pytest.approx(value), (
                f"{refl_name=!r} real {axis=!r}: {refl.reals[axis]=} != {value=}"
            )
        # Compare wavelength.
        assert refl.wavelength == pytest.approx(cfg_refl["wavelength"]), (
            f"{refl_name=!r} {refl.wavelength=} != {cfg_refl['wavelength']=}"
        )

    assert len(fourc.core.constraints) == len(config["constraints"])
    for key, constraint in fourc.core.constraints.items():
        assert key in config["constraints"]
        cfg = config["constraints"][key]
        assert cfg["class"] == constraint.__class__.__name__
        for field in constraint._fields:
            assert field in cfg, f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
            assert cfg[field] == getattr(constraint, field), (
                f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
            )


@pytest.mark.parametrize(
    "diffractometer, clear, restore, file, context",
    [
        pytest.param(
            e4cv,
            True,
            True,
            E4CV_CONFIG_FILE,
            does_not_raise(),
            id="e4cv-clear-restore",
        ),
        pytest.param(
            e4cv,
            True,
            False,
            E4CV_CONFIG_FILE,
            does_not_raise(),
            id="e4cv-clear-no-restore",
        ),
        pytest.param(
            sim2c,
            True,
            True,
            E4CV_CONFIG_FILE,
            pytest.raises(ConfigurationError, match=re.escape("solver mismatch")),
            id="sim2c-solver-mismatch",
        ),
        pytest.param(
            sim2c,
            True,
            True,
            "this file does not exist",
            pytest.raises(FileExistsError, match=re.escape("this file does not exist")),
            id="sim2c-file-not-exist",
        ),
        pytest.param(
            None,
            True,
            True,
            E4CV_CONFIG_FILE,
            pytest.raises(AssertionError, match=re.escape("False")),
            id="none-diffractometer",
        ),
    ],
)
def test_restore(diffractometer, clear, restore, file, context):
    with context:
        assert isinstance(diffractometer, DiffractometerBase)
        diffractometer.restore(
            file,
            clear=clear,
            restore_constraints=restore,
        )


# ---------------------------------------------------------------------------
# Issue #390: extras / safety controls round-trip
# ---------------------------------------------------------------------------


def _hardware_e4cv(name="hw_e4cv"):
    """Return an E4CV diffractometer whose reals are EpicsMotor instances.

    The EPICS PVs never need to connect for these tests; we only care
    about ``is_simulator == False``.
    """
    return creator(
        name=name,
        reals=dict(
            omega="NO_IOC:m1",
            chi="NO_IOC:m2",
            phi="NO_IOC:m3",
            tth="NO_IOC:m4",
        ),
    )


def _e6c_with_extras(name="e6c", h2=0.0, k2=0.0, l2=0.0, psi=0.0):
    """Return an E6C diffractometer with the named extras pre-set."""
    sim = creator(name=name, geometry="E6C")
    # psi_constant is the canonical mode that exposes h2/k2/l2/psi extras.
    sim.core.mode = "psi_constant_vertical"
    sim.core.extras = dict(h2=h2, k2=k2, l2=l2, psi=psi)
    return sim


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(reals=None, expected=True),
            does_not_raise(),
            id="default-creator-is-simulator",
        ),
        pytest.param(
            dict(
                reals=dict(
                    omega="NO_IOC:m1",
                    chi="NO_IOC:m2",
                    phi="NO_IOC:m3",
                    tth="NO_IOC:m4",
                ),
                expected=False,
            ),
            does_not_raise(),
            id="all-epicsmotor-reals-not-simulator",
        ),
        pytest.param(
            dict(
                reals=dict(
                    omega=None,
                    chi=None,
                    phi="NO_IOC:m3",
                    tth=None,
                ),
                expected=False,
            ),
            does_not_raise(),
            id="mixed-reals-not-simulator",
        ),
    ],
)
def test_is_simulator_property(parms, context):
    with context:
        kwargs = dict(name="probe")
        if parms["reals"] is not None:
            kwargs["reals"] = parms["reals"]
        diff = creator(**kwargs)
        assert diff.is_simulator is parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(extras=dict(h2=1.0, k2=0.0, l2=0.0, psi=12.5)),
            does_not_raise(),
            id="e6c-psi-constant",
        ),
        pytest.param(
            dict(extras=dict(h2=0.0, k2=2.0, l2=0.0, psi=-5.0)),
            does_not_raise(),
            id="e6c-k2-only",
        ),
        pytest.param(
            dict(extras=dict(h2=1.5, k2=2.5, l2=3.5, psi=42.0)),
            does_not_raise(),
            id="e6c-mixed",
        ),
    ],
)
def test_extras_round_trip_simulator(parms, context, tmp_path):
    """#390: non-zero extras survive export -> restore on a simulator,
    and the values reach the solver (not just Core._extras)."""
    with context:
        orig = _e6c_with_extras(name="e6c_orig", **parms["extras"])
        cfg_file = tmp_path / "e6c.yml"
        orig.export(cfg_file, comment="extras round-trip")

        fresh = creator(name="e6c_fresh", geometry="E6C")
        fresh.core.mode = "psi_constant_vertical"
        # All defaults (simulator) restore extras.
        fresh.restore(cfg_file)

        # Core-side state matches.
        for axis, value in parms["extras"].items():
            assert fresh.core.extras[axis] == pytest.approx(value), (
                f"core extras: {axis=} got={fresh.core.extras[axis]!r}"
                f" expected={value!r}"
            )
        # Solver-side state matches: the dirty-flag fix means update_solver
        # has actually pushed the extras to the backend.
        fresh.core.update_solver()
        solver_extras = fresh.core.solver.extras
        for axis, value in parms["extras"].items():
            assert solver_extras[axis] == pytest.approx(value), (
                f"solver extras: {axis=} got={solver_extras[axis]!r} expected={value!r}"
            )


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(extras=dict(h2=1.0, k2=2.0, l2=0.0, psi=15.0)),
            does_not_raise(),
            id="extras-via-simulator-from-config",
        ),
    ],
)
def test_extras_round_trip_via_simulator_from_config(parms, context):
    """#390: simulator_from_config(diff) preserves non-zero extras."""
    from ...run_utils import simulator_from_config

    with context:
        orig = _e6c_with_extras(name="e6c_orig", **parms["extras"])
        sim = simulator_from_config(orig)

        for axis, value in parms["extras"].items():
            assert sim.core.extras[axis] == pytest.approx(value), (
                f"sim core extras: {axis=} got={sim.core.extras[axis]!r}"
                f" expected={value!r}"
            )
        sim.core.update_solver()
        for axis, value in parms["extras"].items():
            assert sim.core.solver.extras[axis] == pytest.approx(value)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(corrupt_axis="bogus_axis"),
            pytest.raises(ConfigurationError, match=re.escape("extra axis mismatch:")),
            id="unknown-axis-validated-early",
        ),
    ],
)
def test_extras_validation_raises_configuration_error(parms, context, tmp_path):
    """#390: unknown extras → clear ConfigurationError, not opaque KeyError."""
    with context:
        orig = _e6c_with_extras(name="e6c_orig", h2=1.0)
        cfg_file = tmp_path / "e6c_bad.yml"
        orig.export(cfg_file)

        cfg = load_yaml_file(cfg_file)
        # Inject an axis name the live solver does not know.
        cfg["axes"]["extra_axes"][parms["corrupt_axis"]] = 1.0

        fresh = creator(name="e6c_fresh", geometry="E6C")
        fresh.core.mode = "psi_constant_vertical"
        fresh.restore(cfg)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(restore_extras=False),
            does_not_raise(),
            id="explicit-restore-extras-False",
        ),
        pytest.param(
            dict(restore_extras=True),
            does_not_raise(),
            id="explicit-restore-extras-True",
        ),
    ],
)
def test_restore_extras_kwarg_honored(parms, context, tmp_path):
    """#390: ``restore_extras`` kwarg gates the extras restore."""
    with context:
        saved = dict(h2=4.0, k2=5.0, l2=6.0, psi=33.0)
        orig = _e6c_with_extras(name="e6c_orig", **saved)
        cfg_file = tmp_path / "e6c.yml"
        orig.export(cfg_file)

        fresh = creator(name="e6c_fresh", geometry="E6C")
        fresh.core.mode = "psi_constant_vertical"
        # Pre-set non-zero extras so we can detect whether they were touched.
        sentinel = dict(h2=99.0, k2=99.0, l2=99.0, psi=99.0)
        fresh.core.extras = sentinel

        fresh.restore(cfg_file, **parms)

        for axis in saved:
            if parms["restore_extras"]:
                assert fresh.core.extras[axis] == pytest.approx(saved[axis])
            else:
                assert fresh.core.extras[axis] == pytest.approx(sentinel[axis])


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="hardware-default-skips-dangerous-sections",
        ),
    ],
)
def test_restore_safe_defaults_on_hardware(parms, context, tmp_path):
    """#390: hardware-backed diffractometer skips dangerous sections by
    default and emits a single UserWarning."""
    with context:
        # Build a saved config from a simulator with non-default state.
        sim_src = creator(name="sim_src")
        sim_src.beam.wavelength.put(1.234)
        sim_src.add_sample("alt", 5.0)
        cfg_file = tmp_path / "sim_src.yml"
        sim_src.export(cfg_file)

        hw = _hardware_e4cv()
        assert hw.is_simulator is False
        original_wavelength = hw.beam.wavelength.get()
        original_sample_name = hw.sample.name
        original_sample_count = len(hw.samples)

        with pytest.warns(UserWarning, match=re.escape("hardware-backed")):
            hw.restore(cfg_file)

        # Wavelength NOT changed.
        assert hw.beam.wavelength.get() == pytest.approx(original_wavelength)
        # Sample NOT changed (samples block skipped).
        assert hw.sample.name == original_sample_name
        assert len(hw.samples) == original_sample_count


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                restore_samples=True,
                restore_wavelength=True,
                restore_extras=True,
                clear=True,
            ),
            does_not_raise(),
            id="explicit-opt-in-on-hardware",
        ),
    ],
)
def test_restore_explicit_opt_in_on_hardware(parms, context, tmp_path):
    """#390: explicit kwargs override safe defaults; no warning emitted."""
    import warnings

    with context:
        sim_src = creator(name="sim_src")
        sim_src.beam.wavelength.put(1.234)
        sim_src.add_sample("alt", 5.0)
        cfg_file = tmp_path / "sim_src.yml"
        sim_src.export(cfg_file)

        hw = _hardware_e4cv()
        assert hw.is_simulator is False

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Explicit opt-in must not trigger our hardware-backed warning.
            # (Other unrelated UserWarnings would also surface as errors;
            # there are none on this code path.)
            hw.restore(cfg_file, **parms)

        # All requested sections were applied.
        assert hw.beam.wavelength.get() == pytest.approx(1.234)
        assert "alt" in hw.samples


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="setter-on-hardware-is-conservative",
        ),
    ],
)
def test_configuration_setter_safe_on_hardware(parms, context, tmp_path):
    """#390: assigning ``diff.configuration = cfg`` on a hardware-backed
    diffractometer must not silently change inputs to the next forward()."""
    with context:
        sim_src = creator(name="sim_src")
        sim_src.beam.wavelength.put(1.234)
        sim_src.add_sample("alt", 5.0)
        cfg = sim_src.configuration

        hw = _hardware_e4cv()
        original_wavelength = hw.beam.wavelength.get()
        original_sample_name = hw.sample.name

        with pytest.warns(UserWarning, match=re.escape("hardware-backed")):
            hw.configuration = cfg

        assert hw.beam.wavelength.get() == pytest.approx(original_wavelength)
        assert hw.sample.name == original_sample_name


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(preset_mode="psi_constant_vertical", presets=dict(omega=12.34)),
            does_not_raise(),
            id="presets-round-trip",
        ),
    ],
)
def test_presets_round_trip(parms, context, tmp_path):
    """Mode presets survive export -> restore."""
    with context:
        orig = creator(name="e6c_orig", geometry="E6C")
        orig.core._mode_presets[parms["preset_mode"]] = {
            str(k): float(v) for k, v in parms["presets"].items()
        }
        cfg_file = tmp_path / "presets.yml"
        orig.export(cfg_file)

        fresh = creator(name="e6c_fresh", geometry="E6C")
        fresh.restore(cfg_file)

        restored = fresh.core._mode_presets.get(parms["preset_mode"], {})
        for axis, value in parms["presets"].items():
            assert restored.get(axis) == pytest.approx(value)
