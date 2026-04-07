import math
import pathlib
import re
from contextlib import nullcontext as does_not_raise

import pytest

from ... import __version__
from ...diffract import DiffractometerBase
from ...diffract import creator
from ...misc import ConfigurationError
from ...misc import load_yaml_file
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
