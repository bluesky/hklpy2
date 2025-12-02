import math
import pathlib
from contextlib import nullcontext as does_not_raise

import pytest

from ... import __version__
from ...diffract import DiffractometerBase
from ...diffract import creator
from ...misc import ConfigurationError
from ...misc import load_yaml_file
from ...tests.common import assert_context_result
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
        ["_header.datetime", None],
        ["_header.hklpy2_version", __version__],
        ["_header.python_class", e4cv.__class__.__name__],
        ["axes.axes_xref", e4cv.core.axes_xref],
        ["axes.extra_axes", e4cv.core.all_extras],
        ["axes.pseudo_axes", e4cv.pseudo_axis_names],
        ["axes.real_axes", e4cv.real_axis_names],
        ["beam.energy_units", e4cv.beam.energy_units.get()],
        ["beam.energy", e4cv.beam.energy.get()],
        ["beam.source_type", e4cv.beam.source_type.get()],
        ["beam.wavelength_units", e4cv.beam.wavelength_units.get()],
        ["beam.wavelength", e4cv.beam.wavelength.get()],
        ["constraints.chi.high_limit", 180.2],
        ["constraints.omega.label", "omega"],
        ["constraints.tth.low_limit", -180.2],
        ["name", e4cv.name],
        ["sample_name", e4cv.sample.name],
        ["samples.sample.lattice.a", 1],
        ["samples.sample.lattice.alpha", 90],
        ["samples.sample.name", "sample"],
        ["samples.sample.reflections_order", []],
        ["samples.sample.reflections", {}],
        ["samples.sample.U", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
        ["samples.sample.UB", [[twopi, 0, 0], [0, twopi, 0], [0, 0, twopi]]],
        ["samples.vibranium.name", "vibranium"],
        ["samples.vibranium.reflections_order", "r040 r004".split()],
        ["samples.vibranium.reflections_order", "r040 r004".split()],
        ["samples.vibranium.reflections.r004.name", "r004"],
        ["samples.vibranium.reflections.r004.pseudos.h", 0],
        ["samples.vibranium.reflections.r004.pseudos.k", 0],
        ["samples.vibranium.reflections.r004.pseudos.l", 4],
        ["samples.vibranium.reflections.r004.reals.chi", 90],
        ["samples.vibranium.U", e4cv.sample.U],
        ["samples.vibranium.UB", e4cv.sample.UB],
        ["solver.engine", e4cv.core.solver.engine_name],
        ["solver.geometry", e4cv.core.geometry],
        ["solver.name", e4cv.core.solver_name],
        ["solver.real_axes", e4cv.core.solver_real_axis_names],
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
                assert cfg[field] != getattr(
                    constraint, field
                ), f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
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
    for refl in fourc.sample.reflections.order:
        assert refl in fourc.sample.reflections
    # TODO: compare reflections

    assert len(fourc.core.constraints) == len(config["constraints"])
    for key, constraint in fourc.core.constraints.items():
        assert key in config["constraints"]
        cfg = config["constraints"][key]
        assert cfg["class"] == constraint.__class__.__name__
        for field in constraint._fields:
            assert field in cfg, f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"
            assert cfg[field] == getattr(
                constraint, field
            ), f"{key=!r}  {field=!r}  {constraint=!r}  {cfg=!r}"


@pytest.mark.parametrize(
    "diffractometer, clear, restore, file, context, expected",
    [
        [e4cv, True, True, E4CV_CONFIG_FILE, does_not_raise(), None],
        [e4cv, True, False, E4CV_CONFIG_FILE, does_not_raise(), None],
        [
            sim2c,
            True,
            True,
            E4CV_CONFIG_FILE,
            pytest.raises(ConfigurationError),
            "solver mismatch",
        ],
        [
            sim2c,
            True,
            True,
            "this file does not exist",
            pytest.raises(FileExistsError),
            "this file does not exist",
        ],
        [None, True, True, E4CV_CONFIG_FILE, pytest.raises(AssertionError), "False"],
    ],
)
def test_restore(diffractometer, clear, restore, file, context, expected):
    with context as reason:
        assert isinstance(diffractometer, DiffractometerBase)
        diffractometer.restore(
            file,
            clear=clear,
            restore_constraints=restore,
        )
    assert_context_result(expected, reason)
