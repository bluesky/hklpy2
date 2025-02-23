"""
Tests for the NSLS-II 'tardis' (6-circle).
"""

from contextlib import nullcontext as does_not_raise

import pytest

from ..diffract import DiffractometerBase
from ..geom import creator
from ..operations.misc import load_yaml_file
from .common import TESTS_DIR
from .common import assert_context_result

ORIENTED_KCF_SAMPLE_CONFIG_YAML = TESTS_DIR / "tardis-kcf.yml"
TARDIS_TEST_MODE = "lifting_detector_mu"


@pytest.fixture
def tardis():
    """'tardis' is an E6C with renamed real axes."""
    diffractometer = creator(
        name="tardis",
        geometry="E6C",
        solver="hkl_soleil",
        reals=dict(
            theta=None,  # TODO: verify, this is the ONLY renamed axis?
            omega=None,
            chi=None,
            phi=None,
            delta=None,
            gamma=None,
        ),
    )
    diffractometer._wavelength.energy_units = "eV"
    diffractometer._wavelength.wavelength_units = "angstrom"
    diffractometer.operator.solver.mode = TARDIS_TEST_MODE
    diffractometer.operator.constraints["gamma"].limits = -5, 180
    return diffractometer


def test_basic(tardis):
    context = does_not_raise()
    expected = None
    with context as reason:
        assert isinstance(tardis, DiffractometerBase)
        assert tardis.pseudo_axis_names == "h k l".split()
        assert tardis.real_axis_names == "theta omega chi phi delta gamma".split()
        assert tardis.operator.solver.mode == TARDIS_TEST_MODE
        assert tardis.operator.constraints["gamma"].limits == (-5, 180)
        assert list(tardis.samples) == ["sample"]
    assert_context_result(expected, reason)


def test_oriented_kcf(tardis):
    context = does_not_raise()
    expected = None
    with context as reason:
        assert isinstance(tardis, DiffractometerBase)
        tardis.operator.restore(ORIENTED_KCF_SAMPLE_CONFIG_YAML, clear=True)
        assert tardis.operator.solver.mode == TARDIS_TEST_MODE
        assert tardis.operator.constraints["gamma"].limits == (-5, 180)
        assert tardis._wavelength.wavelength_units == "angstrom"
        assert tardis._wavelength.energy_units == "eV"
        assert list(tardis.samples) == ["sample", "KCF"]
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "config, ppos, rpos, context, expected",
    [
        [
            ORIENTED_KCF_SAMPLE_CONFIG_YAML,
            (0, 0, 1.1),
            (
                101.56806493825435,
                0.0,
                0.0,
                0.0,
                42.02226419522791,
                176.69158155966787,
            ),
            does_not_raise(),
            None,
        ],
    ],
)
def test_reachable(tardis, config, ppos, rpos, context, expected):
    with context as reason:
        assert isinstance(tardis, DiffractometerBase)
        tardis.operator.restore(config, clear=True)

        assert tardis._wavelength.energy_units == "eV"
        assert tardis._wavelength.wavelength_units == "angstrom"

        if tardis.wavelength.get() == 1:  # FIXME: wavelength & units not restored
            tardis.wavelength.put(13.317314715359826)
        assert round(tardis._wavelength.energy, 3) == 931.0
        assert round(tardis.wavelength.get(), 3) == 13.317
        assert tardis.operator.sample.name == "KCF"

        # FIXME: UB not restored
        cfg = load_yaml_file(config)
        tardis.operator.solver.UB = cfg["samples"]["KCF"]["UB"]
        assert round(tardis.operator.solver.UB[0][0], 6) == 0.728792

        tardis.move(ppos)
        assert round(tardis.h.position, 3) == 0
        assert round(tardis.k.position, 3) == 0
        assert round(tardis.l.position, 3) == 1.1
        assert round(tardis.theta.position, 3) == 101.568
        assert round(tardis.omega.position, 3) == 0
        assert round(tardis.chi.position, 3) == 0
        assert round(tardis.phi.position, 3) == 0
        assert round(tardis.delta.position, 3) == -137.978
        assert round(tardis.gamma.position, 3) == 3.308

    assert_context_result(expected, reason)
