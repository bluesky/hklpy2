"""
Tests for the NSLS-II 'tardis' (6-circle).
"""

from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..diffract import DiffractometerBase
from ..diffract import creator
from .common import TESTS_DIR
from .common import assert_context_result

TARDIS_CONFIG_YAML = TESTS_DIR / "tardis.yml"
TARDIS_SOLVER_MODE = "lifting_detector_mu"


@pytest.fixture
def tardis():
    """'tardis' is an E6C with renamed real axes."""
    diffractometer = creator(
        name="tardis",
        geometry="E6C",
        solver="hkl_soleil",
        reals=dict(
            mu=None,
            theta=None,
            chi=None,
            phi=None,
            delta=None,
            gamma=None,
        ),
    )
    diffractometer.beam.energy_units.put("eV")
    diffractometer.beam.wavelength_units.put("angstrom")
    diffractometer.core.mode = TARDIS_SOLVER_MODE
    diffractometer.core.constraints["gamma"].limits = -5, 180
    return diffractometer


def test_basic(tardis):
    context = does_not_raise()
    expected = None
    with context as reason:
        assert isinstance(tardis, DiffractometerBase)
        assert tardis.pseudo_axis_names == "h k l".split()
        assert tardis.real_axis_names == "mu theta chi phi delta gamma".split()
        assert tardis.core.mode == TARDIS_SOLVER_MODE
        assert tardis.core.constraints["gamma"].limits == (-5, 180)
        assert list(tardis.samples) == ["sample"]
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "sample, wavelength, ppos, rpos, digits, context, expected",
    [
        [
            "KCF",
            13.317314715359826,
            (0, 0, 1.1),
            (
                101.56806493825435,
                0.0,
                0.0,
                0.0,
                42.02226419522791,
                176.69158155966787,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.60911,
            (6, 0, 0),
            (
                60.993465989902,
                0.0,
                0.0,
                0.0,
                75.845217742871,
                -1.583950160798,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.60954,
            (3, 2, 0),
            (
                26.173823508277,
                0.0,
                0.0,
                0.0,
                53.052076190366,
                -0.843799584044,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.60954,
            (4, 5, 0),
            (
                42.549266257243,
                0.0,
                0.0,
                0.0,
                106.318942250347,
                -1.18540715326,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.60954,
            (5, 4, 0),
            (
                49.892321938069,
                0.0,
                0.0,
                0.0,
                106.320529873965,
                -1.42365604908,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.61198,
            (4, 1, 0),
            (
                40.220,
                0.0,
                0.0,
                0.0,
                56.097,
                -1.08366,
            ),
            3,
            does_not_raise(),
            None,
        ],
        [
            "esrf_sample",
            1.61198,
            (4, 4, 0),
            (
                38.3762,
                0.0,
                0.0,
                0.0,
                90.6303,
                -1.161318,
            ),
            3,
            does_not_raise(),
            None,
        ],
    ],
)
def test_restore_and_move(sample, wavelength, ppos, rpos, digits, context, expected):
    pytest.skip("FIXME: Rework tardis test_restore_and_move.")
    with context as reason:
        tardis = creator(
            name="tardis",
            geometry="E6C",
            solver="hkl_soleil",
            reals=dict(
                theta=None,
                mu=None,
                chi=None,
                phi=None,
                delta=None,
                gamma=None,
            ),
            labels=["tardis"],
        )
        assert isinstance(tardis, DiffractometerBase)
        tardis.restore(TARDIS_CONFIG_YAML, clear=True)

        assert tardis.beam.energy_units.get() == "eV"
        assert tardis.beam.wavelength_units.get() == "angstrom"

        # set the test parameters
        tardis.beam.wavelength.put(wavelength)
        tardis.sample = sample
        tardis.core.mode = TARDIS_SOLVER_MODE

        tardis.move(ppos)

        for axis, value in zip(tardis.pseudo_positioners, ppos):
            (
                assert_almost_equal(
                    axis.position,
                    value,
                    decimal=digits,
                ),
                f"{axis=} {value=}",
            )
        for axis, value in zip(tardis.real_positioners, rpos):
            (
                assert_almost_equal(
                    axis.position,
                    value,
                    decimal=digits,
                ),
                f"{axis=} {value=}",
            )

    assert_context_result(expected, reason)


def test_axis_inversion():
    pytest.skip("FIXME: Rework tardis test_axis_inversion.")
    tardis = creator(
        name="tardis",
        geometry="E6C",
        solver="hkl_soleil",
        reals=dict(
            theta=None,
            mu=None,
            chi=None,
            phi=None,
            delta=None,
            gamma=None,
        ),
        labels=["tardis"],
    )
    assert isinstance(tardis, DiffractometerBase)
    tardis.restore(TARDIS_CONFIG_YAML, clear=True)

    tardis.beam.wavelength.put(13.317314547644292)
    tardis.sample = "KCF"
    tardis.core.mode = TARDIS_SOLVER_MODE

    # ppos = (0, 0, 1.1)
    # rpos = (
    #     101.56806493825435,
    #     0.0,
    #     0.0,
    #     0.0,
    #     42.02226419522791,
    #     -176.69158155966787,  # invert gamma for this test
    # )

    # # TODO: #38 support inverted axes
    # # hklpy v1 code from here
    # tardis.calc.inverted_axes = ["gamma"]
    # tardis.calc.physical_positions = rpos

    # assert not tardis.calc["omega"].inverted
    # gamma = tardis.calc["gamma"]
    # assert gamma.inverted
    # assert_almost_equal(gamma.limits, (-180.0, 5.0))  # inverted from (-5, 180)
    # gamma.limits = (-180.0, 5.0)
    # assert_almost_equal(gamma.limits, (-180.0, 5.0))  # inverted from (-5, 180)

    # assert_almost_equal(tardis.calc.physical_positions, rpos)
    # assert_almost_equal(tardis.calc.inverse(rpos), ppos)
