"""
Regression test for issue #190.

Test the presets feature for constant-axis modes.

When computing ``forward()`` reflections using a mode that holds one or more
axes at constant values, the corresponding axes of the diffractometer can be
set to specific values using presets, independent of the current motor positions.
"""

from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..blocks.lattice import SI_LATTICE_PARAMETER
from ..diffract import creator
from .common import assert_context_result


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(phi_preset=45.0),
            does_not_raise(),
            id="preset phi=45",
        ),
        pytest.param(
            dict(phi_preset=0.0),
            does_not_raise(),
            id="preset phi=0",
        ),
        pytest.param(
            dict(phi_preset=90.0),
            does_not_raise(),
            id="preset phi=90",
        ),
    ],
)
def test_presets_get_set(parms, context):
    """
    Test that presets can be set and retrieved for the current mode.
    """
    with context as reason:
        e4cv = creator()
        e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
        e4cv.beam.wavelength.put(1.54)

        or1 = e4cv.add_reflection(
            (4, 0, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
            wavelength=1.54,
            name="r400",
        )
        or2 = e4cv.add_reflection(
            (0, 4, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
            wavelength=1.54,
            name="r040",
        )
        e4cv.core.calc_UB(or1, or2)

        e4cv.core.mode = "constant_phi"

        e4cv.core.presets = {"phi": parms["phi_preset"]}

        assert "phi" in e4cv.core.presets
        assert e4cv.core.presets["phi"] == parms["phi_preset"]

        assert e4cv.core._mode_presets["constant_phi"]["phi"] == parms["phi_preset"]

    assert_context_result(None, reason)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                mode1="constant_phi",
                mode2="constant_omega",
                phi_preset=45.0,
                omega_preset=30.0,
            ),
            does_not_raise(),
            id="presets per mode",
        ),
    ],
)
def test_presets_per_mode(parms, context):
    """
    Test that presets are remembered for each mode separately.
    """
    with context as reason:
        e4cv = creator()
        e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
        e4cv.beam.wavelength.put(1.54)

        or1 = e4cv.add_reflection(
            (4, 0, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
            wavelength=1.54,
            name="r400",
        )
        or2 = e4cv.add_reflection(
            (0, 4, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
            wavelength=1.54,
            name="r040",
        )
        e4cv.core.calc_UB(or1, or2)

        e4cv.core.mode = parms["mode1"]
        e4cv.core.presets = {"phi": parms["phi_preset"]}

        e4cv.core.mode = parms["mode2"]
        e4cv.core.presets = {"omega": parms["omega_preset"]}

        e4cv.core.mode = parms["mode1"]
        assert e4cv.core.presets.get("phi") == parms["phi_preset"]
        assert "omega" not in e4cv.core.presets

        e4cv.core.mode = parms["mode2"]
        assert e4cv.core.presets.get("omega") == parms["omega_preset"]
        assert "phi" not in e4cv.core.presets

    assert_context_result(None, reason)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                phi_motor=10.0,
                phi_preset=45.0,
            ),
            does_not_raise(),
            id="forward uses preset not motor",
        ),
    ],
)
def test_forward_uses_preset(parms, context):
    """
    Test that forward() uses the preset value, not the current motor position.
    """
    with context as reason:
        e4cv = creator()
        e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
        e4cv.beam.wavelength.put(1.54)

        or1 = e4cv.add_reflection(
            (4, 0, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
            wavelength=1.54,
            name="r400",
        )
        or2 = e4cv.add_reflection(
            (0, 4, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
            wavelength=1.54,
            name="r040",
        )
        e4cv.core.calc_UB(or1, or2)

        e4cv.core.mode = "constant_phi"

        e4cv.phi.move(parms["phi_motor"])

        e4cv.core.presets = {"phi": parms["phi_preset"]}

        for solution in e4cv.core.forward(dict(h=1, k=1, l=1)):
            assert_almost_equal(solution.phi, parms["phi_preset"], decimal=4)

    assert_context_result(None, reason)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                phi_motor=10.0,
                clear_mode="constant_phi",
            ),
            does_not_raise(),
            id="clear presets for mode",
        ),
        pytest.param(
            dict(
                phi_motor=10.0,
                clear_mode=None,
            ),
            does_not_raise(),
            id="clear all presets",
        ),
    ],
)
def test_clear_presets(parms, context):
    """
    Test that clear_presets() removes presets.
    """
    with context as reason:
        e4cv = creator()
        e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
        e4cv.beam.wavelength.put(1.54)

        or1 = e4cv.add_reflection(
            (4, 0, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
            wavelength=1.54,
            name="r400",
        )
        or2 = e4cv.add_reflection(
            (0, 4, 0),
            dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
            wavelength=1.54,
            name="r040",
        )
        e4cv.core.calc_UB(or1, or2)

        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = {"phi": 45.0}

        assert len(e4cv.core.presets) > 0

        e4cv.core.clear_presets(parms["clear_mode"])

        if parms["clear_mode"] is None:
            assert len(e4cv.core._mode_presets) == 0
        else:
            assert parms["clear_mode"] not in e4cv.core._mode_presets

    assert_context_result(None, reason)


def test_presets_in_config():
    """
    Test that presets are included in the configuration dictionary.
    """
    e4cv = creator()
    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.beam.wavelength.put(1.54)

    or1 = e4cv.add_reflection(
        (4, 0, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
        wavelength=1.54,
        name="r400",
    )
    or2 = e4cv.add_reflection(
        (0, 4, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
        wavelength=1.54,
        name="r040",
    )
    e4cv.core.calc_UB(or1, or2)

    e4cv.core.mode = "constant_phi"
    e4cv.core.presets = {"phi": 45.0}

    config = e4cv.core._asdict()

    assert "presets" in config
    assert "constant_phi" in config["presets"]
    assert config["presets"]["constant_phi"]["phi"] == 45.0


def test_presets_from_config():
    """
    Test that presets can be restored from a configuration dictionary.
    """
    e4cv = creator()
    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.beam.wavelength.put(1.54)

    or1 = e4cv.add_reflection(
        (4, 0, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
        wavelength=1.54,
        name="r400",
    )
    or2 = e4cv.add_reflection(
        (0, 4, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=90),
        wavelength=1.54,
        name="r040",
    )
    e4cv.core.calc_UB(or1, or2)

    config = {
        "_header": {},
        "name": e4cv.name,
        "axes": {
            "pseudo_axes": ["h", "k", "l"],
            "real_axes": ["omega", "chi", "phi", "tth"],
            "axes_xref": {"omega": "omega", "chi": "chi", "phi": "phi", "tth": "tth"},
            "extra_axes": {},
        },
        "digits": 4,
        "sample_name": "silicon",
        "samples": {},
        "constraints": {},
        "solver": {},
        "beam": {},
        "presets": {"constant_phi": {"phi": 45.0}},
    }

    e4cv.core._fromdict(config)

    e4cv.core.mode = "constant_phi"
    assert e4cv.core.presets.get("phi") == 45.0


def test_constant_axis_names():
    """
    Test that constant_axis_names returns the correct axes for the current mode.
    """
    e4cv = creator()
    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.beam.wavelength.put(1.54)

    e4cv.core.mode = "constant_phi"

    constant_axes = e4cv.core.constant_axis_names
    assert "phi" in constant_axes


def test_solver_constant_axis_names():
    """
    Test that solver_constant_axis_names returns the correct axes.
    """
    e4cv = creator()
    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.beam.wavelength.put(1.54)

    e4cv.core.mode = "constant_phi"

    constant_axes = e4cv.core.solver_constant_axis_names
    assert "phi" in constant_axes


def test_solver_written_axis_names():
    """
    Test that solver_written_axis_names returns the correct axes.
    """
    e4cv = creator()
    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.beam.wavelength.put(1.54)

    e4cv.core.mode = "constant_phi"

    written_axes = e4cv.core.solver_written_axis_names
    assert "phi" not in written_axes
    assert "omega" in written_axes
    assert "tth" in written_axes
