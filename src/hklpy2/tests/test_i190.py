"""
Regression test for issue #190.

Test the presets feature for constant-axis modes.

When computing ``forward()`` reflections using a mode that holds one or more
axes at constant values, the corresponding axes of the diffractometer can be
set to specific values using presets, independent of the current motor positions.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest
from numpy.testing import assert_almost_equal

from ..blocks.lattice import SI_LATTICE_PARAMETER
from ..diffract import creator
from ..exceptions import NoForwardSolutions


@pytest.fixture()
def e4cv():
    """Create an E4CV diffractometer with UB matrix."""
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
    return e4cv


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
        pytest.param(
            dict(phi_preset="not_a_number"),
            pytest.raises(
                ValueError,
                match=re.escape("could not convert string to float: 'not_a_number'"),
            ),
            id="preset phi=string raises ValueError",
        ),
    ],
)
def test_presets_get_set(e4cv, parms, context):
    """
    Test that presets can be set and retrieved for the current mode.
    """
    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = {"phi": parms["phi_preset"]}

        assert "phi" in e4cv.core.presets
        assert e4cv.core.presets["phi"] == parms["phi_preset"]
        assert e4cv.core._mode_presets["constant_phi"]["phi"] == parms["phi_preset"]


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
def test_presets_per_mode(e4cv, parms, context):
    """
    Test that presets are remembered for each mode separately.
    """
    with context:
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


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(phi_motor=10.0, phi_preset=45.0),
            does_not_raise(),
            id="forward uses preset not motor",
        ),
        pytest.param(
            dict(phi_preset=45.0, phi_limits=(80, 100)),
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="forward NoSolutions: preset phi excluded by constraint",
        ),
        pytest.param(
            dict(phi_preset=45.0, tth_limits=(0, 5)),
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="forward NoSolutions: tth constraint excludes solution",
        ),
        pytest.param(
            dict(
                phi_preset=45.0,
                hkl=dict(h=100, k=100, l=100),
            ),
            pytest.raises(NoForwardSolutions, match=re.escape("No solutions.")),
            id="forward NoSolutions: unreachable hkl",
        ),
    ],
)
def test_forward_with_presets(e4cv, parms, context):
    """
    Test that forward() uses the preset value, and raises NoForwardSolutions
    when the preset is excluded by constraints.
    """
    with context:
        e4cv.core.mode = "constant_phi"

        if "phi_limits" in parms:
            e4cv.core.constraints["phi"].limits = parms["phi_limits"]

        if "tth_limits" in parms:
            e4cv.core.constraints["tth"].limits = parms["tth_limits"]

        if "phi_motor" in parms:
            e4cv.phi.move(parms["phi_motor"])

        e4cv.core.presets = {"phi": parms["phi_preset"]}

        hkl = parms.get("hkl", dict(h=1, k=1, l=1))
        result = e4cv.forward(hkl)
        assert_almost_equal(result.phi, parms["phi_preset"], decimal=4)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(first={"phi": 45.0}, second={"phi": 90.0}, expected={"phi": 90.0}),
            does_not_raise(),
            id="second assignment replaces first",
        ),
        pytest.param(
            dict(first={"phi": 45.0}, second={}, expected={}),
            does_not_raise(),
            id="assign empty dict clears presets",
        ),
        pytest.param(
            dict(
                first={"phi": 45.0},
                second={"phi": 90.0, "omega": 10.0},
                expected={"phi": 90.0},
            ),
            does_not_raise(),
            id="non-constant axes silently dropped on replace",
        ),
    ],
)
def test_presets_replace_semantics(e4cv, parms, context):
    """
    Test that the presets setter replaces (not merges) the dict.
    """
    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = parms["first"]
        e4cv.core.presets = parms["second"]
        assert e4cv.core.presets == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="presets in config",
        ),
    ],
)
def test_presets_in_config(e4cv, parms, context):
    """
    Test that presets are included in the configuration dictionary.
    """
    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = {"phi": 45.0}

        config = e4cv.core._asdict()

        assert "presets" in config
        assert "constant_phi" in config["presets"]
        assert config["presets"]["constant_phi"]["phi"] == 45.0


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="presets from config",
        ),
    ],
)
def test_presets_from_config(e4cv, parms, context):
    """
    Test that presets can be restored from a configuration dictionary.
    """
    with context:
        config = {
            "_header": {},
            "name": e4cv.name,
            "axes": {
                "pseudo_axes": ["h", "k", "l"],
                "real_axes": ["omega", "chi", "phi", "tth"],
                "axes_xref": {
                    "omega": "omega",
                    "chi": "chi",
                    "phi": "phi",
                    "tth": "tth",
                },
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


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode="constant_phi", expected=["phi"]),
            does_not_raise(),
            id="constant_phi mode",
        ),
        pytest.param(
            dict(mode="constant_omega", expected=["omega"]),
            does_not_raise(),
            id="constant_omega mode",
        ),
        pytest.param(
            dict(mode="constant_chi", expected=["chi"]),
            does_not_raise(),
            id="constant_chi mode",
        ),
        pytest.param(
            dict(mode="bissector", expected=[]),
            does_not_raise(),
            id="bissector mode no constants",
        ),
    ],
)
def test_constant_axis_names(e4cv, parms, context):
    """
    Test that constant_axis_names returns the correct axes for each mode.
    """
    with context:
        e4cv.core.mode = parms["mode"]

        constant_axes = e4cv.core.constant_axis_names
        assert constant_axes == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode="constant_phi", expected=["phi"]),
            does_not_raise(),
            id="constant_phi solver",
        ),
        pytest.param(
            dict(mode="bissector", expected=[]),
            does_not_raise(),
            id="bissector solver",
        ),
    ],
)
def test_solver_constant_axis_names(e4cv, parms, context):
    """
    Test that solver_constant_axis_names returns the correct axes.
    """
    with context:
        e4cv.core.mode = parms["mode"]

        constant_axes = e4cv.core.solver_constant_axis_names
        assert constant_axes == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(mode="constant_phi", expected=["omega", "chi", "tth"]),
            does_not_raise(),
            id="constant_phi written",
        ),
        pytest.param(
            dict(mode="bissector", expected=["omega", "chi", "phi", "tth"]),
            does_not_raise(),
            id="bissector written",
        ),
        pytest.param(
            dict(
                mode="bissector",
                expected=["omega", "chi", "phi", "tth"],
                hide_axes_w=True,
            ),
            does_not_raise(),
            id="fallback when solver lacks axes_w",
        ),
    ],
)
def test_solver_written_axis_names(e4cv, parms, context):
    """
    Test that solver_written_axis_names returns the correct axes.
    """
    with context:
        e4cv.core.mode = parms["mode"]

        if parms.get("hide_axes_w"):
            solver_class = type(e4cv.core.solver)
            original = solver_class.axes_w
            delattr(solver_class, "axes_w")
            try:
                written_axes = e4cv.core.solver_written_axis_names
            finally:
                solver_class.axes_w = original
        else:
            written_axes = e4cv.core.solver_written_axis_names
        assert written_axes == parms["expected"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(presets={"omega": 100.0, "phi": 45.0}, expected_phi=45.0),
            does_not_raise(),
            id="ignore computed, keep constant",
        ),
        pytest.param(
            dict(presets={"phi": "not_a_number"}),
            pytest.raises(
                ValueError,
                match=re.escape("could not convert string to float: 'not_a_number'"),
            ),
            id="non-numeric preset value raises ValueError",
        ),
    ],
)
def test_presets_ignored_for_computed_axes(e4cv, parms, context):
    """
    Test that presets for computed (non-constant) axes are ignored.
    """
    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = parms["presets"]

        assert "omega" not in e4cv.core.presets
        assert "phi" in e4cv.core.presets
        assert e4cv.core.presets["phi"] == parms["expected_phi"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(presets=[0.0, 0.0, 45.0, 0.0], expected_phi=45.0),
            does_not_raise(),
            id="list input with all axes",
        ),
        pytest.param(
            dict(presets=[1.0, 2.0]),
            pytest.raises(
                ValueError,
                match=re.escape("Expected at least 4 axes, received 2."),
            ),
            id="list input too short raises ValueError",
        ),
        pytest.param(
            dict(presets=[1.0, "bad", 3.0, 4.0]),
            pytest.raises(
                TypeError,
                match=re.escape("Expected a number. Received: 'bad'."),
            ),
            id="list input non-numeric raises TypeError",
        ),
        pytest.param(
            dict(presets="bad_input"),
            pytest.raises(
                TypeError,
                match=re.escape(
                    "Unexpected type: 'bad_input'.  Expected 'AnyAxesType'."
                ),
            ),
            id="string input raises TypeError",
        ),
    ],
)
def test_presets_with_list_input(e4cv, parms, context):
    """
    Test that presets can be set with list/tuple input (all axes).
    """
    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = parms["presets"]

        assert e4cv.core.presets.get("phi") == parms["expected_phi"]


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(),
            does_not_raise(),
            id="empty initially",
        ),
    ],
)
def test_presets_empty_dict_initially(e4cv, parms, context):
    """
    Test that presets is empty dict initially.
    """
    with context:
        e4cv.core.mode = "constant_phi"

        assert e4cv.core.presets == {}


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(phi_preset=45.0),
            does_not_raise(),
            id="wh shows presets",
        ),
    ],
)
def test_presets_wh_output(e4cv, parms, context):
    """
    Test that presets are reported in wh() full output.
    """
    import io
    import sys

    with context:
        e4cv.core.mode = "constant_phi"
        e4cv.core.presets = {"phi": parms["phi_preset"]}

        captured = io.StringIO()
        sys.stdout = captured
        e4cv.wh(full=True)
        sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "presets:" in output
        assert "phi" in output
        assert "45" in output
