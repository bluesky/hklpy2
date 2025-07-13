"""Test the hklpy2.diffract module."""

import math
from collections import deque
from collections import namedtuple
from contextlib import nullcontext as does_not_raise

import bluesky
import pytest
from numpy.testing import assert_almost_equal
from ophyd import EpicsMotor
from ophyd import SoftPositioner
from ophyd.signal import AttributeSignal
from ophyd.sim import noisy_det

from ..backends.base import SolverBase
from ..blocks.reflection import ReflectionError
from ..blocks.sample import Sample
from ..diffract import DiffractometerBase
from ..diffract import creator
from ..diffract import diffractometer_class_factory
from ..misc import ConfigurationError
from ..misc import DiffractometerError
from ..misc import SolverNoForwardSolutions
from ..ops import DEFAULT_SAMPLE_NAME
from ..ops import Core
from ..ops import CoreError
from .common import HKLPY2_DIR
from .common import assert_context_result
from .models import AugmentedFourc
from .models import Fourc
from .models import MultiAxis99
from .models import MultiAxis99NoSolver
from .models import NoOpTh2Th
from .models import TwoC


@pytest.mark.parametrize(
    "config_file",
    ["e4cv_orient.yml", "fourc-configuration.yml"],
)
@pytest.mark.parametrize(
    "pseudos, reals, positioner_class, context, expected",
    [
        [
            [],
            dict(omega="IOC:m1", chi="IOC:m2", phi="IOC:m3", tth="IOC:m4"),
            EpicsMotor,
            does_not_raise(),
            None,
        ],
        [
            [],
            dict(aaa=None, bbb=None, ccc=None),
            SoftPositioner,
            pytest.raises(KeyError),
            "tth",
        ],
        [
            [],
            dict(aaa=None, bbb=None, ccc=None, ddd=None),
            SoftPositioner,
            does_not_raise(),
            None,
        ],
        [
            [],
            dict(aaa=None, bbb=None, ccc=None, ddd=None, eee=None),
            SoftPositioner,
            does_not_raise(),
            None,
        ],
        [
            [],
            dict(aaa="IOC:m1", bbb=None, ccc=None, ddd=None, eee=None),
            (EpicsMotor, SoftPositioner),
            does_not_raise(),
            None,
        ],
        [[], {}, SoftPositioner, does_not_raise(), None],
        [
            "h k".split(),
            {},
            SoftPositioner,
            pytest.raises(ConfigurationError),
            "pseudo axis mismatch",
        ],
        [
            "h2 k2 l2 psi alpha beta".split(),
            {},
            SoftPositioner,
            pytest.raises(ConfigurationError),
            "pseudo axis mismatch",
        ],
    ],
)
def test_creator_reals(
    pseudos, reals, positioner_class, context, expected, config_file
):
    with context as reason:
        sim = creator(pseudos=pseudos, reals=reals)
        assert sim is not None
        for axis in sim.real_axis_names:
            if len(reals) > 0:
                assert axis in reals
            assert isinstance(getattr(sim, axis), positioner_class)
        sim.restore(HKLPY2_DIR / "tests" / config_file)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "setup, context, expected",
    [
        [{}, does_not_raise(), None],
        [dict(forward_solution_function=None), does_not_raise(), None],
        [
            dict(forward_solution_function="hklpy2.misc.pick_closest_solution"),
            does_not_raise(),
            None,
        ],
        [
            dict(forward_solution_function="hklpy2.misc.pick_first_solution"),
            does_not_raise(),
            None,
        ],
        [
            dict(forward_solution_function="cannot.find.this.function"),
            pytest.raises(ModuleNotFoundError),
            "No module named 'cannot'",
        ],
    ],
)
def test_creator_setup(setup, context, expected):
    with context as reason:
        sim = creator(**setup)
        assert sim is not None
    assert_context_result(expected, reason)


def test_DiffractometerBase():
    with pytest.raises((DiffractometerError, ValueError)) as reason:
        DiffractometerBase(name="dbase")
    assert_context_result("Must have at least 1 positioner", reason)


@pytest.mark.parametrize("axis", "h k l".split())
@pytest.mark.parametrize(
    "value, context, expected",
    [
        [-1, does_not_raise(), None],
        [1, does_not_raise(), None],
        [-1.2, does_not_raise(), None],
        [1.2, does_not_raise(), None],
        [12, pytest.raises(SolverNoForwardSolutions), "No forward solutions found."],
        [-12, pytest.raises(SolverNoForwardSolutions), "No forward solutions found."],
    ],
)
def test_limits(axis, value, context, expected):
    with context as reason:
        sim = creator()
        assert hasattr(sim, axis)
        pseudo = getattr(sim, axis)
        assert pseudo.limits == (0, 0)
        assert pseudo.check_value(value) is None

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "base, pseudos, reals, context, expected",
    [
        [Fourc, "h k l".split(), "omega chi phi tth".split(), does_not_raise(), None],
        [
            AugmentedFourc,
            "h k l".split(),
            "omega chi phi tth".split(),
            does_not_raise(),
            None,
        ],
        [
            MultiAxis99NoSolver,
            "p1 p2".split(),
            "r1 r2 r3 r4".split(),
            pytest.raises(AssertionError),
            "where False = isinstance(None, SolverBase)",  # no solver
        ],
        [
            MultiAxis99,
            "p1 p2".split(),
            "r1 r2 r3 r4".split(),
            does_not_raise(),
            None,
        ],
        [NoOpTh2Th, "q".split(), "th tth".split(), does_not_raise(), None],
        [TwoC, "q".split(), "theta ttheta".split(), does_not_raise(), None],
    ],
)
def test_diffractometer_class_models(base, pseudos, reals, context, expected):
    with context as reason:
        sim = base(name="sim")
        assert isinstance(sim, DiffractometerBase)
        assert isinstance(sim.sample, Sample)
        assert isinstance(sim.core, Core)
        assert not sim.moving

        # These property methods _must_ work immediately.
        assert isinstance(sim.position, tuple), f"{type(sim.position)=!r}"
        assert isinstance(sim.real_position, tuple), f"{type(sim.real_position)=!r}"
        assert isinstance(sim.report, dict), f"{type(sim.report)=!r}"

        assert [axis.attr_name for axis in sim._pseudo] == pseudos
        assert [axis.attr_name for axis in sim._real] == reals
        assert list(sim.position._fields) == pseudos
        assert list(sim.real_position._fields) == reals
        assert len(sim.pseudo_axis_names) >= len(pseudos)
        assert len(sim.real_axis_names) >= len(reals)

        assert len(sim.samples) == 1
        assert sim.sample.name == "sample"  # default sample

        sim.core.add_sample("test", 5)
        assert len(sim.samples) == 2
        assert sim.sample.name == "test"

        assert isinstance(sim.core.solver, SolverBase)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "specs, full, mode, config_file, output, context, expected",
    [
        [
            {},
            False,
            None,
            None,
            [
                "wavelength=1.0",
                "h=0, k=0, l=0",
                "omega=0, chi=0, phi=0, tth=0",
            ],
            does_not_raise(),
            None,
        ],
        [
            {},
            True,
            "psi_constant",
            None,
            [
                "diffractometer='e4cv'",
                "HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')",
                "Sample(name='sample', lattice=Lattice(a=1, system='cubic'))",
                "Orienting reflections: []",
                "U=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
                "UB=[[6.283185307179586, 0.0, 0.0], [0.0, 6.283185307179586, 0.0], [0.0, 0.0, 6.283185307179586]]",
                "constraint: -180.0 <= omega <= 180.0",
                "constraint: -180.0 <= chi <= 180.0",
                "constraint: -180.0 <= phi <= 180.0",
                "constraint: -180.0 <= tth <= 180.0",
                "Mode: psi_constant",
                "beam={'class': 'WavelengthXray', 'source_type': 'Synchrotron X-ray Source', 'energy': 12.398419843856837, 'wavelength': 1.0, 'energy_units': 'keV', 'wavelength_units': 'angstrom'}",
                "h=0, k=0, l=0",
                "omega=0, chi=0, phi=0, tth=0",
                "h2=0 k2=0 l2=0 psi=0",
            ],
            does_not_raise(),
            None,
        ],
        [
            {},
            True,
            None,
            HKLPY2_DIR / "tests" / "e4cv_orient.yml",
            [
                "diffractometer='e4cv'",
                "HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')",
                "Sample(name='vibranium', lattice=Lattice(a=6.2832, system='cubic'))",
                "Reflection(name='r400', h=4, k=0, l=0)",
                "Reflection(name='r040', h=0, k=4, l=0)",
                "Reflection(name='r004', h=0, k=0, l=4)",
                "Orienting reflections: ['r040', 'r004']",
                "U=[[0.000279252677, -0.999999961009, -2.2e-11], [-7.7982e-08, 0.0, -1], [0.999999961009, 0.000279252677, -7.7982e-08]]",
                "UB=[[0.000279252677, -0.999999961009, -2.2e-11], [-7.7982e-08, 0.0, -1], [0.999999961009, 0.000279252677, -7.7982e-08]]",
                "constraint: -180.2 <= omega <= 180.2",
                "constraint: -180.2 <= chi <= 180.2",
                "constraint: -180.2 <= phi <= 180.2",
                "constraint: -180.2 <= tth <= 180.2",
                "Mode: bissector",
            ],
            does_not_raise(),
            None,
        ],
        [
            {"reals": "omega chi phi tth huey dewey louie".split()},
            False,
            None,
            None,
            [
                "wavelength=1.0",
                "h=0, k=0, l=0",
                "omega=0, chi=0, phi=0, tth=0",
                "huey=0, dewey=0, louie=0",
            ],
            does_not_raise(),
            None,
        ],
    ],
)
def test_diffractometer_wh(
    specs, full, mode, config_file, output, context, expected, capsys
):
    from ..diffract import creator

    with context as reason:
        gonio = creator(**specs)
        if config_file is not None:
            gonio.restore(config_file)
        if mode is not None:
            gonio.core.mode = mode

        gonio.wh(full=full)
        captured = capsys.readouterr()
        lines = captured.out.splitlines()
        for out in output:
            assert out in lines, f"{out=!r} {lines=}"

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "mode, keys, context, expected, config_file",
    [
        [
            "bissector",
            "h k l omega chi phi tth".split(),
            does_not_raise(),
            None,
            "e4cv_orient.yml",
        ],
        [
            "bissector",
            "h k l omega chi phi tth".split(),
            does_not_raise(),
            None,
            "fourc-configuration.yml",
        ],
    ],
)
def test_full_position(mode, keys, context, expected, config_file):
    from ..diffract import creator

    assert config_file.endswith(".yml")

    with context as reason:
        fourc = creator()
        fourc.restore(HKLPY2_DIR / "tests" / config_file)
        fourc.core.mode = mode
        pos = fourc.full_position()
        assert isinstance(pos, dict)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "pseudos, extras, mode, context, expected",
    [
        [
            dict(h=1, k=1, l=0),
            dict(h2=1, k2=1, l2=1, psi=0),
            "psi_constant",
            does_not_raise(),
            None,
        ],
    ],
)
def test_move_forward_with_extras(pseudos, extras, mode, context, expected):
    from ..diffract import creator

    fourc = creator()
    fourc.restore(HKLPY2_DIR / "tests" / "e4cv_orient.yml")
    fourc.core.mode = mode
    assert fourc.core.mode == mode

    RE = bluesky.RunEngine()

    with context as reason:
        RE(fourc.move_forward_with_extras(pseudos, extras))

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "pos, context, expected",
    [
        [(1, 2, 3, 4), does_not_raise(), None],
        [{"omega": 1, "chi": 2, "phi": 3, "tth": 4}, does_not_raise(), None],
        [
            namedtuple(
                "RealTuple", "omega chi phi tth".split(), defaults=[1, 2, 3, 4]
            )(),
            does_not_raise(),
            None,
        ],
        [(1, 2, 3, 4, 5, 6), pytest.raises(ValueError), "too many args"],
        [
            {"omega": 1, "chi": 2, "phi": 3, "delta": -4},
            pytest.raises(TypeError),
            "unexpected keyword argument 'delta'",
        ],
    ],
)
def test_move_reals(pos, context, expected):
    from ..diffract import creator

    fourc = creator()
    with context as reason:
        fourc.move_reals(pos)

    assert_context_result(expected, reason)


def test_null_core():
    """Tests special cases when diffractometer.core is None."""
    from ..diffract import creator

    fourc = creator()
    assert fourc.core is not None
    assert len(fourc.samples) > 0
    assert fourc.sample is not None
    assert fourc.core.solver is not None

    fourc.core._solver = None
    assert len(fourc.samples) > 0
    assert fourc.sample is not None

    fourc.core = None
    assert len(fourc.samples) == 0
    assert fourc.sample is None


def test_orientation():
    from ..blocks.lattice import SI_LATTICE_PARAMETER
    from ..diffract import creator

    fourc = creator()
    fourc.add_sample("silicon", SI_LATTICE_PARAMETER)
    fourc.beam.wavelength.put(1.0)
    assert math.isclose(
        fourc.beam.wavelength.get(), 1.0, abs_tol=0.01
    ), f"{fourc.beam.wavelength.get()=!r}"

    fourc.add_reflection(
        (4, 0, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
        wavelength=1.54,
        name="(400)",
    )
    fourc.add_reflection(
        (0, 4, 0),
        dict(tth=69.0966, omega=-145.451, chi=90, phi=0),
        wavelength=1.54,
        name="(040)",
    )

    assert math.isclose(
        fourc.beam.wavelength.get(), 1.0, abs_tol=0.01
    ), f"{fourc.beam.wavelength.get()=!r}"
    assert fourc.sample.reflections.order == "(400) (040)".split()

    result = fourc.core.calc_UB(*fourc.sample.reflections.order)
    assert isinstance(result, list)
    assert isinstance(result[0], list)
    assert isinstance(result[0][0], (float, int))

    UB = fourc.sample.UB
    assert len(UB) == 3

    e = -1.157
    UBe = [[0, 0, e], [0, e, 0], [e, 0, 0]]
    assert_almost_equal(UB, UBe, 3)

    result = fourc.forward(4, 0, 0, wavelength=1.54)
    reals = [-145.4509, 0, 0, 69.0982]  # at wavelength = 1.54
    assert_almost_equal(list(result._asdict().values()), reals, 3)

    result = fourc.forward(4, 0, 0, wavelength=1)
    reals = [-158.3920, 0, 0, 43.2161]
    assert_almost_equal(list(result._asdict().values()), reals, 3)

    assert math.isclose(  # still did not change the diffractometer wavelength
        fourc.beam.wavelength.get(), 1.0, abs_tol=0.01
    ), f"{fourc.beam.wavelength.get()=!r}"

    fourc.beam.wavelength.put(1.54)
    result = fourc.inverse(-145, 0, 0, 70)
    pseudos = [4.0456, 0, 0]  # at wavelength = 1.54
    assert_almost_equal(list(result._asdict().values()), pseudos, 3)

    result = fourc.inverse(-145, 0, 0, 70, wavelength=1)
    pseudos = [6.2302, 0, 0]
    assert_almost_equal(list(result._asdict().values()), pseudos, 3)


def test_remove_sample():
    sim = NoOpTh2Th(name="sim")
    assert len(sim.samples) == 1
    try:
        sim.core.remove_sample(DEFAULT_SAMPLE_NAME)
    except CoreError as reason:
        assert_context_result("Cannot remove last sample.", reason)
    assert len(sim.samples) == 1


@pytest.mark.parametrize(
    "name, pseudos, reals, wavelength, replace, num, context, expected",
    [
        ["(100)", (1, 0, 0), (10, 0, 0, 20), 1, True, 1, does_not_raise(), None],
        [
            "(100)",
            (1, 0, 0),
            (10, 0, 0, 20),
            1,
            False,
            1,
            pytest.raises(ReflectionError),
            "Use 'replace=True' to overwrite.",
        ],
        ["r2", (1, 0, 0), (10, 0, 0, 20), 1, True, 1, does_not_raise(), None],
        ["r2", (2, 0, 0), (10, 0, 0, 20), 1, False, 2, does_not_raise(), None],
        ["r2", (1, 0, 0), (10, 10, 0, 20), 1, False, 2, does_not_raise(), None],
        ["(100)", (1, 0, 0), (10, 10, 0, 20), 1, True, 1, does_not_raise(), None],
        [
            "r2",  # different name
            (1, 0, 0),  # same data
            (10, 0, 0, 20),  # same data
            1,  # same data
            False,
            1,
            pytest.raises(ReflectionError),
            "Use 'replace=True' to overwrite.",
        ],
        [
            "r2",  # different name
            (1, 0, 0),  # same data
            (10, 0, 0, 20),  # same data
            1.5,  # different data
            False,
            2,
            does_not_raise(),
            None,
        ],
    ],
)
def test_repeated_reflections(
    name, pseudos, reals, wavelength, replace, num, context, expected
):
    from ..diffract import creator

    e4cv = creator()
    e4cv.add_reflection(
        dict(h=1, k=0, l=0),
        dict(omega=10, chi=0, phi=0, tth=20),
        wavelength=1.0,
        name="(100)",
    )
    assert len(e4cv.sample.reflections) == 1

    with context as reason:
        e4cv.add_reflection(
            pseudos,
            reals,
            name=name,
            wavelength=wavelength,
            replace=replace,
        )
    assert_context_result(expected, reason)
    assert len(e4cv.sample.reflections) == num, f"{e4cv.sample.reflections=!r}"


@pytest.mark.parametrize(
    "scan_kwargs, mode, context, expected",
    [
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                start=5,
                finish=10,
                num=3,
                pseudos=dict(h=1, k=1, l=0),
                reals=None,
                extras=dict(h2=1, k2=1, l2=1, psi=0),
                fail_on_exception=True,
            ),
            "psi_constant",
            does_not_raise(),
            None,
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                start=5,
                finish=10,
                num=3,
                pseudos=dict(h=2, k=-1, l=100),  # l=100 is unreachable
                reals=None,
                extras=dict(h2=2, k2=2, l2=0, psi=0),
                fail_on_exception=True,
            ),
            "psi_constant",
            pytest.raises(SolverNoForwardSolutions),
            "No forward solutions found.",
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                start=5,
                finish=10,
                num=3,
                pseudos=dict(h=2, k=-1, l=0),
                reals=None,
                extras=dict(h2=2, k2=2, l2=0, psi=0),
                fail_on_exception=False,  # ignore error that failed previous test
            ),
            "psi_constant",
            does_not_raise(),
            None,
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                start=5,
                finish=10,
                num=3,
                pseudos=None,
                reals=dict(omega=1, chi=2, phi=3, tth=4),
                extras=dict(h2=2, k2=2, l2=0, psi=5),
                fail_on_exception=True,
            ),
            "psi_constant",
            does_not_raise(),
            None,
        ],
        [
            dict(
                detectors=noisy_det,
            ),
            "psi_constant",
            pytest.raises(KeyError),
            "None not in ['h2', 'k2', 'l2', 'psi']",
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="oddball",
            ),
            "psi_constant",
            pytest.raises(KeyError),
            "'oddball' not in ",
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                pseudos=None,
                reals=None,
            ),
            "psi_constant",
            pytest.raises(ValueError),
            "Must define either pseudos or reals.",
        ],
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                pseudos=dict(h=2, k=-1, l=0),
                reals=dict(omega=1, chi=2, phi=3, tth=4),
            ),
            "psi_constant",
            pytest.raises(ValueError),
            "Cannot define both pseudos and reals.",
        ],
    ],
)
def test_scan_extra(scan_kwargs, mode, context, expected):
    from ..diffract import creator

    fourc = creator()
    fourc.restore(HKLPY2_DIR / "tests" / "e4cv_orient.yml")
    fourc.core.mode = mode
    assert fourc.core.mode == mode

    RE = bluesky.RunEngine()

    with context as reason:
        RE(fourc.scan_extra(**scan_kwargs))

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "scan_kwargs, mode, context, expected",
    [
        [
            dict(
                detectors=[noisy_det],
                axis="psi",
                start=-5,  # expect to fail at psi=0
                finish=555,
                num=2,
                pseudos=dict(h=2, k=-1, l=0),
                reals=None,
                extras=dict(h2=2, k2=2, l2=0, psi=0),
                fail_on_exception=False,  # prepare to print FAIL
            ),
            "psi_constant",
            does_not_raise(),
            None,
        ],
    ],
)
def test_scan_extra_print_fail(scan_kwargs, mode, context, expected, capsys):
    from ..diffract import creator

    fourc = creator()
    fourc.restore(HKLPY2_DIR / "tests" / "e4cv_orient.yml")
    fourc.core.mode = mode
    assert fourc.core.mode == mode

    RE = bluesky.RunEngine()

    with context as reason:
        RE(fourc.scan_extra(**scan_kwargs))

    assert_context_result(expected, reason)

    out, err = capsys.readouterr()
    assert len(err) == 0
    assert "FAIL: psi=555.0 No forward solutions found." in out


def test_set_UB():
    """UB chosen to get hkl ~= (1.0, 0, 0), to 3 digits."""
    from ..diffract import creator

    fourc = creator()

    e = 6.28319  # 2 pi.
    assert_almost_equal(
        fourc.sample.UB,  # Default UB (sent to solver) is 2 pi I
        [[e, 0, 0], [0, e, 0], [0, 0, e]],
        3,
    )

    e = 6.25
    UBe = [[e, 0, 0], [0, e, 0], [0, 0, e]]
    fourc.sample.UB = UBe
    assert_almost_equal(fourc.sample.UB, UBe, 5)

    reals = dict(omega=130, chi=0, phi=90, tth=-100)
    result = fourc.inverse(reals, wavelength=1.54)
    assert math.isclose(result.h, 1.0, abs_tol=0.001), f"{result=!r}"
    assert math.isclose(result.k, 0, abs_tol=0.001), f"{result=!r}"
    assert math.isclose(result.l, 0, abs_tol=0.001), f"{result=!r}"
    assert_almost_equal(fourc.sample.UB, UBe, 3)


def test_e4cv_constant_phi():
    from ..diffract import creator

    e4cv = creator()

    # Approximate the code presented as the example problem.
    refl = dict(h=1, k=1, l=1)

    e4cv.core.mode = "constant_phi"
    phi = 23.4567
    e4cv.phi.move(phi)

    e4cv.core.constraints["phi"].limits = -180, 180

    # Check that phi is held constant in all forward solutions.
    solutions = e4cv.core.forward(refl)
    assert isinstance(solutions, list)
    assert len(solutions) > 0
    for solution in solutions:
        assert solution.__class__.__name__.endswith("RealPos")
        solution = solution._asdict()
        assert_almost_equal(solution["phi"], phi, 4)

    # Check that phi is held constant in forward()
    # Returns a position namedtuple.
    position = e4cv.forward(refl)
    assert isinstance(position, tuple)
    assert_almost_equal(position.phi, phi, 4)


@pytest.mark.parametrize(
    "miller, context, expected",
    [
        [(1, 2, 3), does_not_raise(), None],
        [dict(h=1, k=2, l=3), does_not_raise(), None],
        [[1.0, 2.0, 3.0], does_not_raise(), None],
        [
            None,
            pytest.raises(TypeError),
            "Pseudos must be tuple, list, or dict.",
        ],
        [
            # Tests that h, k, l was omitted, only a position was supplied.
            # This is one of the problems reported.
            namedtuple("PseudoTuple", "a b c d".split())(1, 2, 3, 4),
            pytest.raises(ValueError),
            "Expected 3 pseudos, received ",
        ],
        [
            # Tests that wrong name(s) were supplied.
            namedtuple("PseudoTuple", "three wrong names".split())(1, 2, 4),
            pytest.raises(ValueError),
            "Wrong axis names",
        ],
        [("1", 2, 3), pytest.raises(TypeError), "Must be number, received "],
        [(1, 2, "3"), pytest.raises(TypeError), "Must be number, received "],
        [([1], 2, 3), pytest.raises(TypeError), "Must be number, received "],
        [(object, 2, 3), pytest.raises(TypeError), "Must be number, received "],
        [(None, 2, 3), pytest.raises(TypeError), "Must be number, received "],
        [((1,), 2, 3), pytest.raises(TypeError), "Must be number, received "],
        [deque(), pytest.raises(TypeError), "Unexpected data type"],
    ],
)
def test_miller_args(miller, context, expected):
    """Test the Miller indices arguments: h, k, l."""

    with context as reason:
        e4cv = creator()
        e4cv.add_reflection(miller)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "input, ref, context, expected",
    [
        [
            dict(restore_wavelength=False),
            dict(energy=12.3984, wavelength=1.0),
            does_not_raise(),
            None,
        ],
        [
            dict(restore_wavelength=True),
            dict(energy=8.0509, wavelength=1.54),
            does_not_raise(),
            None,
        ],
    ],
)
def test_restore(input, ref, context, expected):
    from ..incident import A_KEV
    from ..incident import DEFAULT_WAVELENGTH
    from ..misc import load_yaml_file

    with context as reason:
        input["config"] = load_yaml_file(HKLPY2_DIR / "tests" / "e4cv_orient.yml")
        e4cv = creator()
        assert math.isclose(
            e4cv.beam.energy.get(),
            A_KEV / DEFAULT_WAVELENGTH,
            abs_tol=0.001,
        )
        assert math.isclose(
            e4cv.beam.wavelength.get(),
            DEFAULT_WAVELENGTH,
            abs_tol=0.001,
        )
        e4cv.restore(**input)
        assert math.isclose(
            e4cv.beam.energy.get(),
            ref.get("energy", A_KEV / DEFAULT_WAVELENGTH),
            abs_tol=0.001,
        )
        assert math.isclose(
            e4cv.beam.wavelength.get(),
            ref.get("wavelength", DEFAULT_WAVELENGTH),
            abs_tol=0.001,
        )
    assert_context_result(expected, reason)


def test_failed_restore():
    from ..misc import load_yaml_file

    config = load_yaml_file(HKLPY2_DIR / "tests" / "e4cv_orient.yml")
    assert isinstance(config, dict)
    assert "_header" in config
    with does_not_raise():
        e4cv = creator()
        e4cv.restore(config)

    config.pop("_header")
    with pytest.raises(KeyError) as reason:
        e4cv = creator()
        e4cv.restore(config)
    expected = "Configuration is missing '_header' key"
    assert_context_result(expected, reason)

    with pytest.raises(TypeError) as reason:
        e4cv = creator()
        e4cv.restore(12345)
    expected = "Unrecognized configuration"
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "specs, context, expected",
    [
        [{}, does_not_raise(), None],
        [{"pseudos": "axis"}, pytest.raises(TypeError), "Expected a list"],
        [{"reals": "omega chi phi tth".split()}, does_not_raise(), None],
        [
            {
                "reals": dict(
                    omega=None,
                    chi=None,
                    phi=None,
                    tth=None,
                )
            },
            does_not_raise(),
            None,
        ],
        [{"reals": "axis"}, pytest.raises(TypeError), "Expected a dict"],
        [{"aliases": "alias"}, pytest.raises(TypeError), "Expected a dict"],
    ],
)
def test_diffractometer_class_factory(specs, context, expected):
    with context as reason:
        klass = diffractometer_class_factory(**specs)
        assert not isinstance(klass, DiffractometerBase)
        assert issubclass(klass, DiffractometerBase)

        gonio = klass(name="gonio")
        assert isinstance(gonio, DiffractometerBase)
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "solver, geometry",
    [
        ["hkl_soleil", "PETRA3 P09 EH2"],
        ["th_tth", "TH TTH Q"],
    ],
)
def test_signature(solver: str, geometry: str):
    sim = creator(solver=solver, geometry=geometry)
    assert isinstance(sim, DiffractometerBase)

    signature = sim.solver_signature
    assert isinstance(signature, AttributeSignal)

    text = signature.get()
    assert isinstance(text, str)
    assert solver in text
    assert geometry in text
