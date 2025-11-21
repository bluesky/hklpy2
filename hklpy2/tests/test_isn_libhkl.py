"""
Test the ISN diffractometer using interfaces from various levels.

Levels

* [x] test_libhkl: direct calls to gobject-introspection interface
* [x] test_HklSolver: direct calls to the HklSolver class
* [x] test_hklpy2: use the hklpy2.creator
* [x] test_ISN_Diffractometer: use the custom Diffractometer class from ISN
* [x] test_hklpy_v1: test hklpy v1, with local fixes applied

Procedure

* setup geometry
* set wavelength
* set mode
* set sample
* set lattice
* add two orienting reflections
* calculate the UB matrix
* Check inverse() computation with orienting reflections
* Check forward() computation with orienting reflections
"""

import math
import uuid

import numpy as np

import hklpy2
from hklpy2.backends.hkl_soleil import LIBHKL_DETECTOR_TYPE
from hklpy2.backends.hkl_soleil import LIBHKL_USER_UNITS
from hklpy2.backends.hkl_soleil import HklSolver
from hklpy2.backends.hkl_soleil import NoForwardSolutions
from hklpy2.backends.hkl_soleil import libhkl

GEOMETRY = "E6C"
SOLVER = "hkl_soleil"
ENGINE = "hkl"
MODE = "lifting_detector_mu"
WAVELENGTH = 12.3984 / 20.0
SAMPLE_NAME = "GaAs"
SAMPLE_LATTICE_A = 5.75
ROUNDOFF_DIGITS = 12

# Two reflections and UB matrix for (R001, R100)
# Common wavelength, ignore custom axis names, set values in canonical order.
R001 = [(0, 0, 1), (6.18 / 2, 0, 0, 0, 6.18, 0)]
R100 = [(1, 0, 0), (6.18 / 2, 0, 90, 0, 6.18, 0)]
PRESET_REALS = (0, 0, 0, 0, 0, 0)
SCALE = 2 * math.pi / SAMPLE_LATTICE_A
UB_R001_R100 = SCALE * np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
FORWARD_SOLUTIONS = {
    # keys: pseudos (tuple)
    # values: list of reals, in order returned by solver
    (0, 0, 1): [
        [6.18 / 2, 0.0, 0.0, 0.0, 6.18, 0.0],
        [180 - 6.18 / 2, 0.0, 0.0, 0.0, -6.18, 0.0],
        [6.18 / 2, 0.0, 0.0, 0.0, -(180 - 6.18), -180.0],
        [6.18 / 2, 0.0, 0.0, 0.0, -(180 - 6.18), 180.0],
        [180 - 6.18 / 2, 0.0, 0.0, 0.0, 180 - 6.18, -180.0],
        [180 - 6.18 / 2, 0.0, 0.0, 0.0, 180 - 6.18, 180.0],
    ],
    (1, 0, 0): [],  # non-zero chi is reachable
}
REALS_REFERENCE = (20, 0, 0, 0, 40, 0)


def test_libhkl():
    """Test for the ISN diffractometer using the low-level code."""
    det_type = libhkl.DetectorType(LIBHKL_DETECTOR_TYPE)
    detector = libhkl.Detector.factory_new(det_type)
    assert detector is not None

    factory = libhkl.factories()[GEOMETRY]
    assert factory.name_get() == GEOMETRY

    geometry = factory.create_new_geometry()
    assert geometry.name_get() == GEOMETRY

    engine_list = factory.create_new_engine_list()
    engine = engine_list.engine_get_by_name(ENGINE)
    assert engine.name_get() == ENGINE

    engine.current_mode_set(MODE)
    assert engine.current_mode_get() == MODE

    geometry.wavelength_set(WAVELENGTH, LIBHKL_USER_UNITS)
    assert math.isclose(
        geometry.wavelength_get(LIBHKL_USER_UNITS),
        WAVELENGTH,
        abs_tol=0.001,
    )

    geometry.axis_values_set(REALS_REFERENCE, LIBHKL_USER_UNITS)
    assert np.allclose(
        geometry.axis_values_get(LIBHKL_USER_UNITS),
        REALS_REFERENCE,
        atol=0.001,
    )

    sample_name = f"{SAMPLE_NAME}:{str(uuid.uuid4())[:7]}"
    sample = libhkl.Sample.new(sample_name)  # new sample each time
    engine_list.init(geometry, detector, sample)
    print(f"{sample.name_get()=}")

    a, alpha = SAMPLE_LATTICE_A, math.radians(90)
    sample.lattice_set(libhkl.Lattice.new(a, a, a, alpha, alpha, alpha))
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a, a, a, 90, 90, 90],
        atol=0.01,
    )
    assert len(sample.reflections_get()) == 0

    # . add reflections
    geometry.axis_values_set(R001[1], LIBHKL_USER_UNITS)
    r001 = sample.add_reflection(
        geometry,
        detector,
        *list(R001[0]),
    )
    assert r001 is not None
    assert len(sample.reflections_get()) == 1

    geometry.axis_values_set(R100[1], LIBHKL_USER_UNITS)
    r100 = sample.add_reflection(
        geometry,
        detector,
        *list(R100[0]),
    )
    assert r100 is not None
    assert len(sample.reflections_get()) == 2

    # sample.compute_UB_busing_levy(*sample.reflections_get())
    sample.compute_UB_busing_levy(r001, r100)
    matrix = sample.UB_get()
    UB = np.array(
        [[matrix.get(i, j) for j in range(3)] for i in range(3)],
        dtype=float,
    )
    assert np.allclose(UB, UB_R001_R100, atol=0.001)

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    # pseudos = inverse(reals)
    for reflection in (R001, R100):
        reals = reflection[1]
        geometry.axis_values_set(reals, LIBHKL_USER_UNITS)
        engine_list.get()  # reals -> pseudos  (Odd name for this call!)
        pseudos = np.asarray(engine.pseudo_axis_values_get(LIBHKL_USER_UNITS))
        assert np.allclose(pseudos, reflection[0], atol=0.001)

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        # rotate pseudos coordinates from hklpy2 to libhkl
        libhkl_pseudos = np.asarray(pseudos)
        try:
            geometry.axis_values_set(REALS_REFERENCE, LIBHKL_USER_UNITS)
            # Hkl.GeometryList (not a Python dict) has a '.items()' method.
            raw = list(
                engine.pseudo_axis_values_set(
                    libhkl_pseudos,
                    LIBHKL_USER_UNITS,
                ).items()
            )
        except Exception:
            raw = []
        assert len(raw) == len(reals)

        for i, glist_item in enumerate(raw):
            geo = glist_item.geometry_get()
            assert np.allclose(
                geo.axis_values_get(LIBHKL_USER_UNITS),
                reals[i],
                atol=0.01,
            ), f"{i=} {reals[i]=}"


def test_HklSolver():
    """Same tests using the SolverBase subclass."""
    solver = HklSolver(GEOMETRY, engine=ENGINE)
    assert solver is not None
    assert solver.name == SOLVER
    assert solver.geometry == GEOMETRY
    assert solver.engine_name == ENGINE

    solver.wavelength = WAVELENGTH
    assert math.isclose(solver.wavelength, WAVELENGTH, abs_tol=0.001)

    solver.mode = MODE
    assert solver.mode == MODE

    a, alpha = SAMPLE_LATTICE_A, 90
    solver.sample = dict(
        name=SAMPLE_NAME,
        lattice=dict(
            a=a,
            b=a,
            c=a,
            alpha=alpha,
            beta=alpha,
            gamma=alpha,
        ),
        order=[],
        reflections=[],
    )
    assert solver.sample["name"] != SAMPLE_NAME
    assert solver.sample["lattice"]["a"] == SAMPLE_LATTICE_A
    assert len(solver.sample["reflections"]) == 0

    r001 = dict(
        name="r001",
        pseudos=dict(zip(solver.pseudo_axis_names, R001[0])),
        reals=dict(zip(solver.real_axis_names, R001[1])),
        wavelength=WAVELENGTH,
    )

    r100 = dict(
        name="r100",
        pseudos=dict(zip(solver.pseudo_axis_names, R100[0])),
        reals=dict(zip(solver.real_axis_names, R100[1])),
        wavelength=WAVELENGTH,
    )
    assert len(solver.reflections) == 0

    UB = solver.calculate_UB(r001, r100)

    # Now the reflections are defined.
    # Spot check their pseudos.
    refs = solver.reflections
    assert len(refs) == 2

    refs = list(refs.values())
    assert np.allclose(
        list(refs[0]["pseudos"].values()),
        (0, 0, 1),
        atol=0.001,
    )
    assert np.allclose(
        list(refs[1]["pseudos"].values()),
        (1, 0, 0),
        atol=0.001,
    )
    assert np.allclose(
        UB,
        UB_R001_R100,
        atol=0.001,
    )

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    for reflection in (r001, r100):
        pseudos = list(solver.inverse(reflection["reals"]).values())
        assert np.allclose(
            list(pseudos),
            list(reflection["pseudos"].values()),
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        try:
            solver._hkl_geometry.axis_values_set(PRESET_REALS, LIBHKL_USER_UNITS)
            solutions = solver.forward(dict(zip(solver.pseudo_axis_names, pseudos)))
        except NoForwardSolutions:
            solutions = []
        assert len(solutions) == len(reals)


def test_hklpy2():
    psic = hklpy2.creator(geometry=GEOMETRY)
    assert psic.core.solver.name == SOLVER
    assert psic.core.solver.engine_name == ENGINE
    assert psic.core.geometry == GEOMETRY

    psic.core.mode = MODE
    assert psic.core.mode == MODE

    psic.beam.wavelength.put(WAVELENGTH)
    assert math.isclose(psic.beam.wavelength.get(), WAVELENGTH, abs_tol=0.001)

    psic.core.add_sample(SAMPLE_NAME, SAMPLE_LATTICE_A)
    assert psic.core.sample.name == SAMPLE_NAME
    assert f"a={SAMPLE_LATTICE_A}" in str(psic.core.sample)

    r001 = psic.core.add_reflection(*R001, name="r001")
    r100 = psic.core.add_reflection(*R100, name="r100")
    assert len(psic.core.sample.reflections) == 2
    assert np.allclose(
        psic.core.calc_UB(r001, r100),
        UB_R001_R100,
        atol=0.001,
    )

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    # pseudos = inverse(reals)
    for reflection in (r001, r100):
        pseudos = psic.core.inverse(reflection.reals)
        assert np.allclose(
            list(pseudos.values()),
            list(reflection.pseudos.values()),
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # Bump the constraints out just a bit so roundoff or
    # truncation does not cause any solutions to be dropped.
    for axis in psic.real_axis_names:
        psic.core.constraints[axis].limits = -180.01, 180.01

    # list of reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        solutions = psic.core.forward(
            dict(
                zip(
                    psic.pseudo_axis_names,
                    pseudos,
                )
            )
        )
        assert len(solutions) == len(reals)

        if len(solutions):
            assessments = [
                np.allclose(
                    list(sol),
                    reals[i],
                    atol=0.01,
                )
                for i, sol in enumerate(solutions)
            ]
            assert True in assessments, f"{solutions=} {reals=}"


def test_ISN_Diffractometer():
    from ophyd import Component
    from ophyd import SoftPositioner

    from hklpy2 import DiffractometerBase
    from hklpy2.diffract import Hklpy2PseudoAxis

    class Diffractometer(DiffractometerBase):
        """Example custom diffractometer"""

        _real = "mu eta chi phi pitch yaw".split()

        h = Component(Hklpy2PseudoAxis, "", kind="hinted")
        k = Component(Hklpy2PseudoAxis, "", kind="hinted")
        l = Component(Hklpy2PseudoAxis, "", kind="hinted")  # noqa E741

        mu = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        eta = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        chi = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        phi = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        pitch = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        yaw = Component(SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted")
        radius = Component(
            SoftPositioner, limits=(-180, 180), init_pos=0, kind="hinted"
        )

        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                solver=SOLVER,
                geometry=GEOMETRY,
                **kwargs,
            )

    psic = Diffractometer("", name="psic")
    assert psic.connected
    assert "eta" in psic.real_axis_names
    assert "radius" not in psic.real_axis_names

    psic.radius._limits = 0, 1000
    psic.radius.move(800)
    psic.yaw.move(40)
    psic.pitch.move(psic.yaw.position / 2)
    psic.core.mode = MODE
    psic.beam.wavelength.put(WAVELENGTH)

    psic.core.add_sample(SAMPLE_NAME, SAMPLE_LATTICE_A)

    r001 = psic.core.add_reflection(*R001, name="r001")
    r100 = psic.core.add_reflection(*R100, name="r100")
    assert len(psic.core.sample.reflections) == 2
    assert np.allclose(
        psic.core.calc_UB(r001, r100),
        UB_R001_R100,
        atol=0.001,
    )

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    # pseudos = inverse(reals)
    for reflection in (r001, r100):
        pseudos = psic.core.inverse(reflection.reals)
        assert np.allclose(
            list(pseudos.values()),
            list(reflection.pseudos.values()),
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # Bump the constraints out just a bit so roundoff or
    # truncation does not cause any solutions to be dropped.
    for axis in psic.real_axis_names:
        psic.core.constraints[axis].limits = -180.01, 180.01

    # list of reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        solutions = psic.core.forward(
            dict(
                zip(
                    psic.pseudo_axis_names,
                    pseudos,
                )
            )
        )
        assert len(solutions) == len(reals)

        if len(solutions):
            assessments = [
                np.allclose(
                    list(sol),
                    reals[i],
                    atol=0.01,
                )
                for i, sol in enumerate(solutions)
            ]
            assert True in assessments, f"{solutions=} {reals=}"
