"""
Test the ISN diffractometer using interfaces from various levels.

Levels

* [x] test_libhkl: direct calls to gobject-introspection interface
* [x] test_HklSolver: direct calls to the HklSolver class
* [ ] test_hklpy2: use the hklpy2.creator
* [ ] custom: use the custom Diffractometer class from ISN

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
SCALE = 2 * math.pi / SAMPLE_LATTICE_A
UB_R001_R100 = SCALE * np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
FORWARD_SOLUTIONS = {
    # keys: pseudos (tuple)
    # values: list of reals, in order returned by solver
    # (0, 0, 1): [],  # FIXME: Why not at least the R001?
    (0, 0, 1): [
        [6.18/2, 0.0, 0.0, 0.0, 6.18, 0.0],
        [180 - 6.18/2, 0.0, 0.0, 0.0, -6.18, 0.0],
        [6.18/2, 0.0, 0.0, 0.0, -(180 - 6.18), -180.0],
        [6.18/2, 0.0, 0.0, 0.0, -(180 - 6.18), 180.0],
        [180 - 6.18/2, 0.0, 0.0, 0.0, 180 - 6.18, -180.0],
        [180 - 6.18/2, 0.0, 0.0, 0.0, 180 - 6.18, 180.0],
    ],
    (1, 0, 0): [
        [6.18/2, 0, 90, 0, 6.18, 0],
        [180 - 6.18/2, 0, 90, 0, -6.18, 0],
        [6.18/2, 0, 90, 0, -(180 - 6.18), -180],
        [6.18/2, 0, 90, 0, -(180 - 6.18), 180],
        [180 - 6.18/2, 0, 90, 0, 180 - 6.18, -180],
        [180 - 6.18/2, 0, 90, 0, 180 - 6.18, 180],
    ],
}


def test_libhkl():
    """Test for the ISN diffractometer using the low-level code."""
    det_type = libhkl.DetectorType(LIBHKL_DETECTOR_TYPE)
    libhkl_detector = libhkl.Detector.factory_new(det_type)
    assert libhkl_detector is not None

    libhkl_factory = libhkl.factories()[GEOMETRY]
    assert libhkl_factory.name_get() == GEOMETRY

    libhkl_engine_list = libhkl_factory.create_new_engine_list()
    libhkl_engine = libhkl_engine_list.engine_get_by_name(ENGINE)
    assert libhkl_engine.name_get() == ENGINE

    libhkl_geometry = libhkl_factory.create_new_geometry()
    assert libhkl_geometry.name_get() == GEOMETRY

    libhkl_engine.current_mode_set(MODE)
    assert libhkl_engine.current_mode_get() == MODE

    libhkl_geometry.wavelength_set(WAVELENGTH, LIBHKL_USER_UNITS)
    assert math.isclose(
        libhkl_geometry.wavelength_get(LIBHKL_USER_UNITS),
        WAVELENGTH,
        abs_tol=0.001,
    )

    libhkl_geometry.axis_values_set((20, 0, 0, 0, 40, 0), LIBHKL_USER_UNITS)
    assert np.allclose(
        libhkl_geometry.axis_values_get(LIBHKL_USER_UNITS),
        (20, 0, 0, 0, 40, 0),
        atol=0.001,
    )

    sample_name = f"{SAMPLE_NAME}:{str(uuid.uuid4())[:7]}"
    libhkl_sample = libhkl.Sample.new(sample_name)  # new sample each time
    libhkl_engine_list.init(libhkl_geometry, libhkl_detector, libhkl_sample)
    print(f"{libhkl_sample.name_get()=}")

    a, alpha = SAMPLE_LATTICE_A, math.radians(90)
    libhkl_sample.lattice_set(libhkl.Lattice.new(a, a, a, alpha, alpha, alpha))
    assert np.allclose(
        libhkl_sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a, a, a, 90, 90, 90],
        atol=0.01,
    )

    # . remove reflections
    refs = libhkl_sample.reflections_get()
    for ref in refs:
        libhkl_sample.del_reflection(ref)
    assert len(libhkl_sample.reflections_get()) == 0

    # . add reflections
    libhkl_geometry.axis_values_set(R001[1], LIBHKL_USER_UNITS)
    libhkl_r001 = libhkl_sample.add_reflection(
        libhkl_geometry,
        libhkl_detector,
        *(R001[0]),
    )
    assert libhkl_r001 is not None

    libhkl_geometry.axis_values_set(R100[1], LIBHKL_USER_UNITS)
    libhkl_r100 = libhkl_sample.add_reflection(
        libhkl_geometry,
        libhkl_detector,
        *(R100[0]),
    )
    assert libhkl_r100 is not None

    # libhkl_sample.compute_UB_busing_levy(*libhkl_sample.reflections_get())
    libhkl_sample.compute_UB_busing_levy(libhkl_r001, libhkl_r100)
    mat = libhkl_sample.UB_get()
    libhkl_UB = np.array(
        [[mat.get(i, j) for j in range(3)] for i in range(3)],
        dtype=float,
    )
    assert np.allclose(libhkl_UB, UB_R001_R100, atol=0.001)

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    for reflection in (R001, R100):
        libhkl_geometry.axis_values_set(reflection[1], LIBHKL_USER_UNITS)
        libhkl_engine_list.get()  # reals -> pseudos  (Odd name for this call!)
        assert np.allclose(
            libhkl_engine.pseudo_axis_values_get(LIBHKL_USER_UNITS),
            reflection[0],
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations

    # reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        # pseudos = (0, 0, 1)  # forward((h, k, l))
        try:
            # Hkl.GeometryList is not a dict but has a .items() method.
            raw = list(
                libhkl_engine.pseudo_axis_values_set(
                    pseudos,
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
        pseudos=dict(zip("h k l".split(), R100[0])),
        reals=dict(zip(solver.real_axis_names, R100[1])),
        wavelength=WAVELENGTH,
    )
    assert np.allclose(
        solver.calculate_UB(r001, r100),
        UB_R001_R100,
        atol=0.001,
    )

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    for reflection in (r001, r100):
        pseudos = solver.inverse(reflection["reals"])
        assert np.allclose(
            list(pseudos.values()),
            list(reflection["pseudos"].values()),
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        try:
            solutions = solver.forward(
                dict(
                    zip(
                        solver.pseudo_axis_names,
                        pseudos,
                    )
                )
            )
        except Exception:
            solutions = []
        assert len(solutions) == len(reals)

        for i, sol in enumerate(solutions):
            assert np.allclose(
                list(sol.values()),
                reals[i],
                atol=0.01,
            ), f"{i=} {reals[i]=}"


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

        for i, sol in enumerate(solutions):
            # TODO compare with all reals, must match exactly one
            assert np.allclose(
                list(sol),
                reals[i],
                atol=0.01,
            ), f"{i=} {reals[i]=}"


def test_ISN_Diffractometer():
    pass


def test_hklpy_v1():
    """Same procedure, original hklpy (v1) code."""
    # https://blueskyproject.io/hklpy/examples/notebooks/geo_e6c.html
    from hkl import E6C, SimMixin, Lattice, A_KEV
    from ophyd import SoftPositioner
    from ophyd import Component as Cpt

    class SixCircle(SimMixin, E6C):
        """
        Our 6-circle.  Eulerian.
        """

        # the reciprocal axes are defined by SimMixin

        mu = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)
        omega = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)
        chi = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)
        phi = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)
        gamma = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)
        delta = Cpt(SoftPositioner, kind="hinted", limits=(-180, 180), init_pos=0)

    sixc = SixCircle("", name="sixc")
    assert sixc is not None
    assert sixc.geometry_name.get() == GEOMETRY

    sixc.engine.mode = MODE
    assert sixc._mode.get() == MODE

    a0 = SAMPLE_LATTICE_A
    sixc.calc.new_sample(
        SAMPLE_NAME,
        lattice=Lattice(a=a0, b=a0, c=a0, alpha=90, beta=90, gamma=90),
    )
    assert sixc.sample_name.get() == SAMPLE_NAME
    assert math.isclose(sixc.calc.sample.lattice.a, SAMPLE_LATTICE_A, abs_tol=0.001)

    sixc.energy.put(A_KEV / WAVELENGTH)
    assert math.isclose(sixc.energy.get(), A_KEV / WAVELENGTH, abs_tol=0.001)

    r001 = sixc.calc.sample.add_reflection(*(R001[0]), position=R001[1])
    r100 = sixc.calc.sample.add_reflection(*(R100[0]), position=R100[1])
    assert len(sixc.calc.sample.reflections) == 2

    sixc.calc.sample.compute_UB(r001, r100)
    assert np.allclose(
        sixc.calc.sample.UB,
        UB_R001_R100,
        atol=0.001,
    )

    # - - - - - - - - - - - - - - - - Test inverse() calculations
    # pseudos = inverse(reals)
    for reflection in (R001, R100):
        pseudos = sixc.calc.inverse(reflection[1])
        assert np.allclose(
            list(pseudos),
            list(reflection[0]),
            atol=0.001,
        )

    # - - - - - - - - - - - - - - - - Test forward() calculations
    # Bump the constraints out just a bit so roundoff or
    # truncation does not cause any solutions to be dropped.

    for axis in sixc.RealPosition._fields:
        sixc.calc[axis].limits = -180.01, 180.01

    # list of reals = forward(pseudos)
    for pseudos, reals in FORWARD_SOLUTIONS.items():
        solutions = sixc.calc.forward(pseudos)
        assert len(solutions) == len(reals)

        for i, sol in enumerate(solutions):
            # TODO compare with all reals, must match exactly one
            assert np.allclose(
                list(sol),
                reals[i],
                atol=0.01,
            ), f"{i=} {reals[i]=}"
