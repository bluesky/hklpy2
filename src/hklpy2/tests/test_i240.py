"""Test issue #240"""

import pathlib
import uuid

import numpy as np
import yaml

I250_CONFIG_FILE = pathlib.Path(__file__).parent / "configuration_i240.yml"
with open(I250_CONFIG_FILE) as f:
    I250_CONFIG = yaml.safe_load(f)
TOL = 0.001


def test_i240_as_user():
    """Test how a user might execute the code."""
    from .. import creator

    # Get some information from the saved configuration file.
    SAMPLE_NAME = I250_CONFIG["sample_name"]
    SAMPLE = I250_CONFIG["samples"][SAMPLE_NAME]
    SAMPLE_LATTICE_ORIGINAL = SAMPLE["lattice"]["a"]
    SAMPLE_LATTICE_NEW = 5
    assert not np.isclose(SAMPLE_LATTICE_ORIGINAL, SAMPLE_LATTICE_NEW, atol=TOL)

    _pseudos_order = list(I250_CONFIG["axes"]["pseudo_axes"])
    _reals_order = list(I250_CONFIG["axes"]["real_axes"])
    _r1, _r2 = list(SAMPLE["reflections"])[:2]  # just the first two
    R1 = [
        tuple([SAMPLE["reflections"][_r1]["pseudos"][key] for key in _pseudos_order]),
        tuple([SAMPLE["reflections"][_r1]["reals"][key] for key in _reals_order]),
    ]
    R2 = [
        tuple([SAMPLE["reflections"][_r2]["pseudos"][key] for key in _pseudos_order]),
        tuple([SAMPLE["reflections"][_r2]["reals"][key] for key in _reals_order]),
    ]

    polar = creator(
        name=I250_CONFIG["name"],
        geometry=I250_CONFIG["solver"]["geometry"],
        solver=I250_CONFIG["solver"]["name"],
        solver_kwargs=dict(engine=I250_CONFIG["solver"]["engine"]),
    )
    polar.beam.wavelength.put(I250_CONFIG["beam"]["wavelength"])
    polar.add_sample(SAMPLE_NAME, SAMPLE_LATTICE_ORIGINAL)
    assert np.isclose(polar.sample.lattice.a, SAMPLE_LATTICE_ORIGINAL, atol=TOL)
    assert polar.sample.lattice.crystal_system == "cubic"

    polar.add_reflection(pseudos=R1[0], reals=R1[1], name=_r1)
    polar.add_reflection(pseudos=R2[0], reals=R2[1], name=_r2)
    assert len(polar.sample.reflections) == 2

    ub_original = polar.core.calc_UB(*list(polar.sample.reflections)[:2])
    assert np.isclose(
        np.linalg.norm(ub_original),  # ||UB||
        2
        * np.pi
        * np.sqrt(3)
        / SAMPLE_LATTICE_ORIGINAL,  # scales with lattice parameter
        atol=TOL,
    )

    # There are multiple ways to change the lattice parameter:
    # - set a new lattice (as used in issue 240)
    # - set the parameter directly
    polar.sample.lattice = dict(a=SAMPLE_LATTICE_NEW)  # new lattice
    assert np.isclose(polar.sample.lattice.a, SAMPLE_LATTICE_NEW, atol=TOL)
    assert polar.sample.lattice.crystal_system == "cubic"

    # compute UB with this new lattice
    ub_new = polar.core.calc_UB(*list(polar.sample.reflections)[:2])
    assert np.isclose(
        polar.sample.lattice.a,
        SAMPLE_LATTICE_NEW,
        atol=TOL,
    ), "lattice was changed when calculating UB!"
    assert np.isclose(
        np.linalg.norm(ub_new),  # ||UB||
        2 * np.pi * np.sqrt(3) / SAMPLE_LATTICE_NEW,  # scales with lattice parameter
        atol=TOL,
    )


def test_i240_hkl_soleil():
    """Test using hkl_soleil solver directly."""
    from hklpy2.backends.hkl_soleil import HklSolver

    # Configuration parameters (similar to I250_CONFIG)
    SAMPLE_NAME = I250_CONFIG["sample_name"]
    SAMPLE = I250_CONFIG["samples"][SAMPLE_NAME]
    SAMPLE_LATTICE_ORIGINAL = SAMPLE["lattice"]["a"]
    SAMPLE_LATTICE_NEW = 5
    WAVELENGTH = I250_CONFIG["beam"]["wavelength"]
    _r1, _r2 = list(SAMPLE["reflections"])[:2]  # just the first two

    # Create solver instance directly
    solver = HklSolver(
        geometry=I250_CONFIG["solver"]["geometry"],
        engine=I250_CONFIG["solver"]["engine"],
    )

    # Set wavelength
    solver.wavelength = WAVELENGTH

    # Create sample with lattice
    solver.sample = dict(
        name=SAMPLE_NAME,
        lattice=dict(
            a=SAMPLE_LATTICE_ORIGINAL,
            b=SAMPLE_LATTICE_ORIGINAL,
            c=SAMPLE_LATTICE_ORIGINAL,
            alpha=90,
            beta=90,
            gamma=90,
        ),
        order=[],
    )

    # Define reflections
    r1 = dict(
        name=_r1,
        pseudos=SAMPLE["reflections"][_r1]["pseudos"],
        reals=SAMPLE["reflections"][_r1]["reals"],
        wavelength=WAVELENGTH,
    )
    r2 = dict(
        name=_r2,
        pseudos=SAMPLE["reflections"][_r2]["pseudos"],
        reals=SAMPLE["reflections"][_r2]["reals"],
        wavelength=WAVELENGTH,
    )

    # Add reflections to solver
    solver.addReflection(r1)
    solver.addReflection(r2)

    # Calculate UB matrix with original lattice
    ub_original = solver.calculate_UB(r1, r2)
    assert np.isclose(
        np.linalg.norm(ub_original),
        2 * np.pi * np.sqrt(3) / SAMPLE_LATTICE_ORIGINAL,
        atol=0.01,
    )

    # Update lattice parameter
    solver.lattice = dict(
        a=SAMPLE_LATTICE_NEW,
        b=SAMPLE_LATTICE_NEW,
        c=SAMPLE_LATTICE_NEW,
        alpha=90,
        beta=90,
        gamma=90,
    )

    # Calculate UB matrix with new lattice
    ub_new = solver.calculate_UB(r1, r2)
    assert np.isclose(
        np.linalg.norm(ub_new),
        2 * np.pi * np.sqrt(3) / SAMPLE_LATTICE_NEW,
        atol=0.01,
    )


def test_i240_libhkl():
    """Test the libhkl code without the solver API."""
    from hklpy2.backends.hkl_soleil import LIBHKL_DETECTOR_TYPE
    from hklpy2.backends.hkl_soleil import LIBHKL_USER_UNITS
    from hklpy2.backends.hkl_soleil import libhkl

    SAMPLE_NAME = I250_CONFIG["sample_name"]
    SAMPLE = I250_CONFIG["samples"][SAMPLE_NAME]
    SAMPLE_LATTICE_ORIGINAL = SAMPLE["lattice"]["a"]
    SAMPLE_LATTICE_NEW = 5
    ENGINE = I250_CONFIG["solver"]["engine"]
    GEOMETRY = I250_CONFIG["solver"]["geometry"]
    WAVELENGTH = I250_CONFIG["beam"]["wavelength"]
    _pseudos_order = list(I250_CONFIG["axes"]["pseudo_axes"])
    _reals_order = list(I250_CONFIG["axes"]["real_axes"])
    REALS_REFERENCE = (0, 0, 0, 0, 0, 0)
    _r1, _r2 = list(SAMPLE["reflections"])[:2]  # just the first two
    R1 = [
        tuple([SAMPLE["reflections"][_r1]["pseudos"][key] for key in _pseudos_order]),
        tuple([SAMPLE["reflections"][_r1]["reals"][key] for key in _reals_order]),
    ]
    R2 = [
        tuple([SAMPLE["reflections"][_r2]["pseudos"][key] for key in _pseudos_order]),
        tuple([SAMPLE["reflections"][_r2]["reals"][key] for key in _reals_order]),
    ]

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

    geometry.wavelength_set(WAVELENGTH, LIBHKL_USER_UNITS)
    assert np.isclose(
        geometry.wavelength_get(LIBHKL_USER_UNITS),
        WAVELENGTH,
        atol=TOL,
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
    assert str(sample.name_get()).startswith(SAMPLE_NAME)

    a, alpha = SAMPLE_LATTICE_ORIGINAL, np.radians(90)
    sample.lattice_set(libhkl.Lattice.new(a, a, a, alpha, alpha, alpha))
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a, a, a, 90, 90, 90],
        atol=TOL,
    )
    assert len(sample.reflections_get()) == 0

    # . add reflections
    geometry.axis_values_set(R1[1], LIBHKL_USER_UNITS)
    r1 = sample.add_reflection(
        geometry,
        detector,
        *list(R1[0]),
    )
    assert r1 is not None
    assert len(sample.reflections_get()) == 1

    geometry.axis_values_set(R2[1], LIBHKL_USER_UNITS)
    r2 = sample.add_reflection(
        geometry,
        detector,
        *list(R2[0]),
    )
    assert r2 is not None
    assert len(sample.reflections_get()) == 2

    # sample.compute_UB_busing_levy(*sample.reflections_get())
    sample.compute_UB_busing_levy(r1, r2)
    matrix = sample.UB_get()
    UB_reference = np.array(
        [[matrix.get(i, j) for j in range(3)] for i in range(3)],
        dtype=float,
    )
    # Testing against supplied UB is not part of issue 240
    # assert np.allclose(np.linalg.norm(UB_reference), UB_R1_R2_NORM, atol=TOL)
    # assert np.allclose(UB_reference, UB_R1_R2, atol=TOL)
    assert np.isclose(
        np.linalg.norm(UB_reference),  # ||UB||
        2 * np.pi * np.sqrt(3) / a,  # expected
        atol=TOL,
    )

    # Check the lattice again.  Should not change.
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a, a, a, 90, 90, 90],
        atol=TOL,
    )

    # Issue 240 reported problems computing UB with stale lattice.
    # Change the lattice constant.
    a_new = SAMPLE_LATTICE_NEW
    sample.lattice_set(libhkl.Lattice.new(a_new, a_new, a_new, alpha, alpha, alpha))
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a_new, a_new, a_new, 90, 90, 90],
        atol=TOL,
    )

    # sample.compute_UB_busing_levy(*sample.reflections_get())
    sample.compute_UB_busing_levy(r1, r2)
    # Don't expect the lattice to be changed
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a_new, a_new, a_new, 90, 90, 90],
        atol=TOL,
    )

    matrix = sample.UB_get()
    # Still don't expect the lattice to be changed
    assert np.allclose(
        sample.lattice_get().get(LIBHKL_USER_UNITS),
        [a_new, a_new, a_new, 90, 90, 90],
        atol=TOL,
    )

    UB_new = np.array(
        [[matrix.get(i, j) for j in range(3)] for i in range(3)],
        dtype=float,
    )
    assert np.isclose(
        np.linalg.norm(UB_new),  # ||UB||
        2 * np.pi * np.sqrt(3) / a_new,  # scales with lattice parameter
        atol=TOL,
    )


def test_i240_from_config():
    """Test using configuration file."""
    from .. import simulator_from_config

    polar = simulator_from_config(I250_CONFIG_FILE)
    # validate the input first
    assert polar.sample.name == "test"
    assert polar.sample.lattice.crystal_system == "cubic"
    assert np.isclose(polar.sample.lattice.a, 2, atol=TOL)
    assert len(polar.sample.reflections) == 2

    # Compute UB from the first two reflections
    ub0 = polar.core.calc_UB(*list(polar.sample.reflections)[:2])
    norm0 = np.linalg.norm(ub0)
    assert np.isclose(
        norm0,
        2 * np.pi * np.sqrt(3) / polar.sample.lattice.a,
        atol=TOL,
    )

    # change the lattice constant
    polar.sample.lattice = dict(a=5)
    assert np.isclose(polar.sample.lattice.a, 5, atol=TOL)
    assert polar.sample.name == "test"  # unchanged
    assert polar.sample.lattice.crystal_system == "cubic"  # unchanged

    # Again, Compute UB from the first two reflections
    ub1 = polar.core.calc_UB(*list(polar.sample.reflections)[:2])
    norm1 = np.linalg.norm(ub1)
    assert np.isclose(
        norm1,
        2 * np.pi * np.sqrt(3) / polar.sample.lattice.a,
        atol=TOL,
    )

    # Are they different?
    assert not np.isclose(ub0, ub1, atol=TOL).all()
    assert not np.isclose(norm0, norm1, atol=TOL)
