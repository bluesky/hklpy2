import math

import numpy as np
import pytest
from pyRestTable import Table

from ...misc import IDENTITY_MATRIX_3X3
from ...ops import Core
from ...tests.common import assert_context_result
from .. import hkl_soleil


def test_version():
    assert "libhkl" in dir(hkl_soleil)
    libhkl = hkl_soleil.libhkl
    assert isinstance(libhkl.VERSION, str)
    assert "HklSolver" in dir(hkl_soleil)

    solver = hkl_soleil.HklSolver("E4CV")
    assert isinstance(solver.version, str)
    assert solver.version == libhkl.VERSION


def kryptonite():
    """Make a kryptonite sample for E4CV."""
    from ...blocks.lattice import Lattice
    from ...blocks.reflection import Reflection
    from ...blocks.sample import Sample

    core = Core(None, default_sample=False)
    sample = Sample(core, "kryptonite", Lattice(0.01))  # should be interesting
    r1 = Reflection(
        name="r1",
        pseudos=dict(h=1, k=0, l=0),
        reals=dict(omega=1, chi=0, phi=0, tth=2),
        wavelength=1.54,
        geometry="E4CV",
        pseudo_axis_names="h k l".split(),
        real_axis_names="omega chi phi tth".split(),
    )
    sample.reflections.add(r1)
    return sample


def test_hkl_soleil():
    arr = hkl_soleil.libhkl.Matrix.new_euler(0, 0, 0)
    assert hkl_soleil.to_hkl_matrix(arr) == arr

    arr = np.array([1, 2, 3])
    np.testing.assert_array_equal(hkl_soleil.to_numpy_matrix(arr), arr)


def test_HklSolver():
    solver = hkl_soleil.HklSolver(geometry="E4CV", engine="hkl")
    assert solver.wavelength == 1.54
    assert solver.axes_c == []
    assert solver.axes_r == ["omega", "chi", "phi", "tth"]
    assert solver.axes_w == ["omega", "chi", "phi", "tth"]
    assert solver.engines == ["hkl", "psi", "q", "incidence", "emergence"]

    assert solver._sample is None  # pre-requisite for next assertions
    assert solver.U == IDENTITY_MATRIX_3X3
    assert solver.UB == IDENTITY_MATRIX_3X3
    assert solver.calculate_UB(None, None) is None

    with pytest.raises(TypeError) as reason:
        solver.addReflection(1.0)
    assert_context_result("Must supply", reason)

    with pytest.raises(KeyError) as reason:
        solver.extras = dict(trombone=0)
    assert_context_result("Unexpected dictionary key received", reason)

    with pytest.raises(ValueError) as reason:
        solver.inverse(dict(a=1, b=2, c=3, d=4))
    assert_context_result("Wrong dictionary keys received", reason)

    with pytest.raises(TypeError) as reason:
        solver.inverse(dict(omega="1", chi=0, phi=0, tth=0))
    assert_context_result("All values must be numbers", reason)

    with pytest.raises(TypeError) as reason:
        solver.lattice = 1.0
    assert_context_result("Must supply", reason)

    with pytest.raises(ValueError) as reason:
        solver.refineLattice([])
    assert_context_result("Must provide 3 or more reflections", reason)

    with pytest.raises(TypeError) as reason:
        solver.sample = "kryptonite"
    assert_context_result("Must supply", reason)
    solver._sample = kryptonite()


@pytest.mark.parametrize(
    "gname, ename, reals",
    [
        ["E4CV", "hkl", ["omega", "chi", "phi", "tth"]],
        ["E4CH", "hkl", ["omega", "chi", "phi", "tth"]],
        ["E6C", "hkl", ["mu", "omega", "chi", "phi", "gamma", "delta"]],
        ["K4CV", "hkl", ["komega", "kappa", "kphi", "tth"]],
        ["K6C", "hkl", ["mu", "komega", "kappa", "kphi", "gamma", "delta"]],
        ["PETRA3 P09 EH2", "hkl", ["mu", "omega", "chi", "phi", "delta", "gamma"]],
        ["PETRA3 P23 4C", "hkl", ["omega_t", "mu", "gamma", "delta"]],
        [
            "PETRA3 P23 6C",
            "hkl",
            ["omega_t", "mu", "omega", "chi", "phi", "gamma", "delta"],
        ],
        ["ZAXIS", "hkl", ["mu", "omega", "delta", "gamma"]],
    ],
)
def test_engine(gname, ename, reals):
    libhkl = hkl_soleil.libhkl
    assert libhkl is not None

    factories = libhkl.factories()
    assert len(factories) > 1
    assert gname in factories

    factory = factories[gname]
    assert factory is not None

    geometry = factory.create_new_geometry()
    assert geometry.name_get() == gname

    r_axes = geometry.axis_names_get()
    assert r_axes == reals, f"{r_axes=}"

    # tip: libhkl will dump core if engines is combined into following:
    #   factory.create_new_engine_list().engine_get_by_name(ename)
    engines = factory.create_new_engine_list()
    assert engines is not None

    engine = engines.engine_get_by_name(ename)
    assert engine is not None
    assert engine.name_get() == ename

    p_axes = engine.pseudo_axis_names_get()
    assert p_axes == "h k l".split(), f"{p_axes=}"


def test_geometries():
    solver = hkl_soleil.HklSolver("E4CV")  # to get the geometries
    assert solver is not None

    glist = solver.geometries()
    assert len(glist) >= 18
    for gname in "E4CV E4CH E6C K4CV K6C ZAXIS".split():
        assert gname in glist, f"{gname=}  {glist=}"


def test_affine():
    """Test the lattice parameter refinement."""
    from ... import SI_LATTICE_PARAMETER
    from ... import creator
    from ...blocks.lattice import SI_LATTICE_PARAMETER_UNCERTAINTY

    e4cv = creator(name="e4cv")
    assert e4cv is not None

    e4cv.add_sample("silicon", SI_LATTICE_PARAMETER)
    e4cv.add_reflection(
        (4, 0, 0),
        dict(tth=69.1, omega=-145.5, chi=0, phi=0),
        wavelength=1.54,
        name="r1",
    )
    e4cv.add_reflection(
        (0, 4, 0),
        dict(tth=69.1, omega=-145.5, chi=90, phi=0),
        wavelength=1.54,
        name="r2",
    )
    e4cv.add_reflection(
        (0, 0, 4),
        dict(tth=69.1, omega=-145.5, chi=0, phi=90),
        wavelength=1.54,
        name="r3",
    )
    assert len(e4cv.sample.reflections) == 3

    # as-defined, sample is cubic with precise lattice parameter
    tol = SI_LATTICE_PARAMETER_UNCERTAINTY
    assert math.isclose(e4cv.sample.lattice.a, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(e4cv.sample.lattice.b, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(e4cv.sample.lattice.c, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(e4cv.sample.lattice.alpha, 90, abs_tol=tol)
    assert math.isclose(e4cv.sample.lattice.beta, 90, abs_tol=tol)
    assert math.isclose(e4cv.sample.lattice.gamma, 90, abs_tol=tol)

    refined = e4cv.core.refine_lattice()

    # sample lattice was not changed
    assert refined != e4cv.sample.lattice

    # refined lattice parameter is not so precise
    assert not math.isclose(refined.a, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert not math.isclose(refined.b, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert not math.isclose(refined.c, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert not math.isclose(refined.alpha, 90, abs_tol=tol)
    assert not math.isclose(refined.beta, 90, abs_tol=tol)
    assert not math.isclose(refined.gamma, 90, abs_tol=tol)

    # relax the precision quite a bit
    tol = 0.1
    assert math.isclose(refined.a, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(refined.b, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(refined.c, SI_LATTICE_PARAMETER, abs_tol=tol)
    assert math.isclose(refined.alpha, 90, abs_tol=tol)
    assert math.isclose(refined.beta, 90, abs_tol=tol)
    assert math.isclose(refined.gamma, 90, abs_tol=tol)


def test_summary_dict():
    from ... import creator

    k4cv = creator(name="k4cv", geometry="K4CV")
    assert k4cv is not None

    # TODO: expand this test
    summary = k4cv.core.solver._summary_dict
    assert isinstance(summary, dict)


def test_summary():
    from ... import creator

    k4cv = creator(name="k4cv", geometry="K4CV")
    assert k4cv is not None

    # TODO: expand this test
    summary = k4cv.core.solver_summary
    assert isinstance(summary, Table)


@pytest.mark.parametrize("geometry", ["APS POLAR", "ZAXIS"])
def test__details(geometry):
    from ... import creator

    diffractometer = creator(name="diffractometer", geometry=geometry)
    assert diffractometer is not None

    review = diffractometer.core.solver._details
    assert isinstance(review, dict)
    keys = "name geometry engine sample lattice U UB wavelength mode extras".split()
    assert list(sorted(review.keys())) == sorted(keys)
    assert review["engine"] == "hkl"
    assert review["geometry"] == geometry
    assert review["name"] == "hkl_soleil"


def test_reflections():
    from ... import SI_LATTICE_PARAMETER
    from ... import creator

    sim = creator()
    sim.add_sample("silicon", SI_LATTICE_PARAMETER)
    sim.add_reflection(
        (4, 0, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
        wavelength=1.54,
        name="(400)",
    )
    sim.add_reflection(
        (0, 4, 0),
        dict(tth=69.0966, omega=-145.451, chi=90, phi=0),
        wavelength=1.54,
        name="(040)",
    )
    sim.core.update_solver()
    reflections = sim.core.solver.reflections
    assert isinstance(reflections, dict)
    for k, refl in reflections.items():
        assert refl.get("name") == k
        assert "pseudos" in refl
        assert "reals" in refl
        assert isinstance(refl.get("wavelength"), float)


def test_sample_property():
    from ... import SI_LATTICE_PARAMETER
    from ... import creator

    sim = creator()
    sim.add_sample("silicon", SI_LATTICE_PARAMETER)
    sim.add_reflection(
        (4, 0, 0),
        dict(tth=69.0966, omega=-145.451, chi=0, phi=0),
        wavelength=1.54,
        name="(400)",
    )
    sim.add_reflection(
        (0, 4, 0),
        dict(tth=69.0966, omega=-145.451, chi=90, phi=0),
        wavelength=1.54,
        name="(040)",
    )
    sample = sim.core.update_solver()
    assert sample is None

    sim.core.update_solver()
    sample = sim.core.solver.sample
    assert sample is not None
    # In hkl_soleil, each call to add_sample creates a new random name
    assert sample.get("name") != "silicon"
    assert math.isclose(sample["lattice"]["a"], SI_LATTICE_PARAMETER, abs_tol=0.01)
    assert math.isclose(sample["lattice"]["b"], SI_LATTICE_PARAMETER, abs_tol=0.01)
    assert math.isclose(sample["lattice"]["c"], SI_LATTICE_PARAMETER, abs_tol=0.01)
    assert math.isclose(sample["lattice"]["alpha"], 90, abs_tol=0.01)
    assert math.isclose(sample["lattice"]["beta"], 90, abs_tol=0.01)
    assert math.isclose(sample["lattice"]["gamma"], 90, abs_tol=0.01)
    assert len(sample.get("reflections")) == 2
