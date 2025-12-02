from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from ...misc import LatticeError
from ...tests.common import assert_context_result
from ..lattice import Lattice


@pytest.mark.parametrize(
    "system, a, others, context, expected",
    [
        ["cubic", 5, dict(), does_not_raise(), None],
        ["hexagonal", 4, dict(c=3, gamma=120), does_not_raise(), None],
        ["rhombohedral", 4, dict(alpha=80.2), does_not_raise(), None],
        ["rhombohedral", 4, dict(alpha=120), does_not_raise(), None],
        ["tetragonal", 4, dict(c=3), does_not_raise(), None],
        ["orthorhombic", 4, dict(b=5, c=3), does_not_raise(), None],
        ["monoclinic", 4, dict(b=5, c=3, beta=75), does_not_raise(), None],
        [
            "triclinic",
            4,
            dict(b=5, c=3, alpha=75, beta=85, gamma=95),
            does_not_raise(),
            None,
        ],
        [
            "hexagonal",
            4,
            dict(gamma=120),  # hexagonal needs a != c
            pytest.raises(LatticeError),
            "Unrecognized crystal system:",
        ],
    ],
)
def test_repr(system, a, others, context, expected):
    lattice = Lattice(a, **others)
    assert lattice is not None

    with context as reason:
        rep = repr(lattice)
        assert rep.startswith("Lattice(")
        assert "a=" in rep
        assert "system=" in rep
        assert repr(system) in rep, f"{system=!r} lattice={rep!r}"
        assert rep.endswith(")")
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "args, kwargs, expected",
    [
        [[5], {}, (5, 5, 5, 90, 90, 90)],  # cubic
        [[4], dict(c=3.0, gamma=120), (4, 4, 3, 90, 90, 120)],  # hexagonal
        [[4], dict(alpha=80.1), (4, 4, 4, 80.1, 80.1, 80.1)],  # rhombohedral
        [[4], dict(c=3), (4, 4, 3, 90, 90, 90)],  # tetragonal
        [[4, 5, 3], {}, (4, 5, 3, 90, 90, 90)],  # orthorhombic
        [[4, 5, 3], dict(beta=75), (4, 5, 3, 90, 75, 90)],  # monoclinic
        [[4, 5, 3, 75, 85, 95], {}, (4, 5, 3, 75, 85, 95)],  # triclinic
    ],
)
def test_crystal_classes(args, kwargs, expected):
    """
    Test each of the 7 crystal lattice types for correct lattices.
    """
    assert isinstance(expected, (list, tuple))
    latt = Lattice(*args, **kwargs)
    assert isinstance(latt, Lattice)
    # Compare only the canonical six lattice parameters (a,b,c,alpha,beta,gamma).
    d = latt._asdict()
    canonical = tuple(d[k] for k in ("a", "b", "c", "alpha", "beta", "gamma"))
    assert list(canonical) == list(expected), f"{latt=!r} {canonical=!r}"


def test_equal():
    l1 = Lattice(4.000_1)
    l2 = Lattice(4.000_0)
    l1.digits = 3
    assert l1 == l2

    l1.digits = 4
    assert l1 != l2

    assert l1 != dict(a=4, alpha=90)


@pytest.mark.parametrize(
    "config, context, expected",
    [
        [
            dict(
                a=3,
                b=4,
                c=5,
                alpha=75.0,
                beta=85.0,
                gamma=95.0,
            ),
            does_not_raise(),
            None,
        ],
        [
            dict(
                a=3,
                b=4,
                c=5,
                alpha=75.0,
                beta=85.0,
                # gamma=95.0,
            ),
            pytest.raises(KeyError),
            "gamma",
        ],
        [
            dict(
                a=3,
                b=4,
                c=5,
                alpha=75.0,
                beta=85.0,
                gamma=95.0,
                delta=1,  # ignored, no error
            ),
            does_not_raise(),
            None,
        ],
        [
            dict(
                able=3,
                baker=4,
                charlie=5,
                echo=75.0,
                foxtrot=85.0,
                # gamma=95.0,
            ),
            pytest.raises(KeyError),
            "'a'",
        ],
    ],
)
def test_fromdict(config, context, expected):
    with context as reason:
        assert isinstance(config, dict)
        lattice = Lattice(1)
        for k in "a b c alpha beta gamma".split():
            assert getattr(lattice, k) != config[k], f"{k=!r}  {lattice=!r}"
        lattice._fromdict(config)
        for k in "a b c alpha beta gamma".split():
            assert getattr(lattice, k) == config[k], f"{k=!r}  {lattice=!r}"

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "set_value, context, expected",
    [
        ("angstrom", does_not_raise(), "angstrom"),
        ("not_a_unit", pytest.raises(Exception), None),
    ],
)
def test_length_units_property_and_validation(set_value, context, expected):
    # Create and set inside the context so exceptions from construction or
    # assignment are captured by the parametrized context manager.
    with context:
        lat = Lattice(3.0)
        assert hasattr(lat, "length_units")
        lat.length_units = set_value

    if expected is not None:
        assert lat.length_units == expected


@pytest.mark.parametrize(
    "set_value, context, expected",
    [
        ("degrees", does_not_raise(), "degrees"),
        ("not_a_unit", pytest.raises(Exception), None),
    ],
)
def test_angle_units_property_and_validation(set_value, context, expected):
    # Create and set inside the context so exceptions from construction or
    # assignment are captured by the parametrized context manager.
    with context:
        lat = Lattice(3.0)
        assert hasattr(lat, "angle_units")
        lat.angle_units = set_value

    if expected is not None:
        assert lat.angle_units == expected


def test_lattice_eq_fallback_raw_comparison(monkeypatch):
    """Force convert_units to raise and ensure equality falls back to raw values."""
    import hklpy2.blocks.lattice as lattice_mod

    def bad_convert_units(value, from_u, to_u):
        raise Exception("conversion failed")

    monkeypatch.setattr(lattice_mod, "convert_units", bad_convert_units)

    l1 = Lattice(4.0, length_units="angstrom", angle_units="degrees")
    l2 = Lattice(4.0, length_units="angstrom", angle_units="degrees")

    # Should fall back to raw comparison and succeed
    assert l1 == l2

    # Now make a different lattice; fallback comparison should detect inequality
    l2b = Lattice(5.0, length_units="angstrom", angle_units="degrees")
    assert not (l1 == l2b)


@pytest.mark.parametrize(
    "params, context, expected",
    [
        (
            {
                "a": 1.0,
                "b": 1.0,
                "c": 1.0,
                "alpha": 160.0,
                "beta": 160.0,
                "gamma": 1.0,
                "tol": 1e-6,
            },
            pytest.raises(ValueError),
            "Inconsistent lattice parameters",
        ),
        (
            {"a": 3.0},
            does_not_raise(),
            None,
        ),
    ],
)
def test_lattice_defensive_check(params, context, expected):
    """
    Defensive check in Lattice.__init__ that raises for the specific
    pathological regime (alpha>150, beta>150, gamma<2 with small tol).
    Also verify a normal construction succeeds.
    """
    with context as reason:
        _lat = Lattice(**params)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "params, context, expected",
    [
        (
            {"a": 1.0, "b": 1.0, "c": 1.0, "alpha": 1.0, "beta": 179.0, "gamma": 1e-6},
            pytest.raises(ValueError),
            "Inconsistent lattice parameters produce imaginary 'c_z'",
        ),
    ],
)
def test_compute_cartesian_inconsistent_cz_raises(params, context, expected):
    """
    Create parameters that lead to a significantly negative c_z_sq and verify
    compute_cartesian_lattice raises the expected ValueError.

    The Lattice constructor itself can raise for these extreme parameters
    (because it calls compute_cartesian_lattice()). To ensure the explicit
    call site inside the test is executed (and thus covered), construct a
    benign lattice inside the context and then overwrite its angle values
    before calling compute_cartesian_lattice(). This guarantees the
    compute_cartesian_lattice() invocation in the test body is reached and
    that the expected exception is raised there.
    """
    with context as reason:
        # Start from a safe lattice so construction does not raise.
        lat = Lattice(1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0)
        # Overwrite attributes to the pathological regime used to trigger the
        # c_z inconsistency in compute_cartesian_lattice().
        lat.alpha = params["alpha"]
        lat.beta = params["beta"]
        lat.gamma = params["gamma"]
        # Now call the method that is expected to raise.
        lat.compute_cartesian_lattice()

    assert_context_result(expected, reason)


def test_compute_B_invalid_shape_raises():
    """
    Force an invalid cartesian matrix shape to exercise the shape check in
    compute_B(). Execution follows the required context-manager pattern.
    """
    with pytest.raises(ValueError) as reason:
        lat = Lattice(2.0)
        lat.cartesian_lattice_matrix = np.zeros((2, 2))
        lat.compute_B()

    assert_context_result("Matrix must be 3x3", reason)


@pytest.mark.parametrize(
    "angle_units, context, expected",
    [
        ("degrees", does_not_raise(), "degrees"),
        ("radians", does_not_raise(), "radians"),
        ("not_a_unit", pytest.raises(Exception), None),
    ],
)
def test_angle_units_validation_in_init(angle_units, context, expected):
    """
    Setting angle_units during construction should validate via the setter.
    Construction and any potential exception are executed inside the context.
    """
    with context as ctx:
        lat = Lattice(3.0, angle_units=angle_units)
        # access property to ensure setter produced the internal canonical value
        _ = lat.angle_units

    assert_context_result(expected, ctx)
    if expected is not None:
        assert lat.angle_units == expected


@pytest.mark.parametrize(
    "length_units, context, expected",
    [
        ("angstrom", does_not_raise(), "angstrom"),
        ("nm", does_not_raise(), "nm"),
        ("not_a_unit", pytest.raises(Exception), None),
    ],
)
def test_length_units_validation_in_init(length_units, context, expected):
    """
    Setting length_units during construction should validate via the setter.
    Construction and any potential exception are executed inside the context.
    """
    with context as ctx:
        lat = Lattice(3.0, length_units=length_units)
        _ = lat.length_units

    assert_context_result(expected, ctx)
    if expected is not None:
        assert lat.length_units == expected


@pytest.mark.parametrize(
    "l1_kwargs, l2_kwargs, context, expect_equal",
    [
        # equal via unit conversion: 10 angstrom == 1 nm
        (
            {"a": 10.0, "length_units": "angstrom"},
            {"a": 1.0, "length_units": "nm"},
            does_not_raise(),
            True,
        ),
        # not equal via unit conversion: 10 angstrom != 2 nm
        (
            {"a": 10.0, "length_units": "angstrom"},
            {"a": 2.0, "length_units": "nm"},
            does_not_raise(),
            False,
        ),
        # angles expressed in different units but equal: 90 deg == pi/2 rad
        (
            {"a": 1.0, "alpha": 90.0, "angle_units": "degrees"},
            {"a": 1.0, "alpha": np.pi / 2, "angle_units": "radians"},
            does_not_raise(),
            True,
        ),
        # slightly different values should not be equal with default digits
        (
            {"a": 4.0001, "length_units": "angstrom"},
            {"a": 4.0, "length_units": "angstrom"},
            does_not_raise(),
            False,
        ),
    ],
)
def test_equality_within_context(l1_kwargs, l2_kwargs, context, expect_equal):
    """
    __eq__ should perform unit conversion; both construction and comparison
    are executed inside the provided context.
    """
    with context as ctx:
        l1 = Lattice(**l1_kwargs)
        l2 = Lattice(**l2_kwargs)
        result = l1 == l2

    assert_context_result(None, ctx)
    assert result is expect_equal


@pytest.mark.parametrize(
    "params, context, expected_msg",
    [
        # Valid cubic lattice: should not raise
        (
            dict(a=1.0, b=1.0, c=1.0, alpha=90.0, beta=90.0, gamma=90.0, tol=1e-12),
            does_not_raise(),
            None,
        ),
        # Parameters chosen to produce an inconsistent (imaginary) c_z
        # Use small gamma so sin(gamma) small and alpha/beta large to produce negative c_z_sq beyond limit.
        (
            dict(a=1.0, b=1.0, c=1.0, alpha=160.0, beta=160.0, gamma=1.0, tol=1e-6),
            pytest.raises(ValueError),
            "Inconsistent lattice parameters",
        ),
    ],
)
def test_constructor_detects_inconsistent_cartesian(params, context, expected_msg):
    """
    Constructing lattices that would produce an imaginary c_z should raise
    during initialization. Construction is executed inside the parametrized context.
    """
    with context as ctx:
        Lattice(**params)

    assert_context_result(expected_msg, ctx)


@pytest.mark.parametrize(
    "cart_a, context, expected_msg",
    [
        (np.eye(2), pytest.raises(ValueError), "Matrix must be 3x3"),
        (np.zeros((3, 3)), pytest.raises(ValueError), "Unit cell volume too small"),
    ],
)
def test_compute_B_rejects_bad_internal_A(cart_a, context, expected_msg):
    """
    compute_B validates the internal cartesian matrix shape and determinant.
    We replace the instance attribute and call compute_B inside the context.
    """
    lat = Lattice(2.0)
    with context as ctx:
        lat.cartesian_lattice_matrix = cart_a
        lat.compute_B()
    assert_context_result(expected_msg, ctx)


def test_compute_cartesian_and_B_on_cubic_inside_context():
    """
    Basic functional test on a cubic lattice: compute_cartesian_lattice
    and compute_B are executed inside a success context.
    """
    with does_not_raise() as ctx:
        lat = Lattice(1.0)
        A = lat.compute_cartesian_lattice()
        assert A.shape == (3, 3)
        # For cubic a=1 we expect the standard orthonormal basis
        assert np.allclose(A, np.eye(3))
        B1 = lat.compute_B(with_2pi=True)
        B2 = lat.compute_B(with_2pi=False)
        assert np.allclose(B1, 2.0 * np.pi * B2)
    assert_context_result(None, ctx)


@pytest.mark.parametrize(
    "params, context, expected",
    [
        ({"a": 0.0}, pytest.raises(ValueError), "Lattice lengths must be positive."),
        ({"a": -1.0}, pytest.raises(ValueError), "Lattice lengths must be positive."),
    ],
)
def test_invalid_lengths(params, context, expected):
    """Lattice lengths must be positive."""
    with context as reason:
        _lat = Lattice(**params)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "params, context, expected",
    [
        (
            {"a": 1.0, "alpha": 0.0},
            pytest.raises(ValueError),
            "Lattice angles must be within range 0 .. 180.",
        ),
        (
            {"a": 1.0, "alpha": 180.0},
            pytest.raises(ValueError),
            "Lattice angles must be within range 0 .. 180.",
        ),
    ],
)
def test_invalid_angles_range(params, context, expected):
    """Angles must be within (0,180)."""
    with context as reason:
        _lat = Lattice(**params)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "gamma, context, expected",
    [
        (
            1e-13,
            pytest.raises(ValueError),
            "Lattice gamma angle is too close to 0 or 180 degrees.",
        ),
    ],
)
def test_gamma_too_close(gamma, context, expected):
    """Gamma values with sin(gamma) smaller than tol should raise."""
    with context as reason:
        _lat = Lattice(1.0, gamma=gamma)

    assert_context_result(expected, reason)


def test_init_handles_float_conversion_exception(monkeypatch):
    """
    Exercise the except branch in __init__ where float(...) conversion fails.

    Monkeypatch the module-level 'float' used by the lattice module to raise,
    and replace compute_cartesian_lattice / compute_B so construction can
    complete without invoking the real implementations.
    """
    import hklpy2.blocks.lattice as lattice_mod

    def _bad_float(x):
        raise RuntimeError("forced float failure")

    # Replace the module-level float name so the try: float(...) raises.
    monkeypatch.setattr(lattice_mod, "float", _bad_float, raising=False)

    # Avoid calling the real heavy numeric methods after __init__'s try/except.
    monkeypatch.setattr(
        lattice_mod.Lattice, "compute_cartesian_lattice", lambda self: np.eye(3)
    )
    monkeypatch.setattr(
        lattice_mod.Lattice, "compute_B", lambda self, with_2pi=True: np.eye(3)
    )

    from contextlib import nullcontext as does_not_raise

    with does_not_raise() as reason:
        lat = lattice_mod.Lattice(
            3.0, b=3.0, c=3.0, alpha=100.0, beta=100.0, gamma=100.0
        )

    assert_context_result(None, reason)
    assert isinstance(lat, lattice_mod.Lattice)
