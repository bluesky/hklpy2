from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...misc import INTERNAL_WAVELENGTH_UNITS
from ...misc import ConfigurationError
from ...tests.common import assert_context_result
from ...tests.models import add_oriented_vibranium_to_e4cv
from ..reflection import DEFAULT_REFLECTION_DIGITS
from ..reflection import Reflection
from ..reflection import ReflectionError
from ..reflection import ReflectionsDict

e4cv_r400_config_yaml = """
    name: r400
    geometry: E4CV
    pseudos:
        h: 4
        k: 0
        l: 0
    reals:
        omega: -145.451
        chi: 0
        phi: 0
        tth: 69.066
    wavelength: 1.54
    digits: 4
"""
r100_parms = [
    "(100)",
    dict(h=1, k=0, l=0),
    dict(omega=10, chi=0, phi=0, tth=20),
    1.0,
    "E4CV",
    "h k l".split(),
    "omega chi phi tth".split(),
]
r010_parms = [
    "(010)",
    dict(h=0, k=1, l=0),
    dict(omega=10, chi=-90, phi=0, tth=20),
    1.0,
    "E4CV",
    "h k l".split(),
    "omega chi phi tth".split(),
]
# These are the same reflection (in content)
r_1 = ["r1", {"a": 1, "b": 2}, dict(c=1, d=2), 1, "abcd", ["a", "b"], ["c", "d"]]
r_2 = ["r2", {"a": 1, "b": 2}, dict(c=1, d=2), 1, "abcd", ["a", "b"], ["c", "d"]]
r_3 = ["r3", {"a": 1, "b": 2}, dict(c=1, d=2), 1, "abcd", ["a", "b"], ["c", "d"]]
# different ones
r_4 = ["r4", {"a": 1, "b": 3}, dict(c=1, d=2), 1, "abcd", ["a", "b"], ["c", "d"]]
r_5 = ["r5", {"a": 1, "b": 4}, dict(c=1, d=2), 1, "abcd", ["a", "b"], ["c", "d"]]


@pytest.mark.parametrize(
    "name, pseudos, reals, wavelength, geometry, pseudo_axis_names, real_axis_names, context, expect",
    [
        r100_parms + [does_not_raise(), None],  # good case
        r010_parms + [does_not_raise(), None],  # good case
        [
            1,  # wrong type
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply str",
        ],
        [
            None,  # wrong type
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply str",
        ],
        [
            "one",
            [1, 0, 0],  # wrong type
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply dict",
        ],
        [
            "one",
            dict(hh=1, kk=0, ll=0),  # wrong keys
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError),
            "pseudo axis 'hh' unknown",
        ],
        [
            "one",
            dict(h=1, k=0, l=0, m=0),  # extra key
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError),
            "pseudo axis 'm' unknown",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            [10, 0, 0, 20],  # wrong type
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply dict,",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(theta=10, chi=0, phi=0, tth=20),  # wrong key
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError),
            "real axis 'theta' unknown",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            "1.0",  # wrong type
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply number,",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            None,  # wrong type
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError),
            "Must supply number,",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            -1,  # not allowed
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError),
            "Must be >=0,",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            0,  # not allowed: will cause DivideByZero later
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError),
            "Must be >=0,",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1,
            None,  # allowed
            "h k l".split(),
            "omega chi phi tth".split(),
            does_not_raise(),
            None,
        ],
        [
            "one",
            dict(a=1, b=2),
            dict(c=10, d=0, e=20),
            1,
            "test",  # allowed
            "a b".split(),
            "c d e".split(),
            does_not_raise(),
            None,
        ],
        [
            "one",
            dict(h=1, l=0),  # missing pseudo
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ReflectionError),
            "Missing pseudo axis",
        ],
        [
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, tth=20),  # missing real
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ReflectionError),
            "Missing real axis",
        ],
    ],
)
def test_Reflection(
    name,
    pseudos,
    reals,
    wavelength,
    geometry,
    pseudo_axis_names,
    real_axis_names,
    context,
    expect,
):
    with context as reason:
        refl = Reflection(
            name,
            pseudos,
            reals,
            wavelength,
            geometry,
            pseudo_axis_names,
            real_axis_names,
        )
    if expect is not None:
        assert expect in str(reason), f"{reason}"
    else:
        refl_dict = refl._asdict()
        for k in "name pseudos reals wavelength geometry".split():
            assert k in refl_dict, f"{k=}"

        text = repr(refl)
        assert text.startswith("Reflection(")
        assert f"{name=!r}" in text, f"{text}"
        for key in refl.pseudos.keys():
            assert f"{key}=" in text, f"{text}"
        assert text.endswith(")")


@pytest.mark.parametrize(
    "parms, representation, context, expected",
    [
        [[r100_parms], "(100)", does_not_raise(), None],
        [[r010_parms], "(010)", does_not_raise(), None],
        [[r100_parms, r010_parms], "(100)", does_not_raise(), None],
        [[r_1], "r1", does_not_raise(), None],
        [[r_2], "r2", does_not_raise(), None],
        [[r_1, r_4], "r4", does_not_raise(), None],
    ],
)
def test_ReflectionsDict(parms, representation, context, expected):
    db = ReflectionsDict()
    assert len(db._asdict()) == 0

    with context as reason:
        for i, refl in enumerate(parms, start=1):
            with pytest.raises(TypeError) as exc:
                db.add(refl)
            assert "Unexpected reflection=" in str(exc)

            db.add(Reflection(*refl))
            assert len(db._asdict()) == i
            assert len(db.order) == i

            r1 = list(db.values())[0]
            db.setor([r1])
            assert len(db._asdict()) == i  # unchanged
            assert len(db.order) == 1

            db.set_orientation_reflections([r1])
            assert len(db._asdict()) == i  # unchanged
            assert len(db.order) == 1

            db.order = [r1.name]
            assert len(db._asdict()) == i  # unchanged
            assert len(db.order) == 1

        assert representation in repr(db)

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "parms, context, expected",
    [
        [[r100_parms], does_not_raise(), None],
        [[r010_parms], does_not_raise(), None],
        [[r100_parms, r010_parms], does_not_raise(), None],
        [[r_1], does_not_raise(), None],
        [[r_2], does_not_raise(), None],
        [[r_1, r_2], pytest.raises(ReflectionError), "matches one or more existing"],
        [[r_1, r_4], does_not_raise(), None],
        [
            [r100_parms, r010_parms, r_1, r_4],
            pytest.raises(ValueError),
            "geometry does not match previous reflections",
        ],
        [
            [r100_parms, r_2],
            pytest.raises(ValueError),
            "geometry does not match previous reflections",
        ],
    ],
)
def test_IncompatibleReflectionsDict(parms, context, expected):
    db = ReflectionsDict()
    assert len(db._asdict()) == 0

    with context as reason:
        for i, refl in enumerate(parms, start=1):
            r = Reflection(*refl)
            assert r is not None
            db.add(r)
            assert len(db) == i

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "reflection, context, expected",
    [
        [r_1, pytest.raises(ReflectionError), "is known."],
        [r_2, pytest.raises(ReflectionError), "matches one or more existing"],
    ],
)
def test_duplicate_reflection(reflection, context, expected):
    with context as reason:
        db = ReflectionsDict()
        db.add(Reflection(*r_1))
        db.add(Reflection(*reflection))

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "reflections, order, context, expected",
    [
        [[r_1, r_4, r_5], ["r1", "r4"], does_not_raise(), None],
        [[r_1, r_4, r_5], ["r5", "r4"], does_not_raise(), None],
        [
            [r_1, r_4, r_5],
            ["r5"],
            pytest.raises(ReflectionError),
            "Need at least two reflections to swap.",
        ],
        [
            [r_1, r_4, r_5],
            [],
            pytest.raises(ReflectionError),
            "Need at least two reflections to swap.",
        ],
    ],
)
def test_swap(reflections, order, context, expected):
    db = ReflectionsDict()
    original_order = []
    for params in reflections:
        ref = Reflection(*params)
        db.add(ref)
        original_order.append(ref.name)
    assert db.order == original_order

    with context as reason:
        db.order = order
        assert db.order == order, f"{db.order=!r}"
        db.swap()
        assert db.order == list(reversed(order)), f"{db.order=!r}"

    if expected is None:
        assert reason is None
    else:
        assert expected in str(reason)


@pytest.mark.parametrize(
    "config, context, expected",
    [
        [
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            does_not_raise(),
            None,
        ],
        [
            {
                "name": "wrong_r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(ConfigurationError),
            "Mismatched name for reflection",
        ],
        [
            {
                "name": "r400",
                "geometry": "wrong_E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(ConfigurationError),
            "Mismatched geometry for reflection",
        ],
        [
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"wrong_h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(ConfigurationError),
            "Mismatched pseudo axis names for reflection",
        ],
        [
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"wrong_omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(ConfigurationError),
            "Mismatched real axis names for reflection",
        ],
    ],
)
def test_fromdict(config, context, expected):
    with context as reason:
        assert isinstance(config, dict)
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        r400 = e4cv.sample.reflections["r400"]
        assert isinstance(r400, Reflection)
        r400._fromdict(config)

    assert_context_result(expected, reason)


def test_wrong_real_names():
    expected = "do not match diffractometer"
    with pytest.raises(ReflectionError) as reason:
        e4cv = creator(name="e4cv")
        Reflection(
            name="r400",
            geometry="E4CV",
            pseudos={"h": 4, "k": 0, "l": 0},
            reals={"aaaa_omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
            wavelength=1.54,
            pseudo_axis_names="h k l".split(),
            real_axis_names="aaaa_omega chi phi tth".split(),
            core=e4cv.core,
        )
    assert_context_result(expected, reason)


# === Combined parametrized tests for Reflection arithmetic (__add__ and __sub__) ===


@pytest.mark.parametrize(
    "left,right,ctx,expect_pseudos,expect_reals",
    [
        # successes for addition
        pytest.param(
            r_1,
            r_2,
            does_not_raise(),
            {"a": 2, "b": 4},
            {"c": 2, "d": 4},
            id="add_identical",
        ),
        pytest.param(
            r_1,
            r_4,
            does_not_raise(),
            {"a": 2, "b": 5},
            {"c": 2, "d": 4},
            id="add_diff",
        ),
        # error cases for addition (non-Reflection operand)
        pytest.param(
            r_1, 5, pytest.raises(TypeError), None, None, id="add_type_error_int"
        ),
        pytest.param(
            r_1,
            ["not", "a", "reflection"],
            pytest.raises(TypeError),
            None,
            None,
            id="add_type_error_list",
        ),
    ],
)
def test_reflection_add(left, right, ctx, expect_pseudos, expect_reals):
    r1 = Reflection(*left)
    # for non-Reflection right operands, we do not construct Reflection(*right)
    with ctx:
        r2 = Reflection(*right)
        if isinstance(ctx, type(does_not_raise())):
            # success path: compute and assert
            r3 = r1 + r2
            assert "plus" in r3.name
            assert r3.pseudos == expect_pseudos
            assert r3.reals == expect_reals


@pytest.mark.parametrize(
    "left,right,ctx,expect_pseudos,expect_reals",
    [
        # successes for subtraction
        pytest.param(
            r_1,
            r_2,
            does_not_raise(),
            {"a": 0, "b": 0},
            {"c": 0, "d": 0},
            id="sub_identical",
        ),
        pytest.param(
            r_4,
            r_1,
            does_not_raise(),
            {"a": 0, "b": 1},
            {"c": 0, "d": 0},
            id="sub_diff",
        ),
        # error cases for subtraction (non-Reflection operand)
        pytest.param(
            r_1, 5, pytest.raises(TypeError), None, None, id="sub_type_error_int"
        ),
        pytest.param(
            r_1,
            ["not", "a", "reflection"],
            pytest.raises(TypeError),
            None,
            None,
            id="sub_type_error_list",
        ),
    ],
)
def test_reflection_sub(left, right, ctx, expect_pseudos, expect_reals):
    r1 = Reflection(*left)
    with ctx:
        r2 = Reflection(*right)
        r3 = r1 - r2
        assert "minus" in r3.name
        assert r3.pseudos == expect_pseudos
        assert r3.reals == expect_reals


@pytest.mark.parametrize(
    "init_kwargs, expect_digits, expect_wavelength_units, context, expected",
    [
        (
            {},
            DEFAULT_REFLECTION_DIGITS,
            INTERNAL_WAVELENGTH_UNITS,
            does_not_raise(),
            None,
        ),
        (
            {"digits": 6, "wavelength_units": "angstrom"},
            6,
            "angstrom",
            does_not_raise(),
            None,
        ),
    ],
)
def test_reflection_digits_and_wavelength_units_defaults(
    init_kwargs, expect_digits, expect_wavelength_units, context, expected
):
    """Ensure digits and wavelength_units default behavior and preservation."""
    pseudos = {"h": 1.0}
    reals = {"x": 0.0}

    with context:
        r = Reflection(
            "r_defaults",
            pseudos,
            reals,
            1.0,
            "geo",
            list(pseudos.keys()),
            list(reals.keys()),
            **init_kwargs,
        )

    assert r.digits == expect_digits
    assert r.wavelength_units == expect_wavelength_units


@pytest.mark.parametrize(
    "config, explicit_digits, context, expected",
    [
        (
            {
                "r1": {
                    "name": "r1",
                    "geometry": "geo",
                    "pseudos": {"h": 1.0},
                    "reals": {"x": 0.0},
                    "wavelength": 1.0,
                }
            },
            None,
            does_not_raise(),
            None,
        ),
        (
            {
                "r2": {
                    "name": "r2",
                    "geometry": "geo",
                    "pseudos": {"h": 1.0},
                    "reals": {"x": 0.0},
                    "wavelength": 1.0,
                    "digits": 8,
                }
            },
            8,
            does_not_raise(),
            None,
        ),
    ],
)
def test_reflectionsdict_fromdict_defaults(config, explicit_digits, context, expected):
    """Test ReflectionsDict._fromdict handles missing digits and wavelength_units."""
    rd = ReflectionsDict()
    with context:
        rd._fromdict(config)

    # get the single reflection by name
    name = list(config.keys())[0]
    r = rd[name]
    if explicit_digits is None:
        assert r.digits == DEFAULT_REFLECTION_DIGITS
    else:
        assert r.digits == explicit_digits
    assert r.wavelength_units == INTERNAL_WAVELENGTH_UNITS


# ---------------------------------------------------------------------------
# Ensure non-Reflection operands raise TypeError for __add__ and __sub__
# ---------------------------------------------------------------------------


def _make_simple_reflection(name: str = "r"):
    pseudos = {"a": 1.0, "b": 2.0}
    reals = {"x": 0.0, "y": 0.0}
    return Reflection(
        name, pseudos, reals, 1.0, "geo", list(pseudos.keys()), list(reals.keys())
    )


@pytest.mark.parametrize("bad", [5, "string", [1, 2, 3], {"not": "refl"}])
def test_add_type_error_for_non_reflection_operand(bad):
    r = _make_simple_reflection("r1")
    with pytest.raises(TypeError) as exc:
        _ = r + bad
    assert "Unsupported operand type(s) for +" in str(exc.value)


@pytest.mark.parametrize("bad", [5, "string", [1, 2, 3], {"not": "refl"}])
def test_sub_type_error_for_non_reflection_operand(bad):
    r = _make_simple_reflection("r1")
    with pytest.raises(TypeError) as exc:
        _ = r - bad
    assert "Unsupported operand type(s) for -" in str(exc.value)


def test_add_and_sub_success_case():
    r1 = _make_simple_reflection("r1")
    r2 = _make_simple_reflection("r2")
    r3 = r1 + r2
    assert "plus" in r3.name
    assert r3.pseudos["a"] == 2.0
    r4 = r2 - r1
    assert "minus" in r4.name
    assert r4.reals["x"] == 0.0


@pytest.mark.parametrize(
    "initial_units, expect_units, context, expected",
    [
        ("angstrom", "angstrom", does_not_raise(), None),
        (None, INTERNAL_WAVELENGTH_UNITS, does_not_raise(), None),
    ],
)
def test_asdict_fromdict_preserves_wavelength_units(
    initial_units, expect_units, context, expected
):
    pseudos = {"h": 1, "k": 0, "l": 0}
    reals = {"omega": 0, "chi": 0, "phi": 0, "tth": 0}
    if initial_units is None:
        r = Reflection(
            "r1", pseudos, reals, 1.0, "geo", list(pseudos.keys()), list(reals.keys())
        )
    else:
        r = Reflection(
            "r1",
            pseudos,
            reals,
            1.0,
            "geo",
            list(pseudos.keys()),
            list(reals.keys()),
            wavelength_units=initial_units,
        )

    d = r._asdict()
    assert d["wavelength_units"] == expect_units

    # create a new Reflection with same name/geometry to test _fromdict
    r2 = Reflection(
        "r1", pseudos, reals, 1.0, "geo", list(pseudos.keys()), list(reals.keys())
    )
    with context:
        r2._fromdict(d)
    assert r2.wavelength_units == expect_units


@pytest.mark.parametrize(
    "wl1,u1,wl2,u2,context,expect_eq",
    [
        (1.0, "angstrom", 0.1, "nanometer", does_not_raise(), True),
        (1.0, "angstrom", 1.1, "angstrom", does_not_raise(), False),
    ],
)
def test_eq_converts_wavelength_units(wl1, u1, wl2, u2, context, expect_eq):
    pseudos = {"h": 1, "k": 0, "l": 0}
    reals = {"omega": 0, "chi": 0, "phi": 0, "tth": 0}
    with context:
        r1 = Reflection(
            "ra",
            pseudos,
            reals,
            wl1,
            "geo",
            list(pseudos.keys()),
            list(reals.keys()),
            wavelength_units=u1,
        )
        r2 = Reflection(
            "rb",
            pseudos,
            reals,
            wl2,
            "geo",
            list(pseudos.keys()),
            list(reals.keys()),
            wavelength_units=u2,
        )

    if expect_eq:
        assert r1 == r2
    else:
        assert not (r1 == r2)


@pytest.mark.parametrize(
    "value,from_u,to_u,context,expected",
    [
        (1.0, "angstrom", "nanometer", does_not_raise(), 0.1),
        (1.0, "nanometer", "angstrom", does_not_raise(), 10.0),
        (1.0, "not_a_unit", "angstrom", pytest.raises(Exception), None),
    ],
)
def test_convert_units_helper(value, from_u, to_u, context, expected):
    from ...misc import convert_units

    with context:
        result = convert_units(value, from_u, to_u)
        if expected is not None:
            assert result == pytest.approx(expected)
