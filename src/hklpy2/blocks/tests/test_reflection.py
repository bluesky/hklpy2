import re
from contextlib import nullcontext as does_not_raise

import pint
import pytest

from ...diffract import creator
from ...misc import INTERNAL_LENGTH_UNITS
from ...misc import ConfigurationError
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
    "name, pseudos, reals, wavelength, geometry, pseudo_axis_names, real_axis_names, context",
    [
        r100_parms + [does_not_raise()],  # good case
        r010_parms + [does_not_raise()],  # good case
        pytest.param(
            1,  # wrong type
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply str")),
            id="name-int-type-error",
        ),
        pytest.param(
            None,  # wrong type
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply str")),
            id="name-none-type-error",
        ),
        pytest.param(
            "one",
            [1, 0, 0],  # wrong type
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply dict")),
            id="pseudos-list-type-error",
        ),
        pytest.param(
            "one",
            dict(hh=1, kk=0, ll=0),  # wrong keys
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError, match=re.escape("pseudo axis 'hh' unknown")),
            id="pseudos-wrong-keys",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0, m=0),  # extra key
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError, match=re.escape("pseudo axis 'm' unknown")),
            id="pseudos-extra-key",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            [10, 0, 0, 20],  # wrong type
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply dict,")),
            id="reals-list-type-error",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(theta=10, chi=0, phi=0, tth=20),  # wrong key
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError, match=re.escape("real axis 'theta' unknown")),
            id="reals-wrong-key",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            "1.0",  # wrong type
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply number,")),
            id="wavelength-str-type-error",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            None,  # wrong type
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(TypeError, match=re.escape("Must supply number,")),
            id="wavelength-none-type-error",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            -1,  # not allowed
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError, match=re.escape("Must be >=0,")),
            id="wavelength-negative",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            0,  # not allowed: will cause DivideByZero later
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ValueError, match=re.escape("Must be >=0,")),
            id="wavelength-zero",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, phi=0, tth=20),
            1,
            None,  # allowed
            "h k l".split(),
            "omega chi phi tth".split(),
            does_not_raise(),
            id="geometry-none-allowed",
        ),
        pytest.param(
            "one",
            dict(a=1, b=2),
            dict(c=10, d=0, e=20),
            1,
            "test",  # allowed
            "a b".split(),
            "c d e".split(),
            does_not_raise(),
            id="custom-geometry-allowed",
        ),
        pytest.param(
            "one",
            dict(h=1, l=0),  # missing pseudo
            dict(omega=10, chi=0, phi=0, tth=20),
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ReflectionError, match=re.escape("Missing pseudo axis")),
            id="missing-pseudo-axis",
        ),
        pytest.param(
            "one",
            dict(h=1, k=0, l=0),
            dict(omega=10, chi=0, tth=20),  # missing real
            1.0,
            "E4CV",
            "h k l".split(),
            "omega chi phi tth".split(),
            pytest.raises(ReflectionError, match=re.escape("Missing real axis")),
            id="missing-real-axis",
        ),
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
):
    with context:
        refl = Reflection(
            name,
            pseudos,
            reals,
            wavelength,
            geometry,
            pseudo_axis_names,
            real_axis_names,
        )
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
        pytest.param([r100_parms], "(100)", does_not_raise(), None, id="single-r100"),
        pytest.param([r010_parms], "(010)", does_not_raise(), None, id="single-r010"),
        pytest.param(
            [r100_parms, r010_parms], "(100)", does_not_raise(), None, id="r100-r010"
        ),
        pytest.param([r_1], "r1", does_not_raise(), None, id="single-r1"),
        pytest.param([r_2], "r2", does_not_raise(), None, id="single-r2"),
        pytest.param([r_1, r_4], "r4", does_not_raise(), None, id="r1-r4"),
    ],
)
def test_ReflectionsDict(parms, representation, context, expected):
    db = ReflectionsDict()
    assert len(db._asdict()) == 0

    with context:
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


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(other=ReflectionsDict()),
            does_not_raise(),
            id="equal to another empty ReflectionsDict",
        ),
        pytest.param(
            dict(other={}),
            does_not_raise(),
            id="not equal to plain dict returns NotImplemented",
        ),
    ],
)
def test_ReflectionsDict_eq(parms, context):
    with context:
        db = ReflectionsDict()
        other = parms["other"]
        if isinstance(other, ReflectionsDict):
            assert db == other
        else:
            assert db.__eq__(other) is NotImplemented


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param([r100_parms], does_not_raise(), id="single-r100"),
        pytest.param([r010_parms], does_not_raise(), id="single-r010"),
        pytest.param([r100_parms, r010_parms], does_not_raise(), id="r100-r010"),
        pytest.param([r_1], does_not_raise(), id="single-r1"),
        pytest.param([r_2], does_not_raise(), id="single-r2"),
        pytest.param(
            [r_1, r_2],
            pytest.raises(
                ReflectionError, match=re.escape("matches one or more existing")
            ),
            id="duplicate-content",
        ),
        pytest.param([r_1, r_4], does_not_raise(), id="r1-r4-compatible"),
        pytest.param(
            [r100_parms, r010_parms, r_1, r_4],
            pytest.raises(
                ValueError,
                match=re.escape("geometry does not match previous reflections"),
            ),
            id="mixed-geometry-4",
        ),
        pytest.param(
            [r100_parms, r_2],
            pytest.raises(
                ValueError,
                match=re.escape("geometry does not match previous reflections"),
            ),
            id="mixed-geometry-2",
        ),
    ],
)
def test_IncompatibleReflectionsDict(parms, context):
    db = ReflectionsDict()
    assert len(db._asdict()) == 0

    with context:
        for i, refl in enumerate(parms, start=1):
            r = Reflection(*refl)
            assert r is not None
            db.add(r)
            assert len(db) == i


@pytest.mark.parametrize(
    "reflection, context",
    [
        pytest.param(
            r_1,
            pytest.raises(ReflectionError, match=re.escape("is known.")),
            id="same-name",
        ),
        pytest.param(
            r_2,
            pytest.raises(
                ReflectionError, match=re.escape("matches one or more existing")
            ),
            id="same-content",
        ),
    ],
)
def test_duplicate_reflection(reflection, context):
    with context:
        db = ReflectionsDict()
        db.add(Reflection(*r_1))
        db.add(Reflection(*reflection))


@pytest.mark.parametrize(
    "reflections, order, context",
    [
        pytest.param([r_1, r_4, r_5], ["r1", "r4"], does_not_raise(), id="swap-r1-r4"),
        pytest.param([r_1, r_4, r_5], ["r5", "r4"], does_not_raise(), id="swap-r5-r4"),
        pytest.param(
            [r_1, r_4, r_5],
            ["r5"],
            pytest.raises(
                ReflectionError,
                match=re.escape("Need at least two reflections to swap."),
            ),
            id="swap-single-reflection-error",
        ),
        pytest.param(
            [r_1, r_4, r_5],
            [],
            pytest.raises(
                ReflectionError,
                match=re.escape("Need at least two reflections to swap."),
            ),
            id="swap-empty-order-error",
        ),
    ],
)
def test_swap(reflections, order, context):
    db = ReflectionsDict()
    original_order = []
    for params in reflections:
        ref = Reflection(*params)
        db.add(ref)
        original_order.append(ref.name)
    assert db.order == original_order

    with context:
        db.order = order
        assert db.order == order, f"{db.order=!r}"
        db.swap()
        assert db.order == list(reversed(order)), f"{db.order=!r}"


@pytest.mark.parametrize(
    "config, context",
    [
        pytest.param(
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            does_not_raise(),
            id="valid-config",
        ),
        pytest.param(
            {
                "name": "wrong_r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(
                ConfigurationError, match=re.escape("Mismatched name for reflection")
            ),
            id="mismatched-name",
        ),
        pytest.param(
            {
                "name": "r400",
                "geometry": "wrong_E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(
                ConfigurationError,
                match=re.escape("Mismatched geometry for reflection"),
            ),
            id="mismatched-geometry",
        ),
        pytest.param(
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"wrong_h": 4, "k": 0, "l": 0},
                "reals": {"omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(
                ConfigurationError,
                match=re.escape("Mismatched pseudo axis names for reflection"),
            ),
            id="mismatched-pseudo-axes",
        ),
        pytest.param(
            {
                "name": "r400",
                "geometry": "E4CV",
                "pseudos": {"h": 4, "k": 0, "l": 0},
                "reals": {"wrong_omega": -145.451, "chi": 0, "phi": 0, "tth": 69.066},
                "wavelength": 1.54,
                "digits": 4,
            },
            pytest.raises(
                ConfigurationError,
                match=re.escape("Mismatched real axis names for reflection"),
            ),
            id="mismatched-real-axes",
        ),
    ],
)
def test_fromdict(config, context):
    with context:
        assert isinstance(config, dict)
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        r400 = e4cv.sample.reflections["r400"]
        assert isinstance(r400, Reflection)
        r400._fromdict(config)


def test_wrong_real_names():
    with pytest.raises(ReflectionError, match=re.escape("do not match diffractometer")):
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
    "init_kwargs, expect_digits, expect_wavelength_units, context",
    [
        pytest.param(
            {},
            DEFAULT_REFLECTION_DIGITS,
            INTERNAL_LENGTH_UNITS,
            does_not_raise(),
            id="defaults",
        ),
        pytest.param(
            {"digits": 6, "wavelength_units": "angstrom"},
            6,
            "angstrom",
            does_not_raise(),
            id="explicit-values",
        ),
    ],
)
def test_reflection_digits_and_wavelength_units_defaults(
    init_kwargs, expect_digits, expect_wavelength_units, context
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
    "config, explicit_digits, context",
    [
        pytest.param(
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
            id="no-explicit-digits",
        ),
        pytest.param(
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
            id="explicit-digits-8",
        ),
    ],
)
def test_reflectionsdict_fromdict_defaults(config, explicit_digits, context):
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
    assert r.wavelength_units == INTERNAL_LENGTH_UNITS


# ---------------------------------------------------------------------------
# Ensure non-Reflection operands raise TypeError for __add__ and __sub__
# ---------------------------------------------------------------------------


def _make_simple_reflection(name: str = "r"):
    pseudos = {"a": 1.0, "b": 2.0}
    reals = {"x": 0.0, "y": 0.0}
    return Reflection(
        name, pseudos, reals, 1.0, "geo", list(pseudos.keys()), list(reals.keys())
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(5, id="int"),
        pytest.param("string", id="str"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param({"not": "refl"}, id="dict"),
    ],
)
def test_add_type_error_for_non_reflection_operand(bad):
    r = _make_simple_reflection("r1")
    with pytest.raises(TypeError) as exc:
        _ = r + bad
    assert "Unsupported operand type(s) for +" in str(exc.value)


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(5, id="int"),
        pytest.param("string", id="str"),
        pytest.param([1, 2, 3], id="list"),
        pytest.param({"not": "refl"}, id="dict"),
    ],
)
def test_sub_type_error_for_non_reflection_operand(bad):
    r = _make_simple_reflection("r1")
    with pytest.raises(TypeError, match=re.escape("unsupported operand type(s) for -")):
        _ = r - bad


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
    "initial_units, expect_units, context",
    [
        pytest.param("angstrom", "angstrom", does_not_raise(), id="angstrom"),
        pytest.param(None, INTERNAL_LENGTH_UNITS, does_not_raise(), id="default-units"),
    ],
)
def test_asdict_fromdict_preserves_wavelength_units(
    initial_units, expect_units, context
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
        pytest.param(
            1.0,
            "angstrom",
            0.1,
            "nanometer",
            does_not_raise(),
            True,
            id="angstrom-eq-nanometer",
        ),
        pytest.param(
            1.0,
            "angstrom",
            1.1,
            "angstrom",
            does_not_raise(),
            False,
            id="different-wavelength",
        ),
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
        pytest.param(
            1.0,
            "angstrom",
            "nanometer",
            does_not_raise(),
            0.1,
            id="angstrom-to-nanometer",
        ),
        pytest.param(
            1.0,
            "nanometer",
            "angstrom",
            does_not_raise(),
            10.0,
            id="nanometer-to-angstrom",
        ),
        pytest.param(
            1.0,
            "not_a_unit",
            "angstrom",
            pytest.raises(Exception),
            None,
            id="invalid-unit-error",
        ),
    ],
)
def test_convert_units_helper(value, from_u, to_u, context, expected):
    from ...misc import convert_units

    with context:
        result = convert_units(value, from_u, to_u)
        if expected is not None:
            assert result == pytest.approx(expected)


def test_reflections_to_solver_converts_per_reflection_units():
    """Ensure _reflections_to_solver converts each reflection's wavelength
    from its own units into the solver internal units."""
    from ...diffract import creator
    from ...misc import INTERNAL_LENGTH_UNITS
    from ...misc import convert_units

    # create a minimal diffractometer/core to use the conversion helper
    dif = creator(name="testdif")
    core = dif.core

    pseudos = dict(h=1, k=0, l=0)
    reals = dict(omega=0, chi=0, phi=0, tth=0)

    # Reflection with 1.0 angstrom
    rA = Reflection(
        "ra",
        pseudos,
        reals,
        1.0,
        "geo",
        list(pseudos),
        list(reals),
        wavelength_units="angstrom",
    )
    # Reflection with 0.1 nanometer (equal to 1.0 angstrom)
    rB = Reflection(
        "rb",
        pseudos,
        reals,
        0.1,
        "geo",
        list(pseudos),
        list(reals),
        wavelength_units="nanometer",
    )

    out = core._reflections_to_solver([rA, rB])
    assert len(out) == 2
    # both wavelengths converted to INTERNAL_LENGTH_UNITS should be equal
    wl0 = out[0]["wavelength"]
    wl1 = out[1]["wavelength"]
    assert wl0 == pytest.approx(wl1)
    assert wl0 == pytest.approx(convert_units(1.0, "angstrom", INTERNAL_LENGTH_UNITS))


@pytest.mark.parametrize(
    "explicit_units,beam_units,expect_units",
    [
        pytest.param("nanometer", None, "nanometer", id="explicit-nanometer"),
        pytest.param(None, "angstrom", "angstrom", id="beam-angstrom-fallback"),
    ],
)
def test_add_reflection_wavelength_units_preference(
    explicit_units, beam_units, expect_units
):
    """Combined test for explicit wavelength_units preference and beam-unit fallback."""
    sim = creator()
    # if a beam unit is provided, set it on the diffractometer
    if beam_units is not None:
        sim.beam.wavelength_units.set(beam_units)

    r = sim.core.add_reflection(
        (1, 0, 0),
        (0, 0, 0, 0),
        wavelength=1.0,
        wavelength_units=explicit_units,
        name="r_test",
    )
    assert r.wavelength_units == expect_units


@pytest.mark.parametrize(
    "r1_kwargs, r2_kwargs, expect_eq, expect_exception",
    [
        # Same pseudos, reals, wavelength, units, digits
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            True,
            None,
            id="same-content-equal",
        ),
        # Same values, different units (convertible)
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=0.1,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="nanometer",
            ),
            True,
            None,
            id="convertible-units-equal",
        ),
        # Different pseudos
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 2.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            False,
            None,
            id="different-pseudos-not-equal",
        ),
        # Different reals
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 1.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            False,
            None,
            id="different-reals-not-equal",
        ),
        # Different wavelength (not convertible)
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=2.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            False,
            None,
            id="different-wavelength-not-equal",
        ),
        # Exception: wavelength_units not convertible
        pytest.param(
            dict(
                name="r1",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="angstrom",
            ),
            dict(
                name="r2",
                pseudos={"a": 1.0, "b": 2.0},
                reals={"x": 0.0, "y": 0.0},
                wavelength=1.0,
                geometry="geo",
                pseudo_axis_names=["a", "b"],
                real_axis_names=["x", "y"],
                digits=4,
                wavelength_units="not_a_unit",
            ),
            False,
            pytest.raises(Exception),
            id="unconvertible-units-exception",
        ),
    ],
)
def test_reflection_eq(r1_kwargs, r2_kwargs, expect_eq, expect_exception):
    r1 = Reflection(**r1_kwargs)
    if expect_exception is not None:
        with expect_exception:
            _ = r1 == Reflection(**r2_kwargs)
    else:
        r2 = Reflection(**r2_kwargs)
        assert (r1 == r2) is expect_eq


@pytest.mark.parametrize(
    "left,right,expected,context",
    [
        # Identical reflections, same units
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            True,
            does_not_raise(),
            id="identical-same-units",
        ),
        # Identical reflections, same units, but supply a non-Core 'core' kwarg
        # to exercise the make_reflection helper branch that assigns kwargs['core'].
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
                "SOME_CORE",
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
                "SOME_CORE",
            ],
            True,
            does_not_raise(),
            id="identical-with-core-kwarg",
        ),
        # Identical except for wavelength units, convertible
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
                None,
                4,
                "angstrom",
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                0.1,
                "geo",
                ["a", "b"],
                ["x", "y"],
                None,
                4,
                "nanometer",
            ],
            True,
            does_not_raise(),
            id="convertible-units-equal",
        ),
        # Different pseudos
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            [
                "r1",
                {"a": 2.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            False,
            does_not_raise(),
            id="different-pseudos",
        ),
        # Different reals
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 1.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            False,
            does_not_raise(),
            id="different-reals",
        ),
        # Different wavelength, same units
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                2.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
            ],
            False,
            does_not_raise(),
            id="different-wavelength",
        ),
        # Unconvertible units triggers fallback (should fail)
        pytest.param(
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
                None,
                4,
                "not_a_unit",
            ],
            [
                "r1",
                {"a": 1.0, "b": 2.0},
                {"x": 0.0, "y": 0.0},
                1.0,
                "geo",
                ["a", "b"],
                ["x", "y"],
                None,
                4,
                "angstrom",
            ],
            False,
            pytest.raises(pint.errors.UndefinedUnitError),
            id="unconvertible-units-error",
        ),
    ],
)
def test_reflection_eq_deeper_test(left, right, expected, context):
    # allow for optional wavelength_units and digits in params
    def make_reflection(params):
        args = params[:7]
        kwargs = {}
        if len(params) > 7:
            if params[7] is not None:
                kwargs["core"] = params[7]
        if len(params) > 8:
            if params[8] is not None:
                kwargs["digits"] = params[8]
        if len(params) > 9:
            if params[9] is not None:
                kwargs["wavelength_units"] = params[9]
        return Reflection(*args, **kwargs)

    with context:
        r1 = make_reflection(left)
        r2 = make_reflection(right)
        assert (r1 == r2) is expected


def test_reflection_eq_fallback_raw_comparison(monkeypatch):
    # Simulate convert_units raising Exception to trigger fallback
    r1 = Reflection(
        "r1",
        {"a": 1.0},
        {"x": 0.0},
        1.0,
        "geo",
        ["a"],
        ["x"],
        wavelength_units="angstrom",
    )
    r2 = Reflection(
        "r1",
        {"a": 1.0},
        {"x": 0.0},
        1.0,
        "geo",
        ["a"],
        ["x"],
        wavelength_units="angstrom",
    )
    import hklpy2.blocks.reflection as reflection_mod

    def bad_convert_units(value, from_u, to_u):
        raise Exception("conversion failed")

    monkeypatch.setattr(reflection_mod, "convert_units", bad_convert_units)
    # Should fall back to raw comparison, which will succeed here
    assert r1 == r2

    # Now, make r2 wavelength different, fallback should fail
    r2b = Reflection(
        "r1",
        {"a": 1.0},
        {"x": 0.0},
        2.0,
        "geo",
        ["a"],
        ["x"],
        wavelength_units="angstrom",
    )
    assert not (r1 == r2b)


def test_reflection_repr_paren():
    """Simple sanity check: repr(Reflection) ends with a closing parenthesis."""
    r = Reflection(
        "r_repr",
        {"h": 1.0},
        {"x": 0.0},
        1.0,
        "geo",
        ["h"],
        ["x"],
    )
    text = repr(r)
    assert text.endswith(")")
