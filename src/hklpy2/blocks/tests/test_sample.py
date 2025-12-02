from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...misc import IDENTITY_MATRIX_3X3
from ...misc import CoreError
from ...misc import load_yaml
from ...misc import unique_name
from ...tests.common import assert_context_result
from ...tests.models import add_oriented_vibranium_to_e4cv
from ..lattice import Lattice
from ..reflection import ReflectionsDict
from ..sample import Sample


@pytest.mark.parametrize(
    "context, expected",
    [
        [pytest.raises(TypeError), "expected Core"],
    ],
)
def test_sample_constructor_no_core(context, expected):
    with context as reason:
        Sample(None, "test", Lattice(4))
    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "lattice, sname, context, expect",
    [
        [Lattice(4), "sample name", does_not_raise(), None],
        [Lattice(4), None, does_not_raise(), None],
        [None, None, pytest.raises(TypeError), "Must supply Lattice"],
        [
            None,  # <-- not a Lattice
            None,
            pytest.raises(TypeError),
            "Must supply Lattice() object,",
        ],
        [
            (1, 2),  # <-- not a Lattice
            None,
            pytest.raises(TypeError),
            "Must supply Lattice() object,",
        ],
        [
            dict(a=1, b=2, c=3, alpha=4, beta=5, gamma=6),  # <-- dict is acceptable
            None,
            does_not_raise(),
            None,
        ],
        [
            Lattice(4),
            12345,  # <-- not a str
            pytest.raises(TypeError),
            "Must supply str,",
        ],
    ],
)
def test_sample_constructor(lattice, sname, context, expect):
    with context as excuse:
        sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
        sample = Sample(sim.core, sname, lattice)
        assert sample is not None

        if sname is None:
            assert isinstance(sample.name, str)
            assert len(sample.name) == len(unique_name())
        else:
            assert sample.name == sname
        assert isinstance(sample.lattice, Lattice), f"{sample.lattice=}"
        assert isinstance(sample.reflections, ReflectionsDict)

        rep = sample._asdict()
        assert isinstance(rep, dict)
        assert isinstance(rep.get("name"), str)
        assert isinstance(rep.get("lattice"), dict)
        assert isinstance(rep.get("reflections"), dict)
        assert isinstance(rep.get("U"), list)
        assert isinstance(rep.get("UB"), list)
        assert len(rep.get("U")) == 3
        assert len(rep.get("UB")) == 3
        assert len(rep.get("U")[0]) == 3
        assert len(rep.get("UB")[0]) == 3

    if expect is not None:
        assert expect in str(excuse), f"{excuse=} {expect=}"


def test_repr():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    rep = repr(sim.sample)
    assert rep.startswith("Sample(")
    assert "name=" in rep
    assert "lattice=" in rep
    assert "system=" in rep
    assert rep.endswith(")")


@pytest.mark.parametrize(
    "context, expected",
    [
        [pytest.raises(TypeError), "Must supply ReflectionsDict"],
    ],
)
def test_reflections_fail(context, expected):
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    with context as reason:
        sim.sample.reflections = None
    assert_context_result(expected, reason)


def test_fromdict():
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    text = """
    name: vibranium
    lattice:
      a: 6.283185307179586
      b: 6.283185307179586
      c: 6.283185307179586
      alpha: 90.0
      beta: 90.0
      gamma: 90.0
    reflections:
      r400:
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
      r040:
        name: r040
        geometry: E4CV
        pseudos:
          h: 0
          k: 4
          l: 0
        reals:
          omega: -145.451
          chi: 0
          phi: 90
          tth: 69.066
        wavelength: 1.54
        digits: 4
      r004:
        name: r004
        geometry: E4CV
        pseudos:
          h: 0
          k: 0
          l: 4
        reals:
          omega: -145.451
          chi: 90
          phi: 0
          tth: 69.066
        wavelength: 1.54
        digits: 4
    reflections_order:
    - r040
    - r004
    U:
    - - 0.000279252677
      - -0.999999961009
      - -2.2e-11
    - - -7.7982e-08
      - -0.0
      - -1.0
    - - 0.999999961009
      - 0.000279252677
      - -7.7982e-08
    UB:
    - - 0.000279252677
      - -0.999999961009
      - -2.2e-11
    - - -7.7982e-08
      - 0.0
      - -1.0
    - - 0.999999961009
      - 0.000279252677
      - -7.7982e-08
    digits: 6
    """
    config = load_yaml(text)
    assert isinstance(config, dict), f"{config=!r}"
    assert len(config) == 7

    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")

    cfg_latt = Lattice(1)
    cfg_latt._fromdict(config["lattice"])
    sample = Sample(sim.core, "unit", Lattice(1))
    assert sample.name != config["name"]
    assert sample.digits != config["digits"]
    assert sample.lattice != cfg_latt, f"{sample.lattice=!r}  {cfg_latt=!r}"
    assert len(sample.reflections) == 0
    assert len(sample.reflections.order) == 0
    assert sample.U != config["U"]
    assert sample.UB != config["UB"]

    sample._fromdict(config)
    assert sample.name == config["name"]
    assert sample.digits == config["digits"]
    assert sample.lattice == cfg_latt, f"{sample.lattice=!r}  {cfg_latt=!r}"
    assert len(sample.reflections) == 3
    assert sample.reflections.order == config["reflections_order"]
    assert sample.U == config["U"]
    assert sample.UB == config["UB"]


@pytest.mark.parametrize(
    "remove, context, expected",
    [
        [None, does_not_raise(), None],
        ["r004", pytest.raises(CoreError), ""],
        ["wrong", pytest.raises(KeyError), ""],
    ],
)
def test_refine_lattice(remove, context, expected):
    with context as reason:
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        if remove is not None:
            e4cv.sample.reflections.pop(remove)
        e4cv.sample.refine_lattice()

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "rname, context, expected",
    [
        ["r400", does_not_raise(), None],
        ["r1", pytest.raises(KeyError), "Reflection 'r1' is not found"],
    ],
)
def test_remove_reflection(rname, context, expected):
    with context as reason:
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        e4cv.core.calc_UB("r040", "r400")
        e4cv.sample.remove_reflection(rname)
        assert rname not in e4cv.sample.reflections.order

    assert_context_result(expected, reason)


@pytest.mark.parametrize(
    "name, value, context, expected",
    [
        [
            "U",
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            pytest.raises(ValueError),
            "columns must be normalized",
        ],
        [
            "U",
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
            pytest.raises(ValueError),
            "rows must be normalized",
        ],
        ["U", [1, 2, "3"], pytest.raises(TypeError), "must be numerical"],
        ["U", [1, 2, 3], pytest.raises(ValueError), "must by 3x3."],
        ["U", IDENTITY_MATRIX_3X3, does_not_raise(), None],
        ["UB", [1, 2, "3"], pytest.raises(TypeError), "must be numerical"],
        ["UB", [1, 2, 3], pytest.raises(ValueError), "must by 3x3."],
        ["UB", IDENTITY_MATRIX_3X3, does_not_raise(), None],
    ],
)
def test_matrix_validation(name, value, context, expected):
    with context as reason:
        e4cv = creator(name="e4cv")
        if name == "U":
            e4cv.sample.U = value
        else:
            e4cv.sample.UB = value

    assert_context_result(expected, reason)
