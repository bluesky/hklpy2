import re
from contextlib import nullcontext as does_not_raise

import pytest

from ...diffract import creator
from ...exceptions import CoreError
from ...utils import IDENTITY_MATRIX_3X3
from ...utils import load_yaml
from ...utils import unique_name
from ...tests.models import add_oriented_vibranium_to_e4cv
from ..lattice import Lattice
from ..reflection import ReflectionsDict
from ..sample import Sample


@pytest.mark.parametrize(
    "context",
    [
        pytest.param(
            pytest.raises(TypeError, match=re.escape("expected Core")), id="no-core"
        ),
    ],
)
def test_sample_constructor_no_core(context):
    with context:
        Sample(None, "test", Lattice(4))


@pytest.mark.parametrize(
    "lattice, sname, context, expect",
    [
        pytest.param(
            Lattice(4), "sample name", does_not_raise(), None, id="valid-named"
        ),
        pytest.param(Lattice(4), None, does_not_raise(), None, id="valid-unnamed"),
        pytest.param(
            None,
            None,
            pytest.raises(TypeError),
            "Must supply Lattice",
            id="none-lattice",
        ),
        pytest.param(
            None,  # <-- not a Lattice
            None,
            pytest.raises(TypeError),
            "Must supply Lattice() object,",
            id="none-not-lattice",
        ),
        pytest.param(
            (1, 2),  # <-- not a Lattice
            None,
            pytest.raises(TypeError),
            "Must supply Lattice() object,",
            id="tuple-not-lattice",
        ),
        pytest.param(
            dict(a=1, b=2, c=3, alpha=4, beta=5, gamma=6),  # <-- dict is acceptable
            None,
            does_not_raise(),
            None,
            id="dict-lattice",
        ),
        pytest.param(
            Lattice(4),
            12345,  # <-- not a str
            pytest.raises(TypeError),
            "Must supply str,",
            id="int-name",
        ),
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
    "context",
    [
        pytest.param(
            pytest.raises(TypeError, match=re.escape("Must supply ReflectionsDict")),
            id="wrong-type",
        ),
    ],
)
def test_reflections_fail(context):
    sim = creator(name="sim", solver="th_tth", geometry="TH TTH Q")
    with context:
        sim.sample.reflections = None


def test_fromdict():
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
    "remove, context",
    [
        pytest.param(None, does_not_raise(), id="all-reflections"),
        pytest.param("r004", pytest.raises(CoreError), id="remove-r004"),
        pytest.param("wrong", pytest.raises(KeyError), id="remove-nonexistent"),
    ],
)
def test_refine_lattice(remove, context):
    with context:
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        if remove is not None:
            e4cv.sample.reflections.pop(remove)
        e4cv.sample.refine_lattice()


@pytest.mark.parametrize(
    "rname, context",
    [
        pytest.param("r400", does_not_raise(), id="existing-reflection"),
        pytest.param(
            "r1",
            pytest.raises(KeyError, match=re.escape("Reflection 'r1' is not found")),
            id="missing-reflection",
        ),
    ],
)
def test_remove_reflection(rname, context):
    with context:
        e4cv = creator(name="e4cv")
        add_oriented_vibranium_to_e4cv(e4cv)
        e4cv.core.calc_UB("r040", "r400")
        e4cv.sample.remove_reflection(rname)
        assert rname not in e4cv.sample.reflections.order


@pytest.mark.parametrize(
    "name, value, context",
    [
        pytest.param(
            "U",
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            pytest.raises(ValueError, match=re.escape("columns must be normalized")),
            id="U-cols-not-normalized",
        ),
        pytest.param(
            "U",
            [[1, 1, 0], [1, 0, 0], [1, 0, 0]],
            pytest.raises(ValueError, match=re.escape("rows must be normalized")),
            id="U-rows-not-normalized",
        ),
        pytest.param(
            "U",
            [1, 2, "3"],
            pytest.raises(TypeError, match=re.escape("must be numerical")),
            id="U-non-numerical",
        ),
        pytest.param(
            "U",
            [1, 2, 3],
            pytest.raises(ValueError, match=re.escape("must be 3x3.")),
            id="U-not-3x3",
        ),
        pytest.param("U", IDENTITY_MATRIX_3X3, does_not_raise(), id="U-identity"),
        pytest.param(
            "UB",
            [1, 2, "3"],
            pytest.raises(TypeError, match=re.escape("must be numerical")),
            id="UB-non-numerical",
        ),
        pytest.param(
            "UB",
            [1, 2, 3],
            pytest.raises(ValueError, match=re.escape("must be 3x3.")),
            id="UB-not-3x3",
        ),
        pytest.param("UB", IDENTITY_MATRIX_3X3, does_not_raise(), id="UB-identity"),
    ],
)
def test_matrix_validation(name, value, context):
    with context:
        e4cv = creator(name="e4cv")
        if name == "U":
            e4cv.sample.U = value
        else:
            e4cv.sample.UB = value
