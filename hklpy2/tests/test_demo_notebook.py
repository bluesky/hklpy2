from math import pi

import pytest

import hklpy2

from ..ops import DEFAULT_SAMPLE_NAME
from .common import assert_context_result


@pytest.fixture
def fourc():
    sim = hklpy2.creator(
        aliases=dict(
            pseudos="h k l".split(),
            reals="theta chi phi ttheta".split(),
        ),
        pseudos="h k l h2 k2 l2".split(),
        reals=dict(
            theta=None,
            chi=None,
            phi=None,
            ttheta={
                "class": "ophyd.SoftPositioner",
                "limits": (-170, 170),
                "init_pos": 0,
                "kind": "hinted",
            },
            psi={
                "class": "ophyd.SoftPositioner",
                "limits": (-170, 170),
                "init_pos": 0,
                "kind": "hinted",
            },
            energy={
                "class": "ophyd.SoftPositioner",
                "limits": (5, 35),
                "init_pos": 12.4,
                "kind": "hinted",
            },
        ),
        solver="hkl_soleil",
        geometry="E4CV",
        solver_kwargs={"engine": "hkl"},
        name="fourc",
    )
    yield sim


def test_as_in_demo_notebook(fourc):
    assert "E4CV" in fourc.core.geometries()
    assert "hkl_soleil" in fourc.core.solver_signature
    assert "E4CV" in fourc.core.solver_signature
    assert fourc.beam.wavelength.get() == 1.0
    assert fourc.core.axes_xref == {
        "h": "h",
        "k": "k",
        "l": "l",
        "theta": "omega",
        "chi": "chi",
        "phi": "phi",
        "ttheta": "tth",
    }

    assert fourc.pseudo_axis_names == ["h", "k", "l"]
    assert fourc.real_axis_names == ["theta", "chi", "phi", "ttheta"]
    assert fourc.core.solver_pseudo_axis_names == ["h", "k", "l"]
    assert fourc.core.solver_real_axis_names == "omega chi phi tth".split()
    assert fourc.core.solver_extra_axis_names == []

    expected = "{'position': Hklpy2DiffractometerPseudoPos(h=0, k=0, l=0)}"
    assert str(fourc.report) == expected, f"{fourc.report=!r}"

    assert len(fourc.samples) == 1
    assert fourc.sample.name == DEFAULT_SAMPLE_NAME

    try:
        fourc.core.remove_sample("vibranium")
    except KeyError as reason:
        assert_context_result("not in sample list", reason)
    assert len(fourc.samples) == 1

    fourc.add_sample("vibranium", 2 * pi, digits=3, replace=True)
    assert len(fourc.samples) == 2

    fourc.add_sample("vibranium", 2 * pi, digits=3, replace=True)
    assert len(fourc.samples) == 2
    assert fourc.sample.name == "vibranium"

    fourc.sample = DEFAULT_SAMPLE_NAME
    assert fourc.sample.name == DEFAULT_SAMPLE_NAME

    fourc.sample = "vibranium"
    assert fourc.sample.name == "vibranium"

    assert len(fourc.sample.reflections.order) == 0
    fourc.add_reflection((1, 0, 0), (10, 0, 0, 20), name="r1")
    fourc.add_reflection((0, 1, 0), (10, -90, 0, 20), name="r2")
    assert len(fourc.sample.reflections.order) == 2
    assert fourc.sample.reflections.order == "r1 r2".split()

    fourc.sample.reflections.swap()
    assert fourc.sample.reflections.order == "r2 r1".split()


def test_add_reflections_simple():
    fourc = hklpy2.creator(name="fourc")
    fourc.add_reflection((1, 0, 0), (10, 0, 0, 20), name="r1")
    fourc.add_reflection((0, 1, 0), (10, -90, 0, 20), name="r2")
    assert len(fourc.sample.reflections.order) == 2
    assert fourc.sample.reflections.order == "r1 r2".split()

    fourc.sample.reflections.swap()
    assert fourc.sample.reflections.order == "r2 r1".split()
