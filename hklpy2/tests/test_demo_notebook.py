from math import pi

import pytest

import hklpy2

from ..diffract import Hklpy2PseudoAxis
from ..ops import DEFAULT_SAMPLE_NAME
from .common import assert_context_result


@pytest.fixture
def fourc():
    from ophyd import Component as Cpt
    from ophyd import Kind
    from ophyd import SoftPositioner

    NORMAL_HINTED = Kind.hinted | Kind.normal

    class Fourc(hklpy2.DiffractometerBase):
        """Test case."""

        _pseudo = "h k l".split()
        _real = "theta chi phi ttheta".split()

        # pseudo-space axes, in order expected by hkl_soleil E4CV, engine="hkl"
        h = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741
        k = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741
        l = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741

        # real-space axes, in order expected by hkl_soleil E4CV
        # using different names
        theta = Cpt(SoftPositioner, limits=(-180, 180), init_pos=0, kind=NORMAL_HINTED)
        chi = Cpt(SoftPositioner, limits=(-180, 180), init_pos=0, kind=NORMAL_HINTED)
        phi = Cpt(SoftPositioner, limits=(-180, 180), init_pos=0, kind=NORMAL_HINTED)
        ttheta = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=NORMAL_HINTED)

        # pseudo-space extra axes used in a couple modes
        h2 = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741
        k2 = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741
        l2 = Cpt(Hklpy2PseudoAxis, kind=NORMAL_HINTED)  # noqa: E741

        # real-space extra axis used in a couple modes
        psi = Cpt(SoftPositioner, limits=(-170, 170), init_pos=0, kind=NORMAL_HINTED)

        # another Component, not used (yet)
        energy = Cpt(SoftPositioner, limits=(5, 35), init_pos=12.4, kind=NORMAL_HINTED)

        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                solver="hkl_soleil",
                geometry="E4CV",
                solver_kwargs={"engine": "hkl"},
                **kwargs,
            )

    fourc = Fourc(name="fourc")
    yield fourc


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

    expected = "{'position': FourcPseudoPos(h=0, k=0, l=0)}"
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
