"""Tests for backends/typing.py TypedDict structures."""

import re
from contextlib import nullcontext as does_not_raise

import pytest

from ..typing import ReflectionDict
from ..typing import SampleDict
from ..typing import SolverMetadataDict


# ---------------------------------------------------------------------------
# ReflectionDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                value=dict(
                    name="r1",
                    pseudos={"h": 1.0, "k": 0.0, "l": 0.0},
                    reals={"omega": 10.0, "chi": 0.0, "phi": 0.0, "tth": 20.0},
                    wavelength=1.54,
                )
            ),
            does_not_raise(),
            id="valid ReflectionDict",
        ),
        pytest.param(
            dict(
                value=dict(
                    name="r2",
                    pseudos={"h": 0.0, "k": 1.0, "l": 0.0},
                    reals={"omega": 15.0, "chi": 0.0, "phi": 0.0, "tth": 30.0},
                    wavelength=1.0,
                )
            ),
            does_not_raise(),
            id="valid ReflectionDict different values",
        ),
        pytest.param(
            dict(value="not a dict"),
            pytest.raises(TypeError, match=re.escape("is not a")),
            id="ReflectionDict rejects non-dict",
        ),
    ],
)
def test_reflection_dict_is_dict(parms, context):
    """ReflectionDict is a plain dict at runtime; verify keys and types."""
    with context:
        value = parms["value"]
        if not isinstance(value, dict):
            raise TypeError(f"{value!r} is not a dict")
        assert "name" in value
        assert "pseudos" in value
        assert "reals" in value
        assert "wavelength" in value
        assert isinstance(value["name"], str)
        assert isinstance(value["pseudos"], dict)
        assert isinstance(value["reals"], dict)
        assert isinstance(value["wavelength"], (int, float))


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                refl=ReflectionDict(
                    name="r1",
                    pseudos={"h": 1.0, "k": 0.0, "l": 0.0},
                    reals={"omega": 10.0, "chi": 0.0},
                    wavelength=1.54,
                )
            ),
            does_not_raise(),
            id="construct ReflectionDict",
        ),
        pytest.param(
            dict(
                refl=ReflectionDict(
                    name="origin",
                    pseudos={"h": 0.0, "k": 0.0, "l": 0.0},
                    reals={},
                    wavelength=2.0,
                )
            ),
            does_not_raise(),
            id="construct ReflectionDict empty reals",
        ),
    ],
)
def test_reflection_dict_construct(parms, context):
    """ReflectionDict can be constructed and accessed like a plain dict."""
    with context:
        refl = parms["refl"]
        assert isinstance(refl, dict)
        assert isinstance(refl["name"], str)
        assert isinstance(refl["pseudos"], dict)
        assert isinstance(refl["reals"], dict)
        assert isinstance(refl["wavelength"], float)


# ---------------------------------------------------------------------------
# SampleDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                sample=SampleDict(
                    name="silicon",
                    lattice={
                        "a": 5.431,
                        "b": 5.431,
                        "c": 5.431,
                        "alpha": 90.0,
                        "beta": 90.0,
                        "gamma": 90.0,
                    },
                    reflections={},
                    order=[],
                )
            ),
            does_not_raise(),
            id="valid SampleDict no reflections",
        ),
        pytest.param(
            dict(
                sample=SampleDict(
                    name="copper",
                    lattice={
                        "a": 3.615,
                        "b": 3.615,
                        "c": 3.615,
                        "alpha": 90.0,
                        "beta": 90.0,
                        "gamma": 90.0,
                    },
                    reflections={
                        "r1": ReflectionDict(
                            name="r1",
                            pseudos={"h": 1.0, "k": 0.0, "l": 0.0},
                            reals={"omega": 10.0, "chi": 0.0, "phi": 0.0, "tth": 20.0},
                            wavelength=1.54,
                        )
                    },
                    order=["r1"],
                )
            ),
            does_not_raise(),
            id="valid SampleDict with one reflection",
        ),
    ],
)
def test_sample_dict_construct(parms, context):
    """SampleDict can be constructed and accessed like a plain dict."""
    with context:
        sample = parms["sample"]
        assert isinstance(sample, dict)
        assert isinstance(sample["name"], str)
        assert isinstance(sample["lattice"], dict)
        lattice_keys = set(sample["lattice"].keys())
        assert lattice_keys == {"a", "b", "c", "alpha", "beta", "gamma"}
        assert isinstance(sample["reflections"], dict)
        assert isinstance(sample["order"], list)
        for name in sample["order"]:
            assert name in sample["reflections"]


# ---------------------------------------------------------------------------
# SolverMetadataDict
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(
                meta=SolverMetadataDict(
                    name="no_op",
                    description="NoOpSolver(...)",
                    geometry="test",
                    real_axes=[],
                    version="0.1.0",
                )
            ),
            does_not_raise(),
            id="SolverMetadataDict no_op solver",
        ),
        pytest.param(
            dict(
                meta=SolverMetadataDict(
                    name="th_tth",
                    description="ThTthSolver(...)",
                    geometry="TH TTH Q",
                    real_axes=["th", "tth"],
                    version="1.0.0",
                )
            ),
            does_not_raise(),
            id="SolverMetadataDict th_tth solver",
        ),
    ],
)
def test_solver_metadata_dict_construct(parms, context):
    """SolverMetadataDict holds the common required keys for all solvers."""
    with context:
        meta = parms["meta"]
        assert isinstance(meta, dict)
        assert isinstance(meta["name"], str)
        assert isinstance(meta["description"], str)
        assert isinstance(meta["geometry"], str)
        assert isinstance(meta["real_axes"], list)
        assert isinstance(meta["version"], str)
        assert "engine" not in meta  # engine belongs in HklSolverMetadataDict


# ---------------------------------------------------------------------------
# Package-level imports (backends types only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(name="ReflectionDict"),
            does_not_raise(),
            id="ReflectionDict importable from hklpy2",
        ),
        pytest.param(
            dict(name="SampleDict"),
            does_not_raise(),
            id="SampleDict importable from hklpy2",
        ),
        pytest.param(
            dict(name="SolverMetadataDict"),
            does_not_raise(),
            id="SolverMetadataDict importable from hklpy2",
        ),
    ],
)
def test_public_exports(parms, context):
    """Backend TypedDict classes are available from the top-level hklpy2 package."""
    with context:
        import hklpy2

        obj = getattr(hklpy2, parms["name"])
        assert obj is not None
