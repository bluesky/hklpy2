# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Tests for issue #388: round-trip of auxiliary sub-devices (scalar +
nested ``PseudoPositioner``) through the export / restore schema (v2).
"""

import re
import warnings
from contextlib import nullcontext as does_not_raise

import pytest

from ..blocks.configure import CONFIG_SCHEMA_VERSION
from ..devices import describe_aux
from ..devices import make_aux_pseudo_positioner_class
from ..exceptions import ConfigurationError
from ..run_utils import _AUX_RECONSTRUCTORS
from ..run_utils import _normalize_aux_record
from ..run_utils import register_aux_reconstructor
from ..run_utils import simulator_from_config
from .test_diffract import _build_gonio_with_nested_pseudo

# ---------------------------------------------------------------------------
# Schema-v2 export: describe_aux() and Core._asdict()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {
                "aux_name": "tablex",
                "expected": {"name": "tablex", "category": "scalar"},
            },
            does_not_raise(),
            id="scalar-aux-record",
        ),
        pytest.param(
            {
                "aux_name": "ana",
                "expected": {
                    "name": "ana",
                    "category": "pseudo_positioner",
                    "pseudos": ["energy"],
                    "reals": ["theta"],
                    "class_name": "_MiniAnalyzer",
                },
            },
            does_not_raise(),
            id="pseudo-positioner-aux-record",
        ),
    ],
)
def test_describe_aux(parms, context):
    """``describe_aux()`` produces v2 records with the right shape."""
    with context:
        gonio = _build_gonio_with_nested_pseudo()
        record = describe_aux(gonio, parms["aux_name"])

        for key, value in parms["expected"].items():
            if key == "class_name":
                assert record["class"] == value
            else:
                assert record[key] == value


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"check": "schema-version-bumped"},
            does_not_raise(),
            id="schema-version-is-2",
        ),
        pytest.param(
            {"check": "aux-axes-is-list-of-records"},
            does_not_raise(),
            id="aux-axes-record-form",
        ),
    ],
)
def test_export_schema_v2(parms, context):
    """``Core._asdict()`` writes schema v2 with record-form aux axes."""
    with context:
        gonio = _build_gonio_with_nested_pseudo()
        cfg = gonio.configuration

        if parms["check"] == "schema-version-bumped":
            assert cfg["_header"]["config_schema_version"] == CONFIG_SCHEMA_VERSION
            assert CONFIG_SCHEMA_VERSION == 2

        elif parms["check"] == "aux-axes-is-list-of-records":
            aux = cfg["axes"]["auxiliary_axes"]
            assert all(isinstance(rec, dict) for rec in aux)
            assert {rec["name"] for rec in aux} == {"tablex", "ana"}
            ana = next(r for r in aux if r["name"] == "ana")
            assert ana["category"] == "pseudo_positioner"
            assert ana["pseudos"] == ["energy"]
            assert ana["reals"] == ["theta"]


# ---------------------------------------------------------------------------
# Schema-v1 / v2 input normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"entry": "tablex", "expected": {"name": "tablex", "category": "scalar"}},
            does_not_raise(),
            id="legacy-string-form",
        ),
        pytest.param(
            {
                "entry": {"name": "tablex"},
                "expected": {"name": "tablex", "category": "scalar"},
            },
            does_not_raise(),
            id="record-without-category-defaults-scalar",
        ),
        pytest.param(
            {
                "entry": {
                    "name": "ana",
                    "category": "pseudo_positioner",
                    "pseudos": ["energy"],
                    "reals": ["theta"],
                },
                "expected": {"name": "ana", "category": "pseudo_positioner"},
            },
            does_not_raise(),
            id="full-pseudo-positioner-record",
        ),
        pytest.param(
            {"entry": {"category": "scalar"}},
            pytest.raises(
                ConfigurationError, match=re.escape("missing a non-empty 'name'")
            ),
            id="missing-name-raises",
        ),
        pytest.param(
            {"entry": 42},
            pytest.raises(
                ConfigurationError, match=re.escape("must be a str (legacy) or dict")
            ),
            id="bogus-type-raises",
        ),
    ],
)
def test_normalize_aux_record(parms, context):
    """Legacy-vs-record normalisation, plus error paths."""
    with context:
        record = _normalize_aux_record(parms["entry"])
        for key, value in parms["expected"].items():
            assert record[key] == value


def test_normalize_aux_record_unknown_category_warns_and_degrades():
    """Unknown category emits ``UserWarning`` and degrades to ``scalar``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = _normalize_aux_record(
            {"name": "weird", "category": "no-such-category"}
        )

    assert record["category"] == "scalar"
    matching = [
        w
        for w in caught
        if issubclass(w.category, UserWarning) and "no-such-category" in str(w.message)
    ]
    assert len(matching) == 1, f"expected one UserWarning, got: {matching!r}"


# ---------------------------------------------------------------------------
# End-to-end round-trip via simulator_from_config()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"check": "aux-names-preserved", "expected_names": {"tablex", "ana"}},
            does_not_raise(),
            id="round-trip-aux-names",
        ),
        pytest.param(
            {
                "check": "ana-substructure-preserved",
                "expected_pseudos": ["energy"],
                "expected_reals": ["theta"],
            },
            does_not_raise(),
            id="round-trip-pseudo-positioner-substructure",
        ),
        pytest.param(
            {"check": "wh-full-does-not-raise"},
            does_not_raise(),
            id="round-trip-wh-full",
        ),
    ],
)
def test_simulator_from_config_aux_roundtrip(parms, context):
    """Nested PseudoPositioner aux survives export -> simulator_from_config."""
    with context:
        gonio = _build_gonio_with_nested_pseudo()
        cfg = gonio.configuration
        sim = simulator_from_config(cfg)

        if parms["check"] == "aux-names-preserved":
            assert set(sim.auxiliary_axis_names) == parms["expected_names"]

        elif parms["check"] == "ana-substructure-preserved":
            from ophyd import PseudoPositioner

            assert isinstance(sim.ana, PseudoPositioner)
            assert [p.attr_name for p in sim.ana.pseudo_positioners] == parms[
                "expected_pseudos"
            ]
            assert [r.attr_name for r in sim.ana.real_positioners] == parms[
                "expected_reals"
            ]

        elif parms["check"] == "wh-full-does-not-raise":
            sim.wh(full=True)


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"legacy_form": ["tablex"]}, does_not_raise(), id="v1-flat-list-of-strings"
        )
    ],
)
def test_simulator_from_config_accepts_v1_aux_form(parms, context):
    """Legacy v1 flat-list ``auxiliary_axes`` still loads without error."""
    with context:
        # Build a minimal v1-shaped config and feed it through
        # simulator_from_config().  We start from a v2 export and rewrite
        # only the auxiliary_axes field.
        from hklpy2 import creator

        donor = creator(name="donor")
        cfg = donor.configuration
        cfg["axes"]["auxiliary_axes"] = parms["legacy_form"]

        sim = simulator_from_config(cfg)
        for legacy_name in parms["legacy_form"]:
            assert legacy_name in sim.auxiliary_axis_names


# ---------------------------------------------------------------------------
# Public hook: register_aux_reconstructor()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"category": "my_cat", "builder": lambda rec: None},
            does_not_raise(),
            id="register-and-use",
        ),
        pytest.param(
            {"category": "", "builder": lambda rec: None},
            pytest.raises(ValueError, match=re.escape("non-empty str")),
            id="empty-category-rejected",
        ),
        pytest.param(
            {"category": "x", "builder": "not-callable"},
            pytest.raises(TypeError, match=re.escape("builder must be callable")),
            id="non-callable-builder-rejected",
        ),
    ],
)
def test_register_aux_reconstructor(parms, context):
    """Registry rejects bad inputs, accepts good ones, and is consulted on read."""
    with context:
        register_aux_reconstructor(parms["category"], parms["builder"])
        # On success the entry is queryable and the normaliser accepts it.
        assert _AUX_RECONSTRUCTORS[parms["category"]] is parms["builder"]

        record = _normalize_aux_record({"name": "x", "category": parms["category"]})
        # No degrade-to-scalar warning path:
        assert record["category"] == parms["category"]

        # Cleanup so we don't leak state between parametrize cases.
        _AUX_RECONSTRUCTORS.pop(parms["category"], None)


# ---------------------------------------------------------------------------
# Synthetic PseudoPositioner builder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"name": "aux1", "pseudos": ["e"], "reals": ["t"]},
            does_not_raise(),
            id="single-pseudo-single-real",
        ),
        pytest.param(
            {"name": "aux2", "pseudos": ["a", "b"], "reals": ["x", "y", "z"]},
            does_not_raise(),
            id="multi-pseudo-multi-real",
        ),
        pytest.param(
            {"name": "bad", "pseudos": [], "reals": ["t"]},
            pytest.raises(ValueError, match=re.escape("at least one pseudo")),
            id="empty-pseudos-raises",
        ),
        pytest.param(
            {"name": "bad", "pseudos": ["e"], "reals": []},
            pytest.raises(ValueError, match=re.escape("at least one real")),
            id="empty-reals-raises",
        ),
    ],
)
def test_make_aux_pseudo_positioner_class(parms, context):
    """Synthetic class is well-formed and forward/inverse return zeros."""
    with context:
        cls = make_aux_pseudo_positioner_class(
            name=parms["name"], pseudos=parms["pseudos"], reals=parms["reals"]
        )
        instance = cls(name=parms["name"])
        assert [p.attr_name for p in instance.pseudo_positioners] == parms["pseudos"]
        assert [r.attr_name for r in instance.real_positioners] == parms["reals"]
        # forward/inverse return zero-valued tuples of the right length.
        rp = instance.forward(instance.PseudoPosition(*([1.0] * len(parms["pseudos"]))))
        assert tuple(rp) == tuple([0.0] * len(parms["reals"]))
        pp = instance.inverse(instance.RealPosition(*([1.0] * len(parms["reals"]))))
        assert tuple(pp) == tuple([0.0] * len(parms["pseudos"]))


# ---------------------------------------------------------------------------
# define_real_axis() / make_component() callable-class acceptance (#388)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"spec": {"class": 42}},
            pytest.raises(
                TypeError,
                match=re.escape("must be a dotted import path (str) or a callable"),
            ),
            id="define-real-axis-bogus-class",
        ),
        pytest.param(
            {"spec": {"class": "ophyd.SoftPositioner"}},
            does_not_raise(),
            id="define-real-axis-string-class",
        ),
    ],
)
def test_define_real_axis_class_kind(parms, context):
    """``define_real_axis`` rejects non-str non-callable ``class`` values."""
    from ..devices import define_real_axis

    with context:
        define_real_axis(parms["spec"], {"labels": []})


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"call_name": 42},
            pytest.raises(TypeError, match=re.escape("must be a dotted import path")),
            id="make-component-bogus-call-name",
        ),
        pytest.param(
            {"call_name": "ophyd.SoftPositioner"},
            does_not_raise(),
            id="make-component-string-call-name",
        ),
    ],
)
def test_make_component_call_name_kind(parms, context):
    """``make_component`` rejects non-str non-callable ``call_name`` values."""
    from ..devices import make_component

    with context:
        make_component(parms["call_name"])


# ---------------------------------------------------------------------------
# simulator_from_config(): aux name overlapping a real axis is left alone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            {"overlap_name": "omega"},
            does_not_raise(),
            id="aux-name-collides-with-real-axis",
        )
    ],
)
def test_simulator_from_config_aux_overlapping_real(parms, context):
    """An aux record whose name matches a real axis is silently skipped."""
    from hklpy2 import creator

    with context:
        donor = creator(name="donor")
        cfg = donor.configuration
        # Inject an aux record whose name shadows a real axis.
        cfg["axes"]["auxiliary_axes"] = [
            {"name": parms["overlap_name"], "category": "scalar"}
        ]

        sim = simulator_from_config(cfg)
        # The real-axis omega remains a SoftPositioner (not overwritten),
        # and is not surfaced as an auxiliary.
        assert parms["overlap_name"] in sim.real_axis_names
        assert parms["overlap_name"] not in sim.auxiliary_axis_names
