# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""
Bluesky run-engine and databroker integration for |hklpy2|.

These utilities write diffractometer configuration into Bluesky run metadata
and retrieve orientation information from previously recorded runs.

.. autosummary::

    ~ConfigurationRunWrapper
    ~simulator_from_config
    ~get_run_orientation
    ~list_orientation_runs
"""

import logging
import pathlib
import sys
import warnings
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any
from typing import Mapping
from typing import Sequence

import pandas as pd
import tqdm
from deprecated.sphinx import versionadded
from deprecated.sphinx import versionchanged
from ophyd import Device

from .utils import DEFAULT_START_KEY
from .utils import MISSING_HEADER_KEY_MSG
from .utils import load_yaml_file
from .typing import KeyValueMap

logger = logging.getLogger(__name__)

__all__ = [
    "ConfigurationRunWrapper",
    "get_run_orientation",
    "list_orientation_runs",
    "register_aux_reconstructor",
    "simulator_from_config",
]


# ---------------------------------------------------------------------------
# Reserved keys in the persisted ``solver:`` configuration block.
#
# Every other key found in ``solver:`` at restore time is forwarded into
# ``solver_kwargs`` so that solver-specific construction state persisted via
# a ``_metadata`` override survives the export/restore round-trip
# (issue #405).  Out-of-tree solver authors should avoid collisions with
# this set when extending ``_metadata``.
#
# Reserved key meanings:
#   * ``name``         — selects the solver class
#   * ``geometry``     — selects the solver geometry
#   * ``real_axes``    — solver-canonical real-axis ordering
#   * ``description``  — informational only
#   * ``version``      — informational only
#   * ``mode``         — re-applied post-construction by
#                        ``Diffractometer.restore()``; must not also flow
#                        as a construction kwarg
# ---------------------------------------------------------------------------
_RESERVED_SOLVER_KEYS: frozenset[str] = frozenset(
    {"name", "description", "geometry", "real_axes", "version", "mode"}
)


# ---------------------------------------------------------------------------
# Auxiliary-axis reconstruction (issue #388)
# ---------------------------------------------------------------------------

#: Built-in / user-registered callables that turn an aux record (``dict``
#: with ``name`` / ``category`` / …) into a ``creator()`` ``reals=`` value
#: (``None`` for a scalar SoftPositioner, or a dict-spec describing a
#: nested PseudoPositioner stand-in).  Keyed by the record's
#: ``category``.  Use :func:`register_aux_reconstructor` to register a
#: custom builder; built-ins live in :data:`_BUILTIN_AUX_RECONSTRUCTORS`.
_AUX_RECONSTRUCTORS: dict[str, Callable[[Mapping[str, Any]], Any]] = {}


@versionadded(
    version="0.7.0",
    reason="Public hook for restoring custom auxiliary sub-devices.  See :issue:`388`.",
)
def register_aux_reconstructor(
    category: str,
    builder: Callable[[Mapping[str, Any]], Any],
) -> None:
    """
    Register a builder for a given aux-record ``category``.

    The ``builder`` receives a normalised record (``dict`` with at least
    ``name`` and ``category``) and returns whatever
    :func:`~hklpy2.diffract.creator` accepts as a ``reals=`` value:

    * ``None`` for a scalar :class:`~ophyd.SoftPositioner` (the
      historic default);
    * a ``dict`` spec of the form ``{"class": <callable or import path>,
      ...kwargs}`` for any nested device.

    Built-in categories — ``"scalar"`` and ``"pseudo_positioner"`` —
    may be overridden by a later registration.

    PARAMETERS

    category : str
        The ``category`` value the builder handles (e.g. a custom
        ``"device_group"``).
    builder : callable
        ``builder(record) -> creator-reals-spec``.
    """
    if not isinstance(category, str) or not category:
        raise ValueError(f"category must be a non-empty str; got {category!r}")
    if not callable(builder):
        raise TypeError(f"builder must be callable; got {type(builder).__name__!r}")
    _AUX_RECONSTRUCTORS[category] = builder


def _normalize_aux_record(entry: Any) -> dict:
    """Coerce a v1 (str) or v2 (dict) aux entry into the v2 record form."""
    from .exceptions import ConfigurationError

    if isinstance(entry, str):  # legacy v1 form
        return {"name": entry, "category": "scalar"}
    if not isinstance(entry, Mapping):
        raise ConfigurationError(f"auxiliary_axes entry must be a str (legacy) or dict; got {entry!r}")
    rec = dict(entry)
    if "name" not in rec or not isinstance(rec["name"], str) or not rec["name"]:
        raise ConfigurationError(f"auxiliary_axes record is missing a non-empty 'name': {entry!r}")
    rec.setdefault("category", "scalar")
    if rec["category"] not in _AUX_RECONSTRUCTORS:
        warnings.warn(
            (
                f"Unknown auxiliary_axes category {rec['category']!r}"
                f" for {rec['name']!r}; falling back to scalar SoftPositioner."
            ),
            UserWarning,
            stacklevel=3,
        )
        rec["category"] = "scalar"
    return rec


def _build_scalar_aux(record: Mapping[str, Any]) -> Any:
    """Built-in builder for ``category == 'scalar'``: a SoftPositioner."""
    return None  # creator() interprets None as a default SoftPositioner


def _build_pseudo_positioner_aux(record: Mapping[str, Any]) -> dict:
    """Built-in builder for ``category == 'pseudo_positioner'``."""
    from .devices import make_aux_pseudo_positioner_class

    cls = make_aux_pseudo_positioner_class(
        name=record["name"],
        pseudos=record.get("pseudos") or [],
        reals=record.get("reals") or [],
    )
    return {"class": cls, "kind": "hinted"}


# Register built-ins.  Users may override via register_aux_reconstructor().
_AUX_RECONSTRUCTORS["scalar"] = _build_scalar_aux
_AUX_RECONSTRUCTORS["pseudo_positioner"] = _build_pseudo_positioner_aux


class ConfigurationRunWrapper:
    """
    Write configuration of supported device(s) to a bluesky run.

    EXAMPLE::

        crw = ConfigurationRunWrapper(sim4c2)
        RE.preprocessors.append(crw.wrapper)
        RE(bp.rel_scan([noisy], m1, -1.2, 1.2, 11))

    Disable the preprocessor::

        crw.enable = False  # 'True' to enable

    Remove the last preprocessor::

        RE.preprocessors.pop()

    Add another diffractometer::

        crw.devices.append(e4cv)

    .. autosummary::

        ~device_names
        ~devices
        ~enable
        ~known_bases
        ~start_key
        ~validate
        ~wrapper
    """

    devices: Sequence[Device] = []
    """List of devices to be reported."""

    known_bases: Sequence[Device] = []
    """
    Known device base classes.

    Any device (base class) that reports its configuration dictionary in
    the `.read_configuration()` method can be added to this tuple.
    """

    start_key: str = DEFAULT_START_KEY
    """Top-level key in run's metadata dictionary."""

    def __init__(self, *devices, knowns=None) -> None:
        """
        Constructor.

        EXAMPLES::

            ConfigurationRunWrapper(sim4c)
            ConfigurationRunWrapper(e4cv, e6c)

        PARAMETERS

        devices : list
            List of supported objects to be reported.
        knowns : list
            List of base classes that identify supported objects.
            (default: :class:`hklpy2.DiffractometerBase`)
        """
        from .diffract import DiffractometerBase as hklpy2_DiffractometerBase

        self.enable = True
        self.known_bases = knowns or [hklpy2_DiffractometerBase]
        self.validate(devices)
        self.devices = list(devices)

    @property
    def device_names(self) -> list[str]:
        """Return list of configured device names."""
        return [dev.name for dev in self.devices]

    @property
    def enable(self) -> bool:
        """Is it permitted to write device configuration?"""
        return self._enable

    @enable.setter
    def enable(self, state: bool) -> None:
        """Set permit to write configuration."""
        self._enable = state

    def validate(self, devices: Sequence[Device]) -> None:
        """Verify all are recognized objects."""
        for dev in devices:
            if not isinstance(dev, tuple(self.known_bases)):
                raise TypeError(f"{dev} is not a recognized object.")

    def wrapper(self, plan: Iterator):
        """
        Bluesky plan wrapper (preprocessor).

        Writes device(s) configuration to start document metadata.

        Example::

            crw = ConfigurationRunWrapper(e4cv)
            RE.preprocessors.append(crw.wrapper)
        """
        from bluesky import preprocessors as bpp

        if not self._enable or len(self.devices) == 0:
            # Nothing to do here, move on.
            return (yield from plan)

        self.validate(self.devices)

        cfg = {dev.name: dev.configuration for dev in self.devices}

        return (yield from bpp.inject_md_wrapper(plan, {self.start_key: cfg}))


@versionadded(version="0.2.3", reason="Retrieve diffractometer orientation from a Tiled run.")
@versionchanged(version="0.4.0", reason="Exported from top-level ``hklpy2`` namespace.")
def get_run_orientation(
    run: Any,
    name=None,
    start_key: str = DEFAULT_START_KEY,
) -> KeyValueMap:
    """
    Return the orientation information dictionary from a run.

    EXAMPLE::

        In [3]: get_run_orientation(cat[9752], name="sim4c2")
        Out[3]:
        {'_header': {'datetime': '2025-02-27 15:54:33.364719',
        'hklpy2_version': '0.0.26.dev72+gcf9a65a.d20250227',
        'python_class': 'Hklpy2Diffractometer',
        'source_type': 'X-ray',
        'energy_units': 'keV',
        'energy': 12.398419843856837,
        'wavelength_units': 'angstrom',
        'wavelength': 1.0},
        'name': 'sim4c2',
        'axes': {'pseudo_axes': ['h', 'k', 'l'],
        'real_axes': ['omega', 'chi', 'phi', 'tth'],
        'axes_xref': {'h': 'h',
        'k': 'k',
        'l': 'l',
        'omega': 'omega',
        'chi': 'chi',
        'phi': 'phi',
        'tth': 'tth'},
        'extra_axes': {}},
        'sample_name': 'sample',
        'samples': {'sample': {'name': 'sample',
        'lattice': {'a': 1,
            'b': 1,
            'c': 1,
            'alpha': 90.0,
            'beta': 90.0,
            'gamma': 90.0},
        'reflections': {},
        'reflections_order': [],
        'U': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'UB': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        'digits': 4}},
        'constraints': {'omega': {'label': 'omega',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'chi': {'label': 'chi',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'phi': {'label': 'phi',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'},
        'tth': {'label': 'tth',
        'low_limit': -180.0,
        'high_limit': 180.0,
        'class': 'LimitsConstraint'}},
        'solver': {'name': 'hkl_soleil',
        'description': "HklSolver(name='hkl_soleil', version='5.1.2', geometry='E4CV', engine_name='hkl', mode='bissector')",
        'geometry': 'E4CV',
        'real_axes': ['omega', 'chi', 'phi', 'tth'],
        'version': '5.1.2',
        'engine': 'hkl'}}


    Parameters
    ----------
    run : object
        Bluesky run object.
    name : str
        (optional)
        Name of the diffractometer. (default=None, returns all available.)
    start_key : str
        Metadata key where the orientation information is stored in the start
        document.  (default="diffractometers")
    """
    info = run.metadata["start"].get(start_key, {})
    if isinstance(name, str):
        info = info.get(name, {})
    return info


@versionadded(version="0.2.3", reason="List runs that contain diffractometer orientation data.")
@versionchanged(version="0.4.0", reason="Exported from top-level ``hklpy2`` namespace.")
def list_orientation_runs(
    catalog: Any,
    limit: int = 10,
    start_key: str = DEFAULT_START_KEY,
    **kwargs: Mapping,
) -> pd.DataFrame:
    """
    List the runs with orientation information.

    EXAMPLE::

        In [42]: list_orientation_runs(cat, limit=5, date="_header.datetime")
        Out[42]:
            scan_id      uid  sample diffractometer geometry      solver                        date
        0      9752  41f71e9  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        1      9751  36e38bc  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        2      9750  62e425d  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:54:33.364719
        3      9749  18b11f0  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:53:55.958929
        4      9748  bf9912f  sample         sim4c2     E4CV  hkl_soleil  2025-02-27 15:53:55.958929

    Returns
    -------
    Table of orientation runs: Pandas DataFrame object

    Parameters
    ----------
    catalog : object
        Catalog of bluesky runs.
    limit : int
        Limit the list to at most ``limit`` runs. (default=10)
        It could take a long time to search an entire catalog.
    start_key : str
        Metadata key where the orientation information is stored in the start
        document.  (default="diffractometers")
    **kwargs : dict[str:str]
        Keyword parameters describing data column names to be displayed. The
        value of each column name is the dotted path to the orientation
        information (in the start document's metadata).
    """
    buffer = []
    _count = 0
    columns = dict(
        sample="sample_name",
        diffractometer="name",
        geometry="solver.geometry",
        solver="solver.name",
    )
    columns.update(**kwargs)
    try:
        container = catalog.v2  # data broker catalog
    except AttributeError:
        container = catalog  # tiled Container
    limit = min(limit, len(container))
    with tqdm.tqdm(total=limit, file=sys.stdout, leave=False) as progress_bar:
        for full_uid in container:
            _count += 1
            run = container[full_uid]
            start_md = run.metadata.get("start", {})
            info = get_run_orientation(run, start_key=start_key)
            if info is not None:

                def get_subdict_value(biblio, full_key):
                    value = biblio
                    for key in full_key.split("."):
                        value = (value or {}).get(key)
                    return value

                for device in sorted(info):
                    orientation = info[device]
                    row = dict(
                        scan_id=start_md.get("scan_id", 0),
                        uid=full_uid[:7],
                    )
                    for f, addr in columns.items():
                        value = get_subdict_value(orientation, addr)
                        if value is not None:
                            row[f] = value
                    buffer.append(row)

            progress_bar.update()
            if _count >= limit:
                break
    return pd.DataFrame(buffer)


@versionadded(
    version="0.4.0",
    reason="Create a simulated diffractometer from a saved configuration.",
)
@versionchanged(
    version="0.6.1",
    reason="Accept a ``DiffractometerBase`` instance directly.",
)
@versionchanged(
    version="0.7.0",
    reason=(
        "Restore nested ``PseudoPositioner`` auxiliaries as structural "
        "stand-ins (forward/inverse return zeros).  ``axes.auxiliary_axes`` "
        "is now a list of ``{name, category, …}`` records; the legacy flat "
        "list of names is still accepted on read.  See :issue:`388`."
    ),
)
@versionchanged(
    version="0.7.1",
    reason=(
        "Forward every non-reserved key in the ``solver:`` block as a "
        "``solver_kwargs`` entry so that solver-specific construction "
        "state persisted via ``_metadata`` survives the round-trip.  "
        "See :issue:`405`."
    ),
)
def simulator_from_config(config):
    """
    Create a simulated diffractometer from a saved configuration.

    All axes are soft positioners — no hardware connections are made.
    Auxiliary axes and solver mode saved by
    :meth:`~hklpy2.diffract.DiffractometerBase.export` are restored
    automatically.

    If the diffractometer requires auxiliary axes that are not in the
    configuration file, use :func:`~hklpy2.diffract.creator` with
    :meth:`~hklpy2.diffract.DiffractometerBase.restore` instead::

        sim = hklpy2.creator(name="e4cv", reals=dict(..., extra_axis=None))
        sim.restore("e4cv-config.yml")

    PARAMETERS

    config : dict, str, pathlib.Path, or DiffractometerBase
        One of:

        * a configuration dictionary,
        * a path to a YAML configuration file previously saved with
          ``diffractometer.export()``, or
        * a :class:`~hklpy2.diffract.DiffractometerBase` instance (its
          current configuration snapshot will be used).

    RETURNS

    DiffractometerBase
        A fully configured simulated diffractometer instance.

    EXAMPLE::

        >>> import hklpy2
        >>> sim = hklpy2.simulator_from_config("e4cv-config.yml")
        >>> sim.wh()

        Or directly from an existing diffractometer::

        >>> sim = hklpy2.simulator_from_config(diffractometer)

    .. rubric:: Solver construction kwargs

    Every key in the persisted ``solver:`` block *except* the reserved
    set ``{"name", "description", "geometry", "real_axes", "version",
    "mode"}`` is forwarded to the solver's constructor as a
    ``solver_kwargs`` entry.  This lets a solver persist arbitrary
    construction state by overriding its ``_metadata`` property and
    accepting the matching keyword in its ``__init__``.  See
    :issue:`405`.  Solver authors are responsible for choosing
    ``_metadata`` key names that do not collide with the reserved set.

    SEE ALSO

        :func:`~hklpy2.diffract.creator` — create a diffractometer from scratch.
    """
    from .diffract import DiffractometerBase
    from .diffract import creator

    if isinstance(config, DiffractometerBase):
        logger.debug(
            "simulator_from_config: snapshotting diffractometer %r",
            getattr(config, "name", None),
        )
        config = config.configuration
    if isinstance(config, (str, pathlib.Path)):
        logger.debug("simulator_from_config: loading from file %r", str(config))
        config = load_yaml_file(config)
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a dict, path to a YAML file, or DiffractometerBase instance. Received: {type(config)!r}"
        )
    if "_header" not in config:
        raise KeyError(MISSING_HEADER_KEY_MSG)
    from .blocks.configure import _check_schema_version

    _check_schema_version(config["_header"])

    solver_cfg = config.get("solver", {})
    # Default solver name remains "hkl_soleil" because the config
    # does not include a way to discover a system-wide default solver
    # (see #372 discussion); the geometry default now flows from the
    # solver's own default_geometry() classmethod.
    solver_name = solver_cfg.get("name") or "hkl_soleil"
    from .solver_utils import get_solver

    geometry = solver_cfg.get("geometry") or get_solver(solver_name).default_geometry()

    # Generic forwarding (issue #405): every non-reserved key in the
    # ``solver:`` block flows through to the solver constructor via
    # ``solver_kwargs``.  This subsumes the previous ``engine`` special
    # case and lets out-of-tree solvers persist construction state by
    # overriding ``_metadata`` -- no new abstract API is required.
    solver_kwargs: dict = {k: v for k, v in solver_cfg.items() if k not in _RESERVED_SOLVER_KEYS}
    if solver_kwargs:
        logger.debug("simulator_from_config: forwarding solver_kwargs=%r", solver_kwargs)

    axes_cfg = config.get("axes", {})
    axes_xref = axes_cfg.get("axes_xref", {})
    pseudo_axes = axes_cfg.get("pseudo_axes", [])
    real_axes = [ax for ax in axes_cfg.get("real_axes", []) if ax not in set(pseudo_axes)]

    # Sort diffractometer real axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name) and
    # solver.real_axes (solver canonical order).
    solver_real_order = solver_cfg.get("real_axes", [])
    if solver_real_order:
        solver_to_diff_real = {v: k for k, v in axes_xref.items() if k in real_axes}
        real_axes = [solver_to_diff_real[s] for s in solver_real_order if s in solver_to_diff_real]

    # Sort diffractometer pseudo axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name).
    # The solver canonical pseudo order is derived from the xref values for pseudos.
    pseudo_set = set(pseudo_axes)
    solver_to_diff_pseudo = {v: k for k, v in axes_xref.items() if k in pseudo_set}
    # Preserve the solver-canonical order already encoded in axes_xref values;
    # fall back to the order in axes.pseudo_axes if no xref is available.
    pseudo_solver_order = [axes_xref.get(p, p) for p in pseudo_axes]
    pseudo_axes_ordered = [
        solver_to_diff_pseudo[s] for s in pseudo_solver_order if s in solver_to_diff_pseudo
    ] or pseudo_axes

    reals_dict = dict.fromkeys(real_axes)

    # Restore auxiliary axes saved in the config.  Backward-compatible:
    #   * absent in very old files (handled by .get() default);
    #   * v1 schema: flat list of names (str) — normalised to scalar records;
    #   * v2 schema: list of {name, category, ...} records.
    # See issue #388.
    for entry in axes_cfg.get("auxiliary_axes", []):
        record = _normalize_aux_record(entry)
        aux_name = record["name"]
        if aux_name in reals_dict:
            # An aux that overlaps with a real axis is not an auxiliary;
            # leave the real-axis entry alone.
            continue
        builder = _AUX_RECONSTRUCTORS[record["category"]]
        reals_dict[aux_name] = builder(record)

    # Pass _real and _pseudo so creator() maps axes in solver-expected order
    # even when diffractometer names differ from solver canonical names.
    diffractometer_name = config.get("name", geometry.lower())

    logger.debug(
        "simulator_from_config: creating %r solver=%r geometry=%r",
        diffractometer_name,
        solver_name,
        geometry,
    )
    sim = creator(
        name=diffractometer_name,
        solver=solver_name,
        geometry=geometry,
        solver_kwargs=solver_kwargs,
        reals=reals_dict,
        _real=real_axes if real_axes else None,
        _pseudo=pseudo_axes_ordered if pseudo_axes_ordered else None,
    )

    # restore_mode=True is safe here: simulator_from_config() always produces
    # a simulator with no hardware connections, so mode changes cannot cause
    # unexpected motion.
    sim.restore(config, restore_mode=True)
    return sim
