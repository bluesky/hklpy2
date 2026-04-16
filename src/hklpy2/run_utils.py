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
from collections.abc import Iterator
from typing import Any
from typing import Mapping
from typing import Sequence
from typing import Union

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
    "simulator_from_config",
    "get_run_orientation",
    "list_orientation_runs",
]


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


@versionadded(
    version="0.2.3", reason="Retrieve diffractometer orientation from a Tiled run."
)
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


@versionadded(
    version="0.2.3", reason="List runs that contain diffractometer orientation data."
)
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
def simulator_from_config(config: Union[dict, str, pathlib.Path]):
    """
    Create a simulated diffractometer from a saved configuration.

    All axes are soft positioners — no hardware connections are made.
    Auxiliary axes saved by :meth:`~hklpy2.diffract.DiffractometerBase.export`
    are restored automatically.

    If the diffractometer requires auxiliary axes that are not in the
    configuration file, use :func:`~hklpy2.diffract.creator` with
    :meth:`~hklpy2.diffract.DiffractometerBase.restore` instead::

        sim = hklpy2.creator(name="e4cv", reals=dict(..., extra_axis=None))
        sim.restore("e4cv-config.yml")

    PARAMETERS

    config : dict, str, or pathlib.Path
        Configuration dictionary, or path to a YAML configuration file
        previously saved with ``diffractometer.export()``.

    RETURNS

    DiffractometerBase
        A fully configured simulated diffractometer instance.

    EXAMPLE::

        >>> import hklpy2
        >>> sim = hklpy2.simulator_from_config("e4cv-config.yml")
        >>> sim.wh()

    SEE ALSO

        :func:`~hklpy2.diffract.creator` — create a diffractometer from scratch.
    """
    from .diffract import creator

    if isinstance(config, (str, pathlib.Path)):
        logger.debug("simulator_from_config: loading from file %r", str(config))
        config = load_yaml_file(config)
    if not isinstance(config, dict):
        raise TypeError(
            f"Expected a dict or path to a YAML file. Received: {type(config)!r}"
        )
    if "_header" not in config:
        raise KeyError(MISSING_HEADER_KEY_MSG)

    solver_cfg = config.get("solver", {})
    solver_name = solver_cfg.get("name", "hkl_soleil")
    geometry = solver_cfg.get("geometry", "E4CV")

    solver_kwargs: dict = {}
    engine = solver_cfg.get("engine")
    if engine is not None:
        solver_kwargs["engine"] = engine

    axes_cfg = config.get("axes", {})
    axes_xref = axes_cfg.get("axes_xref", {})
    pseudo_axes = axes_cfg.get("pseudo_axes", [])
    real_axes = [
        ax for ax in axes_cfg.get("real_axes", []) if ax not in set(pseudo_axes)
    ]

    # Sort diffractometer real axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name) and
    # solver.real_axes (solver canonical order).
    solver_real_order = solver_cfg.get("real_axes", [])
    if solver_real_order:
        solver_to_diff_real = {v: k for k, v in axes_xref.items() if k in real_axes}
        real_axes = [
            solver_to_diff_real[s]
            for s in solver_real_order
            if s in solver_to_diff_real
        ]

    # Sort diffractometer pseudo axis names into the order the solver expects,
    # using axes_xref (diffractometer_name -> solver_canonical_name).
    # The solver canonical pseudo order is derived from the xref values for pseudos.
    pseudo_set = set(pseudo_axes)
    solver_to_diff_pseudo = {v: k for k, v in axes_xref.items() if k in pseudo_set}
    # Preserve the solver-canonical order already encoded in axes_xref values;
    # fall back to the order in axes.pseudo_axes if no xref is available.
    pseudo_solver_order = [axes_xref.get(p, p) for p in pseudo_axes]
    pseudo_axes_ordered = [
        solver_to_diff_pseudo[s]
        for s in pseudo_solver_order
        if s in solver_to_diff_pseudo
    ] or pseudo_axes

    reals_dict = {name: None for name in real_axes}

    # Restore auxiliary axes saved in the config (backward-compatible: absent in old files).
    for name in axes_cfg.get("auxiliary_axes", []):
        if name not in reals_dict:
            reals_dict[name] = None

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

    sim.restore(config)
    return sim
