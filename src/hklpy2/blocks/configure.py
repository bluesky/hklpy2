"""
Export and restore sample UB matrix and other diffractometer configuration.

.. autosummary::

    ~Configuration

.. seealso:: # https://pyyaml.org/wiki/PyYAMLDocumentation
"""

import logging
from typing import Any

from deprecated.sphinx import versionchanged

from ..exceptions import ConfigurationError
from ..typing import KeyValueMap

logger = logging.getLogger(__name__)


class Configuration:
    """
    Manage diffractometer configurations.

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~_valid
    """

    def __init__(self, diffractometer: object) -> None:
        self.diffractometer = diffractometer

    def _asdict(self) -> KeyValueMap:
        """Return diffractometer's configuration as a dict."""
        return self.diffractometer.core._asdict()

    @versionchanged(
        version="0.6.2",
        reason=(
            "Add ``restore_samples`` / ``restore_extras`` / "
            "``restore_presets`` flags so callers can scope which sections "
            "of the configuration are applied.  See :issue:`390`."
        ),
    )
    def _fromdict(
        self,
        config: KeyValueMap,
        clear: bool = True,
        restore_constraints: bool = True,
        restore_samples: bool = True,
        restore_extras: bool = True,
        restore_presets: bool = True,
    ) -> None:
        """Restore diffractometer's configuration from a dict."""
        self._valid(config)  # will raise if invalid

        if clear:
            self.diffractometer.core.reset_constraints()
            self.diffractometer.core.reset_samples()

        if restore_constraints:
            controls = {}
            oper = self.diffractometer.core  # alias
            for axis, v in config["constraints"].items():
                axis_canonical = config["axes"]["axes_xref"][axis]
                axis_local = oper.axes_xref_reversed[axis_canonical]
                v["label"] = axis_local  # must match
                controls[axis_local] = v
            config["constraints"] = controls
        else:
            config["constraints"] = {}

        self.diffractometer.core._fromdict(
            config,
            restore_samples=restore_samples,
            restore_extras=restore_extras,
            restore_presets=restore_presets,
        )

    @versionchanged(
        version="0.6.2",
        reason=(
            "Validate ``extra_axes`` early so a mismatch raises a clear "
            "``ConfigurationError`` instead of a deep ``KeyError`` from "
            "``Core._validate_extras``.  See :issue:`390`."
        ),
    )
    def _valid(self, config: KeyValueMap) -> bool:
        """Validate incoming configuration for current diffractometer."""

        def compare(incoming: Any, existing: any, template: str) -> None:
            if incoming != existing:
                raise ConfigurationError(template % (incoming, existing))

        compare(
            config.get("solver", {}).get("name"),
            self.diffractometer.core.solver_name,
            "solver mismatch: incoming=%r existing=%r",
        )
        if "engine" in dir(self.diffractometer.core.solver):
            compare(
                config.get("solver", {}).get("engine"),
                self.diffractometer.core.solver.engine_name,
                "engine mismatch: incoming=%r existing=%r",
            )
        compare(
            config.get("solver", {}).get("geometry"),
            self.diffractometer.core.geometry,
            "geometry mismatch: incoming=%r existing=%r",
        )
        compare(
            config.get("axes", {}).get("pseudo_axes"),
            # ignore any extra pseudos
            self.diffractometer.pseudo_axis_names[
                : len(
                    # long line
                    self.diffractometer.core.solver_pseudo_axis_names
                )
            ],
            "pseudo axis mismatch: incoming=%r existing=%r",
        )
        compare(
            config.get("solver", {}).get("real_axes"),
            self.diffractometer.core.solver_real_axis_names,
            "solver real axis mismatch: incoming=%r existing=%r",
        )
        # Validate extras early so a mismatch raises a clear
        # ConfigurationError before Core._validate_extras would raise the
        # opaque KeyError.
        incoming_extras = config.get("axes", {}).get("extra_axes") or {}
        existing_extras = self.diffractometer.core.all_extras or {}
        unexpected = sorted(set(incoming_extras) - set(existing_extras))
        if unexpected:
            raise ConfigurationError(
                f"extra axis mismatch: unexpected extras {unexpected!r}"
                f" not in solver-known extras {sorted(existing_extras)!r}"
            )
