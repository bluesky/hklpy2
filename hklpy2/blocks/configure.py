"""
Export and restore sample UB matrix and other diffractometer configuration.

.. autosummary::

    ~Configuration

From **hklpy**, these TODO items:

- bluesky runs support
- stack support
- set constraints

.. seealso:: # https://pyyaml.org/wiki/PyYAMLDocumentation
"""

import logging
from typing import Any

from ..misc import ConfigurationError
from ..misc import KeyValueMap

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

    def _fromdict(
        self,
        config: KeyValueMap,
        clear: bool = True,
        restore_constraints: bool = True,
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

        self.diffractometer.core._fromdict(config)

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
