"""
Limitations on acceptable positions for computed 'forward()' solutions.

Computation of the real-space axis positions given a set of reciprocal-space
coordinates can have many solutions. One or more constraints (Constraint)
(a.k.a, cut points), together with a choice of operating *mode*, can:

* Limit the range of ``forward()`` solutions accepted for that positioner.
* Declare the value to use when the positioner should be kept constant. (not
  implemented yet)

.. autosummary::

    ~RealAxisConstraints
    ~ConstraintBase
    ~LimitsConstraint

From **hklpy**, these TODO items:

- _constraints_dict
- _constraints_for_databroker
- _push_current_constraints
- _set_constraints
"""

from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional

from ..misc import NUMERIC, KeyValueMap
from ..misc import ConfigurationError
from ..misc import ConstraintsError

ENDPOINT_TOLERANCE: float = 1e-7  # for comparisons, less than motion step size
UNDEFINED_LABEL: str = "undefined"


class ConstraintBase(ABC):
    """
    Base class for all constraints for selecting 'forward()' solutions.

    .. autosummary::

        ~_asdict
        ~valid
    """

    _fields: List[str] = []
    label: str = UNDEFINED_LABEL

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        content = [f"{k}={v}" for k, v in self._asdict().items()]
        return f"{self.__class__.__name__}({', '.join(content)})"

    def _asdict(self) -> KeyValueMap:
        """Return a new dict which maps field names to their values."""
        result = {k: getattr(self, k) for k in self._fields}
        result["class"] = self.__class__.__name__
        return result

    def _fromdict(self, config: KeyValueMap, core: Optional[Any] = None):
        """Redefine this constraint from a (configuration) dictionary."""
        from ..ops import Core

        if self.__class__.__name__ != config["class"]:
            raise ConfigurationError(
                f"Wrong configuration class {self.__class__.__name__}({self.label!r})."
                f" Received: {config!r}"
            )

        if isinstance(core, Core):
            # Validate with solver.
            axis = config["label"]
            axes_local = list(core.diffractometer.real_axis_names)
            axes_solver = list(core.solver.real_axis_names)
            if axis not in axes_local + axes_solver:
                raise KeyError(
                    f"Constraint label {axis=}"
                    f" not found in diffractometer reals: {axes_local}"
                    f" or solver's reals {axes_solver}."
                )

        for k in self._fields:
            if k in config:
                setattr(self, k, config[k])
            else:
                raise ConfigurationError(
                    f"Missing key for {self.__class__.__name__}({self.label!r})."
                    f" Expected key: {k!r}."
                    f" Received configuration: {config!r}"
                )

    @abstractmethod
    def valid(self, **values: Dict[str, NUMERIC]) -> bool:
        """
        Is this constraint satisifed by current value(s)?

        PARAMETERS

        values *dict*:
            Dictionary of current 'axis: value' pairs for comparison.
        """
        # return True


class LimitsConstraint(ConstraintBase):
    """
    Value must fall between low & high limits.

    Parameters
    ----------
    low_limit : float
        Lowest acceptable value for this axis when computing real-space solutions
        from given reciprocal-space positions.
    high_limit : float
        Highest acceptable value for this axis when computing real-space solutions
        from given reciprocal-space positions.
    label : str
        Name of the axis for these limits.

    .. autosummary::

        ~limits
        ~valid
    """

    def __init__(
        self,
        low_limit: Optional[float] = -180,
        high_limit: Optional[float] = 180,
        label: Optional[str] = None,
    ) -> None:
        if label is None:
            raise ConstraintsError("Must provide a value for 'label'.")

        self.label = label
        self._fields = "label low_limit high_limit".split()

        if low_limit is None:
            low_limit = -180
        if high_limit is None:
            high_limit = 180

        self.low_limit, self.high_limit = sorted(
            map(float, [low_limit, high_limit]),
        )

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        return f"{self.low_limit} <= {self.label} <= {self.high_limit}"

    @property
    def limits(self) -> tuple[float, float]:
        """Return the low and high limits of this constraint."""
        return (self.low_limit, self.high_limit)

    @limits.setter
    def limits(self, values: tuple[float, float]) -> None:
        if len(values) != 2:
            raise ConstraintsError(f"Use exactly two values.  Received: {values!r}")
        self.low_limit, self.high_limit = sorted(map(float, values))

    def valid(self, **values: Dict[str, NUMERIC]) -> bool:
        """
        True if low <= value <= high.

        PARAMETERS

        reals *dict*:
            Dictionary of current 'axis: value' pairs for comparison.
        """
        if self.label not in values:
            raise ConstraintsError(
                f"Supplied values ({values!r}) did not include this"
                f" constraint's label {self.label!r}."
            )

        value = values[self.label]
        # return self.low_limit <= values[self.label] <= self.high_limit
        return (
            (value + ENDPOINT_TOLERANCE) >= self.low_limit
            # .
            and (value - ENDPOINT_TOLERANCE) <= self.high_limit
        )


class RealAxisConstraints(dict):
    """
    Constraints for every (real) axis of the diffractometer.

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~valid
    """

    def __init__(self, reals: List[str]) -> None:
        for k in reals:
            self[k] = LimitsConstraint(label=k)

    def __repr__(self) -> str:
        """Return a nicely-formatted string."""
        return str([str(c) for c in self.values()])

    def _asdict(self) -> KeyValueMap:
        """Return all constraints as a dictionary."""
        return {k: c._asdict() for k, c in self.items()}

    def _fromdict(self, config: KeyValueMap, core=None) -> None:
        """Redefine existing constraints from a (configuration) dictionary."""
        for k, v in config.items():
            self[k]._fromdict(v, core=core)

    def valid(self, **reals: Dict[str, NUMERIC]) -> bool:
        """Are all constraints satisfied?"""
        findings = [constraint.valid(**reals) for constraint in self.values()]
        return False not in findings
