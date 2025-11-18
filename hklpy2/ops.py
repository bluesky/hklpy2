"""
Operate the diffractometer using a |solver| library and geometry.

Intermediate layer between DiffractometerBase Device and backend |solver|
library.

.. autosummary::

    ~Core
"""

import datetime
import logging
from collections.abc import Iterable
from typing import List
from typing import Optional
from typing import Union

from .backends.base import SolverBase
from .blocks.configure import Configuration
from .blocks.constraints import RealAxisConstraints
from .blocks.lattice import Lattice
from .blocks.reflection import Reflection
from .blocks.sample import Sample
from .misc import AnyAxesType
from .misc import AxesDict
from .misc import CoreError
from .misc import NoForwardSolutions
from .misc import axes_to_dict
from .misc import convert_units
from .misc import solver_factory
from .misc import unique_name

__all__ = ["Core"]

Number = Union[int, float]
logger = logging.getLogger(__name__)
DEFAULT_EXTRA_VALUE = 0
DEFAULT_SAMPLE_NAME = "sample"


class Core:
    """
    Core operations of a diffractometer, coordinating with sample & |solver|.

    PARAMETERS

    diffractometer (DiffractometerBase):
        The diffractometer parent.
    default_sample (bool):
        If 'True' (default), create a 'sample' with 1 angstrom cubic lattice.

    .. rubric:: Python Methods

    .. autosummary::

        ~_asdict
        ~_fromdict
        ~_validate_pseudos
        ~add_reflection
        ~add_sample
        ~assign_axes
        ~calc_UB
        ~forward
        ~geometries
        ~inverse
        ~local_pseudo_axes
        ~local_real_axes
        ~refine_lattice
        ~remove_sample
        ~request_solver_update
        ~reset_constraints
        ~reset_samples
        ~set_solver
        ~standardize_pseudos
        ~standardize_reals
        ~update_solver

    .. rubric:: Python Properties

    .. autosummary::

        ~all_extras
        ~extras
        ~geometry
        ~mode
        ~modes
        ~sample
        ~solver
        ~solver_extra_axis_names
        ~solver_name
        ~solver_pseudo_axis_names
        ~solver_real_axis_names
        ~solver_signature
        ~solver_summary
    """

    from .blocks.sample import Sample

    def __init__(self, diffractometer, default_sample: bool = True) -> None:
        self.axes_xref = {}  # cross-reference:  diffractometer name : solver name
        self.diffractometer = diffractometer
        self._extras = {}  # Dictionary of any extra solver axis (across all modes).
        self._mode = None
        self._sample_name = None
        self._samples = {}
        self._solver = None
        self.constraints = None
        self.configuration = None
        self.request_solver_update()

        if default_sample:
            # first sample is cubic, no reflections
            self.add_sample(DEFAULT_SAMPLE_NAME, 1)

    def _asdict(self):
        """Describe the diffractometer as a dictionary."""
        from .__init__ import __version__

        config = {
            "_header": {
                "datetime": str(datetime.datetime.now()),
                "hklpy2_version": __version__,
                "python_class": self.diffractometer.__class__.__name__,
            },
            "name": self.diffractometer.name,
            "axes": {
                "pseudo_axes": self.diffractometer.pseudo_axis_names,
                "real_axes": self.diffractometer.real_axis_names,
                "axes_xref": self.axes_xref,
                "extra_axes": self.all_extras,
            },
            "digits": self.diffractometer.digits,
            "sample_name": self.sample.name,
            "samples": {k: v._asdict() for k, v in self._samples.items()},
            "constraints": self.constraints._asdict(),
            "solver": self.solver._metadata,
            "beam": self.diffractometer.beam._asdict(),
        }

        if "engine_name" in dir(self.solver):
            config["solver"]["engine"] = self.solver.engine_name

        return config

    def _axes_names_s2d(self, axis_dict: dict[str, float]) -> dict[str, float]:
        """Convert keys of axis dictionary from solver to diffractometer."""
        reverse = self.axes_xref_reversed
        return {reverse[k]: v for k, v in axis_dict.items()}

    def _axes_names_d2s(self, axis_dict: dict[str, float]) -> dict[str, float]:
        """Convert keys of axis dictionary from diffractometer to solver."""
        return {self.axes_xref[k]: v for k, v in axis_dict.items()}

    def _fromdict(self, config):
        """Redefine diffractometer from a (configuration) dictionary."""
        # Since this code might raise, validate first.
        extras = self._validate_extras(config["axes"]["extra_axes"], self.all_extras)
        if len(extras) > 0:
            self._extras.update(extras)

        digits = config.get("digits")
        if isinstance(digits, int) and 0 <= digits:
            self.diffractometer.digits = digits

        for key, sample in config["samples"].items():
            sample_object = self.add_sample(key, 1, replace=True)
            sample_object._fromdict(sample, core=self)
        sname = config.get("sample_name")
        if sname is not None:
            self.sample = sname

        for key, constraint in config["constraints"].items():
            if (
                constraint["class"] == "LimitsConstraint"
                # .
                and constraint["label"] in config["axes"]["real_axes"]
            ):
                # By convention, the 'key' here is the axis name when config was written.
                axis_canonical = config["axes"]["axes_xref"][key]
                axis_local = self.axes_xref_reversed[axis_canonical]
                constraint["label"] = axis_local
        self.constraints._fromdict(config["constraints"], core=self)

    def _validate_extras(
        self,
        values: dict[str, Number],
        expected: dict[str, Number],
    ) -> dict[str, Number]:
        """Validate that the supplied extras are acceptable."""
        extras, unexpected = {}, []
        for key, value in values.items():
            if key in expected:
                extras[key] = value
            else:
                unexpected.append(key)
        if len(unexpected) > 0:
            raise KeyError(
                f"Unexpected extra axis name(s) {unexpected!r}."
                # ..
                f"  Expected names: {expected}."
            )
        return extras

    def _validate_pseudos(self, pseudos) -> bool:
        """Validate that the supplied pseudos are acceptable."""
        if not isinstance(pseudos, Iterable):
            raise TypeError(
                "Pseudos must be tuple, list, or dict."
                # Always show the input.
                f"  Received {pseudos!r}"
            )
        if not isinstance(pseudos, (dict, list, set, tuple)):
            raise TypeError(f"Unexpected data type: {pseudos}")

        expected_names = self.local_pseudo_axes
        if len(pseudos) != len(expected_names):
            raise ValueError(
                f"Expected {len(expected_names)} pseudos,"
                # Always show the input.
                f" received {pseudos}"
            )

        original = pseudos  # Keep the original for reporting.
        if hasattr(pseudos, "_asdict"):
            pseudos = pseudos._asdict()
        if isinstance(pseudos, (list, set, tuple)):
            # Expect values are provided in canonical order.
            pseudos = {
                axis: value
                # rewrite as dictionary
                for axis, value in zip(expected_names, pseudos)
            }
        for axis in expected_names:
            if axis not in pseudos:
                raise ValueError(f"Wrong axis names: received {original}")
            if not isinstance(pseudos[axis], (float, int)):
                raise TypeError(f"Must be number, received {original}")

        return True

    def add_reflection(
        self,
        pseudos: AnyAxesType,
        reals: Union[AnyAxesType, None] = None,
        wavelength=None,
        wavelength_units: str = None,
        name=None,
        replace: bool = False,
    ) -> Reflection:
        """
        Add a new reflection.

        PARAMETERS

        pseudos various:
            Pseudo-space axes and values.
        reals various:
            Dictionary of real-space axes and values.
        wavelength float:
            Wavelength of incident radiation. The created Reflection will
            use ``wavelength_units`` from the optional parameter below; if
            that is ``None``, the diffractometer's current
            ``beam.wavelength_units`` is used. When preparing data for the
            solver, Core converts each reflection's wavelength into the
            solver's internal units (``INTERNAL_LENGTH_UNITS``).
        wavelength_units str or None:
            Units for the ``wavelength`` value (e.g. "angstrom"). If
            provided, this overrides the diffractometer beam units for the
            new Reflection; otherwise the beam units are used. The
            Reflection stores this string and Core will convert values for
            solver consumption.
        name str:
            Reference name for this reflection.  If ``None``, a random name will
            be assigned.
        replace bool:
            When ``True``, replace existing reflection of this name.
            (default: ``False``)
        """
        from .blocks.reflection import Reflection

        self._validate_pseudos(pseudos)
        wavelength = wavelength or self.diffractometer.beam.wavelength.get()

        logger.debug(
            "name=%r, geometry=%r, wavelength=%r",
            name,
            self.geometry,
            wavelength,
        )

        pnames = self.local_pseudo_axes
        rnames = self.local_real_axes
        pdict = self.standardize_pseudos(pseudos)
        rdict = self.standardize_reals(reals)
        logger.debug(
            "pdict=%r, rdict=%r, pnames=%r, rnames=%r",
            pdict,
            rdict,
            pnames,
            rnames,
        )
        # Thread the current beam wavelength units into the new Reflection
        # prefer an explicit wavelength_units argument; otherwise use beam units
        wl_units = (
            wavelength_units
            if wavelength_units is not None
            else self.diffractometer.beam.wavelength_units.get()
        )
        refl = Reflection(
            name or unique_name(),
            pdict,
            rdict,
            wavelength,
            self.geometry,
            pnames,
            rnames,
            wavelength_units=wl_units,
        )
        self.sample.reflections.add(refl, replace=replace)
        return refl

    def add_sample(
        self,
        name: str,
        a: float,
        b: float = None,
        c: float = None,
        alpha: float = 90.0,  # degrees
        beta: float = None,  # degrees
        gamma: float = None,  # degrees
        digits: int = 4,
        replace: bool = False,
    ) -> Sample:
        """Add a new sample."""
        if name in self.samples:
            if not replace:
                raise CoreError(f"Sample {name=!r} already defined.")
        lattice = Lattice(a, b, c, alpha, beta, gamma, digits=digits)
        self._samples[name] = Sample(self, name, lattice)
        self.sample = name
        self.request_solver_update(True)  # 58  test_diffract line 410: 2 pi a
        if self.solver is not None:
            self.solver.sample = self.to_solver_units()["sample"]
        return self.sample

    @property
    def all_extras(self) -> list[str]:
        """Sorted dictionary of |solver| extra parameters in any mode."""
        return self._extras

    def assign_axes(self, pseudos: list[str], reals: list[str]) -> None:
        """
        Designate attributes for use by the PseudoPositioner class.

        Result is re-definition of 'self.axes_xref'.
        """
        pseudos = pseudos or []
        reals = reals or []

        def itemize(label, select, full):
            keys = [name for name, _obj in full]
            for attr in select:
                if attr not in keys:
                    raise KeyError(f"Unknown {label}={attr!r}.  Known: {keys!r}")
            return keys

        def rebuild_axes_xref(dnames, snames):
            for dname, sname in zip(dnames, snames):
                self.axes_xref[dname] = sname
                both_p_r.remove(dname)

        # check for duplicates
        if len(set(pseudos + reals)) != len(pseudos + reals):
            raise ValueError("Axis name cannot be in more than list.")

        if self.solver is None:
            return  # such as initialization

        dfrct = self.diffractometer
        all_pseudos = itemize("pseudo", pseudos, dfrct._get_pseudo_positioners())
        all_reals = itemize("real", reals, dfrct._get_real_positioners())
        both_p_r = all_pseudos + all_reals

        self.axes_xref = {}
        rebuild_axes_xref(pseudos, self.solver.pseudo_axis_names)
        rebuild_axes_xref(reals, self.solver.real_axis_names)
        self.reset_constraints()
        logger.debug("axes_xref=%r", self.axes_xref)
        self.configuration = Configuration(self.diffractometer)

    @property
    def axes_xref_reversed(self):
        """Map axis names from solver to diffractometer."""
        if len(self.axes_xref) == 0:
            if self.solver is not None:
                names = self.solver.pseudo_axis_names + self.solver.real_axis_names
                if len(names) == 0:
                    return {}
            raise CoreError("Did you forget to call `assign_axes()`?")
        return {v: k for k, v in self.axes_xref.items()}

    def calc_UB(
        self, r1: Union[Reflection, str], r2: Union[Reflection, str]
    ) -> List[List[Number]]:
        """
        Calculate and return the UB (orientation) matrix with two reflections.

        The method of Busing & Levy, Acta Cryst 22 (1967) 457.
        """

        def _get(r):
            """Given a reference, get the Reflection object."""
            if isinstance(r, Reflection):
                return r
            reflection = self.sample.reflections.get(r)
            if reflection is None:
                raise KeyError(
                    f"{reflection!r} unknown."
                    # .
                    f"  Knowns: {list(self.sample.reflections)!r}"
                )
            return reflection

        two_reflections = [_get(r1), _get(r2)]
        self.sample.reflections.set_orientation_reflections(two_reflections)

        solver_reflections = self._reflections_to_solver(two_reflections)
        ub = self.solver.calculate_UB(*solver_reflections)
        self.sample.U = self.solver.U
        self.sample.UB = ub
        self.request_solver_update(False)
        return ub

    @property
    def extras(self) -> list[str]:
        """Ordered dictionary of |solver| extra parameters in current mode."""
        every = self.all_extras
        current = {axis: every[axis] for axis in self.solver_extra_axis_names}
        return current

    @extras.setter
    def extras(self, values: dict[str, Number]):
        """Set |solver| extra parameters for the current mode."""
        incoming = self._validate_extras(values, self.extras)
        if len(incoming) > 0:
            self._extras.update(incoming)
            self.request_solver_update(True)

    def forward(self, pseudos: AnyAxesType, wavelength: float = None) -> list:
        """Compute [{names:reals}] from {names: pseudos} (hkl -> angles)."""
        logger.debug(
            "(%s) forward(): pseudos=%r",
            self.__class__.__name__,
            pseudos,
        )

        pdict = self.standardize_pseudos(pseudos)
        reals = self.diffractometer.real_position._asdict()  # Original values.

        self.update_solver(wavelength=wavelength)

        # Filter just the solutions that fit the constraints.
        solutions = []
        try:
            # Unit conversions for presets and new values.
            angle_units_solver = self.solver.ANGLE_UNITS
            angle_units_core = self.diffractometer.reals_units

            presets = {
                axis: convert_units(
                    value,
                    angle_units_core,
                    angle_units_solver,
                )
                for axis, value in reals.items()
            }
            self.solver.set_reals(presets)  # preset reals (issue #155)
            for solution in self.solver.forward(self._axes_names_d2s(pdict)):
                new_reals = self._axes_names_s2d(solution)
                for axis, value in new_reals.items():
                    # Update with converted new value.
                    reals[axis] = convert_units(
                        value,
                        angle_units_solver,
                        angle_units_core,
                    )
                if self.constraints.valid(**reals):
                    solutions.append(self.diffractometer.RealPosition(**reals))
        except NoForwardSolutions:
            pass

        return solutions

    def geometries(self) -> list[str]:
        """Return all available |solver| geometries."""
        # Not a @property since it's a classmethod of a Solver.
        return self.solver.geometries()

    @property
    def geometry(self) -> str:
        """Return the |solver| geometry."""
        return self.solver.geometry

    def inverse(
        self,
        reals: Union[AnyAxesType, None],
        wavelength: float = None,
    ) -> AxesDict:
        """Compute (pseudos) from {names: reals} (angles -> hkl)."""
        logger.debug(
            "(%s) inverse(): reals=%r",
            self.__class__.__name__,
            reals,
        )
        pseudos: AxesDict = {
            axis[0]: 0
            # Original values.
            for axis in self.diffractometer._get_pseudo_positioners()
        }
        if self.solver is None or len(self.axes_xref) == 0:
            # Called from the constructor before solver is defined.
            return pseudos  # current values of pseudos

        # Just the reals expected by the solver.
        # Dictionary in order expected by the solver.
        reals: AxesDict = self.standardize_reals(reals)
        for axis, value in reals.items():
            angle_units_uc = self.diffractometer.reals_units
            angle_units_solver = self.solver.ANGLE_UNITS
            reals[axis] = convert_units(value, angle_units_uc, angle_units_solver)

        self.update_solver(wavelength=wavelength)

        # transform: reals -> pseudos
        spdict: AxesDict = self.solver.inverse(self._axes_names_d2s(reals))

        pseudos.update(self._axes_names_s2d(spdict))  # Update with new values.
        return pseudos

    @property
    def local_pseudo_axes(self) -> list:
        """
        List of the diffractometer pseudo axes expected by the solver.

        This becomes useful when additional pseudo axes are named
        as ophyd Components in the diffractometer.
        """
        if self.solver is None:
            return []
        return [
            self.axes_xref_reversed[k]
            #
            for k in self.solver.pseudo_axis_names
        ]

    @property
    def local_real_axes(self) -> list:
        """
        List of the diffractometer real axes expected by the solver.

        This becomes useful when additional real axes are named
        as ophyd Components in the diffractometer.
        """
        if self.solver is None:
            return []
        return [
            self.axes_xref_reversed[k]
            #
            for k in self.solver.real_axis_names
        ]

    @property
    def mode(self) -> str:
        """Return the current computation mode."""
        if self._mode is None:
            self._mode = self.solver.mode
            self.request_solver_update(True)
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Set the computation mode to be used."""
        if value in self.modes:
            self._mode = value
            self.request_solver_update(True)

    @property
    def modes(self) -> list[str]:
        """Return the list of available |solver| modes."""
        return self.solver.modes

    def refine_lattice(self, *reflections: list[Reflection]) -> Lattice:
        """
        Return the sample lattice computed from 3 or more reflections.

        Do not change the sample lattice.  Let the user decide that.
        """
        if len(reflections) == 0:
            reflections = list(self.sample.reflections.values())
        rnames = [r.name for r in reflections]
        if len(reflections) < 3:
            raise CoreError(
                # fmt: off
                "Must have at least 3 reflections to refine lattice."
                f" Known reflections: {rnames}"
                # fmt: on
            )

        logger.debug("Refining lattice using reflections %r", rnames)
        lattice = self.solver.refineLattice(self._reflections_to_solver(reflections))

        # apply unit conversions solver to lattice after refineLattice
        angle_units_solver = self.solver.ANGLE_UNITS
        angle_units_uc = self.sample.lattice.angle_units
        length_units_solver = self.solver.LENGTH_UNITS
        length_units_uc = self.sample.lattice.length_units
        for parm, value in lattice.items():
            if parm in "a b c".split():
                lattice[parm] = convert_units(
                    value, length_units_solver, length_units_uc
                )
            elif parm in "alpha beta gamma".split():
                lattice[parm] = convert_units(value, angle_units_solver, angle_units_uc)

        return Lattice(**lattice)

    def _reflections_to_solver(self, refl_list: list) -> dict:
        """(internal) Convert list of reflections to dict for solver.

        Each :class:`~hklpy2.blocks.reflection.Reflection` carries its
        own units for real axis angles and wavelength.  Convert these to
        the solver's internal units.
        """
        k = "wavelength"
        wl_units_solver = self.solver.LENGTH_UNITS
        reflections = []
        for refl in refl_list:
            if isinstance(refl, str):
                refl = self.sample.reflections[refl]
            refl = refl._asdict()

            for axis, value in refl["reals"].items():
                angle_units_uc = self.diffractometer.reals_units
                angle_units_solver = self.solver.ANGLE_UNITS
                refl["reals"][axis] = convert_units(
                    value, angle_units_uc, angle_units_solver
                )

            # determine the original wavelength_units: reflection-level, else beam-level
            refl_wl_units = (
                refl.get("wavelength_units")
                or self.diffractometer.beam.wavelength_units.get()
            )
            refl[k] = convert_units(refl[k], refl_wl_units, wl_units_solver)

            reflections.append(refl)
        return reflections

    def remove_sample(self, name):
        """Remove the named sample.  No error if name is not known."""
        if name not in self.samples:
            raise KeyError(f"{name!r} not in sample list:{list(self.samples)}.")
        if len(self.samples) == 1:
            raise CoreError("Cannot remove last sample.")

        self._samples.pop(name)
        self._sample_name = list(self.samples)[0]

    def request_solver_update(self, flag: bool = True) -> None:
        """
        Set (or clear) signal to trigger a solver update.

        Needs to be a method (not a property) so it can be called from a
        wavelength method.
        """
        self._solver_needs_update = flag

    def reset_constraints(self):
        """Restore diffractometer constraints to default settings."""
        self.constraints = RealAxisConstraints(self.diffractometer.real_axis_names)

    def reset_samples(self):
        """Restore diffractometer samples to default settings."""
        self._samples = {}  # Remove all the samples.
        # Create the default sample.
        self.add_sample(DEFAULT_SAMPLE_NAME, 1)

    @property
    def sample(self) -> Sample:
        """Current Sample (Python object)."""
        return self.samples[self._sample_name]

    @sample.setter
    def sample(self, value: str) -> None:
        self._sample_name = value
        if self.solver is not None:
            try:
                self.solver.U = self.sample.U
                self.solver.UB = self.sample.UB
            except AttributeError:
                pass  # property is not settable

    @property
    def samples(self) -> dict:
        """Sample dictionary."""
        return self._samples

    @property
    def solver(self) -> SolverBase:
        """Backend |solver| object."""
        return self._solver

    @property
    def solver_extra_axis_names(self) -> list[str]:
        """Ordered list of any |solver| extra axis names in current mode."""
        self.update_solver()
        return self.solver.extra_axis_names

    @property
    def solver_name(self) -> str:
        """Name of |solver|."""
        return self.solver.name

    @property
    def solver_pseudo_axis_names(self) -> list[str]:
        """Ordered list of |solver| pseudo axis names."""
        self.update_solver()
        return self.solver.pseudo_axis_names

    @property
    def solver_real_axis_names(self) -> list[str]:
        """Ordered list of |solver| real axis names."""
        self.update_solver()
        return self.solver.real_axis_names

    @property
    def solver_signature(self) -> str:
        """Return 'repr(self.solver)' for use as ophyd.AttributeSignal."""
        return repr(self.solver)

    @property
    def solver_summary(self) -> str:
        """Return table of solver's geometry (modes, axes).."""
        return self.solver.summary

    def set_solver(
        self,
        name: str,
        geometry: str,
        **kwargs: dict,
    ) -> SolverBase:
        """
        Create an instance of the backend |solver| library and geometry.

        PARAMETERS

        solver str:
            Name of the |solver| library.
        geometry str:
            Name of the |solver| geometry.
        kwargs dict:
            Any keyword arguments needed by the |solver|.
        """
        logger.debug(
            "(%s) solver=%r, geometry=%r, kwargs=%r",
            self.__class__.__name__,
            name,
            geometry,
            kwargs,
        )
        self._solver = solver_factory(name, geometry, **kwargs)
        self._extras = {
            k: DEFAULT_EXTRA_VALUE
            #
            for k in self.solver.all_extra_axis_names
        }
        self.update_solver()
        return self._solver

    def standardize_pseudos(self, pseudos: AnyAxesType) -> AxesDict:
        """
        Convert user-supplied pseudos into dictionary in solver's order.

        User could provide pseudos in several forms:

        * dict: {"h": 0, "k": 1, "l": -1}
        * namedtuple: (h=0.0, k=1.0, l=-1.0)
        * ordered list: [0, 1, -1]  (for h, k, l)
        * ordered tuple: (0, 1, -1)  (for h, k, l)
        """
        return axes_to_dict(pseudos, self.local_pseudo_axes)

    def standardize_reals(self, reals: Union[AnyAxesType, None]) -> AxesDict:
        """
        Convert user-supplied reals into dictionary in solver's order.

        User could provide reals in several forms:

        * None: current positions
        * dict: {"omega": 120, "chi": 35.3, "phi": 45, "tth": -120}
        * namedtuple: (omega=120, chi=35.3, phi=45, tth=-120)
        * ordered list: [120, 35.3, 45, -120]  (for omega, chi, phi, tth)
        * ordered tuple: (120, 35.3, 45, -120)  (for omega, chi, phi, tth)
        """

        if reals is None:  # write ordered dict
            reals = {
                k: getattr(self.diffractometer, k).position
                # Get from current diffractometer axis positions
                for k in self.local_real_axes
            }

        return axes_to_dict(reals, self.local_real_axes)

    def to_solver_units(self, wavelength: float = None) -> dict:
        """Convert quantities from diffractometer units to solver units."""
        lattice = self.sample.lattice._asdict()

        angle_units_uc = lattice["angle_units"]
        angle_units_solver = self.solver.ANGLE_UNITS
        length_units_uc = lattice["length_units"]
        length_units_solver = self.solver.LENGTH_UNITS
        for k in "a b c  alpha beta gamma".split():
            if k in "a b c".split():
                lattice[k] = convert_units(
                    lattice[k], length_units_uc, length_units_solver
                )
            else:
                lattice[k] = convert_units(
                    lattice[k], angle_units_uc, angle_units_solver
                )

        reflections = self._reflections_to_solver(self.sample.reflections)
        wavelength = wavelength or self.diffractometer.beam.wavelength.get()

        wl_units = self.diffractometer.beam.wavelength_units.get()
        wl_units_solver = self.solver.LENGTH_UNITS
        return dict(
            sample=dict(
                name=self.sample.name,
                lattice=lattice,
                order=self.sample.reflections.order,
                reflections=reflections,
            ),
            wavelength=convert_units(wavelength, wl_units, wl_units_solver),
        )

    def update_solver(self, wavelength: Optional[float] = None) -> None:
        """Update solver data if needed."""
        if self.solver.mode != self.mode or wavelength is not None:
            self.request_solver_update(True)  # force the update

        if self._solver_needs_update:
            std = self.to_solver_units(wavelength)

            self.solver.wavelength = std["wavelength"]
            self.solver.sample = std["sample"]
            self.solver.mode = self.mode

            try:
                self.solver.extras = {
                    axis: self._extras[axis]
                    # multiline
                    for axis in self.solver.extra_axis_names
                }
            except AttributeError:
                pass  # Some solvers have no setter for extras

            try:
                self.solver.U = self.sample.U
                self.solver.UB = self.sample.UB
            except AttributeError:
                pass  # Some solvers have no setter for U & UB

            self.request_solver_update(False)
