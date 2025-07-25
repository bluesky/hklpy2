"""
Base class for all diffractometers

.. autosummary::

    ~creator
    ~diffractometer_class_factory
    ~DiffractometerBase
    ~Hklpy2PseudoAxis
"""

import logging
import pathlib
from collections.abc import Iterable
from typing import Callable
from typing import Optional

import numpy as np
import yaml
from ophyd import Component as Cpt
from ophyd import EpicsMotor
from ophyd import Kind
from ophyd import PositionerBase
from ophyd import PseudoPositioner
from ophyd import PseudoSingle
from ophyd import SoftPositioner
from ophyd.device import required_for_connection
from ophyd.pseudopos import pseudo_position_argument
from ophyd.pseudopos import real_position_argument

from .blocks.reflection import Reflection
from .blocks.sample import Sample
from .incident import WavelengthXray
from .misc import AnyAxesType
from .misc import AxesDict
from .misc import DiffractometerError
from .misc import load_yaml_file
from .misc import pick_first_solution
from .misc import roundoff

__all__ = """
    DiffractometerBase
    diffractometer_class_factory
    creator
""".split()
logger = logging.getLogger(__name__)

DEFAULT_PHOTON_ENERGY_KEV = 8.0
H_OR_N = Kind.hinted | Kind.normal


class Hklpy2PseudoAxis(PseudoSingle):
    """Override to allow auxiliary pseudos."""

    @property
    def position(self):
        """The current position of the positioner in its engineering units

        Returns
        -------
        position
        """
        if self._idx is None:
            return self._position  # This is an auxiliary pseudo axis.
        return self._parent.position[self._idx]

    @required_for_connection(description="{device.name} readback subscription")
    def _sub_proxy_readback(self, obj=None, value=None, **kwargs):
        """Parent callbacks including a position value will be filtered through
        this function and re-broadcast using only the relevant position to this
        pseudo axis.
        """
        if hasattr(value, "__getitem__"):
            if self._idx is not None:  # auxiliary pseudos are ignored here.
                value = value[self._idx]

        return self._run_subs(obj=self, value=value, **kwargs)


class DiffractometerBase(PseudoPositioner):
    """
    Base class for all diffractometers.

    PARAMETERS

    solver : str
        Name of |solver| library.
        (default: unspecified)
    geometry : str
        Name of |solver| geometry.
        (default: unspecified)
    solver_kwargs : Dict(str, Any)
        Any additional keyword arguments needed by |solver| library.
        (default: empty)
    pseudos : List[str]
        List of diffractometer axis names to be used
        as pseudo axes. (default: unspecified)
    reals : List[str]
        List of diffractometer axis names to be used as
        real axes. (default: unspecified)
    forward_solution_function : Callable
        Function to pick one solution from list of possibilities.
        Used by :meth:`~hklpy2.diffract.DiffractometerBase.forward`.
        (default: :func:`~hklpy2.misc.pick_first_solution`)

    .. rubric:: (ophyd) Components

    .. rubric:: Python Attributes

    .. autosummary::

        ~_forward_solution

    .. rubric:: Python Methods

    .. autosummary::

        ~add_reflection
        ~add_sample
        ~export
        ~forward
        ~full_position
        ~inverse
        ~move_dict
        ~move_forward_with_extras
        ~move_inverse_with_extras
        ~move_reals
        ~restore
        ~scan_extra
        ~wh

    .. rubric:: Python Properties

    .. autosummary::
        ~auxiliary_axis_names
        ~configuration
        ~pseudo_axis_names
        ~real_axis_names
        ~sample
        ~samples
    """

    beam = Cpt(WavelengthXray)
    """Incident monochromatic beam."""

    _forward_solution: Callable = pick_first_solution
    """
    Pick a solution from the solution(s) return from
    :meth:`~hklpy2.ops.Core.forward`.

    Choices include:

    * (default) :func:`hklpy2.misc.pick_first_solution`
    * :func:`hklpy2.misc.pick_closest_solution`
    * User-supplied function matching the same interface.

    .. seealso::
        :meth:`hklpy2.diffract.DiffractometerBase.forward`,
        :meth:`hklpy2.ops.Core.forward`

    """

    def __init__(
        self,
        prefix: str = "",
        *,
        solver: str = None,
        geometry: str = None,
        solver_kwargs: dict = {},
        pseudos: list[str] = [],
        reals: list[str] = [],
        forward_solution_function: Optional[Callable] = None,
        **kwargs,
    ):
        from .ops import Core

        self._backend = None
        self._forward_solution = forward_solution_function or pick_first_solution
        self.core = Core(self)

        super().__init__(prefix, **kwargs)

        # After __init__, Core syncs solver with the diffractometer wavelength.
        if isinstance(solver, str) and isinstance(geometry, str):
            self.core.set_solver(solver, geometry, **solver_kwargs)

        if len(pseudos) == 0:
            pseudos = [axis.attr_name for axis in self._pseudo]
        if len(reals) == 0:
            reals = [axis.attr_name for axis in self._real]
        self.core.assign_axes(pseudos, reals)
        self.beam.wavelength_updated_func = self.core.request_solver_update

        for attr in self.auxiliary_axis_names:  # auxiliary
            component = getattr(self, attr)
            if isinstance(component, Hklpy2PseudoAxis):
                if component.position is None:
                    # Set position of all uninitialized auxiliary pseudo axes.
                    component._position = 0.0

    def add_reflection(
        self,
        pseudos,
        reals=None,
        wavelength: float = None,
        name: str = None,
        replace: bool = False,
    ) -> Reflection:
        """
        Add a new reflection with this geometry to the selected sample.

        PARAMETERS

        pseudos various:
            Pseudo-space axes and values.
        reals various:
            Dictionary of real-space axes and values.
        wavelength float:
            Wavelength of incident radiation. If ``None``, diffractometer's
            current wavelength will be assigned.
        name str:
            Reference name for this reflection.
            If ``None``, a random name will be assigned.
        replace bool:
            If ``True``, replace existing reflection matching this name.
            (default: ``False``)
        """
        return self.core.add_reflection(
            pseudos,
            reals,
            wavelength or self.beam.wavelength.get(),
            name,
            replace,
        )

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
        return self.core.add_sample(
            name,
            a,
            b,
            c,
            alpha,
            beta,
            gamma,
            digits,
            replace,
        )

    @property
    def configuration(self) -> dict:
        """Diffractometer configuration (orientation)."""
        return self.core._asdict()

    @configuration.setter
    def configuration(self, config: dict) -> dict:
        """
        Diffractometer configuration (orientation).

        PARAMETERS

        config: dict
            Dictionary of diffractometer configuration, geometry, constraints,
            samples, reflections, orientations, solver, ...
        """
        return self.core._fromdict(config)

    def export(self, file, comment=""):
        """
        Export the diffractometer configuration to a YAML file.

        Example::

            import hklpy2

            e4cv = hklpy2.creator(name="e4cv")
            e4cv.export("e4cv-config.yml", comment="example")
        """
        path = pathlib.Path(file)
        config = self.configuration
        config["_header"].update(dict(file=str(file), comment=str(comment)))
        dump = yaml.dump(
            config,
            indent=2,
            default_flow_style=False,
            sort_keys=False,
        )
        with open(path, "w") as y:
            y.write("#hklpy2 configuration file\n\n")
            y.write(dump)

    def restore(
        self,
        config,
        clear=True,
        restore_constraints=True,
        restore_wavelength=True,
    ):
        """
        Restore diffractometer configuration.

        Example::

            import hklpy2

            e4cv = hklpy2.creator(name="e4cv")
            e4cv.restore("e4cv-config.yml")

        PARAMETERS

        config *dict*, *str*, or *pathlib.Path* object:
            Dictionary with configuration, or name (str or pathlib object) of
            diffractometer configuration YAML file.
        clear *bool*:
            If ``True`` (default), remove any previous configuration of the
            diffractometer and reset it to default values before restoring the
            configuration.

            If ``False``, sample reflections will be append with all reflections
            included in the configuration data for that sample.  Existing
            reflections will not be changed.  The user may need to edit the
            list of reflections after ``restore(clear=False)``.
        restore_constraints *bool*:
            If ``True`` (default), restore any constraints provided.
        restore_wavelength *bool*:
            If ``True`` (default), restore wavelength.

        Note: Can't name this method "import", it's a reserved Python word.
        """
        if isinstance(config, (str, pathlib.Path)):
            config = load_yaml_file(config)
        if not isinstance(config, dict):
            raise TypeError(f"Unrecognized configuration: {config=}")
        header = config.get("_header")
        if header is None:
            raise KeyError("Configuration is missing '_header' key.")
        # Note: python_class key is not testable, could be anything.

        bcfg = config["beam"].copy()
        if not restore_wavelength:
            bcfg.pop("energy", None)
            bcfg.pop("wavelength", None)
        if bcfg.get("energy") is not None and bcfg.get("wavelength") is not None:
            # Don't restore BOTH energy & wavelength
            bcfg.pop("energy", None)
        bcfg["class"] = self.beam.__class__.__name__
        self.beam._fromdict(bcfg)

        self.core.configuration._fromdict(
            config,
            clear=clear,
            restore_constraints=restore_constraints,
        )

    @pseudo_position_argument
    def forward(self, pseudos: dict, wavelength: float = None) -> tuple:
        """Compute real-space coordinates from pseudos (hkl -> angles)."""
        logger.debug("forward: pseudos=%r", pseudos)
        solutions = self.core.forward(pseudos, wavelength=wavelength)
        return self._forward_solution(self.real_position, solutions)

    def full_position(self, digits=4) -> dict:
        """Return dict with positions of pseudos, reals, & extras."""
        from .misc import roundoff

        pdict = self.position._asdict()
        pdict.update(self.real_position._asdict())
        pdict.update(self.core.extras)
        for k in pdict:
            pdict[k] = roundoff(pdict[k], digits)
        return pdict

    @real_position_argument
    def inverse(self, reals: tuple, wavelength: float = None) -> tuple:
        """Compute pseudo-space coordinates from reals (angles -> hkl)."""
        logger.debug("inverse: reals=%r", reals)
        pos = self.core.inverse(reals, wavelength=wavelength)
        return self.PseudoPosition(**pos)  # as created by namedtuple

    def move_dict(self, axes: AxesDict):
        """(plan) Move diffractometer axes to positions in 'axes'."""
        from bluesky import plan_stubs as bps

        from .misc import flatten_lists

        if hasattr(axes, "_fields"):
            # Convert namedtuple to dict
            axes = axes._asdict()

        # Transform axes dict to args for bps.mv(position, value)
        moves = list(
            flatten_lists(
                [[getattr(self, k), v] for k, v in axes.items()]
            )  # move the diffractometer axes
        )
        yield from bps.mv(*moves)

    def move_forward_with_extras(self, pseudos: AnyAxesType, extras: AxesDict):
        """
        (plan stub) Compute forward solution at fixed pseudos and extras.

        EXAMPLE::

            RE(
                move_forward_with_extras(
                    diffractometer,
                    pseudos=dict(h=2, k=1, l=0),
                    extras=dict(h2=2, k2=2, l2=0, psi=25),
                )
            )
        """
        self.core.extras = extras  # before forward()
        self.core.update_solver()
        solution = self.forward(self.core.standardize_pseudos(pseudos))
        yield from self.move_dict(solution)

    def move_inverse_with_extras(self, reals: AnyAxesType, extras: AxesDict):
        """
        (plan stub) Compute inverse solution at fixed reals and extras.

        EXAMPLE::

            RE(
                move_inverse_with_extras(
                    diffractometer,
                    reals=dict(omega=10, chi=0, phi=0, phi=20),
                    extras=dict(h2=2, k2=2, l2=0, psi=25),
                )
            )
        """
        self.core.extras = extras
        self.core.update_solver()
        pseudos = self.inverse(self.core.standardize_reals(reals))
        yield from self.move_dict(pseudos)

    @real_position_argument
    def move_reals(self, reals: AnyAxesType) -> None:
        """(not a plan) Move the real-space axes as specified in 'real_positions'."""
        reals = self.core.standardize_reals(reals)
        for axis_name, position in reals.items():
            hkl_axis = getattr(self, axis_name)
            hkl_axis.move(position)

    def scan_extra(
        self,
        detectors: Iterable,
        axis: Optional[str] = None,  # name of extra parameter to be scanned
        start: Optional[float] = None,
        finish: Optional[float] = None,
        num: Optional[int] = 2,
        *,
        pseudos: Optional[dict] = None,  # h, k, l
        reals: Optional[dict] = None,  # angles
        extras: Optional[
            dict
        ] = {},  # define all but the 'axis', these will remain constant
        fail_on_exception: Optional[bool] = False,
        md: Optional[dict] = None,
    ):
        """
        Scan one extra diffractometer parameter, such as 'psi'.

        * TODO: #107 one **or more** (such as bp.scan)

        * iterate extra positions as described:
            * set extras
            * solution = forward(pseudos)
            * move to solution
            * acquire (trigger) all controls
            * read and record all controls
        """
        import numpy
        from bluesky import plan_stubs as bps
        from bluesky import preprocessors as bpp

        from .misc import dict_device_factory

        self.core.update_solver()

        # validate
        if axis not in self.core.solver_extra_axis_names:
            raise KeyError(f"{axis!r} not in {self.core.solver_extra_axis_names}")
        if pseudos is None and reals is None:
            raise ValueError("Must define either pseudos or reals.")
        if pseudos is not None and reals is not None:
            raise ValueError("Cannot define both pseudos and reals.")

        _md = {
            "diffractometer": {
                "name": self.name,
                "solver_signature": self.core.solver_signature,
                "geometry": self.core.geometry,
                "mode": self.core.mode,
                "extra_axes": self.core.solver_extra_axis_names,
            },
            "axis": axis,
            "start": start,
            "finish": finish,
            "num": num,
            "pseudos": pseudos,
            "reals": reals,
            "extras": extras,
            "transformation": "forward" if reals is None else "inverse",
        }.update(md or {})

        extras[axis] = start
        extras_class = dict_device_factory(extras)
        extras_device = extras_class("", name=f"{self.name}_extras", kind="hinted")

        all_controls = detectors
        all_controls.append(extras_device)
        all_controls = list(set(all_controls))

        signal = getattr(extras_device, axis)  # Pick the 'axis' Component.
        signal.kind = "hinted"

        def position_series(start, finish, num):
            for value in numpy.linspace(start, finish, num=num):
                yield value

        @bpp.stage_decorator(detectors)
        @bpp.run_decorator(md=_md)
        def _inner():
            for value in position_series(start, finish, num):

                def move_axes(pseudos, reals, extras):
                    """Move extras, then reals or pseudos, move to the solution."""
                    if reals is None:
                        yield from self.move_forward_with_extras(pseudos, extras)
                    else:
                        yield from self.move_inverse_with_extras(reals, extras)

                def acquire(objects):
                    """Tell each object to acquire its data."""
                    group = "trigger_control_objects"
                    for item in objects:
                        yield from bps.trigger(item, group=group)
                    yield from bps.wait(group=group)

                def record(objects, stream="primary"):
                    """Read & record each object."""
                    yield from bps.create(stream)
                    for item in objects:
                        yield from bps.read(item)
                    yield from bps.save()

                # note the new axis position, will report later
                extras.update({axis: value})
                yield from bps.mv(signal, value)
                try:
                    yield from move_axes(pseudos, reals, extras)
                    yield from acquire(all_controls)
                    yield from record(all_controls)
                except Exception as reason:
                    if fail_on_exception:
                        raise reason
                    else:
                        # Scan axis beyond limits will trigger this code.
                        print(f"FAIL: {axis}={value} {reason}")  # Inform the user!

        return (yield from _inner())

    # ---- get/set properties

    @property
    def auxiliary_axis_names(self) -> list[str]:
        """
        Names of all auxiliary positioners, in order of appearance.

        Auxiliary axes are all components using (subclasses of
        'ophyd.PositionerBase') that are not pseudos or reals.

        Example::

            >>> fourc.auxiliary_axis_names
            ['h2', 'k2', 'l2']
        """
        pseudos_and_reals = self.pseudo_axis_names + self.real_axis_names
        return [
            # Names of any auxiliary positioners not described above.
            attr
            for attr in self.component_names
            if attr not in pseudos_and_reals
            if isinstance(getattr(self, attr), PositionerBase)
        ]

    @property
    def pseudo_axis_names(self):
        """
        Names of all the pseudo axes, in order of appearance.

        Example::

            >>> fourc.pseudo_axis_names
            ['h', 'k', 'l']
        """
        return [o.attr_name for o in self.pseudo_positioners]

    @property
    def real_axis_names(self):
        """
        Names of all the real axes, in order of appearance.

        Example::

            >>> fourc.real_axis_names
            ['omega', 'chi, 'phi', 'tth']
        """
        return [o.attr_name for o in self.real_positioners]

    @property
    def samples(self):
        """Dictionary of samples."""
        if self.core is None:
            return {}
        return self.core.samples

    @property
    def sample(self):
        """Current sample object."""
        if self.core is None:
            return None
        return self.core.sample

    @sample.setter
    def sample(self, value: str) -> None:
        self.core.sample = value

    def wh(self, digits=4, full=False):
        """Concise report of the current diffractometer positions."""

        if not self.connected:
            raise DiffractometerError(f"Diffractometer {self.name!r} is not connected.")

        def labeled_value(label, value):
            return f"{label}={roundoff(value, digits)}"

        def print_axes(names: list[str], preface: str = ""):
            pairs = []
            for nm in names:
                # Any instance of PositionerBase.
                component = getattr(self, nm)
                value = component.position
                pair = labeled_value(nm, value)
                pairs.append(pair)
            if len(pairs):
                print(preface + ", ".join(pairs))

        def format_array(array):
            """Apply (fixed-point) digits formatting to numpy array."""

            def each(x):
                """Roundoff each float."""
                # return roundoff(x, digits)
                text = f"{x:.{digits}f}"
                if text.startswith("-0."):
                    text = text[1:]
                return float(text)

            # Apply formatting function to each element.
            return np.vectorize(each)(np.array(array)).tolist()

        if full:
            print(f"diffractometer={self.name!r}")
            print(f"{self.core.solver}")
            print(f"{self.sample!r}")
            for v in self.sample.reflections.values():
                print(f"{v}")
            print(f"Orienting reflections: {self.sample.reflections.order}")
            print(f"U={format_array(self.sample.U)}")
            print(f"UB={format_array(self.sample.UB)}")
            for v in self.core.constraints.values():
                print(f"constraint: {v}")
            print(f"Mode: {self.core.mode}")
            beam = self.beam._asdict()
            for key in "energy wavelength".split():
                if key in beam:
                    beam[key] = roundoff(beam[key], digits)
            print(f"beam={beam}")
        else:
            print(f"wavelength={roundoff(self.beam.wavelength.get(), digits)}")

        print_axes(self.pseudo_axis_names, preface="pseudos: ")
        print_axes(self.real_axis_names, preface="reals: ")
        extras = self.core.extras
        if len(extras) > 0:
            value_text = " ".join([labeled_value(k, v) for k, v in extras.items()])
            print("extras: " + value_text)
        print_axes(self.auxiliary_axis_names, preface="auxiliaries: ")


def creator(
    *,
    prefix: str = "",
    name: str = "",
    solver: str = "hkl_soleil",
    geometry: str = "E4CV",
    beam_kwargs: dict[str, object] = {},
    solver_kwargs: dict[str, object] = {},
    pseudos: list = [],
    reals: list[str] | dict[str, str | None] = {},
    aliases: dict[str, list[str]] = {},
    motor_labels: list = ["motors"],
    labels: list = ["diffractometer"],
    class_name: str = "Hklpy2Diffractometer",
    class_bases: list = [DiffractometerBase],
    forward_solution_function: Optional[str] = None,
    **kwargs,
):
    """
    Factory function to create a diffractometer instance.

    EXAMPLES:

    Four-circle diffractometer, vertical orientation, Eulerian rotations,
    canonical real axis names, EPICS motor PVs::

        e4cv = creator(name="e4cv",
            solver="hkl_soleil", geometry="E4CV",
            reals=dict(omega="IOC:m1", chi="IOC:m2", phi="IOC:m3", tth="IOC:m4"),
        )

    Four-circle diffractometer, vertical orientation, Eulerian rotations,
    custom real axis names, simulated positioners::

        sim4c = creator(name="sim4c",
            solver="hkl_soleil", geometry="E4CV",
            reals=dict(uno=None, dos=None, tres=None, cuatro=None),
        )

    (Simplest case to get a simulator.)
    Four-circle diffractometer, vertical orientation, Eulerian rotations,
    canonical real axis names, simulated positioners (all default settings)::

        sim4c = creator(name="sim4c")

    Kappa six-circle diffractometer, simulated motors::

        simk6c = creator(name="simk6c",
            solver="hkl_soleil", geometry="K6C"
        )

    PARAMETERS

    prefix : str
        EPICS PV prefix (default: empty string)
    name : str
        Name of the ophyd diffractometer object to be created. (default: '""')
    solver : str
        Name of the backend solver providing the geometry. (default: '"hkl_soleil"')
    geometry : str
        Name of the diffractometer geometry. (default: '"E4CV"')
    beam_kwargs : dict[str, object]
        Additional configuration for the incident beam.
        (default: '{"class": "hklpy2.incident.WavelengthXray"}')
    solver_kwargs : dict[str, object]
        Additional configuration for the solver. (default: '{"engine": "hkl"}')
    pseudos : list
        Specification of the names of any pseudo axis positioners
        in addition to the ones provided by the solver.

        (default: '[]' which means no additional pseudo axes)
    reals : dict
        Specification of the real axis motors.  Dictionary keys are the motor
        names, values are the EPICS motor PV for that axis.  If the PV is
        'None', use a simulated positioner.

        The dictionary can be empty or must have at least the canonical number of
        real axes.  The order of the axes is important.  The names provided will
        be mapped to the canonical order defined by the solver.  Components will
        be created for any extra *reals*.

        (default: '{}' which means use the canonical names for the real axes and
        use simulated positioners)
    aliases: dict[str, list[str]]
        Aliases of diffractometer axes for solver's pseudos and reals.

        (default: '{}' which means use the first diffractometer axes from each to match the solver.)
    motor_labels : list
        Ophyd object labels for each real positioner. (default: '["motors"]')
    labels : list
        Ophyd object labels for the diffractometer object. (default: '["diffractometer"]')
    class_name : str
        Name to use for the diffractometer class.
        (default: '"Hklpy2Diffractometer"')
    class_bases : list
        List of base classes to use for the diffractometer class.
        (default: '[DiffractometerBase]')
    forward_solution_function : str
        Name of function to pick one solution from list of possibilities.
        Used by :meth:`~hklpy2.diffract.DiffractometerBase.forward`.
        (default: :func:`~hklpy2.misc.pick_first_solution`)

        Will be assigned to :attr:`hklpy2.diffract.DiffractometerBase._forward_solution`.
    kwargs : any
        Additional keyword arguments will be added when constructing
        the new diffractometer object.
    """
    # print(f"creator({solver=!r}, {geometry=!r})")
    DiffractometerClass = diffractometer_class_factory(
        solver=solver,
        geometry=geometry,
        beam_kwargs=beam_kwargs,
        solver_kwargs=solver_kwargs,
        pseudos=pseudos,
        reals=reals,
        motor_labels=motor_labels,
        class_name=class_name,
        class_bases=class_bases,
        aliases=aliases,
        forward_solution_function=forward_solution_function,
    )
    if name == "":
        name = geometry.lower()
    return DiffractometerClass(prefix, name=name, labels=labels, **kwargs)


def diffractometer_class_factory(
    *,
    solver: str = "hkl_soleil",
    geometry: str = "E4CV",
    beam_kwargs: dict[str, object] = {},
    solver_kwargs: dict[str, object] = {"engine": "hkl"},
    pseudos: list = [],
    reals: list[str] | dict[str, str | None] = {},
    motor_labels: list = ["motors"],
    class_name: str = "Hklpy2Diffractometer",
    class_bases: list = [DiffractometerBase],
    aliases: dict[str, list[str]] = {},
    forward_solution_function: Optional[str] = None,
) -> DiffractometerBase:
    """
    Build a custom class for this diffractometer geometry.

    PARAMETERS

    solver : str
        Name of the backend solver providing the geometry. (default: '"hkl_soleil"')
    geometry : str
        Name of the diffractometer geometry. (default: '"E4CV"')
    beam_kwargs : dict[str, object]
        Additional configuration for the incident beam.
        (default: '{"class": "hklpy2.incident.WavelengthXray"}')
    solver_kwargs : str
        Additional configuration for the solver. (default: '{"engine": "hkl"}')
    pseudos : list
        Specification of the names of any pseudo axis positioners
        in addition to the ones provided by the solver.

        (default: '[]' which means no additional pseudo axes)
    reals : dict
        Specification of the real axis motors.  Dictionary keys are the motor
        names, values are the EPICS motor PV for that axis.  If the PV is
        'None', use a simulated positioner.

        The dictionary can be empty or must have at least the canonical number of
        real axes.  The order of the axes is important.  The names provided will
        be mapped to the canonical order defined by the solver.  Components will
        be created for any extra *reals*.

        (default: '{}' which means use the canonical names for the real axes and
        use simulated positioners)
    motor_labels : list
        Ophyd object labels for each real positioner. (default: '["motors"]')
    class_name : str
        Name to use for the diffractometer class.
        (default: '"Hklpy2Diffractometer"')
    class_bases : list
        List of base classes to use for the diffractometer class.
        (default: '[DiffractometerBase]')
    aliases: dict[str, list[str]]
        Aliases of diffractometer axes for solver's pseudos and reals.

        (default: '{}' which means use the first diffractometer axes from each to match the solver.)
    forward_solution_function : str
        Name of function to pick one solution from list of possibilities.
        Used by :meth:`~hklpy2.diffract.DiffractometerBase.forward`.
        (default: :func:`~hklpy2.misc.pick_first_solution`)

        Will be assigned to :attr:`hklpy2.diffract.DiffractometerBase._forward_solution`.
    """
    from .misc import dynamic_import
    from .misc import solver_factory

    # print(f"diffractometer_class_factory({solver=!r}, {geometry=!r})")
    # Validation.  Fail early, fail hard.
    if not isinstance(pseudos, list):
        raise TypeError(f"Expected a list.  Received {pseudos=!r}")
    if not isinstance(reals, dict):
        if isinstance(reals, list):
            reals = {axis: None for axis in reals}
        else:
            raise TypeError(f"Expected a dict.  Received {reals=!r}")
    if not isinstance(aliases, dict):
        raise TypeError(f"Expected a dict.  Received {aliases=!r}")

    def make_component_axis(axis_type, labels=[], pv=None):
        if axis_type == "pseudo":
            return Cpt(Hklpy2PseudoAxis, "", kind=H_OR_N)
        elif axis_type == "real":
            if pv is None:
                return Cpt(
                    SoftPositioner,
                    limits=(-180, 180),
                    init_pos=0,
                    kind=H_OR_N,
                    labels=motor_labels,
                )
            else:
                return Cpt(EpicsMotor, pv, kind=H_OR_N, labels=motor_labels)

    factory_class_attributes = {}  # Set defaults for this custom class.
    aliases = {}

    beam_class = beam_kwargs.pop("class", "hklpy2.incident.WavelengthXray")
    if isinstance(beam_class, str):
        beam_class = dynamic_import(beam_class)
    factory_class_attributes["beam"] = Cpt(beam_class, **beam_kwargs)

    if forward_solution_function is None:
        forward_solution_function = "hklpy2.misc.pick_first_solution"
    factory_class_attributes["_forward_solution"] = dynamic_import(
        forward_solution_function,
    )

    # Find the chosen solver.  It describes its various axes.
    solver_object = solver_factory(solver, geometry, **solver_kwargs)

    for space in "pseudos reals".split():
        singular = space.rstrip("s")
        if space == "pseudos":
            solver_axes = solver_object.pseudo_axis_names
            all_axes = pseudos if len(pseudos) > 0 else solver_axes
            for axis in all_axes:
                factory_class_attributes[axis] = make_component_axis(singular)
        else:
            solver_axes = solver_object.real_axis_names
            all_axes = list(reals) if len(reals) > 0 else solver_axes
            for axis in all_axes:
                factory_class_attributes[axis] = make_component_axis(
                    singular,
                    labels=motor_labels,
                    pv=reals.get(axis, None),
                )

        defaults = all_axes[: len(solver_axes)]
        factory_class_attributes[f"_{singular}"] = aliases.get(space, defaults)

    def constructor(
        self,
        prefix: str = "",
        *,
        solver: str = solver,
        geometry: str = geometry,
        solver_kwargs: dict = solver_kwargs,
        pseudos: list[str] = factory_class_attributes["_pseudo"],
        reals: list[str] = factory_class_attributes["_real"],
        **kwargs,
    ):
        DiffractometerBase.__init__(
            self,
            prefix=prefix,
            solver=solver,
            geometry=geometry,
            solver_kwargs=solver_kwargs,
            pseudos=pseudos,
            reals=reals,
            **kwargs,
        )

    factory_class_attributes["__init__"] = constructor
    return type(class_name, tuple(class_bases), factory_class_attributes)
