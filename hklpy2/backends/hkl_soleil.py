r"""
"hkl_soleil" solver, provides **Hkl**, Synchrotron Soleil.

Example::

    >>> import hklpy2
    >>> SolverClass = hklpy2.get_solver("hkl_soleil")
    >>> libhkl_solver = SolverClass(geometry="E4CV")
    >>> solver
    HklSolver(name='hkl_soleil', version='v5.0.0.3434', geometry='E4CV', engine='hkl', mode='bissector')

:home: https://people.debian.org/~picca/hkl/hkl.html
:source: https://repo.or.cz/hkl.git
:conda-forge: https://anaconda.org/conda-forge

..  caution:: The ``hkl_soleil`` |solver| is not available
    for Windows or Mac OS.  The underlying |libhkl| support
    library is only provided
    for Linux 64-bit OS at this time.

.. note:: To hold an axis or extra parameter constant (current or specified value):
    choose the mode and set the parameter before the forward() transformation.

.. note:: To scan using ``psi`` and ``hkl2``, see
    :doc:`/examples/hkl_soleil-e6c-psi`.

.. autosummary::

    ~HklSolver
"""

# Notes:
#
# - 'fit'
#     While this parameter is used by *libhkl* to adjust lattice parameters when
#     refining from more than 2 reflections, it is not used in the calculation of
#     rotation angles from reciprocal-space coordinates.

import logging
import math
import platform
from typing import Any
from typing import Dict
from typing import List

import numpy as np
from pyRestTable import Table

from ..misc import IDENTITY_MATRIX_3X3
from ..misc import NoForwardSolutions
from ..misc import check_value_in_list
from ..misc import istype
from ..misc import roundoff
from ..misc import unique_name
from .base import NamedFloatDict
from .base import SolverBase
from .base import SolverMatrix3x3
from .base import SolverReflection
from .base import SolverSample
from .hkl_soleil_utils import setup_libhkl

logger = logging.getLogger(__name__)

libhkl = setup_libhkl(platform.system(), "Hkl", "5.0")
AXES_READ = 0
AXES_WRITTEN = 1
LIBHKL_DETECTOR_TYPE = 0
LIBHKL_UNITS = {
    "default": libhkl.UnitEnum.DEFAULT,
    "user": libhkl.UnitEnum.USER,
}
LIBHKL_USER_UNITS = LIBHKL_UNITS["user"]
ROUNDOFF_DIGITS = 12


def roundoff_list(values: List[float], digits=ROUNDOFF_DIGITS) -> List[float]:
    """Prevent underflows and '-0' for all numbers in a list."""
    return [roundoff(v, digits) for v in values]


# TODO: Want to use libhkl.Matrix here instead of Any
def hkl_euler_matrix(
    euler_x: List[float],
    euler_y: List[float],
    euler_z: List[float],
) -> Any:
    """Convert into matrix form."""
    return libhkl.Matrix.new_euler(euler_x, euler_y, euler_z)


# TODO: Want to use libhkl.Matrix here instead of Any
def to_hkl(arr: np.ndarray) -> Any:
    """Convert a numpy ndarray to an hkl ``Matrix``

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    Hkl.Matrix
    """

    if isinstance(arr, libhkl.Matrix):
        return arr

    arr = np.array(arr)

    hklm = hkl_euler_matrix(0, 0, 0)
    hklm.init(*arr.flatten())
    return hklm


# TODO: Want to use libhkl.Matrix here instead of Any
def to_numpy(mat: Any) -> np.ndarray:
    """Convert an hkl ``Matrix`` to a numpy ndarray

    Parameters
    ----------
    mat : Hkl.Matrix

    Returns
    -------
    ndarray
    """
    if isinstance(mat, np.ndarray):
        return mat

    ret = [
        [mat.get(i, j) for j in range(3)]
        # .
        for i in range(3)
    ]

    return np.array(ret, dtype=float)


class HklSolver(SolverBase):
    """
    ``"hkl_soleil"`` (Linux x86_64 only) |libhkl|.

    Wraps the |libhkl| library, written by Frédéric-Emmanuel PICCA (Soleil),
    with support for many common diffractometer geometries.

    PARAMETERS

    geometry: str
        Name of geometry.
    engine: str
        Name of computation engine.  (default: ``"hkl"``)
    mode: str
        Name of operating mode.  (default: current mode)
    pseudos: list[PseudoPositioner]
        List of pseudo positioners. (default: ``[]``)
    reals: list[PositionerBase]
        List of real positioners. (default: ``[]``)

    .. note:: The lists of ``pseudos`` and ``reals`` are the
       corresponding axes of the diffractometer, in the order expected by
       the |solver| geometry.  The diffractometer can use names that are
       different from the names expected by the engine here.  The
       :class:`~hklpys.ops.core` class will convert between the two
       sets of names.

    .. rubric:: Python Methods

    .. autosummary::

        ~addReflection
        ~calculate_UB
        ~forward
        ~geometries
        ~inverse
        ~refineLattice
        ~removeAllReflections
        ~set_reals
        ~_details

    .. rubric:: Python Properties

    .. autosummary::

        ~_summary_dict
        ~axes_c
        ~axes_r
        ~axes_w
        ~engine
        ~engine_name
        ~engines
        ~extra_axis_names
        ~extras
        ~geometry
        ~lattice
        ~mode
        ~modes
        ~pseudo_axis_names
        ~real_axis_names
        ~sample
        ~summary
        ~UB
        ~wavelength
    """

    name = "hkl_soleil"
    version = libhkl.VERSION

    def __init__(
        self,
        geometry: str,
        *,
        engine: str = "hkl",
        mode: str = "",
        **kwargs,
    ) -> None:
        self._hkl_engine = None
        self._sample = None

        super().__init__(geometry, **kwargs)

        # Preface libhkl object names with "_hkl".
        # Note: must keep the '_hkl_engine_list' object as class attribute or
        # random core dumps, usually when accessing 'engine.name_get()'.
        self._hkl_detector = libhkl.Detector.factory_new(
            libhkl.DetectorType(LIBHKL_DETECTOR_TYPE),
        )
        self._hkl_factory = libhkl.factories()[geometry]
        self._hkl_engine_list = self._hkl_factory.create_new_engine_list()  # note!
        self._hkl_engine = self._hkl_engine_list.engine_get_by_name(engine)
        self._hkl_geometry = self._hkl_factory.create_new_geometry()

    def __repr__(self) -> str:
        args = [
            f"{s}={getattr(self, s)!r}"
            # .
            for s in "name version geometry engine_name mode".split()
        ]
        return f"{self.__class__.__name__}({', '.join(args)})"

    def addReflection(self, reflection: SolverReflection) -> None:
        """Add coordinates of a diffraction condition (a reflection)."""
        if not istype(reflection, SolverReflection):
            raise TypeError(
                f"Must supply {SolverReflection!r} object, received {reflection!r}",
            )

        logger.debug("reflection: %r", reflection)
        pseudos = list(reflection["pseudos"].values())
        reals = list(reflection["reals"].values())
        w0 = self.wavelength
        self.wavelength = reflection["wavelength"]
        self._hkl_geometry.axis_values_set(reals, LIBHKL_USER_UNITS)
        self._sample.add_reflection(self._hkl_geometry, self._hkl_detector, *pseudos)
        self.wavelength = w0

    @property
    def axes_c(self) -> list[str]:
        """
        HKL real axis names.

        Held constant during 'forward()' computation.
        """
        # Do NOT sort.
        return [axis for axis in self.axes_r if axis not in self.axes_w]

    @property
    def axes_r(self) -> list[str]:
        """HKL real axis names (read-only)."""
        return self.engine.axis_names_get(AXES_READ)  # Do NOT sort.

    @property
    def axes_w(self) -> list[str]:
        """
        HKL real axis names.

        Updated by 'forward()' computation.
        """
        return self.engine.axis_names_get(AXES_WRITTEN)  # Do NOT sort.

    def calculate_UB(
        self,
        r1: SolverReflection,
        r2: SolverReflection,
    ) -> SolverMatrix3x3:
        """
        Calculate the UB (orientation) matrix with two reflections.

        The method of Busing & Levy, Acta Cryst 22 (1967) 457.
        """
        if self._sample is None:
            return
        # Remove all reflections first
        self.removeAllReflections()
        self.addReflection(r1)
        self.addReflection(r2)
        self._sample.compute_UB_busing_levy(*self._sample.reflections_get())
        logger.debug("%r reflections", len(self._sample.reflections_get()))
        return self.UB

    # TODO: Want to use libhkl.Engine here instead of Any
    @property
    def engine(self) -> Any:
        """Selected computational engine for this geometry."""
        return self._hkl_engine

    @property
    def engine_name(self) -> str:
        """Name of selected computational engine for this geometry."""
        return self.engine.name_get()

    @property
    def engines(self) -> list[str]:
        """List of the computational engines available in this geometry."""
        return [engine.name_get() for engine in self._hkl_engine_list.engines_get()]

    @property
    def extra_axis_names(self) -> list[str]:
        """
        Ordered list of any extra parameter names (such as x, y, z).

        Depends on selected geometry, engine, and mode.
        """
        return self.engine.parameters_names_get()  # Do NOT sort.

    @property
    def extras(self) -> NamedFloatDict:
        """
        Ordered dictionary of any extra parameters.

        Depends on selected geometry, engine, and mode.
        """
        return dict(
            zip(
                self.extra_axis_names,
                self.engine.parameters_values_get(LIBHKL_USER_UNITS),
            )
        )

    @extras.setter
    def extras(self, values: NamedFloatDict) -> None:
        known_names = self.extra_axis_names
        for k in values.keys():
            if k not in known_names:
                raise KeyError(
                    f"Unexpected dictionary key received: {k!r}"
                    # .
                    f" Expected one of these: {known_names!r}"
                )
        logger.debug("extras.setter(): values=%s", values)
        for k, v in values.items():
            p = self.engine.parameter_get(k)
            p.value_set(v, LIBHKL_USER_UNITS)
            self.engine.parameter_set(k, p)

    def forward(self, pseudos: NamedFloatDict) -> List[NamedFloatDict]:
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        from gi.repository import GLib  # noqa: E402, F401  # W0611

        logger.debug("(%r) forward(%r)", __name__, pseudos)

        try:
            # Hkl.GeometryList is not a dict.
            # Still, it has a .items() method.
            raw = list(
                self.engine.pseudo_axis_values_set(
                    list(pseudos.values()),
                    LIBHKL_USER_UNITS,
                ).items()
            )

            solutions = []
            for glist_item in raw:
                geo = glist_item.geometry_get()
                sol = dict(
                    zip(
                        geo.axis_names_get(),
                        roundoff_list(geo.axis_values_get(LIBHKL_USER_UNITS)),
                    )
                )
                solutions.append(sol)
            return solutions

        except GLib.GError as exc:
            raise NoForwardSolutions("No forward solutions found.") from exc

    @classmethod
    def geometries(cls) -> list[str]:
        """
        List all geometries that have one or more computational engines.

        Geometry is not usable without a computational engine.
        """

        factories = libhkl.factories()

        def num_engines(geometry):
            factory = factories[geometry]
            engine_list = factory.create_new_engine_list()
            engines = [engine.name_get() for engine in engine_list.engines_get()]
            return len(engines)

        return sorted(
            [
                geometry
                #
                for geometry in factories
                if num_engines(geometry) > 0
            ]
        )

    def inverse(self, reals: NamedFloatDict) -> NamedFloatDict:
        """Compute tuple of pseudos from reals (angles -> hkl)."""
        logger.debug("{__name__=} inverse(reals=%r)", reals)
        if list(reals) != self.real_axis_names:
            raise ValueError(
                f"Wrong dictionary keys received: {list(reals)!r}"
                # .
                f" Expected: {self.real_axis_names!r}"
            )
        if False in [isinstance(v, (float, int)) for v in reals.values()]:
            # fmt: off
            raise TypeError(
                "All values must be numbers."
                f"  Received: {reals!r}"
            )
            # fmt: on

        reals = list(reals.values())
        self._hkl_geometry.axis_values_set(reals, LIBHKL_USER_UNITS)

        self._hkl_engine_list.get()  # reals -> pseudos  (Odd name for this call!)

        # Assemble the dictionary
        pdict = dict(
            zip(
                self.engine.pseudo_axis_names_get(),
                roundoff_list(self.engine.pseudo_axis_values_get(LIBHKL_USER_UNITS)),
            )
        )
        return pdict

    @property
    def lattice(self) -> NamedFloatDict:
        """
        Dictionary of crystal lattice parameters.
        """
        values = self._sample.lattice_get().get(LIBHKL_USER_UNITS)
        keys = "a b c alpha beta gamma".split()
        return {k: getattr(values, k) for k in keys}

    @lattice.setter
    def lattice(self, value: NamedFloatDict):
        if not istype(value, dict):
            raise TypeError(f"Must supply {NamedFloatDict} object, received {value!r}")

        logger.debug("lattice.setter(): value=%s", value)
        self._sample.lattice_set(
            libhkl.Lattice.new(
                value["a"],
                value["b"],
                value["c"],
                math.radians(value["alpha"]),
                math.radians(value["beta"]),
                math.radians(value["gamma"]),
            )
        )
        logger.debug(
            "sample lattice: %r",
            self._sample.lattice_get().get(LIBHKL_USER_UNITS),
        )

    @property
    def mode(self) -> str:
        """Name of the current geometry operating mode."""
        return self.engine.current_mode_get()

    @mode.setter
    def mode(self, value: str):
        check_value_in_list("Mode", value, self.modes, blank_ok=True)
        if value == "":
            return  # keep current mode
        logger.debug("mode.setter(): value=%s", value)
        self.engine.current_mode_set(value)

    @property
    def modes(self) -> list[str]:
        """List of the geometry operating modes."""
        if self.engine is None:
            return []
        return self.engine.modes_names_get()

    @property
    def pseudo_axis_names(self) -> list[str]:
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        return self.engine.pseudo_axis_names_get()  # Do NOT sort.

    @property
    def real_axis_names(self) -> list[str]:
        """Ordered list of the real axis names (such as th, tth)."""
        return self._hkl_geometry.axis_names_get()  # Do NOT sort.

    @property
    def reflections(self) -> Dict[str, SolverReflection]:
        """List of defined reflections (no store reflection names in libhkl)."""
        rlist = {}
        for refl in self._sample.reflections_get():
            values = refl.hkl_get()
            name = f"x{id(refl):x}"  # make up a name from object's id
            rlist[name] = dict(
                name=name,
                pseudos={k: getattr(values, k) for k in self.pseudo_axis_names},
                reals=dict(
                    zip(
                        refl.geometry_get().axis_names_get(),
                        refl.geometry_get().axis_values_get(1),
                    )
                ),
                wavelength=refl.geometry_get().wavelength_get(LIBHKL_USER_UNITS),
            )
        return rlist

    def refineLattice(self, reflections: list[SolverReflection]) -> NamedFloatDict:
        """
        Refine the lattice parameters from a list of reflections.

        hkl_soleil uses a simplex method.
        """
        if len(reflections) < 3:
            raise ValueError("Must provide 3 or more reflections to refine lattice.")
        self.removeAllReflections()
        for r in reflections:
            self.addReflection(r)

        self._sample.affine()  # refine the lattice

        return self.lattice

    def removeAllReflections(self) -> None:
        """Remove all reflections."""
        refs = self._sample.reflections_get()
        for ref in refs:
            self._sample.del_reflection(ref)

    @property
    def sample(self) -> SolverSample:
        """
        Crystalline sample.  libhkl's sample object.
        """
        sample = dict(
            name=self._sample.name_get(),
            lattice=self.lattice,
            reflections=self.reflections,
        )
        return sample

    @sample.setter
    def sample(self, value: SolverSample):
        if not istype(value, dict):
            raise TypeError(f"Must supply {SolverSample} object, received {value!r}")

        # Just drop the old sample and make a new one.
        # Python knows its correct name.
        # Doesn't matter what name is used by libhkl. Use a unique name.
        logger.debug("sample.setter(): value=%s", value)
        sample = libhkl.Sample.new(unique_name())  # new sample each time
        self._sample = sample
        self._hkl_engine_list.init(self._hkl_geometry, self._hkl_detector, sample)
        logger.debug(
            "sample name=%r, libhkl name=%r",
            value["name"],
            sample.name_get(),
        )

        self.lattice = value["lattice"]

        logger.debug("%r ordering reflections: %r", str(self), value["order"])
        for name in value["order"]:
            for refl in value["reflections"]:
                if refl["name"] == name:
                    self.addReflection(refl)
                    break

    def set_reals(self, reals: NamedFloatDict) -> None:
        """Set the values of all the real axes."""
        self._hkl_geometry.axis_values_set(
            (list(reals.values())),
            LIBHKL_USER_UNITS,
        )

    @property
    def _summary_dict(self):
        """Return a summary of the geometry (engines, modes, axes)"""
        geometry_name = self.geometry
        description = {"name": geometry_name}
        factories = libhkl.factories()

        factory = factories[geometry_name]
        engine_list = factory.create_new_engine_list()

        engines = {engine.name_get(): engine for engine in engine_list.engines_get()}
        description["engines"] = {}
        for engine_name, engine in engines.items():
            eng_desc = {
                "pseudos": engine.pseudo_axis_names_get(),
                "reals": {},
                "modes": {},
            }
            description["engines"][engine_name] = eng_desc
            eng_desc["reals"] = engine.axis_names_get(AXES_READ)
            extras = []
            for mode_name in engine.modes_names_get():
                engine.current_mode_set(mode_name)
                eng_desc["modes"][mode_name] = {
                    "extras": engine.parameters_names_get(),
                    "reals": engine.axis_names_get(AXES_WRITTEN),
                }
                extras += eng_desc["modes"][mode_name]["extras"]
            eng_desc["extras"] = list(sorted(set(extras)))
        return description

    @property
    def summary(self) -> Table:
        """
        Table of engines, modes, & axes for this geometry.

        EXAMPLE::

            >>> fourc = hklpy2.creator(name="fourc", geometry="E4CV")
            >>> print(fourc.core.solver_summary)
            ========= ================== ================== ==================== ==================== ===============
            engine    mode               pseudo(s)          real(s)              writable(s)          extra(s)
            ========= ================== ================== ==================== ==================== ===============
            hkl       bissector          h, k, l            omega, chi, phi, tth omega, chi, phi, tth
            hkl       constant_omega     h, k, l            omega, chi, phi, tth chi, phi, tth
            hkl       constant_chi       h, k, l            omega, chi, phi, tth omega, phi, tth
            hkl       constant_phi       h, k, l            omega, chi, phi, tth omega, chi, tth
            hkl       double_diffraction h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
            hkl       psi_constant       h, k, l            omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2, psi
            psi       psi                psi                omega, chi, phi, tth omega, chi, phi, tth h2, k2, l2
            q         q                  q                  tth                  tth
            incidence incidence          incidence, azimuth omega, chi, phi                           x, y, z
            emergence emergence          emergence, azimuth omega, chi, phi, tth                      x, y, z
            ========= ================== ================== ==================== ==================== ===============
        """
        table = Table()
        table.labels = "engine mode pseudo(s) real(s) writable(s) extra(s)".split()
        for engine_name, engine in self._summary_dict["engines"].items():
            for mode_name, mode in engine["modes"].items():
                row = [
                    engine_name,
                    mode_name,
                    ", ".join(engine["pseudos"]),
                    ", ".join(engine["reals"]),
                    ", ".join(mode["reals"]),
                    ", ".join(mode["extras"]),
                ]
                table.addRow(row)
        return table

    @property
    def U(self) -> SolverMatrix3x3:
        """
        Relative orientation of crystal on diffractometer.

        Rotation matrix,  (3x3).
        """
        if self._sample is None:
            return IDENTITY_MATRIX_3X3
        matrix = to_numpy(self._sample.U_get())
        return matrix.round(decimals=ROUNDOFF_DIGITS).tolist()

    @U.setter
    def U(self, value: SolverMatrix3x3) -> None:
        if self._sample is not None:
            logger.debug("U.setter(): value=%s", value)
            self._sample.U_set(to_hkl(value))

    @property
    def UB(self) -> SolverMatrix3x3:
        """Orientation matrix (3x3)."""
        if self._sample is None:
            return IDENTITY_MATRIX_3X3
        matrix = to_numpy(self._sample.UB_get())
        return matrix.round(decimals=ROUNDOFF_DIGITS).tolist()

    @UB.setter
    def UB(self, value: SolverMatrix3x3) -> None:
        if self._sample is not None:
            logger.debug("UB.setter(): value=%s", value)
            self._sample.UB_set(to_hkl(value))

    @property
    def wavelength(self) -> float:
        """Monochromatic wavelength."""
        return self._hkl_geometry.wavelength_get(LIBHKL_USER_UNITS)

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        logger.debug("wavelength.setter(): value=%s", value)
        return self._hkl_geometry.wavelength_set(value, LIBHKL_USER_UNITS)

    @property
    def _details(self) -> dict:
        """(internal use) Current settings for diagnostic review."""
        return dict(
            name=self.name,
            geometry=self.geometry,
            engine=self.engine_name,
            sample=self._sample.name_get(),
            lattice=self.lattice,
            U=self.U,
            UB=self.UB,
            wavelength=self.wavelength,
            mode=self.mode,
            extras=self.extras,
        )
