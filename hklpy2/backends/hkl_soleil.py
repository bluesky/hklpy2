"""
Backend: Hkl (``"hkl_soleil"``)

.. autosummary::

    ~HklSolver
"""

import gi

gi.require_version("Hkl", "5.0")

from gi.repository import GLib  # noqa: E402, F401
from gi.repository import Hkl as libhkl  # noqa: E402

from .base import SolverBase  # noqa: E402


class HklSolver(SolverBase):
    """
    ``"hkl_soleil"`` (Linux x86_64 only) |libhkl|.

    |solver| with support for many common diffractometer geoemtries.
    Wraps the |libhkl| library from Frédéric-Emmanuel PICCA (Soleil).

    .. autosummary::

        ~addReflection
        ~addSample
        ~calculateOrientation
        ~forward
        ~geometries
        ~inverse
        ~modes
        ~pseudo_axis_names
        ~real_axis_names
        ~refineLattice
        ~setGeometry
        ~setLattice
        ~setMode
    """

    __name__ = "hkl_soleil"
    __version__ = libhkl.VERSION

    def __init__(self) -> None:
        super().__init__()
        self.gname = None

        self.detector = libhkl.Detector.factory_new(libhkl.DetectorType(0))
        self._engine = None
        self._engines = None
        self._factory = None
        self._geometry = None
        self.user_units = libhkl.UnitEnum.USER

    def addReflection(self, pseudos, reals, wavelength):
        """Add information about a reflection."""
        pass  # TODO

    def addSample(self, sample):
        """Add a sample."""
        pass  # TODO

    def calculateOrientation(self, r1, r2):
        """Calculate the UB (orientation) matrix from two reflections."""
        pass  # TODO

    def forward(self):
        """Compute list of solutions(reals) from pseudos (hkl -> [angles])."""
        return []  # TODO

    @property
    def geometries(self):
        """
        Ordered list of the geometry names.

        For |libhkl|, each geometry may have zero or more computational
        *engines*. Each engine has its own set of pseudos and reals.  Some of
        the engines have additional (optional) axes.

        So the gemoetry *names* include both the geometry and its engine, such
        as `"E4CV, hkl"`.

        TODO: confirm with the |libhkl| docs
        """
        geometries = []
        for fname, factory in libhkl.factories().items():
            # Underlying library raises error for the merged call:
            #   factory.create_new_engine_list().engines_get()
            # MUST do this in two parts here (and elsewhere).
            engines = factory.create_new_engine_list()
            for engine in engines.engines_get():
                geometries.append(f"{fname}, {engine.name_get()}")
        return sorted(set(geometries))

    @property
    def geometry(self):
        """Diffractometer geometry, such as `"E4CV, hkl"`."""
        return self._geometry

    @geometry.setter
    def geometry(self, value):
        if not isinstance(value, (type(None), str)):
            raise TypeError(f"Must supply str, received {value!r}")
        # self._geometry = value
        if value not in self.geometries:
            raise KeyError(f"Geometry {value} unknown.")
        gname, engine = [s.strip() for s in value.split(",")]
        self.setGeometry(gname, engine=engine)  # TODO: refactor all of setGeometry here

    def inverse(self):
        """Compute tuple of pseudos from reals (angles -> hkl)."""
        return tuple()  # TODO

    @property
    def modes(self):
        """List of the geometry operating modes."""
        return []  # TODO

    @property
    def pseudo_axis_names(self):
        """Ordered list of the pseudo axis names (such as h, k, l)."""
        if self._engine is not None:
            return self._engine.pseudo_axis_names_get()

    @property
    def real_axis_names(self):
        """Ordered list of the real axis names (such as th, tth)."""
        if self._geometry is not None:
            return self._geometry.axis_names_get()

    def refineLattice(self, reflections):
        """Refine the lattice parameters from a list of reflections."""
        pass  # TODO

    def setGeometry(self, gname, engine="hkl"):
        self._factory = libhkl.factories()[gname]
        self.gname = gname
        self._geometry = self._factory.create_new_geometry()
        self._engines = self._factory.create_new_engine_list()
        self._engine = self._engines.engine_get_by_name(engine)
        return self._geometry

    def setLattice(self, lattice):
        """Define the sample's lattice parameters."""
        pass  # TODO

    def setMode(self, mode):
        """
        Define the geometry's operating mode.

        A mode defines constraints on the solutions provided by the
        :meth:`forward` computation.
        """
        pass  # TODO
