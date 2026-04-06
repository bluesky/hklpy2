"""
Exercise Hkl's (libhkl) Python API.
"""

import gi
import numpy
import pyRestTable

gi.require_version("Hkl", "5.0")
from gi.repository import Hkl  # noqa: E402

DEFAULT_DIGITS = 9
UNITS = Hkl.UnitEnum.USER


class Table(pyRestTable.Table):
    def add(self, dd: dict):
        keys = list(dd.keys())
        if len(self.labels) == 0:
            self.labels = keys
        if keys == self.labels:
            self.addRow(list(dd.values()))
        else:
            KeyError(
                "All rows must have same keys."
                f"  Received {keys!r}, expected {self.labels!r}."
            )


def to_hkl(arr):
    import numpy as np

    if isinstance(arr, Hkl.Matrix):
        return arr

    arr = np.array(arr)

    hklm = Hkl.Matrix.new_euler(0, 0, 0)
    hklm.init(*arr.flatten())
    return hklm


def to_numpy(mat) -> numpy.ndarray:
    if isinstance(mat, numpy.ndarray):
        return mat

    ret = numpy.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ret[i, j] = mat.get(i, j)

    return ret


class Diffractometer:
    def __init__(self, geometry, engine="hkl") -> None:
        self.detector = Hkl.Detector.factory_new(Hkl.DetectorType(0))
        self.factory = Hkl.factories()[geometry]
        self.engines = self.factory.create_new_engine_list()
        self.geometry = self.factory.create_new_geometry()
        self.sample = Hkl.Sample.new("sample")
        self.engines.init(self.geometry, self.detector, self.sample)
        self.engine = self.engines.engine_get_by_name(engine)
        self._solutions = []

    def _get_as_dict(
        self, obj: object, part: str, digits: int = DEFAULT_DIGITS
    ) -> dict:
        names = getattr(obj, f"{part}_names_get")()
        values = getattr(obj, f"{part}_values_get")(UNITS)
        return dict(zip(names, self._roundoff(values, digits=digits)))

    def _roundoff(self, array: list, digits: int = 9):
        return [round(value, ndigits=digits) or 0.0 for value in array]

    def forward(self, *pseudos: list) -> list:
        self._solutions = self.engine.pseudo_axis_values_set(
            pseudos,
            UNITS,
        )
        return self._solutions

    def inverse(self, *reals: list) -> dict:
        self.angles = reals
        self.engines.get()  # reals -> pseudos  (Odd name for this call!)
        return self.pseudos

    @property
    def angles(self) -> dict:
        return self._get_as_dict(self.geometry, "axis")

    @angles.setter
    def angles(self, values: list) -> None:
        self.geometry.axis_values_set(values, UNITS)

    @property
    def extras(self) -> dict:
        return self._get_as_dict(self.engine, "parameters")

    @extras.setter
    def extras(self, pdict: dict) -> None:
        pnames = self.engine.parameters_names_get()
        for k, v in pdict.items():
            if k not in pnames:
                raise KeyError(f"Unknown parameter name {k!r}.")

            p = self.engine.parameter_get(k)
            p.value_set(v, UNITS)
            self.engine.parameter_set(k, p)

    @property
    def info(self) -> dict:
        result = dict(
            Hkl=Hkl.VERSION,
            # numpy=numpy.__version__,
            geometry=self.geometry.name_get(),
            engine=self.engine.name_get(),
            mode=self.mode,
            wavelength=self.wavelength,
            sample=self.sample_name,
            lattice=self.lattice,
        )
        return result

    @property
    def lattice(self) -> dict:
        lattice = self.sample.lattice_get()
        lattice = lattice.get(UNITS)
        lattice = {k: getattr(lattice, k) for k in "a b c alpha beta gamma".split()}
        return lattice

    @lattice.setter
    def lattice(self, parameters: dict) -> None:
        a, b, c, alpha, beta, gamma = parameters

        lattice = self.sample.lattice_get()
        lattice.set(a, b, c, alpha, beta, gamma, UNITS)
        self.sample.lattice_set(lattice)

    @property
    def mode(self) -> str:
        return self.engine.current_mode_get()

    @mode.setter
    def mode(self, value: str) -> None:
        self.engine.current_mode_set(value)

    @property
    def modes(self) -> list:
        return self.engine.modes_names_get()

    @property
    def pseudos(self) -> dict:
        return self._get_as_dict(self.engine, "pseudo_axis", digits=4)

    @pseudos.setter
    def pseudos(self, values: list) -> None:
        self.engine.pseudo_axis_values_set(
            values,
            UNITS,
        )

    @property
    def sample_name(self) -> str:
        return self.sample.name_get()

    @sample_name.setter
    def sample_name(self, value: str) -> None:
        self.sample.name_set(value)

    @property
    def solutions(self) -> list:
        def sdict(sol):
            geo = sol.geometry_get()
            return self._get_as_dict(geo, "axis", digits=3)

        return [sdict(sol) for sol in self._solutions.items()]

    @property
    def UB(self) -> list:
        matrix = to_numpy(self.sample.UB_get())
        return matrix.round(decimals=5).tolist()

    @UB.setter
    def UB(self, values: list[list[float]]) -> None:
        self.sample.UB_set(to_hkl(values))

    @property
    def wavelength(self) -> float:
        return self.geometry.wavelength_get(UNITS)

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        self.geometry.wavelength_set(value, UNITS)

    @property
    def wh(self) -> None:
        def show(axes):
            keys, values = [], []
            for k, v in axes.items():
                keys.append(f"{k:8s}")
                values.append(f"{str(round(v, ndigits=3) or 0.0):8s}")
            print(" ".join(keys))
            print(" ".join(values))
            print()

        show(self.pseudos)
        show(self.angles)


def psi_scan(start, finish, np):
    e4cv = Diffractometer("E4CV", engine="hkl")

    a0 = 5.431
    e4cv.wavelength = 1.54
    e4cv.sample_name = "silicon"
    e4cv.lattice = a0, a0, a0, 90, 90, 90
    e4cv.mode = "psi_constant"

    e4cv.angles = (30, 0, 0, 60)
    e4cv.extras = dict(h2=1, k2=0, l2=0, psi=0)
    e4cv.pseudos = (1, 0, 1)

    print(e4cv.info)
    print(f"{e4cv.UB=!r}")
    e4cv.wh

    print()
    print(f"Scan psi from {start} to {finish} with {np} points. {e4cv.mode=!r}")
    table = Table()
    for psi in numpy.linspace(start, finish, num=np):
        e4cv.extras = dict(psi=round(psi, ndigits=1))  # only update psi
        e4cv.forward(1, 0, 1)
        if len(e4cv.solutions) > 0:
            results = e4cv.pseudos
            results.update(e4cv.extras)
            results.update(e4cv.solutions[0])
            table.add(results)
    print(table)


def omega_constant(omega):
    e4cv = Diffractometer("E4CV", engine="hkl")

    a0 = 5.431
    e4cv.wavelength = 1.54
    e4cv.sample_name = "silicon"
    e4cv.lattice = a0, a0, a0, 90, 90, 90
    e4cv.mode = "constant_omega"

    e4cv.angles = (omega, 0, 0, 60)

    e4cv.forward(1, 0, 1)
    if len(e4cv.solutions) > 0:
        results = e4cv.pseudos
        results.update(e4cv.solutions[0])
        print(results)

    print(e4cv.info)

    reals = e4cv.solutions[0]
    print(f"{reals=!r}")
    print(f"{e4cv.inverse(*list(reals.values()))=!r}")

    start, finish, np = -1.00028, 1.0001, 16
    print()
    print(f"Scan h from {start} to {finish} with {np} points. {e4cv.mode=!r}")
    table = Table()
    for h1 in numpy.linspace(start, finish, num=np):
        e4cv.forward(h1, 0, 1)
        reals = e4cv.solutions[0]
        results = e4cv.pseudos
        results.update(reals)
        table.add(results)
    print(table)


if __name__ == "__main__":
    psi_scan(-140.0018, 140.01, 16)
    omega_constant(35)
