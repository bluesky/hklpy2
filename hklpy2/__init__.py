"""
Package-level initialization.
"""

# -----------------------------------------------------------------------------
# copyright (c) 2023-2024, UChicago Argonne, LLC
#
# Distributed under the terms of the
# Argonne National Laboratory Open Source License.
#
# The full license is in the file LICENSE, distributed with this software.
# -----------------------------------------------------------------------------

__settings_orgName__ = "prjemian"
__package_name__ = "hklpy2"

try:
    from setuptools_scm import get_version

    __version__ = get_version(root="..", relative_to=__file__)
    del get_version
except (LookupError, ModuleNotFoundError):
    from importlib.metadata import version

    __version__ = version(__package_name__)
    del version


class Hklpy2Error(Exception):
    """Any exception from the |hklpy2| package."""


from .backends import SolverBase  # noqa: E402, F401
from .diffract import DiffractometerBase  # noqa: E402, F401
from .lattice import SI_LATTICE_PARAMETER  # noqa: E402, F401
from .lattice import Lattice  # noqa: E402, F401
from .misc import SOLVER_ENTRYPOINT_GROUP  # noqa: E402, F401
from .misc import SolverError  # noqa: E402, F401
from .misc import check_value_in_list  # noqa: E402, F401
from .misc import get_solver  # noqa: E402, F401
from .misc import solver_factory  # noqa: E402, F401
from .misc import solvers  # noqa: E402, F401
from .reflection import Reflection  # noqa: E402, F401
from .reflection import ReflectionsDict  # noqa: E402, F401
from .sample import Sample  # noqa: E402, F401
from .wavelength_support import A_KEV  # noqa: E402, F401
from .wavelength_support import ConstantMonochromaticWavelength  # noqa: E402, F401
from .wavelength_support import MonochromaticXrayWavelength  # noqa: E402, F401
from .wavelength_support import WavelengthError  # noqa: E402, F401
