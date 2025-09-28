"""
Package-level initialization.
"""

# -----------------------------------------------------------------------------
# copyright (c) 2023-2025, UChicago Argonne, LLC
#
# Distributed under the terms of the
# Argonne National Laboratory Open Source License.
#
# The full license is in the file LICENSE, distributed with this software.
# -----------------------------------------------------------------------------

__settings_orgName__ = "bluesky"
__package_name__ = "hklpy2"


def _get_version():
    """Make the version code testable."""
    import importlib.metadata
    import importlib.util

    text = importlib.metadata.version(__package_name__)

    if importlib.util.find_spec("setuptools_scm") is not None:
        """Preferred source of package version information."""
        import setuptools_scm

        try:
            text = setuptools_scm.get_version(root="..", relative_to=__file__)
        except LookupError:
            pass  # TODO: How to test this?  See GH prjemian/Chewacla repo.

    return text


__version__ = _get_version()  # Must define before these imports.
from .backends import SolverBase  # noqa: E402, F401
from .blocks.configure import Configuration  # noqa: E402, F401
from .blocks.lattice import SI_LATTICE_PARAMETER  # noqa: E402, F401
from .diffract import DiffractometerBase  # noqa: E402, F401
from .diffract import creator  # noqa: E402, F401, F403
from .diffract import diffractometer_class_factory  # noqa: E402, F401, F403
from .incident import A_KEV  # noqa: E402, F401
from .misc import SOLVER_ENTRYPOINT_GROUP  # noqa: E402, F401
from .misc import ConfigurationRunWrapper  # noqa: E402, F401
from .misc import SolverError  # noqa: E402, F401
from .misc import get_solver  # noqa: E402, F401
from .misc import solver_factory  # noqa: E402, F401
from .misc import solvers  # noqa: E402, F401
