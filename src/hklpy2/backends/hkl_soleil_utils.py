# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Additional support for hkl_soleil solver."""

from ..exceptions import SolverError


def setup_libhkl(system: str, library: str, version: str) -> object:
    """
    Setup current session to load 'library' with the 'gi' package.

    This function was written so the procedure here can test as if it
    was running on other OS.

    .. seealso: https://softwareengineering.stackexchange.com/questions/222383

    Example::

        gi_require_library(platform.system(), "Hkl", "5.0")
    """
    if system != "Linux":
        raise SolverError("'hkl_soleil' only available for linux 64-bit.")

    try:
        import gi
    except (ImportError, ModuleNotFoundError) as e:
        raise SolverError("Cannot import 'gi' (gobject-introspection) library.") from e

    try:
        gi.require_version(library, version)
    except Exception as exinfo:
        raise SolverError(f"Cannot load 'gi' library: {library}, {version}") from exinfo

    from gi.repository import Hkl as Hkl  # noqa: N813

    return Hkl
