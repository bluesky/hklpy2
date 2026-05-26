# Copyright (c) 2023-2026 UChicago Argonne, LLC
# SPDX-License-Identifier: LicenseRef-UChicago-Argonne-LLC-License
"""Code that is common to several tests."""

import pathlib

HKLPY2_DIR = pathlib.Path(__file__).parent.parent
IOC_PREFIX = "hklpy2:"
PV_ENERGY = f"{IOC_PREFIX}energy"
PV_WAVELENGTH = f"{IOC_PREFIX}wavelength"
TESTS_DIR = HKLPY2_DIR / "tests"
