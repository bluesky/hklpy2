#!/usr/bin/env python3
"""
Update the copyright end year in tracked source files to the current year.

Designed to be used as a pre-commit hook (local repo hook).  Exit codes follow
the pre-commit convention:

* 0 – nothing changed (all files already have the correct year).
* 1 – one or more files were modified; pre-commit will mark the hook as
      "failed" so the developer sees the diff and stages the updated files
      before retrying the commit.

Files checked (paths relative to the repository root):

* pyproject.toml              – ``[tool.copyright]`` value
* src/hklpy2/__init__.py      – comment header
* docs/source/index.rst       – summary table row

The script rewrites the pattern ``<START_YEAR>-<OLD_YEAR>`` →
``<START_YEAR>-<CURRENT_YEAR>`` wherever it appears in those files, leaving
everything else untouched.
"""

from __future__ import annotations

import pathlib
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Files that contain the copyright year span, relative to the repo root.
TARGET_FILES: list[pathlib.Path] = [
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "src" / "hklpy2" / "__init__.py",
    REPO_ROOT / "docs" / "source" / "index.rst",
]

# Matches e.g. "2023-2025" and captures the start year and old end year.
YEAR_RANGE_PATTERN = re.compile(r"(\d{4})-(\d{4})")

CURRENT_YEAR = str(datetime.now().year)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def update_file(path: pathlib.Path, current_year: str) -> bool:
    """
    Replace stale end-years in *path* with *current_year*.

    Returns True if the file was modified.
    """
    original = path.read_text(encoding="utf-8")

    def _replace(match: re.Match) -> str:
        start, end = match.group(1), match.group(2)
        if end == current_year:
            return match.group(0)  # already up to date
        return f"{start}-{current_year}"

    updated = YEAR_RANGE_PATTERN.sub(_replace, original)

    if updated == original:
        return False

    path.write_text(updated, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    changed: list[pathlib.Path] = []

    for filepath in TARGET_FILES:
        if not filepath.exists():
            print(f"WARNING: {filepath} not found – skipping.", file=sys.stderr)
            continue
        if update_file(filepath, CURRENT_YEAR):
            changed.append(filepath)
            print(
                f"Updated copyright end year to {CURRENT_YEAR} in: {filepath.relative_to(REPO_ROOT)}"
            )

    if changed:
        print(
            "\nCopyright year(s) updated. Stage the changed file(s) and re-run the commit.",
            file=sys.stderr,
        )
        return 1  # signal pre-commit that something changed

    return 0


if __name__ == "__main__":
    sys.exit(main())
