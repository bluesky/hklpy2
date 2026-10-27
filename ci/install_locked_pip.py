"""Install the PyPI entries recorded in a conda-lock file."""

import subprocess
import sys
from pathlib import Path

import yaml


def main(lockfile: Path) -> None:
    lock = yaml.safe_load(lockfile.read_text())
    urls = [
        package["url"] for package in lock["package"] if package["manager"] == "pip"
    ]
    if urls:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", *urls], check=True
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
