"""Create missing supported CI environment input files."""

import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "code.yml"
ENVIRONMENTS = ROOT / "ci"
TEMPLATE = ENVIRONMENTS / "environment-3.11.yml"


def supported_versions() -> list[str]:
    """Return supported numeric Python versions from the test matrix."""
    workflow = WORKFLOW.read_text()
    matrix = re.search(
        r"python-version:\n(?P<versions>.*?)(?:\n\s*include:|\n\s*max-parallel:)",
        workflow,
        re.DOTALL,
    )
    if matrix is None:
        raise RuntimeError(f"Could not find Python matrix in {WORKFLOW}")

    return re.findall(
        r"^\s*-\s*[\"\'](?P<version>3\.\d+)[\"\']\s*$", matrix["versions"], re.MULTILINE
    )


def create_missing_environment_files() -> list[Path]:
    """Create environment inputs for matrix versions not yet represented."""
    template = TEMPLATE.read_text()
    created = []
    for version in supported_versions():
        path = ENVIRONMENTS / f"environment-{version}.yml"
        if path.exists():
            continue
        path.write_text(template.replace("python=3.11", f"python={version}", 1))
        created.append(path)
    return created


if __name__ == "__main__":
    for path in create_missing_environment_files():
        print(f"created {path.relative_to(ROOT)}")
