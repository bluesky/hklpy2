<!-- Generated guidance for AI coding agents working on hklpy2 -->

# hklpy2 — Copilot instructions (concise)

This file gives focused, actionable guidance for automated coding agents and Copilot-style assistants working on the hklpy2 repository.

Key goals
- Make small, focused, and reversible changes. Prefer feature branches and PRs.
- Always keep tests green for behavioral changes; add tests for any changed behavior.

Quick environment & commands
- Install: create a venv and run `pip install -e .[all]` from the repo root.
- Formatting/linting: repository uses pre-commit. Run `pre-commit run --all-files` before commits.
- Tests: run `pytest ./hklpy2` (or run targeted tests under `hklpy2/*/tests`).

Project structure highlights
- `hklpy2/` — package source. Important subpackages:
  - `backends/` — hardware/solver backends (e.g. `hkl_soleil.py`).
  - `blocks/` — high-level domain objects (e.g. `reflection.py`, `lattice.py`, `sample.py`).
  - `tests/` and per-module `tests/` directories — prefer targeted tests close to code.
- Docs live under `docs/source/` and include API and conceptual guides used by maintainers.

Conventions and patterns
- Keep public API stable. Many callers construct `Reflection` and `ReflectionsDict`; preserve dict-based APIs and axis-name ordering.
- Tests use parametrized pytest patterns and expect in-context execution (see `AGENTS.md`). Use `does_not_raise()` (nullcontext) for success paths.
- Code style: concise type annotations where useful. Respect project pyproject.toml settings for formatting.

Behavioral change rules (strict)
- Any behavioral change must include unit tests demonstrating the change.
- Small interface adjustments may be tolerated if they are backwards-compatible or well-documented in the PR description.

Safe edit checklist for agents
1. Read related files before editing (e.g., editing `blocks/reflection.py` — read `ops.py`, `misc.py`, and `blocks/tests/test_reflection.py`).
2. Make a minimal patch and run local tests for the modified area.
3. Run `pre-commit run --all-files` and fix issues.
4. If tests fail, iterate locally until targeted tests pass.

Agent pytest style (for automated agents)
------------------------------------------------

- Agents must write tests using parametrized pytest patterns and explicit context managers for expected success/failure.
- Prefer more parameter sets to minimize the number of test functions.
- Use `from contextlib import nullcontext as does_not_raise` for success cases and `pytest.raises(...)` for expected exceptions.
- Construct objects and perform assignments that may raise inside the `with context:` block. Place assertions about object state after the `with` when the case expects success.
- Use the project's helper `assert_context_result(expected, reason)` where available to standardize result checks.

Example (brief):

```py
@pytest.mark.parametrize(
  "set_value, context, expected",
  [
    ("angstrom", does_not_raise(), "angstrom"),
    ("not_a_unit", pytest.raises(Exception), None),
  ],
)
def test_length_units_property_and_validation(set_value, context, expected):
  with context:
    lat = Lattice(3.0)
    lat.length_units = set_value

  if expected is not None:
    assert lat.length_units == expected
```

This makes tests explicit and machine-friendly for automated agents.

## Enforcement

PRs opened or modified by automated agents must follow the "Agent pytest style" described above. Reviewers and CI will check for this pattern (test parametrization, use of context managers for expected outcomes, and the `assert_context_result` helper). Changes from agents that do not comply may be requested for revision or reverted.

Integration points & external deps
- `pint` is used for units (see `reflection.py`).
- Backends may interface with EPICS/solvers; avoid adding runtime hardware calls in unit tests — mock them instead.

Where to look for examples
- `hklpy2/blocks/tests/test_reflection.py` — tests that illustrate expected `Reflection` behavior.
- `hklpy2/backends/tests/` — tests for backend behavior and solver interactions.

When unsure
- Open a small issue instead of making large design changes. When in doubt, keep changes minimal and reversible.

Contact
- Use the PR description to summarize the agent-made changes and list tests added/updated.
