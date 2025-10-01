# AI Agent advice for hklpy2

<https://agents.md>

## Purpose

Goal: Short guide for coding agents (auto-formatters, linters, CI bots, test runners, codegen agents) working on this Python project.

## Code Style

- Concise type annotations
- code location described in pyproject.toml
- style information described in pyproject.toml
- `pre-commit run --all-files`


## Agent pytest style (for automated agents)

---

- Agents must write tests using parametrized pytest patterns and explicit context managers for expected success/failure.
- Prefer more parameter sets to minimize the number of test functions.
- Use `from contextlib import nullcontext as does_not_raise` for success cases and `pytest.raises(...)` for expected exceptions.
- Construct objects and perform assignments that may raise inside the `with context:` block. Place assertions about object state after the `with` when the case expects success.
- Use the project's helper `assert_context_result(expected, reason)` where available to standardize result checks.
- Example pattern (brief):

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

## Agent behavior rules

- Always follow the project's formatting, linting, and typing configs.
- Make minimal, focused changes; prefer separate commits per concern.
- Add or update tests for any behavioral change.
- Include clear commit messages and PR descriptions.
- If uncertain about design, open an issue instead of making large changes.
- Respect branch protection: push to feature branches and create PRs.

## Test style

- All test code for MODULE.py goes in file tests/test_MODULE.py
- tests should be written and organized using the project's test style guidelines.
- use parametrized pytests
- Prefer parameter sets that simulate user interactions
- all tests run code within context
- Store test code modules in submodule/tests/ directory
- maximize code coverage
- Use parametrized pytests
  - Generate additional parameters and sets to minimize the number of test functions.
  - Place all functional code in a parametrized context.
    - use parameter for does_not_raise() or pytest.raises(...) as fits the parameter set
      - `from contextlib import nullcontext as does_not_raise`
    - do not separate success and errors tests into different test functions
    - do not separate success and errors tests using try..except

## Inputs & outputs

- Inputs: file diffs, test results, config files, repository metadata
- Outputs: patch/commit, tests, updated docs, CI status

## Running locally

- Setup: create virtualenv, `pip install -e .[all]`
- Common commands:
  - Format & Lint: `pre-commit run --all-files`
  - Test: `pytest ./hklpy2`

## CI integration

- Format and lint in pre-commit job
- Run tests and dependency audit on PRs.

## Minimal example PR checklist

- Runs formatting and linting locally
- Adds/updates tests for changes
- Includes changelog entry if behavior changed
- Marks PR as draft if large refactor

## Notes

- Keep agent actions small, reversible, and reviewable.
- When updating a file, verify that a change has actually been made by comparing
  the mtime before and after the edits.

## Code Coverage

- Aim for 100% coverage, but prioritize meaningful tests over simply hitting every line.
