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

- use parametrized pytests
- all tests run code within context
- do not separate success and errors tests into different test functions
- Store test code modules in submodule/tests/ directory
- maximize code coverage

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

See GitHub Chewacla repo to improve coverage in `hklpy2/__init__.py`

```console
# 2025-09-26
Name                                             Stmts   Miss    Cover   Missing
--------------------------------------------------------------------------------
hklpy2/__init__.py                                  28      2  92.857%   31-32
hklpy2/backends/__init__.py                          1      0 100.000%
hklpy2/backends/base.py                            123      0 100.000%
hklpy2/backends/hkl_soleil.py                      281      0 100.000%
hklpy2/backends/hkl_soleil_utils.py                 14      0 100.000%
hklpy2/backends/no_op.py                            37      0 100.000%
hklpy2/backends/tests/__init__.py                    0      0 100.000%
hklpy2/backends/tests/test_base.py                  80      0 100.000%
hklpy2/backends/tests/test_hkl_soleil.py           182      0 100.000%
hklpy2/backends/tests/test_hkl_soleil_utils.py      24      0 100.000%
hklpy2/backends/tests/test_no_op.py                 11      0 100.000%
hklpy2/backends/tests/test_th_tth_q.py              83      0 100.000%
hklpy2/backends/th_tth_q.py                         89      0 100.000%
hklpy2/blocks/__init__.py                            0      0 100.000%
hklpy2/blocks/configure.py                          35      0 100.000%
hklpy2/blocks/constraints.py                        74      0 100.000%
hklpy2/blocks/lattice.py                            72      0 100.000%
hklpy2/blocks/reflection.py                        182      2  98.901%   121, 178
hklpy2/blocks/sample.py                            100      0 100.000%
hklpy2/blocks/tests/__init__.py                      0      0 100.000%
hklpy2/blocks/tests/test_configure.py              106      0 100.000%
hklpy2/blocks/tests/test_constraints.py             92      0 100.000%
hklpy2/blocks/tests/test_lattice.py                 42      0 100.000%
hklpy2/blocks/tests/test_reflection.py             147      0 100.000%
hklpy2/blocks/tests/test_sample.py                 107      0 100.000%
hklpy2/blocks/tests/test_solver.py                  40     12  70.000%   45-61
hklpy2/diffract.py                                 314      0 100.000%
hklpy2/incident.py                                 130      0 100.000%
hklpy2/misc.py                                     305      0 100.000%
hklpy2/ops.py                                      354      0 100.000%
hklpy2/tests/__init__.py                             1      0 100.000%
hklpy2/tests/common.py                               9      0 100.000%
hklpy2/tests/models.py                              74      0 100.000%
hklpy2/tests/test_backends.py                       23      0 100.000%
hklpy2/tests/test_demo_notebook.py                  75      0 100.000%
hklpy2/tests/test_diffract.py                      318      0 100.000%
hklpy2/tests/test_e4cv.py                           40      0 100.000%
hklpy2/tests/test_incident.py                       71      0 100.000%
hklpy2/tests/test_init.py                           11      0 100.000%
hklpy2/tests/test_misc.py                          269     23  91.450%   656, 660-670, 712, 716-721, 741-746
hklpy2/tests/test_ops.py                           158      0 100.000%
hklpy2/tests/test_tardis.py                         52      0 100.000%
hklpy2/tests/test_user.py                          240      0 100.000%
hklpy2/user.py                                     130      0 100.000%
--------------------------------------------------------------------------------
TOTAL                                             4524     39  99.138%
```
