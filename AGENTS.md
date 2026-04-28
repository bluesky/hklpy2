# AI Agent advice for hklpy2

<https://agents.md>

## Purpose

Goal: Short guide for coding agents (auto-formatters, linters, CI bots, test runners, codegen agents) working on this Python project.

## Code Style

- Concise type annotations
- code location described in pyproject.toml
- style information described in pyproject.toml
- `pre-commit run --all-files`


## Agent pytest style (for automated agents) - MANDATORY

---

**CRITICAL: All test code MUST follow this pattern. Tests not following this pattern will be rejected.**

### Requirements

1. **ALWAYS use parametrized pytest** with `parms, context` as the parameter names
2. **ALWAYS use `pytest.param()`** for each parameter set with `id="..."`
3. **ALWAYS use context managers**: `does_not_raise()` for success, `pytest.raises(...)` for failures
4. **ALWAYS put all functional code and assertions inside `with context:` block**
5. **ALWAYS use `match=re.escape(...)` with `pytest.raises(...)`** for exception matching
6. **ALWAYS include failure test cases** - parameter sets that are expected to raise exceptions must use `pytest.raises(...)`
7. **NEVER create separate test functions** for success vs failure cases
8. **NEVER use try/except** for test logic
9. **NEVER use the deprecated `assert_context_result()` helper**

### Import requirements

```python
from contextlib import nullcontext as does_not_raise
import pytest
```

### Correct pattern (copy this exactly):

```py
@pytest.mark.parametrize(
    "parms, context",
    [
        pytest.param(
            dict(some_param=value1),
            does_not_raise(),
            id="description of test case 1",
        ),
        pytest.param(
            dict(some_param=invalid_value),
            pytest.raises(SomeError, match=re.escape("expected message")),
            id="description of test case 2",
        ),
    ],
)
def test_function_name(parms, context):
    with context:
        # ALL code that might raise goes HERE
        result = object_under_test.method(**parms)
        # ALL assertions go HERE (inside the with block)
        assert result.expected_attribute == some_value
```

### Common mistakes to avoid

- ❌ NOT this:
  ```py
  def test_something():
      # setup code...
      # test code...
  ```

- ❌ NOT this:
  ```py
  def test_success_case():
      # code...
      assert result == expected

  def test_failure_case():
      with pytest.raises(...):
          # code...
  ```

- ❌ NOT this:
  ```py
  @pytest.mark.parametrize(...)
  def test_something(values):
      try:
          result = do_something(values)
      except SomeError:
          # wrong!
  ```

- ✅ ALWAYS this:
  ```py
  @pytest.mark.parametrize(
      "parms, context",
      [
          pytest.param(dict(value=valid), does_not_raise(), id="valid case"),
          pytest.param(dict(value=invalid), pytest.raises(Error), id="error case"),
      ],
  )
  def test_something(parms, context):
      with context:
          result = do_something(**parms)
  ```

## Enforcement

PRs opened or modified by automated agents must follow the "Agent pytest style" described above. Reviewers and CI will check for this pattern (test parametrization and use of context managers for expected outcomes, both successful and failed). Changes from agents that do not comply may be requested for revision or reverted.

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
- When a parameter set contains axis positions (real or pseudo), prefer a
  `dict` with named axes over a bare tuple, so the intent is self-documenting:
  ```py
  # preferred
  dict(reals=dict(omega=-145, chi=0, phi=0, tth=69), ...)
  # avoid
  dict(reals=(-145, 0, 0, 69), ...)
  ```
  Use `sim.inverse(**parms["reals"])` (or `sim.forward(**parms["pseudos"])`)
  to unpack the dict in the test body.  A bare tuple is acceptable only when
  the axis names are not meaningful in context (e.g. a generic zeros vector).

## Inputs & outputs

- Inputs: file diffs, test results, config files, repository metadata
- Outputs: patch/commit, tests, updated docs, CI status

## Running locally

- Setup: create virtualenv, `pip install -e .[all]`
- Common commands:
  - Format & Lint: `make style` or `pre-commit run --all-files`
  - Test: `pytest ./hklpy2`

### pre-commit on NFS home directories

The pre-commit cache (`~/.cache/pre-commit`) lives on NFS on this system
(`aquila:/export/beams1`).  Setuptools' wheel-build cleanup uses `os.rmdir()`
which fails non-deterministically on NFS because directory-entry removals are
not immediately visible.  Symptom: `[Errno 39] Directory not empty` during
`pip install` for any pre-commit hook environment.

**Fix:** keep the pre-commit cache on local disk:

```bash
export PRE_COMMIT_HOME=/tmp/pre-commit-JEMIAN   # in ~/.bashrc and ~/.profile
```

This is already set in `~/.bashrc`, `~/.profile`, and hardcoded in
`.git/hooks/pre-commit` (between the `# end templated` line and the `HERE=`
line).  If `pre-commit install` regenerates the hook, re-add that line:

```bash
# Use local disk for pre-commit cache (avoids NFS rmdir failures)
export PRE_COMMIT_HOME=/tmp/pre-commit-JEMIAN
```

The `Makefile` `pre` target also exports this variable automatically.

## CI integration

- Format and lint in pre-commit job
- Run tests and dependency audit on PRs.

## Minimal example PR checklist

- Runs formatting and linting locally
- Adds/updates tests for changes
- Updates `RELEASE_NOTES.rst` under the current development heading
- Marks PR as draft if large refactor

## Release Notes

- Update `RELEASE_NOTES.rst` as part of every PR that introduces a new
  feature, fix, enhancement, or maintenance change.
- Add the entry under the current development version heading (the topmost
  unreleased section inside the ``.. comment`` block).
- Entries must be **terse**: one line preferred, two lines maximum.
  State *what* changed, not *how* or *why*. Omit implementation details,
  lists of function names, and inline API cross-references unless the
  function name is the entire point of the entry.
- Always end with the issue or PR reference: ``:issue:`N``` or ``:pr:`N```.
- Good examples::

    * Add how-to guide: azimuthal (ψ) scans via ``psi_constant`` mode. (:issue:`188`)
    * Fix ``examples/hkl_soleil-e6c-psi.ipynb``: full axis constraints,
      phi discontinuity, pre-scan verification, matplotlib plot. (:issue:`337`)

- Bad examples (too wordy)::

    * Add how-to guide for azimuthal (ψ) scans: scan the azimuthal angle
      at fixed *(h, k, l)* using the ``psi_constant`` mode of the
      ``hkl_soleil`` solver via
      :meth:`~hklpy2.diffract.DiffractometerBase.scan_extra`; includes
      realistic motor constraints and pre-scan verification. (:issue:`188`)

- Use the appropriate subsection and keep subsections in the logical order
  defined at the top of ``RELEASE_NOTES.rst``: Notice, Breaking Changes, New
  Features, Enhancements, Fixes, Maintenance, Deprecations, New Contributors.
- Sort entries alphabetically within each subsection.  Sort by the
  rendered text — strip RST markup (````backticks````, ``:role:`target```,
  ``**bold**``, etc.) before comparing.  For example,
  ``Add ``scan_psi()```  sorts as ``"Add scan_psi()"`` (after ``"Add s"``),
  placing it after entries that begin with ``"Add h"``.

## Version Decorators

The project uses Sphinx-style version decorators from the ``deprecated``
package to record when public APIs were introduced or changed.  Keep these
in sync with code changes — they appear in the rendered docs and are part
of the public contract.

- Imports::

      from deprecated.sphinx import versionadded
      from deprecated.sphinx import versionchanged
      from deprecated.sphinx import deprecated

- **New public function/class/method** → add ``@versionadded`` with the
  upcoming release version and a one-line ``reason``::

      @versionadded(version="0.6.1", reason="Brief description of the API.")
      def new_function(...):
          ...

- **Behavioral or signature change to an existing public API** → add a
  ``@versionchanged`` decorator (stacked **below** the original
  ``@versionadded``) for the upcoming release.  Do not edit historical
  entries::

      @versionadded(version="0.4.0", reason="Original description.")
      @versionchanged(
          version="0.6.1",
          reason="Accept a ``DiffractometerBase`` instance directly.",
      )
      def existing_function(...):
          ...

- **Decorator order**: ``@versionadded`` first (outermost), then any
  ``@versionchanged`` entries in chronological order beneath it, then the
  function definition.  This matches the existing convention (see
  ``hklpy2.diffract.creator``).

- **Version string**: use the upcoming release tag (the topmost unreleased
  section in ``RELEASE_NOTES.rst``).  If you add a release-notes entry for
  an API change, also add the matching ``@versionchanged``/``@versionadded``
  decorator (and vice versa).

- **Inline forms**: for module/class-level documentation that is not a
  decorated callable, use the RST directive form inside the docstring::

      """
      .. versionadded:: 0.5.0
      .. versionchanged:: 0.6.1
         Description of the change.
      """

- **Deprecations**: use ``@deprecated`` (also from ``deprecated.sphinx``);
  add a corresponding entry in the ``Deprecations`` subsection of
  ``RELEASE_NOTES.rst``.

## Documentation: RST Style

- RST title underlines (and overlines) must be **at least as long** as the
  title text.  Count the characters and match exactly — a one-character
  shortfall is a silent Sphinx build warning that can break rendered output.

## Documentation: Architecture Diagram

The package architecture diagram lives in
`docs/source/concepts/__overview.rst` (the `.. graphviz::` block under
the "Package Architecture" section).  It must be kept in sync with the
actual package structure.  Update it when:

- A new top-level module or class is added (e.g. a new block in
  `hklpy2/blocks/`, a new solver, a new user-facing utility).
- A component moves between layers (user-facing, Core, solver adapter).
- A significant dependency relationship changes (e.g. a new block that
  `Core` manages, a new external library the solver delegates to).
- The four-stage `forward()` pipeline changes (Stages 1–4).

The diagram uses `sphinx.ext.graphviz` (DOT language).  Verify the DOT
syntax locally with `dot -Tsvg <file>` before committing.  Labels must
use only `\n` for line breaks; em-dashes and multi-line string
concatenation are not supported inside the Sphinx graphviz directive.

## Documentation: Glossary

- Keep glossary entries in `docs/source/glossary.rst` sorted alphabetically
  (case-insensitive).
- The glossary uses the Sphinx `.. glossary::` directive (with `:sorted:`).
  Each entry is a term on its own line, with the definition indented below:

  ```rst
  .. glossary::
      :sorted:

      my term
          Definition text here.
  ```

- Use `:term:\`my term\`` to cross-reference glossary entries from other docs pages.
- When adding or moving entries, maintain alphabetical order within the file
  (even though `:sorted:` renders them alphabetically regardless).

## Documentation: Sphinx Index entries

- Each term should have **at most one primary index entry** (`!term`) across the
  entire docs.
- The primary entry belongs on the page with the **most substantive content**
  for that term — typically its dedicated concept or how-to page, not the
  Glossary.
- The Sphinx `.. glossary::` directive automatically generates index entries for
  each term; explicit `.. index::` directives within the glossary are only needed
  for additional cross-references (e.g., `see: preset; presets`).
- Pages that reference a term but are not its primary source (e.g., a summary
  table) should use a plain (non-primary) index entry.

## The Full Workflow

When the user says "the full workflow", execute these steps in order:

1. `pre-commit run --all-files` — format and lint; fix any issues
2. `pytest ./hklpy2` — full test suite; fix any failures
3. Commit all staged changes with a conventional commit message
4. `git push -u origin <branch>` — push to remote
5. Open a PR via `gh pr create`; complete the mandatory PR checklist
   (milestone, project board, status → "In review", assignee, labels)
6. Set issue project status → "In review"
7. Monitor CI via `gh pr checks`; fix any failures with new commits
8. Resolve any code-quality review comments with new commits
9. Merge the PR when all checks are green and reviews approved
10. Local cleanup: `git checkout main && git pull && git branch -d <branch>`

## Notes

- Keep agent actions small, reversible, and reviewable.
- User-facing documentation and examples should be agnostic towards the
  radiation source.  Crystallography is defined in terms of **wavelength**,
  not energy.  Prefer "wavelength" over "energy" in all user-facing prose
  unless the context is explicitly about energy (e.g. monochromator setup).
  Source-agnosticism was a core motivation for hklpy2 over its predecessor.
- When updating a file, verify that a change has actually been made by comparing
  the mtime before and after the edits.

## Code Coverage

- Aim for 100% coverage, but prioritize meaningful tests over simply hitting every line.

## Release Tags and Versioned Docs

When a new release tag is pushed:

1. The docs CI workflow (`docs.yml`) automatically:
   - Builds and publishes docs to `gh-pages/<tag>/`.
   - Updates `gh-pages/latest/_static/switcher.json`: adds the new version
     entry and moves `"preferred": true` to the new tag (removing it from any
     previous entry).
2. Update `docs/source/_static/switcher.json` in the repo to match, so the
   source stays in sync with what is live on `gh-pages`.
3. The `"preferred"` entry in `switcher.json` should always point to the
   latest **tagged** release — not to `"latest"` (the main branch).

## Git Issues, Branches, Commits, and Pull Requests

All non-trivial work follows this lifecycle: **Issue -> Branch -> Commits ->
Pull Request**. Each step is described below with the concrete rules agents
must follow.

### Issues

Every piece of work starts with an issue. Issues answer the most expensive
question in code maintenance: *Why is this change being made?*

- An issue describes the observation, bug, feature request, or maintenance
  task that motivates the work.
- Do not begin coding without a corresponding issue (the only exception is a
  truly trivial fix that needs no explanation).

### Branches

All development happens on feature branches, never directly on `main`.

- **MANDATORY:** Before making *any* change to the working tree (edits,
  new files, or test scaffolding), create the feature branch first.  Do
  not edit on `main`.  Verify with ``git branch --show-current`` before
  the first edit.
- If you discover you have started editing on `main`, immediately stash
  the changes, create the feature branch from `main`, and pop the stash::

      git stash push -m "WIP issue #N"
      git checkout -b <ISSUE_NUMBER>-<concise-title> main
      git stash pop

- **Naming convention**: `<ISSUE_NUMBER>-<CONCISE-TITLE>`
  - The concise title is derived from the issue title, using lowercase words
    separated by hyphens.
  - Example: for issue #42 titled "Add timeout to LDAP queries", the branch
    name is `42-add-ldap-timeout`.
- Create the branch from the current `main`:
  `git checkout -b <branch-name> main`
- Push with tracking: `git push -u origin <branch-name>`

### Commits

Write commit messages following the
[Conventional Commits](https://www.conventionalcommits.org/) style with the
issue number included.

**Format:**

```text
<PREFIX> #<ISSUE_NUMBER> concise subject line

Optional body with additional context.
Agent: <agent name> (<model name>)
```

**Prefix values** (use the one that best describes the change):

| Prefix | Use for |
|--------|---------|
| `feat` | New feature or capability |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code restructuring, no behavior change |
| `style` | Formatting, linting, whitespace |
| `maint` | Maintenance, dependency updates, housekeeping |
| `ci` | CI/CD configuration |
| `test` | Adding or updating tests |

**Examples:**

```text
feat #42 add configurable timeout to LDAP queries

Default timeout is 30 s; configurable via dm_config.ini.
Agent: OpenCode (claudeopus46)
```

```text
docs #15 update AGENTS.md with branching workflow
```

### Project Status

When starting work on an issue, set its project card status to **"In progress"**
and move it to **"In review"** when the PR is opened.  Use the GitHub GraphQL
API via `gh api graphql`.

**Step 1 – look up the project item ID and Status field option IDs:**

```bash
gh api graphql -f query='
{ repository(owner:"bluesky", name:"hklpy2") {
    issue(number:NNN) {
      projectItems(first:5) { nodes {
        id
        project { id title
          fields(first:20) { nodes {
            ... on ProjectV2SingleSelectField { id name options { id name } }
          }}
        }
      }}
    }
  }
}'
```

Note the `itemId`, `projectId`, `fieldId` (for the **Status** field), and the
`singleSelectOptionId` for the desired status (e.g. `"In progress"`).

**Step 2 – set the status:**

```bash
gh api graphql -f query='
mutation {
  updateProjectV2ItemFieldValue(input: {
    projectId: "<projectId>"
    itemId:    "<itemId>"
    fieldId:   "<fieldId>"
    value: { singleSelectOptionId: "<optionId>" }
  }) { projectV2Item { id } }
}'
```

For the **hklpy2 v1.0** project the IDs are stable:

| Status | `singleSelectOptionId` |
|--------|------------------------|
| Backlog | `f75ad846` |
| Ready | `61e4505c` |
| In progress | `47fc9ee4` |
| In review | `df73e18b` |
| Done | `98236657` |

- `projectId`: `PVT_kwDOAtd7Hc4BFQ_A`
- `fieldId` (Status): `PVTSSF_lADOAtd7Hc4BFQ_Azg2p0U4`

### Pull Requests

A Pull Request (PR) describes *how* an issue has been (or will be) addressed.

- Every PR should reference at least one issue.
- Use a bullet list at the top of the PR body to link related issues:

  ```md
  - closes #42
  - #15
  ```

  Using `closes #N` will auto-close the issue when the PR is merged.
- The PR title should be a concise summary of the change.
- Assign the PR to the user running the agent (determine with ``gh api user --jq '.login'``).
- Copy the issue's labels, project(s), milestone, and status to the PR.

  **Mandatory checklist — complete these immediately after `gh pr create`:**

  1. `gh pr edit <N> --milestone "<milestone title>"` — copy milestone from issue.
  2. Add PR to the project board via GraphQL `addProjectV2ItemById`.
  3. Set PR project status to **"In review"** via GraphQL `updateProjectV2ItemFieldValue`.
  4. Set issue project status to **"In review"** via the same mutation.

  These four steps are required for every PR and must not be skipped.

- Sign the PR body with the agent and model name:

  ```md
  Agent: OpenCode (claudesonnet46)
  ```

- PR discussion comments should explain the approach, trade-offs, and any
  open questions.
- Sign all PR and issue comments with the agent and model name:

  ```md
  Agent: OpenCode (claudesonnet46)
  ```
