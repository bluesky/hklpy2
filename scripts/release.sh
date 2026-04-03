#!/usr/bin/env bash
# release.sh - Interactive release checklist for hklpy2
#
# Usage:
#   bash scripts/release.sh [--dry-run] [VERSION]
#
# Options:
#   --dry-run   Show what would be done without making any changes.
#               Checks and read-only commands still run; all writes,
#               commits, pushes, tags, and GitHub API calls are skipped.
#
# If VERSION is omitted the script prompts for it.
#
# The script walks through each release step in order, confirming
# completion before proceeding.  Steps that can be automated are run
# automatically; steps that require human judgment are shown as
# reminders.
#
# Prerequisites (must be installed and on PATH):
#   git, gh, python3, pre-commit
#
# Exit codes:
#   0  - all steps completed (or dry-run finished)
#   1  - user aborted or a required step failed

# Handle --help / -h before anything else (before set -e and git calls).
for _arg in "$@"; do
    case "${_arg}" in
        -h|--help)
            cat <<'EOF'
Usage: bash scripts/release.sh [OPTIONS] [VERSION]

Interactive release checklist for hklpy2.  Walks through every release
step in order, automating what it can and prompting for the rest.

Arguments:
  VERSION       Release version to prepare (e.g. 0.4.1).
                Prompted interactively if omitted.

Options:
  --dry-run     Preview all steps without making any changes.
                Checks and read-only commands still run; file writes,
                commits, pushes, tags, and GitHub API calls are skipped.
  -h, --help    Show this help message and exit.

Steps performed:
  0.  Preflight checks (branch, clean tree, sync with origin, tag existence)
  1.  Milestone and issue triage
  2.  RELEASE_NOTES.rst review
  3.  Update docs/source/_static/switcher.json
  4.  Run pre-commit / style checks
  5.  Run test suite
  6.  Commit release preparation changes
  7.  Push to origin
  8.  Wait for CI to pass
  9.  Create and push the release tag
  10. Create GitHub Release
  11. Monitor CI workflows (docs, PyPI)
  12. conda-forge feedstock update

Prerequisites: git, gh, python3, pre-commit
EOF
            exit 0
            ;;
    esac
done

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
SWITCHER_JSON="${REPO_ROOT}/docs/source/_static/switcher.json"
RELEASE_NOTES="${REPO_ROOT}/RELEASE_NOTES.rst"
BASE_URL="https://blueskyproject.io/hklpy2"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
step()    { echo -e "\n${BOLD}${CYAN}=== Step $1: $2 ===${RESET}"; }
abort()   { error "$*"; exit 1; }
dryrun()  { echo -e "${MAGENTA}[DRY-RUN]${RESET} would run: $*"; }

confirm() {
    # confirm "message"  -> returns 0 if user says yes, 1 if no
    # In dry-run mode always returns 1 (skip the action) after printing intent.
    local msg="$1"
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${MAGENTA}[DRY-RUN]${RESET} skipping: ${msg}"
        return 1
    fi
    local answer
    while true; do
        read -r -p "$(echo -e "${YELLOW}${msg} [y/n]: ${RESET}")" answer
        case "${answer,,}" in
            y|yes) return 0 ;;
            n|no)  return 1 ;;
            *) echo "Please answer y or n." ;;
        esac
    done
}

pause() {
    # pause "instruction" - wait for user to press Enter after completing a manual step.
    # In dry-run mode this is a no-op.
    if [[ "${DRY_RUN}" == "true" ]]; then
        echo -e "${MAGENTA}[DRY-RUN]${RESET} skipping pause: $1"
        return 0
    fi
    echo -e "${YELLOW}ACTION REQUIRED: $1${RESET}"
    read -r -p "Press Enter when done (or Ctrl-C to abort)..."
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || abort "Required command not found: $1"
}

# ---------------------------------------------------------------------------
# Argument / version / dry-run handling
# ---------------------------------------------------------------------------

DRY_RUN="false"
POSITIONAL_ARGS=()

for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN="true" ;;
        *) POSITIONAL_ARGS+=("${arg}") ;;
    esac
done

VERSION="${POSITIONAL_ARGS[0]:-}"
if [[ -z "${VERSION}" ]]; then
    if [[ "${DRY_RUN}" == "true" ]]; then
        read -r -p "$(echo -e "${CYAN}Enter the release version to preview (e.g. 0.4.1): ${RESET}")" VERSION
    else
        read -r -p "$(echo -e "${CYAN}Enter the release version (e.g. 0.4.1): ${RESET}")" VERSION
    fi
fi

[[ -n "${VERSION}" ]] || abort "Version must not be empty."

# Strip a leading 'v' if present
VERSION="${VERSION#v}"

if [[ "${DRY_RUN}" == "true" ]]; then
    echo -e "${MAGENTA}${BOLD}*** DRY-RUN MODE: no files will be written, no commits, pushes, or tags created ***${RESET}"
fi
info "Starting release checklist for version ${BOLD}${VERSION}${RESET}"
echo ""

# ---------------------------------------------------------------------------
# Step 0: Preflight checks
# ---------------------------------------------------------------------------
step 0 "Preflight checks"

require_cmd git
require_cmd gh
require_cmd python3
require_cmd pre-commit

# Must be on main (or a release branch)
CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_BRANCH}" != "main" ]]; then
    warn "Current branch is '${CURRENT_BRANCH}', not 'main'."
    if [[ "${DRY_RUN}" == "true" ]]; then
        warn "[DRY-RUN] ignoring branch check."
    else
        confirm "Continue anyway?" || abort "Aborted by user."
    fi
fi

# Must be clean
if ! git diff --quiet || ! git diff --cached --quiet; then
    warn "Working tree has uncommitted changes:"
    git status --short
    if [[ "${DRY_RUN}" == "true" ]]; then
        warn "[DRY-RUN] ignoring uncommitted changes."
    else
        confirm "Continue anyway?" || abort "Aborted by user. Commit or stash changes first."
    fi
fi

# Must be up-to-date with origin
info "Fetching from origin..."
git fetch origin
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "origin/${CURRENT_BRANCH}" 2>/dev/null || echo '')"
if [[ -n "${REMOTE_SHA}" && "${LOCAL_SHA}" != "${REMOTE_SHA}" ]]; then
    warn "Local branch is not in sync with origin/${CURRENT_BRANCH}."
    if [[ "${DRY_RUN}" == "true" ]]; then
        warn "[DRY-RUN] ignoring sync check."
    else
        confirm "Continue anyway?" || abort "Aborted. Pull or push changes first."
    fi
fi

# Tag must not already exist (informational in dry-run)
if git tag | grep -qx "${VERSION}"; then
    if [[ "${DRY_RUN}" == "true" ]]; then
        warn "[DRY-RUN] Tag '${VERSION}' already exists locally (would abort in real run)."
    else
        abort "Tag '${VERSION}' already exists locally. Nothing to do."
    fi
fi
if git ls-remote --tags origin | grep -q "refs/tags/${VERSION}$"; then
    if [[ "${DRY_RUN}" == "true" ]]; then
        warn "[DRY-RUN] Tag '${VERSION}' already exists on origin (would abort in real run)."
    else
        abort "Tag '${VERSION}' already exists on origin. Nothing to do."
    fi
fi

success "Preflight checks passed."

# ---------------------------------------------------------------------------
# Step 1: Confirm milestone and issues
# ---------------------------------------------------------------------------
step 1 "Milestone and issue triage"
echo "  Verify that all issues/PRs targeting the '${VERSION}' milestone are"
echo "  closed or deferred before tagging."
echo ""
gh milestone list --repo bluesky/hklpy2 2>/dev/null || true
echo ""
pause "Close or defer any open milestone issues, then press Enter."

# ---------------------------------------------------------------------------
# Step 2: Update RELEASE_NOTES.rst
# ---------------------------------------------------------------------------
step 2 "RELEASE_NOTES.rst"
echo "  The file is at: ${RELEASE_NOTES}"
echo "  Ensure the '${VERSION}' section:"
echo "    - Is no longer inside the '.. comment' block"
echo "    - Has the correct 'Released YYYY-MM-DD.' date"
echo "    - Contains all relevant entries"
echo ""
pause "Update RELEASE_NOTES.rst, then press Enter."

# ---------------------------------------------------------------------------
# Step 3: Update switcher.json in source
# ---------------------------------------------------------------------------
step 3 "Update docs/source/_static/switcher.json"

export DRY_RUN
python3 - <<PYEOF
import json, os, re

version = "${VERSION}"
switcher_path = "${SWITCHER_JSON}"
base_url = "${BASE_URL}"
dry_run = os.environ.get("DRY_RUN") == "true"

# A pre-release has a suffix like rc1, a1, b2, etc.
is_prerelease = bool(re.search(r'(a|b|rc)\d+$', version))

with open(switcher_path) as f:
    entries = json.load(f)

versions = [e["version"] for e in entries]

if version in versions:
    # Version already present: update preferred flag only for stable releases.
    if not is_prerelease:
        for e in entries:
            e.pop("preferred", None)
        for e in entries:
            if e["version"] == version:
                e["preferred"] = True
        print(f"  '{version}' already present; marked as preferred.")
    else:
        print(f"  '{version}' already present (pre-release; preferred unchanged).")
else:
    # Insert after the "latest" entry.
    insert_pos = next(
        (i + 1 for i, e in enumerate(entries) if e.get("version") == "latest"),
        1,
    )
    if is_prerelease:
        # Pre-release: add entry without touching preferred on existing entries.
        new_entry = {"version": version, "url": f"{base_url}/{version}/"}
        entries.insert(insert_pos, new_entry)
        print(f"  Added pre-release '{version}' to switcher.json (preferred unchanged).")
    else:
        # Stable release: clear preferred everywhere, mark this one preferred.
        for e in entries:
            e.pop("preferred", None)
        new_entry = {"version": version, "url": f"{base_url}/{version}/", "preferred": True}
        entries.insert(insert_pos, new_entry)
        print(f"  Added stable '{version}' to switcher.json and marked as preferred.")

if dry_run:
    print("  [DRY-RUN] would write switcher.json:")
    print(json.dumps(entries, indent=4))
else:
    with open(switcher_path, "w") as f:
        json.dump(entries, f, indent=4)
        f.write("\n")
    print("  switcher.json written successfully.")
PYEOF

if [[ "${DRY_RUN}" == "true" ]]; then
    dryrun "write ${SWITCHER_JSON}"
else
    success "switcher.json updated."
fi

info "Current switcher.json (on disk):"
python3 -m json.tool "${SWITCHER_JSON}"

# ---------------------------------------------------------------------------
# Step 4: Run pre-commit / style checks
# ---------------------------------------------------------------------------
step 4 "Run pre-commit / style checks"
export PRE_COMMIT_HOME="${PRE_COMMIT_HOME:-/tmp/pre-commit-JEMIAN}"
info "Running: pre-commit run --all-files"
pre-commit run --all-files || {
    warn "pre-commit reported issues. Fix them before continuing."
    confirm "Issues fixed?" || abort "Aborted."
}
success "pre-commit passed."

# ---------------------------------------------------------------------------
# Step 5: Run tests
# ---------------------------------------------------------------------------
step 5 "Run test suite"
if confirm "Run pytest now? (may take a few minutes)"; then
    python3 -m pytest -q ./src
    success "Tests passed."
else
    warn "Skipped. Ensure CI is green on the release commit before tagging."
fi

# ---------------------------------------------------------------------------
# Step 6: Commit release prep changes
# ---------------------------------------------------------------------------
step 6 "Commit release preparation changes"

git status --short
echo ""

if git diff --quiet && git diff --cached --quiet; then
    info "No uncommitted changes; nothing to commit."
else
    if [[ "${DRY_RUN}" == "true" ]]; then
        dryrun "git add -A"
        dryrun "git commit -m 'maint #263 release prep for ${VERSION}: update RELEASE_NOTES.rst and switcher.json'"
    elif confirm "Stage and commit all changes as release prep?"; then
        git add -A
        git commit -m "maint #263 release prep for ${VERSION}

- Move ${VERSION} section out of comment block in RELEASE_NOTES.rst
- Update switcher.json: add ${VERSION} as preferred version

Agent: OpenCode (claudesonnet46)"
        success "Release prep committed."
    else
        warn "Changes not committed. Commit them manually before tagging."
        pause "Commit changes, then press Enter."
    fi
fi

# ---------------------------------------------------------------------------
# Step 7: Push the release-prep commit
# ---------------------------------------------------------------------------
step 7 "Push release-prep commit to origin/${CURRENT_BRANCH}"
if [[ "${DRY_RUN}" == "true" ]]; then
    dryrun "git push origin ${CURRENT_BRANCH}"
elif confirm "Push now?"; then
    git push origin "${CURRENT_BRANCH}"
    success "Pushed to origin/${CURRENT_BRANCH}."
else
    warn "Skipped. Push before tagging."
    pause "Push changes, then press Enter."
fi

# ---------------------------------------------------------------------------
# Step 8: Wait for CI to pass
# ---------------------------------------------------------------------------
step 8 "Wait for CI to pass"
echo "  Open https://github.com/bluesky/hklpy2/actions and confirm all"
echo "  workflows are green on the latest commit before tagging."
echo ""
pause "Confirm CI is green, then press Enter."

# ---------------------------------------------------------------------------
# Step 9: Create and push the tag
# ---------------------------------------------------------------------------
step 9 "Create and push tag ${VERSION}"

echo ""
warn "Pushing the tag will trigger:"
echo "  - docs.yml  : build and publish versioned docs, update switcher.json on gh-pages"
echo "  - pypi.yml  : publish to PyPI"
echo "  - code.yml  : run full CI suite"
echo ""

if [[ "${DRY_RUN}" == "true" ]]; then
    dryrun "git tag -a ${VERSION} -m 'release ${VERSION}'"
    dryrun "git push origin ${VERSION}"
elif confirm "Create annotated tag '${VERSION}' on HEAD?"; then
    git tag -a "${VERSION}" -m "release ${VERSION}"
    success "Tag '${VERSION}' created locally."
    if confirm "Push tag '${VERSION}' to origin now?"; then
        git push origin "${VERSION}"
        success "Tag '${VERSION}' pushed."
    else
        warn "Tag not pushed. Run 'git push origin ${VERSION}' when ready."
        exit 0
    fi
else
    abort "Aborted. Create the tag manually when ready."
fi

# ---------------------------------------------------------------------------
# Step 10: Create GitHub Release
# ---------------------------------------------------------------------------
step 10 "Create GitHub Release"
echo "  This marks the release as non-pre-release on GitHub."
echo ""
if [[ "${DRY_RUN}" == "true" ]]; then
    dryrun "gh release create ${VERSION} --title ${VERSION} --latest"
elif confirm "Create GitHub Release for ${VERSION} now?"; then
    gh release create "${VERSION}" \
        --title "${VERSION}" \
        --notes "See [RELEASE_NOTES.rst](https://github.com/bluesky/hklpy2/blob/main/RELEASE_NOTES.rst) for details." \
        --latest
    success "GitHub Release created."
else
    warn "Skipped. Create the release manually at:"
    echo "  https://github.com/bluesky/hklpy2/releases/new?tag=${VERSION}"
fi

# ---------------------------------------------------------------------------
# Step 11: Monitor CI workflows
# ---------------------------------------------------------------------------
step 11 "Monitor CI workflows"
echo "  Watch the following workflow runs triggered by the tag push:"
echo "    https://github.com/bluesky/hklpy2/actions"
echo ""
echo "  Expected outcomes:"
echo "    docs.yml  - builds docs, publishes to gh-pages/${VERSION}/, updates switcher.json"
echo "    pypi.yml  - publishes hklpy2==${VERSION} to PyPI"
echo "    code.yml  - all tests pass"
echo ""
echo "  If docs.yml fails with a 'fetch first' push error on the switcher.json step,"
echo "  re-run the workflow manually via the Actions UI (workflow_dispatch)."
echo ""
pause "Confirm all CI workflows succeeded, then press Enter."

# ---------------------------------------------------------------------------
# Step 12: conda-forge feedstock
# ---------------------------------------------------------------------------
step 12 "conda-forge feedstock update"
echo "  The conda-forge bot will open a PR at:"
echo "    https://github.com/conda-forge/hklpy2-feedstock/pulls"
echo "  once PyPI publishes the new sdist."
echo ""
echo "  Review and merge the feedstock PR using your fork:"
echo "    https://github.com/prjemian/hklpy2-feedstock"
echo ""
pause "Monitor and merge the feedstock PR, then press Enter to finish."

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
if [[ "${DRY_RUN}" == "true" ]]; then
    echo -e "${MAGENTA}${BOLD}Dry-run for ${VERSION} complete. No changes were made.${RESET}"
    echo ""
    echo "  Re-run without --dry-run to perform the actual release."
else
    echo -e "${GREEN}${BOLD}Release ${VERSION} complete!${RESET}"
    echo ""
    echo "  PyPI:      https://pypi.org/project/hklpy2/${VERSION}/"
    echo "  Docs:      ${BASE_URL}/${VERSION}/"
    echo "  Release:   https://github.com/bluesky/hklpy2/releases/tag/${VERSION}"
    echo "  Feedstock: https://github.com/conda-forge/hklpy2-feedstock"
fi
echo ""
