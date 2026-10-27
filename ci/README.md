# CI Dependency Baseline

The supported unit-test jobs use one `conda-lock` file per Python version:

- `conda-lock-3.11.yml`
- `conda-lock-3.12.yml`
- `conda-lock-3.13.yml`
- `conda-lock-3.14.yml`

The corresponding `environment-*.yml` files are the concise input
specifications. The lockfiles contain exact package URLs and checksums. Do not
edit lockfiles by hand.

The updater reads supported numeric Python versions from the test matrix in
`.github/workflows/code.yml`. It creates any missing environment input from
`environment-3.11.yml`, then generates the matching lockfile. Existing input
files are never overwritten, so version-specific dependency exceptions remain
maintainer-controlled.

The lockfiles are refreshed weekly by the `Update CI dependency baseline`
workflow. The workflow discovers every committed `environment-*.yml` input and
updates its matching lockfile. Adding a new supported numeric Python version to
the matrix is sufficient; the updater creates the missing environment input and
lockfile from the baseline template. The workflow opens a pull request
containing any resolved package changes. The supported unit-test matrix must
pass before the update is merged.

To refresh locally, install `conda-lock` into the active Python environment,
then run the updater and lock commands:

```bash
python -m pip install conda-lock
conda-lock --version
python ci/update_environment_files.py
for input in ci/environment-*.yml; do
    version=${input#ci/environment-}
    version=${version%.yml}
    conda-lock lock \
        --file "$input" \
        --lockfile "ci/conda-lock-${version}.yml" \
        --conda mamba \
        --md input_sha
done
```

To use a refreshed environment locally, install the lockfile and then install
this checkout into that environment:

```bash
conda-lock install --name hklpy2-ci ci/conda-lock-3.13.yml
conda activate hklpy2-ci
pip install --no-deps -e .
```

`conda-lock install` installs both the conda and pip packages recorded in the
unified lockfile. The final `pip install --no-deps -e .` step installs this
checkout without changing the locked dependencies. CI performs both steps
automatically.
