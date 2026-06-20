# Releasing moderndive

Releases are automated. Pushing a `v*` tag runs
[`.github/workflows/release.yml`](.github/workflows/release.yml), which runs the
full test suite, builds the sdist + wheel, checks them with `twine`, and publishes
to PyPI via **Trusted Publishing (OIDC)** — no API tokens.

## One-time setup (PyPI side)

Trusted publishing needs a publisher registered on (Test)PyPI and a matching
GitHub environment. Do this once per index.

1. **Create the GitHub environments**: repo → *Settings → Environments* → add
   `pypi` and `testpypi`. (Optionally require a reviewer so a publish needs
   approval.)
2. **Register the PyPI trusted publisher**: <https://pypi.org/manage/account/publishing/>
   → *Add a pending publisher*:
   - PyPI Project Name: `moderndive`
   - Owner: `moderndive`
   - Repository name: `moderndive-python`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. **Register the TestPyPI publisher** the same way at
   <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`.

## Dry run on TestPyPI (recommended before the first real release)

1. GitHub → *Actions → Release → Run workflow* (this is `workflow_dispatch`). It
   builds and publishes to **TestPyPI** (the `publish-testpypi` job; the real PyPI
   job only runs on a tag).
2. Confirm it installs and imports from TestPyPI:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ moderndive
   python -c "import moderndive as md; print(md.__version__); md.load_bowl()"
   ```
   (The `--extra-index-url` lets the real dependencies resolve from PyPI.)

You can also build and check locally without uploading:
```bash
make build              # uv build
uvx twine check dist/*
unzip -l dist/*.whl | grep -c parquet   # sanity: bundled datasets are present
```

## Cutting a release

1. **Pick the version** (PyPI versions are immutable — you can't re-upload one).
2. **Bump `version`** in `pyproject.toml`.
3. **Update `CHANGELOG.md`**: rename the `## Unreleased` section to
   `## <version> (YYYY-MM-DD)` and start a fresh empty `## Unreleased` above it.
4. **Regenerate the README** if `README.qmd` changed: `make readme`.
5. Commit, then **tag and push**:
   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```
   The tag triggers the release workflow → tests → build → publish to PyPI.
6. Create a GitHub Release for the tag (paste the CHANGELOG section) if you want
   release notes on GitHub.

## Notes

- `CHANGELOG.md` is the single source of truth for release notes (there is no
  separate `NEWS.md`).
- The wheel bundles `moderndive/data/*.parquet`; `README.md` (generated from
  `README.qmd`) is the PyPI long description and its plot uses an absolute URL so
  it renders on PyPI.
- Docs (Read the Docs) build separately from source on every push to `main`; they
  are not part of the PyPI release.
