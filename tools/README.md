# Maintenance tooling

Scripts that keep the Python package in sync with the upstream R packages
(`moderndive`, `infer`) and the source data packages.

## Regenerating the bundled datasets

The `load_*()` datasets are bundled as zstd Parquet under `moderndive/data/`,
generated from the R packages. To refresh them after an upstream data change:

```bash
# 1. Export the datasets from R to CSV (needs R + the upstream packages installed)
Rscript ../ModernDive_book/scripts/export_pilot_data.R   # core book datasets
Rscript tools/export_extra_data.R                        # full-parity datasets + gss

# 2. Convert the CSVs to typed Parquet under moderndive/data/
uv run python tools/build_data.py /tmp/pilot_csv
uv run python tools/build_extra_data.py

# 3. Confirm dimensions still match upstream (see parity check below)
```

`early_january_2023_weather` is **derived** from `weather` inside `build_data.py`
(the R dataset ships those measurement columns as all-`NA`).

## Parity drift checking

Upstream can add/rename datasets or functions. These tools surface that:

- `parity_probe.R` — dumps the current upstream structure (versions, exported
  objects, dataset dims + columns for `moderndive`/`infer`) as JSON.
- `check_parity.py` — compares that JSON against (a) the Python package's bundled
  datasets and (b) `parity_manifest.json`, reporting any drift; exits non-zero
  when drift is found.
- `parity_manifest.json` — the last-reviewed upstream snapshot (the provenance the
  bundled data + API were built against).

Run it locally:

```bash
Rscript tools/parity_probe.R > /tmp/upstream.json
uv run python tools/check_parity.py /tmp/upstream.json            # report drift
uv run python tools/check_parity.py /tmp/upstream.json --update-manifest  # accept current upstream
```

The **Parity drift** GitHub workflow (`.github/workflows/parity-drift.yml`) runs
this weekly against the latest upstream `main` and opens/updates a GitHub issue
when drift appears. After porting an upstream change, refresh the data (above) and
run `--update-manifest` to record the new baseline.

The test suite also asserts (via `tests/test_parity_manifest.py`) that every
`moderndive`/`infer` dataset in the manifest is present in the package with
matching dimensions, so a botched data regeneration fails CI immediately.
