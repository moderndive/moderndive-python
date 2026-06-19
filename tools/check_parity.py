"""Check the Python port for drift against the upstream R moderndive/infer packages.

Workflow::

    Rscript tools/parity_probe.R > upstream.json
    python tools/check_parity.py upstream.json                 # report drift; exit 1 if any
    python tools/check_parity.py upstream.json --update-manifest  # accept current upstream

Drift is anything that means the Python port may have fallen behind upstream:

- a moderndive/infer **version bump** (vs the recorded manifest);
- **exported objects added/removed** upstream (vs the manifest);
- a bundled **dataset missing** from the Python package, or whose **dimensions or
  columns differ** from upstream.

The manifest (``tools/parity_manifest.json``) records the last-reviewed upstream
snapshot — i.e. the provenance the bundled data was built against. After porting
any upstream changes, re-run with ``--update-manifest`` to accept the new state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import moderndive as md

MANIFEST = Path(__file__).resolve().parent / "parity_manifest.json"


def _python_datasets() -> dict[str, dict]:
    """Dimensions + columns of every dataset bundled in the Python package."""
    out = {}
    for name in md.available_datasets():
        df = md.load_dataset(name)
        out[name] = {"rows": df.height, "cols": df.width, "columns": list(df.columns)}
    return out


def _diff(upstream: dict, manifest: dict, py: dict) -> list[str]:
    lines: list[str] = []
    py_names = set(py)
    for pkg, info in upstream.items():
        man = manifest.get(pkg, {})
        if man.get("version") and man["version"] != info["version"]:
            lines.append(f"- **{pkg}**: version changed `{man['version']}` → `{info['version']}`")
        old_exports = set(man.get("exports", [])) - {"%>%"}
        new_exports = set(info["exports"]) - {"%>%"}
        if old_exports:  # only report export drift once we have a baseline
            added = sorted(new_exports - old_exports)
            removed = sorted(old_exports - new_exports)
            if added:
                lines.append(f"- **{pkg}**: new exported object(s) upstream: {added}")
            if removed:
                lines.append(f"- **{pkg}**: exported object(s) removed upstream: {removed}")
        for ds, meta in info["datasets"].items():
            r_dims = [meta["rows"], meta["cols"]]
            if ds not in py_names:
                lines.append(f"- dataset `{ds}` ({pkg}) is **missing** from the Python package "
                             f"(R dims {r_dims})")
                continue
            if [py[ds]["rows"], py[ds]["cols"]] != r_dims:
                lines.append(f"- dataset `{ds}`: dims differ — Python "
                             f"{[py[ds]['rows'], py[ds]['cols']]} vs R {r_dims}")
            missing_cols = sorted(set(meta["columns"]) - set(py[ds]["columns"]))
            if missing_cols:
                lines.append(f"- dataset `{ds}`: columns missing in Python: {missing_cols}")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("upstream", help="JSON produced by tools/parity_probe.R")
    ap.add_argument("--update-manifest", action="store_true",
                    help="overwrite the manifest with the current upstream snapshot")
    args = ap.parse_args()

    upstream = json.loads(Path(args.upstream).read_text())

    if args.update_manifest:
        MANIFEST.write_text(json.dumps(upstream, indent=2, sort_keys=True) + "\n")
        versions = ", ".join(f"{k} {v['version']}" for k, v in upstream.items())
        print(f"Wrote {MANIFEST.name} from current upstream ({versions}).")
        return 0

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    lines = _diff(upstream, manifest, _python_datasets())
    if lines:
        print("# Upstream parity drift detected\n")
        print("\n".join(lines))
        print("\nPort the changes into the Python package, then run "
              "`python tools/check_parity.py <upstream.json> --update-manifest` to accept.")
        return 1
    print("No parity drift: the Python port matches upstream moderndive/infer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
