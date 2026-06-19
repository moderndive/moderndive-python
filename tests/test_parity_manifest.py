"""Guardrail: bundled moderndive/infer datasets must match the recorded upstream.

`tools/parity_manifest.json` is the last-reviewed snapshot of the upstream R
packages (see tools/README.md). If a data regeneration changes a dataset's shape
away from that snapshot, this test fails — catching drift in normal CI without
needing R.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import moderndive as md

_MANIFEST = Path(__file__).resolve().parents[1] / "tools" / "parity_manifest.json"
_DATASETS = json.loads(_MANIFEST.read_text())


def _cases():
    for pkg, info in _DATASETS.items():
        for name, meta in info["datasets"].items():
            yield pytest.param(name, meta["rows"], meta["cols"], id=f"{pkg}:{name}")


@pytest.mark.parametrize("name,rows,cols", _cases())
def test_dataset_matches_upstream_dims(name, rows, cols):
    df = md.load_dataset(name)
    assert (df.height, df.width) == (rows, cols)


def test_manifest_records_upstream_versions():
    # Provenance: the manifest pins which upstream versions the data was built from.
    assert {"moderndive", "infer"} <= set(_DATASETS)
    assert all("version" in info for info in _DATASETS.values())
