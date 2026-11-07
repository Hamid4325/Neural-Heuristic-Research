"""Build the deterministic 70/15/15 split manifest.

Owner: Muhammad Abdullah Saeed.

Writes the manifest to data/processed/split_manifest.json (git-ignored, used by the
loaders) and a copy to results/manifests/split_manifest.json (committed, because
data/ is ignored). Then checks that the manifest agrees with data/processed/*.npz.

Usage (from the repository root):
    python scripts/make_split_manifest.py
    python scripts/make_split_manifest.py --cleaned-dir data/cleaned --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow `python scripts/make_split_manifest.py` to import the `src` package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.data import SPLIT_NAMES, build_split_manifest, manifest_cell_counts  # noqa: E402
from src.utils import configure_logging, save_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the deterministic split manifest")
    parser.add_argument("--cleaned-dir", default="data/cleaned")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/processed/split_manifest.json")
    parser.add_argument("--copy-to", default="results/manifests/split_manifest.json",
                        help="Committed copy of the manifest ('' to skip)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip the cross-check against data/processed/*.npz")
    args = parser.parse_args()

    log = configure_logging()
    manifest = build_split_manifest(args.cleaned_dir, args.seed)

    save_json(args.out, manifest)
    log.info("Wrote %s", args.out)
    if args.copy_to:
        save_json(args.copy_to, manifest)
        log.info("Wrote %s (commit this copy)", args.copy_to)

    log.info("Total mazes: %d (seed %d)", manifest["total_mazes"], manifest["seed"])
    for split in SPLIT_NAMES:
        log.info("  %-5s %5d mazes", split, manifest["splits"][split]["maze_count"])

    if args.skip_verify:
        return 0

    # Cross-check: cells implied by the manifest must equal the rows in train/val/test.npz.
    expected = manifest_cell_counts(manifest, args.cleaned_dir)
    ok = True
    for split in SPLIT_NAMES:
        path = os.path.join(args.processed_dir, f"{split}.npz")
        if not os.path.exists(path):
            log.warning("  %s missing; run data_scripts/features.py to verify", path)
            continue
        with np.load(path) as data:
            actual = len(data["y"])
        match = actual == expected[split]
        ok &= match
        log.info("  %-5s cells: manifest=%d processed=%d -> %s",
                 split, expected[split], actual, "OK" if match else "MISMATCH")
    if not ok:
        log.error("Manifest does not match data/processed. Re-run clean_data.py and features.py.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
