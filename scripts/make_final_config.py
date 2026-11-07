"""Create configs/final.yaml from the tuning winner (plan Section 6.4).

Owner: Muhammad Abdullah Saeed.

`python -m src.tune` writes results/tuning/best_config.{json,yaml}. This script turns that
winner into the committed final training configuration. It refuses to do so when the
tuning ran with the provisional stand-in network, because those results were not obtained
with Hamid's model.

Usage (repository root):
    python scripts/make_final_config.py
    python scripts/make_final_config.py --tuning-dir results/tuning_quick \
        --out results/tuning_quick/final_quick.yaml --allow-standin      # dry run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.config import load_config  # noqa: E402
from src.utils import load_json  # noqa: E402


def build_final_config(tuning_dir: str, out: str, allow_standin: bool = False) -> dict:
    """Write `out` (YAML) from <tuning_dir>/best_config.{json,yaml}; return the metadata."""
    best_json = os.path.join(tuning_dir, "best_config.json")
    best_yaml = os.path.join(tuning_dir, "best_config.yaml")
    for path in (best_json, best_yaml):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run `python -m src.tune` first.")

    meta = load_json(best_json)
    source = meta.get("model_source", "unknown")
    if source != "hamid" and not allow_standin:
        raise ValueError(
            f"The tuning in '{tuning_dir}' used model_source='{source}', not Hamid's model. "
            "Re-run the tuning with src/model.py, or pass --allow-standin for a dry run."
        )

    cfg = load_config(best_yaml)
    header = [
        "# Final training configuration, selected by the staged tuning (src/tune.py).",
        f"# Source: {best_yaml.replace(os.sep, '/')} (stage C run {meta.get('stage_c_run_id')}),",
        f"# model_source: {source}",
        "# Selected hyperparameters: " + json.dumps(meta.get("hyperparameters")),
        "# Used by:",
        "#   python -m src.train --config configs/final.yaml --seed 42 "
        "--output models/neural_heuristic_seed42.pt",
        "",
    ]
    text = "\n".join(header) + yaml.safe_dump(cfg.to_dict(), sort_keys=False)
    parent = os.path.dirname(out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    if load_config(out) != cfg:  # defensive: the written file must load back identically
        raise RuntimeError(f"{out} does not load back to the tuned configuration")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the final training config")
    parser.add_argument("--tuning-dir", default="results/tuning")
    parser.add_argument("--out", default="configs/final.yaml")
    parser.add_argument("--allow-standin", action="store_true",
                        help="Accept a tuning run made with the stand-in network (dry runs only)")
    args = parser.parse_args()
    try:
        meta = build_final_config(args.tuning_dir, args.out, args.allow_standin)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {args.out}")
    print("Selected hyperparameters:", json.dumps(meta["hyperparameters"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
