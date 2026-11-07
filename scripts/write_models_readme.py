"""Write models/README.md (Hugging Face link + SHA256 of every checkpoint), or check it.

Owner: Muhammad Abdullah Saeed (plan Section 6.4, task 6.4.2; Day 5 checksum check).

Write the README after the final training (the Hugging Face user name is yours):
    python scripts/write_models_readme.py --hf-user <your-hf-username>

Check that the files still match the README (also works on files downloaded from
Hugging Face into another folder):
    python scripts/write_models_readme.py --check
    python scripts/write_models_readme.py --check --models-dir hf_check --readme models/README.md

Dry run (uses the files of `python scripts/train_final.py --quick`):
    python scripts/write_models_readme.py --quick
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import load_json  # noqa: E402

QUICK_ROOT = os.path.join("results", "tuning_quick", "final_quick")  # already git-ignored
CKPT_NAME = re.compile(r"neural_heuristic_seed(\d+)\.pt$")
ROW = re.compile(r"\|\s*(neural_heuristic_seed\d+\.pt)\s*\|.*?`([0-9a-f]{64})`")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def collect(models_dir: str, training_dir: str) -> list[dict]:
    """One row per neural_heuristic_seed<S>.pt in `models_dir`, ordered by seed."""
    rows = []
    for path in glob.glob(os.path.join(models_dir, "neural_heuristic_seed*.pt")):
        match = CKPT_NAME.search(os.path.basename(path))
        if not match:
            continue
        name = os.path.basename(path)
        metrics_path = os.path.join(training_dir, name[:-3], "metrics.json")
        metrics = load_json(metrics_path) if os.path.exists(metrics_path) else {}
        rows.append({
            "name": name, "seed": int(match.group(1)), "size": os.path.getsize(path),
            "sha256": sha256_file(path),
            "val_mae": metrics.get("best_val_mae"), "best_epoch": metrics.get("best_epoch"),
        })
    return sorted(rows, key=lambda r: r["seed"])


def render_readme(rows: list[dict], hf_user: str, repo: str) -> str:
    names = ", ".join(r["name"] for r in rows)
    lines = [
        "# Model weights",
        "",
        f"- Hugging Face: https://huggingface.co/{hf_user}/{repo}",
        f"- Files: {names}",
        "- Checkpoint SHA256: see the table below "
        "(recompute with `Get-FileHash <file> -Algorithm SHA256`)",
        '- Load: `torch.load(path, map_location="cpu")`',
        "- Contents: `model_state`, `model_config`, `target_mean`, `target_std`, `bin_edges` "
        "(plus `config`, `git_commit`, `seed`)",
        "- Training settings of every run: `results/training/final_summary.csv`",
        "",
        "| File | Seed | Size (bytes) | SHA256 | Best val MAE | Best epoch |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        mae = "-" if r["val_mae"] is None else f"{r['val_mae']:.4f}"
        epoch = "-" if r["best_epoch"] is None else str(r["best_epoch"])
        lines.append(f"| {r['name']} | {r['seed']} | {r['size']} | `{r['sha256']}` "
                     f"| {mae} | {epoch} |")
    return "\n".join(lines) + "\n"


def check_readme(models_dir: str, readme_path: str) -> list[str]:
    """Compare the SHA256 values written in the README with the files; return problems."""
    if not os.path.exists(readme_path):
        return [f"{readme_path} not found"]
    with open(readme_path, "r", encoding="utf-8") as f:
        listed = dict(ROW.findall(f.read()))
    if not listed:
        return [f"no checkpoint rows found in {readme_path}"]
    problems = []
    for name, expected in sorted(listed.items()):
        path = os.path.join(models_dir, name)
        if not os.path.exists(path):
            problems.append(f"{name}: file not found in {models_dir}")
        elif sha256_file(path) != expected:
            problems.append(f"{name}: SHA256 differs from the README")
    for path in glob.glob(os.path.join(models_dir, "neural_heuristic_seed*.pt")):
        if os.path.basename(path) not in listed:
            problems.append(f"{os.path.basename(path)}: not listed in the README")
    return problems


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write or check models/README.md")
    p.add_argument("--hf-user", default=None, help="Your Hugging Face user name (needed to write)")
    p.add_argument("--repo", default="neural-heuristic-maze")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--training-dir", default="results/training")
    p.add_argument("--readme", default=None, help="README path (default: <models-dir>/README.md)")
    p.add_argument("--check", action="store_true", help="Check hashes instead of writing")
    p.add_argument("--quick", action="store_true", help=f"Dry run inside {QUICK_ROOT}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    models_dir, training_dir = args.models_dir, args.training_dir
    if args.quick:
        models_dir = os.path.join(QUICK_ROOT, "models")
        training_dir = os.path.join(QUICK_ROOT, "training")
    readme: str = args.readme or os.path.join(models_dir, "README.md")

    if args.check:
        problems = check_readme(models_dir, readme)
        for problem in problems:
            print(f"  [FAIL] {problem}")
        if problems:
            return 1
        print(f"All checkpoints in {models_dir} match the SHA256 values in {readme}")
        return 0

    hf_user: Optional[str] = args.hf_user or ("your-hf-username" if args.quick else None)
    if not hf_user:
        print("ERROR: pass --hf-user <your-hf-username>", file=sys.stderr)
        return 1
    rows = collect(models_dir, training_dir)
    if not rows:
        print(f"ERROR: no neural_heuristic_seed*.pt files in {models_dir}", file=sys.stderr)
        return 1
    with open(readme, "w", encoding="utf-8", newline="\n") as f:
        f.write(render_readme(rows, hf_user, args.repo))
    print(f"Wrote {readme} ({len(rows)} checkpoints)")
    for r in rows:
        print(f"  {r['name']}  {r['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
