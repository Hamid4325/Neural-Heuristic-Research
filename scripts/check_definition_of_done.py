"""Check Saeed's "definition of done" (plan Section 6.5) with one command.

Owner: Muhammad Abdullah Saeed.

    [ ] Split manifest committed and tested.
    [ ] Training loop works on CPU and saves a valid checkpoint.
    [ ] Tuning summary committed.
    [ ] Final model trained and uploaded.
    [ ] models/README.md contains link and checksum.
    [ ] Phase 3 training report section drafted.
    [ ] All commits pushed to d2/saeed-training.

Every item prints PASS, WARN or FAIL with the reason. Exit code 1 if anything FAILs.

Usage (repository root):
    python scripts/check_definition_of_done.py
    python scripts/check_definition_of_done.py --run-tests      # also runs pytest -q
    python scripts/check_definition_of_done.py --online         # also checks Hugging Face
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import check_requirements  # noqa: E402
import write_models_readme  # noqa: E402

SEEDS = (42, 43, 44)
PLACEHOLDER_USERS = {"your-hf-username", "<your-hf-username>", "username", "<username>"}
HF_LINK = re.compile(r"https://huggingface\.co/([^/\s)]+)/([^/\s)`]+)")
GENERATED = ("training_macros", "tuning_space_table", "tuning_runs_table", "final_training_table")


@dataclass
class Result:
    name: str
    status: str   # PASS | WARN | FAIL
    detail: str


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------
def git(root: Path, *args: str) -> tuple[int, str]:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def is_tracked(root: Path, relative: str) -> bool:
    return git(root, "ls-files", "--error-unmatch", "--", relative)[0] == 0


def untracked_or_missing(root: Path, paths: list[str]) -> list[str]:
    """Paths that are missing from disk or not committed to Git."""
    return [p for p in paths if not (root / p).exists() or not is_tracked(root, p)]


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def check_manifest(root: Path) -> Result:
    name = "Split manifest committed and tested"
    path = "results/manifests/split_manifest.json"
    bad = untracked_or_missing(root, [path, "tests/test_data.py"])
    if bad:
        return Result(name, "FAIL", f"missing or not committed: {', '.join(bad)}")
    manifest = json.loads((root / path).read_text(encoding="utf-8"))
    counts = [manifest["splits"][s]["maze_count"] for s in ("train", "val", "test")]
    if counts != [1313, 281, 282] or manifest.get("total_mazes") != 1876:
        return Result(name, "FAIL", f"unexpected maze counts {counts} (plan: 1313/281/282)")
    return Result(name, "PASS", "committed; 1313 / 281 / 282 mazes; tests/test_data.py committed")


def check_training(root: Path) -> Result:
    name = "Training loop works on CPU and saves a valid checkpoint"
    bad = untracked_or_missing(root, ["src/train.py", "tests/test_train_smoke.py",
                                      "configs/default.yaml"])
    if bad:
        return Result(name, "FAIL", f"missing or not committed: {', '.join(bad)}")
    checkpoint = root / "models" / "neural_heuristic_seed42.pt"
    if not checkpoint.exists():
        return Result(name, "FAIL", "models/neural_heuristic_seed42.pt not found "
                                    "(run scripts/train_final.py)")
    from train_final import verify_checkpoint
    verify_checkpoint(str(checkpoint), 42, 10)
    return Result(name, "PASS", "code committed; seed 42 checkpoint loads and has all contract keys")


def check_tuning(root: Path) -> Result:
    name = "Tuning summary committed"
    bad = untracked_or_missing(root, ["results/tuning/tuning_summary.csv",
                                      "results/tuning/best_config.json"])
    if bad:
        return Result(name, "FAIL", f"missing or not committed: {', '.join(bad)}")
    with open(root / "results/tuning/tuning_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    sources = {r.get("model_source") for r in rows}
    if sources != {"hamid"}:
        return Result(name, "FAIL", f"rows were made with model_source={sorted(map(str, sources))}"
                                    ", not Hamid's model; re-run `python -m src.tune`")
    counts = {s: sum(1 for r in rows if r["stage"] == s) for s in "ABC"}
    if counts != {"A": 12, "B": 3, "C": 1}:
        return Result(name, "WARN", f"committed, but stage counts are {counts} "
                                    "(plan: A=12, B=3, C=1)")
    return Result(name, "PASS", "committed; stages A=12, B=3, C=1; made with Hamid's model")


def check_final_models(root: Path) -> Result:
    name = "Final models trained (seeds 42, 43, 44)"
    models = [f"models/neural_heuristic_seed{s}.pt" for s in SEEDS]
    absent = [p for p in models if not (root / p).exists()]
    if absent:
        return Result(name, "FAIL", f"checkpoints not found: {', '.join(absent)}")
    committed = ["configs/final.yaml", "results/training/final_summary.json"] + [
        f"results/training/neural_heuristic_seed{s}/history.csv" for s in SEEDS]
    bad = untracked_or_missing(root, committed)
    if bad:
        return Result(name, "FAIL", f"missing or not committed: {', '.join(bad)}")
    summary = json.loads((root / "results/training/final_summary.json").read_text("utf-8"))
    sources = {r.get("model_source") for r in summary["runs"]}
    if sources != {"hamid"}:
        return Result(name, "FAIL", f"final_summary.json has model_source={sorted(map(str, sources))}")
    return Result(name, "PASS", "3 checkpoints exist; final.yaml, summary and curves committed")


def parse_readme(root: Path) -> Optional[tuple[str, str]]:
    path = root / "models" / "README.md"
    if not path.exists():
        return None
    match = HF_LINK.search(path.read_text(encoding="utf-8"))
    return (match.group(1), match.group(2)) if match else None


def check_readme(root: Path) -> Result:
    name = "models/README.md contains link and checksum"
    bad = untracked_or_missing(root, ["models/README.md"])
    if bad:
        return Result(name, "FAIL", "models/README.md missing or not committed")
    link = parse_readme(root)
    if link is None:
        return Result(name, "FAIL", "no https://huggingface.co/<user>/<repo> link found")
    if link[0].lower() in PLACEHOLDER_USERS:
        return Result(name, "FAIL", f"the link still has the placeholder user '{link[0]}'")
    problems = write_models_readme.check_readme(str(root / "models"),
                                                str(root / "models" / "README.md"))
    if problems:
        return Result(name, "FAIL", "; ".join(problems))
    return Result(name, "PASS", f"link https://huggingface.co/{link[0]}/{link[1]}; "
                                "SHA256 values match the local files")


def check_report_section(root: Path) -> Result:
    name = "Phase 3 training report section drafted"
    paths = ["report/sections/phase3_training.tex", "figures/training_curves_final.png"] + [
        f"report/generated/{g}.tex" for g in GENERATED]
    bad = untracked_or_missing(root, paths)
    if bad:
        return Result(name, "FAIL", f"missing or not committed: {', '.join(bad)}")
    macros = (root / "report/generated/training_macros.tex").read_text(encoding="utf-8")
    if "PROVISIONAL" in macros:
        return Result(name, "FAIL", "the generated macros are marked PROVISIONAL (stand-in "
                                    "results); re-run scripts/make_training_report_assets.py")
    return Result(name, "PASS", "section, figure and generated tables committed; not provisional")


def check_git_state(root: Path, branch: str) -> Result:
    name = f"All commits pushed to {branch}"
    current = git(root, "rev-parse", "--abbrev-ref", "HEAD")[1]
    code, changed = git(root, "status", "--porcelain", "--untracked-files=no")
    if changed:
        files = ", ".join(line[3:] for line in changed.splitlines()[:5])
        return Result(name, "FAIL", f"uncommitted changes in tracked files: {files}")
    code, upstream = git(root, "rev-parse", "--abbrev-ref", "@{u}")
    if code != 0:
        return Result(name, "FAIL", f"'{current}' has no upstream; run `git push -u origin {current}`")
    ahead = int(git(root, "rev-list", "--count", "@{u}..HEAD")[1] or 0)
    if ahead:
        return Result(name, "FAIL", f"{ahead} commit(s) not pushed yet; run `git push`")
    _, others = git(root, "ls-files", "--others", "--exclude-standard")
    notes = []
    if current != branch:
        notes.append(f"on branch '{current}', expected '{branch}'")
    if others:
        listed = others.splitlines()
        notes.append(f"{len(listed)} untracked file(s), e.g. {', '.join(listed[:3])}")
    if notes:
        return Result(name, "WARN", "everything tracked is pushed, but " + "; ".join(notes))
    return Result(name, "PASS", f"'{current}' is clean and up to date with {upstream}")


def check_requirements_file(root: Path) -> Result:
    name = "requirements.txt covers every import"
    missing, _ = check_requirements.check(root)
    if missing:
        return Result(name, "FAIL", "not listed: " + ", ".join(sorted(missing)))
    return Result(name, "PASS", "every imported third-party package is listed")


def check_tests(root: Path) -> Result:
    name = "pytest -q passes"
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root,
                          capture_output=True, text=True)
    last = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    return Result(name, "PASS" if proc.returncode == 0 else "FAIL", last)


# ---------------------------------------------------------------------------
# Online checks (Hugging Face)
# ---------------------------------------------------------------------------
def check_hf_public(root: Path, opener: Callable = urllib.request.urlopen) -> Result:
    name = "Hugging Face repository is public"
    link = parse_readme(root)
    if link is None or link[0].lower() in PLACEHOLDER_USERS:
        return Result(name, "FAIL", "models/README.md has no real Hugging Face link")
    url = f"https://huggingface.co/api/models/{link[0]}/{link[1]}"
    try:
        with opener(urllib.request.Request(url), timeout=30) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return Result(name, "FAIL", f"HTTP {exc.code}: the repo is private or does not exist "
                                    "(anonymous access failed)")
    except (urllib.error.URLError, OSError) as exc:
        return Result(name, "WARN", f"could not reach Hugging Face: {exc}")
    if info.get("private"):
        return Result(name, "FAIL", "the repo is marked private")
    return Result(name, "PASS", f"https://huggingface.co/{link[0]}/{link[1]} is publicly readable")


def check_hf_hashes(root: Path, opener: Callable = urllib.request.urlopen) -> Result:
    name = "Hugging Face files match the README SHA256"
    link = parse_readme(root)
    if link is None:
        return Result(name, "FAIL", "models/README.md has no Hugging Face link")
    text = (root / "models" / "README.md").read_text(encoding="utf-8")
    listed = dict(write_models_readme.ROW.findall(text))
    if not listed:
        return Result(name, "FAIL", "no checkpoint hashes found in models/README.md")
    problems = []
    for file_name, expected in sorted(listed.items()):
        url = f"https://huggingface.co/{link[0]}/{link[1]}/resolve/main/{file_name}"
        digest = hashlib.sha256()
        try:
            with opener(urllib.request.Request(url), timeout=60) as resp:
                for block in iter(lambda: resp.read(1 << 20), b""):
                    digest.update(block)
        except urllib.error.HTTPError as exc:
            problems.append(f"{file_name}: HTTP {exc.code} (not uploaded?)")
            continue
        except (urllib.error.URLError, OSError) as exc:
            return Result(name, "WARN", f"could not reach Hugging Face: {exc}")
        if digest.hexdigest() != expected:
            problems.append(f"{file_name}: the uploaded file differs from the README hash")
    if problems:
        return Result(name, "FAIL", "; ".join(problems))
    return Result(name, "PASS", f"{len(listed)} uploaded file(s) match the README SHA256 values")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_checks(root: Path, *, branch: str = "d2/saeed-training", online: bool = False,
               run_tests: bool = False) -> list[Result]:
    checks: list[tuple[str, Callable[[], Result]]] = [
        ("manifest", lambda: check_manifest(root)),
        ("training", lambda: check_training(root)),
        ("tuning", lambda: check_tuning(root)),
        ("final models", lambda: check_final_models(root)),
        ("README", lambda: check_readme(root)),
        ("report section", lambda: check_report_section(root)),
        ("requirements", lambda: check_requirements_file(root)),
        ("git state", lambda: check_git_state(root, branch)),
    ]
    if run_tests:
        checks.append(("tests", lambda: check_tests(root)))
    if online:
        checks += [("hf public", lambda: check_hf_public(root)),
                   ("hf hashes", lambda: check_hf_hashes(root))]
    results = []
    for label, fn in checks:
        try:
            results.append(fn())
        except Exception as exc:  # a broken file must show up as a FAIL, not a crash
            results.append(Result(label, "FAIL", f"check crashed: {type(exc).__name__}: {exc}"))
    return results


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the definition of done")
    p.add_argument("--root", default=str(SCRIPTS_DIR.parent))
    p.add_argument("--branch", default="d2/saeed-training")
    p.add_argument("--online", action="store_true", help="Also check the Hugging Face repo")
    p.add_argument("--run-tests", action="store_true", help="Also run pytest -q")
    args = p.parse_args(argv)

    results = run_checks(Path(args.root), branch=args.branch, online=args.online,
                         run_tests=args.run_tests)
    for r in results:
        print(f"[{r.status}] {r.name}\n        {r.detail}")
    counts = {s: sum(1 for r in results if r.status == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\n{counts['PASS']} passed, {counts['WARN']} warning(s), {counts['FAIL']} failed")
    if counts["FAIL"] == 0 and not args.online:
        print("Run again with --online to check the Hugging Face upload.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
