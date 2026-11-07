"""Tests for the Day 5 tools: check_requirements, verify_reproducibility and
check_definition_of_done. No real dataset, no Hamid model and no network are needed
(Hugging Face access is replaced by a fake opener)."""

import io
import json
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path("scripts").resolve()))

import check_definition_of_done as done  # noqa: E402
import check_requirements as reqs  # noqa: E402
import verify_reproducibility as repro  # noqa: E402
import write_models_readme as readme_tool  # noqa: E402

from src.tune import StandInNet  # noqa: E402
from tests.test_final_tools import finals, tiny_data, tuned  # noqa: E402,F401  (fixtures)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


# ---------------------------------------------------------------------------
# check_requirements.py
# ---------------------------------------------------------------------------
def _tree(tmp_path, files, requirements):
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(requirements, encoding="utf-8")
    return tmp_path


def test_requirements_flags_missing_package_with_pip_name(tmp_path):
    root = _tree(tmp_path, {
        "src/a.py": "import numpy\nimport yaml\nimport os\nfrom src import b\nfrom . import c\n",
        "src/b.py": "def f():\n    import sklearn.metrics\n",
        "scripts/run.py": "import a_local_helper\nimport torch\n",
        "scripts/a_local_helper.py": "",
    }, "numpy>=1.24\n# comment\ntorch>=2.2.0\n")
    missing, unused = reqs.check(root)
    assert set(missing) == {"PyYAML", "scikit-learn"}  # import names mapped to pip names
    assert missing["PyYAML"] == ["src/a.py"] and missing["scikit-learn"] == ["src/b.py"]
    assert unused == set()


def test_requirements_pass_when_everything_is_listed(tmp_path):
    root = _tree(tmp_path, {"src/a.py": "import yaml\nimport huggingface_hub\n"},
                 "PyYAML>=6.0\nhuggingface-hub>=0.23.0\ntqdm>=4.66.0\n")
    missing, unused = reqs.check(root)
    assert missing == {} and unused == {"tqdm"}  # '-' vs '_' and case do not matter


def test_requirements_parser_handles_specifiers_and_extras(tmp_path):
    (tmp_path / "requirements.txt").write_text(
        "numpy>=1.24.0\nRequests[socks]==2.31 ; python_version>'3.8'\n-r other.txt\n\n# x\n",
        encoding="utf-8")
    assert reqs.parse_requirements(tmp_path / "requirements.txt") == {"numpy", "requests"}


def test_requirements_of_this_repository_are_complete():
    missing, _ = reqs.check(Path.cwd())
    assert missing == {}, f"add to requirements.txt: {sorted(missing)}"


# ---------------------------------------------------------------------------
# verify_reproducibility.py
# ---------------------------------------------------------------------------
def _kwargs(root, tiny_data, **extra):
    return dict(models_dir=str(root / "models"), training_dir=str(root / "training"),
                model_factory=StandInNet, data=tiny_data, **extra)


def test_full_repeat_reproduces_the_checkpoint(finals, tiny_data):
    root, _ = finals
    results = repro.verify(44, **_kwargs(root, tiny_data))
    assert all(ok for ok, _ in results), results
    messages = " ".join(m for _, m in results)
    assert "all weights match" in messages and "bin edges are identical" in messages


def test_partial_repeat_compares_history_only(finals, tiny_data):
    root, _ = finals
    results = repro.verify(42, **_kwargs(root, tiny_data, epochs=1))
    assert all(ok for ok, _ in results)
    assert not any("weights" in m and "match" in m for _, m in results)
    assert any("only the first 1 epochs" in m for _, m in results)


def _copy_of(root, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(root, copy)
    return copy


def test_changed_history_is_detected(finals, tiny_data, tmp_path):
    copy = _copy_of(finals[0], tmp_path)
    path = copy / "training" / "neural_heuristic_seed44" / "history.csv"
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    cells = lines[1].split(",")
    cells[header.index("val_mae")] = "0.5"
    path.write_text("\n".join([lines[0], ",".join(cells), *lines[2:]]) + "\n")
    results = repro.verify(44, **_kwargs(copy, tiny_data))
    assert any(not ok and "val_mae" in m for ok, m in results)


def test_changed_weights_are_detected(finals, tiny_data, tmp_path):
    copy = _copy_of(finals[0], tmp_path)
    path = copy / "models" / "neural_heuristic_seed44.pt"
    ckpt = torch.load(path, map_location="cpu")
    first = next(iter(ckpt["model_state"]))
    ckpt["model_state"][first] = ckpt["model_state"][first] + 0.1
    torch.save(ckpt, path)
    results = repro.verify(44, **_kwargs(copy, tiny_data))
    assert any(not ok and "weights" in m for ok, m in results)


def test_missing_files_and_cli_guards(finals, tiny_data, tmp_path):
    with pytest.raises(FileNotFoundError):
        repro.verify(7, **_kwargs(finals[0], tiny_data))
    assert repro.main(["--model", "standin"]) == 1  # stand-in only allowed with --quick


# ---------------------------------------------------------------------------
# check_definition_of_done.py: a throw-away Git repository that satisfies every item
# ---------------------------------------------------------------------------
def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def build_repo(tmp_path, *, hf_user="alice"):
    root = tmp_path / "repo"

    def write(rel, text="x\n"):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    write(".gitignore", "*.pt\n")
    write("requirements.txt", "numpy>=1.24.0\n")
    write("src/train.py", "import numpy\n")
    for rel in ("tests/test_train_smoke.py", "tests/test_data.py", "configs/default.yaml",
                "configs/final.yaml", "report/sections/phase3_training.tex"):
        write(rel)
    write("results/manifests/split_manifest.json", json.dumps({
        "seed": 42, "total_mazes": 1876,
        "splits": {"train": {"maze_count": 1313, "entries": []},
                   "val": {"maze_count": 281, "entries": []},
                   "test": {"maze_count": 282, "entries": []}}}))
    rows = ["run_id,stage,model_source"] + [f"A{i},A,hamid" for i in range(12)] \
        + [f"B{i},B,hamid" for i in range(3)] + ["C,C,hamid"]
    write("results/tuning/tuning_summary.csv", "\n".join(rows) + "\n")
    write("results/tuning/best_config.json", json.dumps({"model_source": "hamid"}))
    write("results/training/final_summary.json", json.dumps(
        {"runs": [{"seed": s, "model_source": "hamid"} for s in (42, 43, 44)], "aggregate": {}}))
    for seed in (42, 43, 44):
        write(f"results/training/neural_heuristic_seed{seed}/history.csv", "epoch\n1\n")
        (root / "models").mkdir(exist_ok=True)
        torch.save({"model_state": {}, "model_config": {}, "target_mean": 0.7, "target_std": 0.3,
                    "bin_edges": [float(i) for i in range(9)], "config": {},
                    "git_commit": "abc", "seed": seed},
                   root / "models" / f"neural_heuristic_seed{seed}.pt")
    rows_ = readme_tool.collect(str(root / "models"), str(root / "results/training"))
    write("models/README.md", readme_tool.render_readme(rows_, hf_user, "neural-heuristic-maze"))
    write("figures/training_curves_final.png", "png")
    for name in done.GENERATED:
        write(f"report/generated/{name}.tex", "% generated\n")

    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _git(root, "checkout", "-q", "-b", "d2/saeed-training")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "work")
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    _git(root, "remote", "add", "origin", str(remote))
    _git(root, "push", "-q", "-u", "origin", "d2/saeed-training")
    return root


def statuses(root, **kw):
    return {r.name: r for r in done.run_checks(root, **kw)}


@needs_git
def test_a_complete_repository_passes_everything(tmp_path):
    root = build_repo(tmp_path)
    results = done.run_checks(root)
    assert len(results) == 8
    assert [r.status for r in results] == ["PASS"] * 8, [(r.name, r.detail) for r in results]
    assert done.main(["--root", str(root)]) == 0


@needs_git
def test_missing_or_uncommitted_files_fail(tmp_path):
    root = build_repo(tmp_path)
    _git(root, "rm", "-q", "results/tuning/tuning_summary.csv")
    _git(root, "commit", "-q", "-m", "remove")
    _git(root, "push", "-q")
    res = statuses(root)
    assert res["Tuning summary committed"].status == "FAIL"
    assert done.main(["--root", str(root)]) == 1


@needs_git
def test_unpushed_and_uncommitted_work_fails(tmp_path):
    root = build_repo(tmp_path)
    (root / "src/train.py").write_text("import numpy\n# changed\n")
    res = statuses(root)["All commits pushed to d2/saeed-training"]
    assert res.status == "FAIL" and "uncommitted" in res.detail
    _git(root, "commit", "-q", "-am", "change")
    res = statuses(root)["All commits pushed to d2/saeed-training"]
    assert res.status == "FAIL" and "not pushed" in res.detail


@needs_git
def test_untracked_files_and_other_branches_only_warn(tmp_path):
    root = build_repo(tmp_path)
    (root / "notes.txt").write_text("hello")
    res = statuses(root)["All commits pushed to d2/saeed-training"]
    assert res.status == "WARN" and "untracked" in res.detail
    assert done.main(["--root", str(root)]) == 0
    other = statuses(root, branch="other")["All commits pushed to other"]
    assert other.status == "WARN"


@needs_git
def test_standin_results_are_rejected(tmp_path):
    root = build_repo(tmp_path)
    path = root / "results/tuning/tuning_summary.csv"
    path.write_text(path.read_text().replace("C,C,hamid", "C,C,standin"))
    res = statuses(root)["Tuning summary committed"]
    assert res.status == "FAIL" and "Hamid" in res.detail


@needs_git
def test_placeholder_user_and_hash_mismatch_fail(tmp_path):
    placeholder = build_repo(tmp_path / "a", hf_user="your-hf-username")
    res = statuses(placeholder)["models/README.md contains link and checksum"]
    assert res.status == "FAIL" and "placeholder" in res.detail

    changed = build_repo(tmp_path / "b")
    (changed / "models/neural_heuristic_seed43.pt").write_bytes(b"corrupted")
    res = statuses(changed)["models/README.md contains link and checksum"]
    assert res.status == "FAIL" and "SHA256 differs" in res.detail


@needs_git
def test_provisional_report_assets_fail(tmp_path):
    root = build_repo(tmp_path)
    (root / "report/generated/training_macros.tex").write_text("PROVISIONAL\n")
    _git(root, "commit", "-q", "-am", "macros")
    _git(root, "push", "-q")
    res = statuses(root)["Phase 3 training report section drafted"]
    assert res.status == "FAIL" and "PROVISIONAL" in res.detail


@needs_git
def test_requirements_gap_fails(tmp_path):
    root = build_repo(tmp_path)
    (root / "src/train.py").write_text("import numpy\nimport scipy\n")
    res = statuses(root)["requirements.txt covers every import"]
    assert res.status == "FAIL" and "scipy" in res.detail


def test_pytest_check_reports_pass_and_fail(tmp_path):
    ok = tmp_path / "ok"
    (ok / "tests").mkdir(parents=True)
    (ok / "tests" / "test_a.py").write_text("def test_a():\n    assert True\n")
    assert done.check_tests(ok).status == "PASS"
    bad = tmp_path / "bad"
    (bad / "tests").mkdir(parents=True)
    (bad / "tests" / "test_a.py").write_text("def test_a():\n    assert False\n")
    assert done.check_tests(bad).status == "FAIL"


# ---------------------------------------------------------------------------
# Online checks with a fake Hugging Face
# ---------------------------------------------------------------------------
class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def fake_hf(root, *, private=False, status=200, overrides=None):
    """An `opener` serving the local checkpoints as if they were on Hugging Face."""
    overrides = overrides or {}

    def opener(request, timeout=0):
        url = request.full_url
        if status != 200:
            raise urllib.error.HTTPError(url, status, "error", {}, None)
        if "/api/models/" in url:
            return FakeResponse(json.dumps({"private": private}).encode())
        name = url.rsplit("/", 1)[-1]
        return FakeResponse(overrides.get(name, (root / "models" / name).read_bytes()))

    return opener


@needs_git
def test_online_public_repo_and_matching_hashes_pass(tmp_path):
    root = build_repo(tmp_path)
    assert done.check_hf_public(root, fake_hf(root)).status == "PASS"
    assert done.check_hf_hashes(root, fake_hf(root)).status == "PASS"


@needs_git
def test_online_private_missing_or_different_uploads_fail(tmp_path):
    root = build_repo(tmp_path)
    assert done.check_hf_public(root, fake_hf(root, private=True)).status == "FAIL"
    assert done.check_hf_public(root, fake_hf(root, status=404)).status == "FAIL"
    assert done.check_hf_hashes(root, fake_hf(root, status=404)).status == "FAIL"
    different = fake_hf(root, overrides={"neural_heuristic_seed44.pt": b"other bytes"})
    res = done.check_hf_hashes(root, different)
    assert res.status == "FAIL" and "seed44.pt" in res.detail


@needs_git
def test_online_network_problem_is_a_warning_not_a_failure(tmp_path):
    root = build_repo(tmp_path)

    def offline(request, timeout=0):
        raise urllib.error.URLError("no network")

    assert done.check_hf_public(root, offline).status == "WARN"
    assert done.check_hf_hashes(root, offline).status == "WARN"


@needs_git
def test_online_checks_need_a_real_link(tmp_path):
    root = build_repo(tmp_path, hf_user="your-hf-username")
    assert done.check_hf_public(root, fake_hf(root)).status == "FAIL"
