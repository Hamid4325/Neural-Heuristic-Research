"""Tests for the Day 4 tools: make_final_config, train_final, write_models_readme and
make_training_report_assets (+ a LaTeX compile test of report/sections/phase3_training.tex).

None of them needs Hamid's model or the real dataset: the pipelines run on small synthetic
data with the built-in StandInNet.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path("scripts").resolve()))

import make_final_config  # noqa: E402
import make_training_report_assets as assets  # noqa: E402
import train_final  # noqa: E402
import write_models_readme as readme_tool  # noqa: E402

from src.config import load_config  # noqa: E402
from src.tune import StandInNet, run_tuning  # noqa: E402
from tests.test_tune import TINY_SPEC, TINY_STAGES, _synthetic  # noqa: E402

SECTION = Path("report/sections/phase3_training.tex")


@pytest.fixture(scope="module")
def tiny_data():
    X_tr, y_tr = _synthetic(5000, 0)
    X_va, y_va = _synthetic(1000, 1)
    return X_tr, y_tr, X_va, y_va


@pytest.fixture(scope="module")
def tuned(tmp_path_factory, tiny_data):
    """A finished (tiny) tuning run made with the stand-in network."""
    root = tmp_path_factory.mktemp("tuned")
    run_tuning(dict(TINY_SPEC), load_config(None), model_factory=StandInNet,
               model_source="standin", data=tiny_data, results_dir=str(root / "tuning"),
               final_checkpoint=str(root / "tuned_best.pt"), stages=TINY_STAGES)
    return root / "tuning"


@pytest.fixture(scope="module")
def finals(tmp_path_factory, tuned, tiny_data):
    """Three final models (seeds 42, 43, 44) trained on the tiny data."""
    root = tmp_path_factory.mktemp("finals")
    cfg = load_config(str(tuned / "best_config.yaml"))
    rows = train_final.train_all(
        cfg, [42, 43, 44], models_dir=str(root / "models"), training_dir=str(root / "training"),
        replicate_cells=1500, model_factory=StandInNet, model_source="standin", data=tiny_data)
    return root, rows


# ---------------------------------------------------------------------------
# make_final_config.py
# ---------------------------------------------------------------------------
def test_final_config_refuses_standin_results(tuned, tmp_path):
    with pytest.raises(ValueError, match="Hamid"):
        make_final_config.build_final_config(str(tuned), str(tmp_path / "final.yaml"))
    assert make_final_config.main.__module__ == "make_final_config"


def test_final_config_dry_run_writes_loadable_yaml(tuned, tmp_path):
    out = tmp_path / "sub" / "final.yaml"
    meta = make_final_config.build_final_config(str(tuned), str(out), allow_standin=True)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Final training configuration")
    assert "python -m src.train --config configs/final.yaml" in text
    assert load_config(str(out)) == load_config(str(tuned / "best_config.yaml"))
    assert meta["hyperparameters"] == json.loads((tuned / "best_config.json").read_text())[
        "hyperparameters"]


def test_final_config_accepts_hamid_results_without_flag(tuned, tmp_path):
    copy = tmp_path / "tuning"
    shutil.copytree(tuned, copy)
    meta = json.loads((copy / "best_config.json").read_text())
    meta["model_source"] = "hamid"
    (copy / "best_config.json").write_text(json.dumps(meta))
    make_final_config.build_final_config(str(copy), str(tmp_path / "final.yaml"))
    assert (tmp_path / "final.yaml").exists()


def test_final_config_missing_tuning_results(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_final_config.build_final_config(str(tmp_path / "nothing"), str(tmp_path / "x.yaml"))


# ---------------------------------------------------------------------------
# train_final.py
# ---------------------------------------------------------------------------
def test_train_all_writes_three_models_and_summary(finals):
    root, rows = finals
    assert [r["seed"] for r in rows] == [42, 43, 44]
    assert [r["role"] for r in rows] == ["primary", "replicate", "replicate"]
    for seed in (42, 43, 44):
        stem = f"neural_heuristic_seed{seed}"
        assert (root / "models" / f"{stem}.pt").exists()
        for name in ("history.csv", "config.yaml", "metrics.json"):
            assert (root / "training" / stem / name).exists()
    summary = json.loads((root / "training" / "final_summary.json").read_text())
    assert summary["aggregate"]["n_seeds"] == 3 and summary["aggregate"]["val_mae_std"] >= 0
    assert (root / "training" / "final_summary.csv").exists()


def test_replicates_use_the_smaller_cell_budget(finals):
    root, _ = finals
    primary = load_config(str(root / "training" / "neural_heuristic_seed42" / "config.yaml"))
    replicate = load_config(str(root / "training" / "neural_heuristic_seed43" / "config.yaml"))
    assert primary.train.max_train_cells_per_epoch is None  # full split
    assert replicate.train.max_train_cells_per_epoch == 1500
    assert primary.train.seed == 42 and replicate.train.seed == 43


def test_verify_checkpoint_detects_problems(finals):
    root, _ = finals
    path = str(root / "models" / "neural_heuristic_seed42.pt")
    assert train_final.verify_checkpoint(path, 42, 10)["seed"] == 42
    with pytest.raises(ValueError, match="seed"):
        train_final.verify_checkpoint(path, 43, 10)
    with pytest.raises(ValueError, match="bin edges"):
        train_final.verify_checkpoint(path, 42, 5)


def test_train_all_resume_does_not_retrain(finals, tiny_data):
    root, first = finals
    built = []

    def counting_factory(cfg):
        built.append(1)
        return StandInNet(cfg)

    cfg = load_config(str(root / "training" / "neural_heuristic_seed42" / "config.yaml"))
    again = train_final.train_all(
        cfg, [42, 43, 44], models_dir=str(root / "models"), training_dir=str(root / "training"),
        replicate_cells=1500, model_factory=counting_factory, model_source="standin",
        data=tiny_data, resume=True)
    assert built == []
    assert [r["best_val_mae"] for r in again] == [r["best_val_mae"] for r in first]


def test_train_final_cli_guards(tmp_path):
    assert train_final.main(["--model", "standin", "--config", "x.yaml"]) == 1  # needs --quick
    assert train_final.main(["--config", str(tmp_path / "missing.yaml")]) == 1


# ---------------------------------------------------------------------------
# write_models_readme.py
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_models(tmp_path):
    models, training = tmp_path / "models", tmp_path / "training"
    models.mkdir()
    for seed in (42, 43, 44):
        (models / f"neural_heuristic_seed{seed}.pt").write_bytes(bytes([seed]) * 1000 + b"x")
    (training / "neural_heuristic_seed42").mkdir(parents=True)
    (training / "neural_heuristic_seed42" / "metrics.json").write_text(
        json.dumps({"best_val_mae": 0.0512, "best_epoch": 7}))
    return models, training


def _run_readme(models, training, *extra):
    return readme_tool.main(["--models-dir", str(models), "--training-dir", str(training), *extra])


def test_readme_has_link_hashes_and_plan_fields(fake_models):
    models, training = fake_models
    assert _run_readme(models, training, "--hf-user", "alice") == 0
    text = (models / "README.md").read_text(encoding="utf-8")
    assert "https://huggingface.co/alice/neural-heuristic-maze" in text
    assert "neural_heuristic_seed42.pt, neural_heuristic_seed43.pt, neural_heuristic_seed44.pt" in text
    assert 'torch.load(path, map_location="cpu")' in text
    assert "model_state" in text and "bin_edges" in text
    for seed in (42, 43, 44):
        name = f"neural_heuristic_seed{seed}.pt"
        assert readme_tool.sha256_file(str(models / name)) in text
    assert "0.0512" in text


def test_readme_check_passes_then_detects_changes(fake_models, capsys):
    models, training = fake_models
    _run_readme(models, training, "--hf-user", "alice")
    assert _run_readme(models, training, "--check") == 0

    (models / "neural_heuristic_seed43.pt").write_bytes(b"corrupted")
    assert _run_readme(models, training, "--check") == 1
    assert "seed43.pt: SHA256 differs" in capsys.readouterr().out

    (models / "neural_heuristic_seed43.pt").unlink()
    assert _run_readme(models, training, "--check") == 1
    assert "file not found" in capsys.readouterr().out


def test_readme_check_flags_unlisted_files_and_missing_readme(fake_models, capsys):
    models, training = fake_models
    assert _run_readme(models, training, "--check") == 1  # no README yet
    _run_readme(models, training, "--hf-user", "alice")
    (models / "neural_heuristic_seed99.pt").write_bytes(b"extra")
    assert _run_readme(models, training, "--check") == 1
    assert "seed99.pt: not listed" in capsys.readouterr().out


def test_readme_needs_user_name_and_files(fake_models, tmp_path):
    models, training = fake_models
    assert _run_readme(models, training) == 1  # no --hf-user
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run_readme(empty, training, "--hf-user", "alice") == 1


# ---------------------------------------------------------------------------
# make_training_report_assets.py
# ---------------------------------------------------------------------------
def test_number_and_escape_helpers():
    assert assets.tex_number(0.0003) == r"$3\times10^{-4}$"
    assert assets.tex_number(0.00001) == "$10^{-5}$"
    assert assets.tex_number(0.0001) == "$10^{-4}$"
    assert assets.tex_number(0) == "0" and assets.tex_number(0.1) == "0.1"
    assert assets.tex_number(1024) == "1024"
    assert assets.tex_escape("smooth_l1") == r"smooth\_l1"
    assert assets.tex_value("hidden_dims", [256, 256, 128]) == "(256,256,128)"


def test_failed_runs_are_marked_in_the_runs_table():
    row = {"stage": "A", "rank_in_stage": "", "status": "failed", "hidden_dims": "16-16",
           "activation": "relu", "dropout": 0.1, "batch_size": 128, "learning_rate": 0.001,
           "weight_decay": 0.0, "reg_loss": "mse", "lambda_aux": 0.1,
           "target_transform": "raw", "best_val_mae": "", "epochs_run": "", "run_id": "A01_x"}
    assert "diverged" in assets.tuning_runs_table([row])


@pytest.fixture(scope="module")
def generated(tmp_path_factory, tuned, finals):
    root, _ = finals
    out = tmp_path_factory.mktemp("assets")
    grid = out / "grid.yaml"
    grid.write_text(yaml.safe_dump({"grid": TINY_SPEC["grid"]}), encoding="utf-8")
    assets.generate(
        tuning_dir=str(tuned), training_dir=str(root / "training"),
        config_path=str(tuned / "best_config.yaml"), grid_path=str(grid),
        figures_dir=str(out / "figures"), generated_dir=str(out / "generated"))
    return out


def test_assets_are_written(generated):
    for name in ("tuning_space_table.tex", "tuning_runs_table.tex",
                 "final_training_table.tex", "training_macros.tex"):
        assert (generated / "generated" / name).stat().st_size > 0
    png = (generated / "figures" / "training_curves_final.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_generated_tables_are_valid_looking_latex(generated):
    for name in ("tuning_space_table.tex", "tuning_runs_table.tex", "final_training_table.tex"):
        text = (generated / "generated" / name).read_text(encoding="utf-8")
        assert text.count(r"\begin{table}") == text.count(r"\end{table}") == 1
        assert r"\caption" in text and r"\label" in text
        visible = re.sub(r"\\label\{[^}]*\}", "", text)  # underscores are fine in label names
        assert not re.search(r"(?<!\\)_", visible), f"unescaped underscore in {name}"
    runs = (generated / "generated" / "tuning_runs_table.tex").read_text(encoding="utf-8")
    assert len([ln for ln in runs.splitlines() if ln.startswith(("A", "B", "C")) and "&" in ln]) == 6


def test_macros_cover_everything_the_section_uses(generated):
    macros = (generated / "generated" / "training_macros.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{(\\tr[A-Za-z]+)\}", macros))
    used = set(re.findall(r"\\(tr[A-Z][A-Za-z]+)", SECTION.read_text(encoding="utf-8")))
    assert {f"\\{name}" for name in used} <= defined
    assert r"\newcommand{\trSeedList}{42, 43 and 44}" in macros
    assert "PROVISIONAL" in macros  # results of a stand-in run are visibly marked


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
def test_phase3_section_compiles(generated, tmp_path):
    shutil.copytree(generated / "generated", tmp_path / "generated")
    shutil.copytree(generated / "figures", tmp_path / "figures")
    (tmp_path / "sections").mkdir()
    shutil.copy(SECTION, tmp_path / "sections" / SECTION.name)
    (tmp_path / "harness.tex").write_text(
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\\usepackage{graphicx}\\usepackage{booktabs}\n"
        "\\usepackage{amsmath}\\usepackage{amssymb}\\usepackage{float}\\usepackage{caption}\n"
        "\\usepackage{subcaption}\\usepackage{enumitem}\\usepackage{xcolor}\n"
        "\\usepackage[hidelinks]{hyperref}\n\\graphicspath{{figures/}}\n"
        "\\begin{document}\n\\section{Phase 3}\n\\input{sections/phase3_training}\n"
        "\\end{document}\n", encoding="utf-8")
    result = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "harness.tex"],
        cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-2000:]
    assert (tmp_path / "harness.pdf").stat().st_size > 10_000
