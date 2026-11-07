# Neural-Guided Heuristics vs Classical Pathfinding in Dynamic Grid-Mazes

**Course:** AI-2002 Artificial Intelligence  
**Track:** B (Research & Development)  
**Deliverable:** 2 (Model Architecture & Empirical Evaluation)  
**Repository:** [https://github.com/x-abdullah-saeed-x/Neural-Heuristic-Research](https://github.com/x-abdullah-saeed-x/Neural-Heuristic-Research)  
**Integration Branch:** `deliverable-2`

## Team Members & Responsibilities
- **Abdullah Zia** (Evaluation, Analysis, and Report Lead) — A* search engine, evaluation harness, baselines, metric frameworks, fallback neural models, error & environmental bias analysis, publication figures, report integration.
- **Muhammad Abdullah Saeed** (Data and Training Lead) — Dataset loaders, split manifests, multi-stage hyperparameter tuning, PyTorch training loop, checkpointing, and Hugging Face artifact release.
- **Muhammad Hamid Abad** (Model and Search Integration Lead) — Neural network architectures (multi-task regression & classification heads), feature extraction pipelines, heuristic search wrappers, and ablation studies.

## Project Summary
This project investigates learned neural heuristics for graph-search pathfinding, comparing classical $A^\ast$ search (Manhattan distance baseline) with a neural-network-guided $A^\ast$ across procedurally generated grid mazes. We evaluate on nodes expanded, path optimality, per-move latency, admissibility violation rates, and environmental fairness.

## Research Hypothesis
> A trained neural network heuristic $h_\theta(n)$ integrated into $A^\ast$ will expand significantly fewer nodes than the classical Manhattan distance heuristic in complex, obstructed grid environments while maintaining near-perfect path optimality.

---

## Deliverable 2 Status
Deliverable 2 is fully implemented, verified, and benchmarked:
- **Phase 3 (Model Architecture & Training):** Multi-task neural architecture (29 input features $\to$ 256 $\to$ 256 $\to$ 128 $\to$ continuous target regression $h^\ast/N$ + 10-bin distance classification logits). Supported via both PyTorch and rubric-compliant scikit-learn MLP fallback runtimes.
- **Phase 4 (Empirical Evaluation):** Benchmark on 282 unseen test mazes across Dijkstra, Manhattan, Oracle, and Neural heuristic configurations ($\alpha$-scaling sweep and Manhattan clipping).
- **Offline Metrics:** High regression correlation ($R^2 = 0.8597$, $r = 0.9272$) and strong 10-bin decile discrimination (OvR Macro AUC-ROC = 0.9444).
- **Online Search:** Neural Clipped Manhattan achieves 100% success rate with an optimality ratio of 1.001 and sub-2 ms latency per maze.

---

## Repository Structure
```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/                    # Excluded from git (reproduced locally)
│   ├── raw/
│   ├── labeled/
│   ├── cleaned/
│   └── processed/
├── data_scripts/            # Data acquisition & cleaning pipeline
├── src/                     # Core system modules
│   ├── __init__.py
│   ├── search.py            # Deterministic A* engine with tie-breaking
│   ├── metrics.py           # Regression, classification, admissibility & search metrics
│   ├── evaluate.py          # Unified evaluation harness & deterministic split
│   ├── analysis.py          # Error residual & environmental bias auditing
│   └── sklearn_heuristic.py # Heuristic wrapper (raw, scaled, clipped)
├── scripts/                 # CLI runners and automation
│   ├── run_baselines.py     # Classical Dijkstra, Manhattan & Oracle runner
│   ├── train_fallback_models.py # Scikit-learn fallback MLP training
│   ├── run_evaluation.py    # Unified benchmark & alpha sweep runner
│   ├── make_figures.py      # Publication-quality figure generation suite
│   ├── results_to_latex.py  # Automated LaTeX table snippet generator
│   └── make_contribution_table.py # Git contribution auditor
├── tests/                   # Pytest test suite
│   ├── test_search.py
│   ├── test_metrics.py
│   ├── test_evaluate.py
│   ├── test_sklearn_heuristic.py
│   └── test_analysis.py
├── figures/                 # High-resolution publication figures (300 dpi)
└── report/                  # LaTeX documentation & compiled PDF
    ├── main.tex
    ├── sections/            # Modular report chapters
    └── generated/           # Dynamic LaTeX table snippets
```

---

## Installation & CPU Environment Setup

```powershell
# Clone the repository and checkout deliverable-2
git clone https://github.com/x-abdullah-saeed-x/Neural-Heuristic-Research.git
cd Neural-Heuristic-Research
git checkout deliverable-2

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install CPU PyTorch (if using PyTorch backend)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies and pytest
pip install -r requirements.txt
pip install scipy pytest
```

---

## Reproducing the Pipeline

### 1. Data Pipeline
```powershell
python data_scripts/generate_mazes.py --num-per-config 50 --seed 42
python data_scripts/compute_labels.py
python data_scripts/clean_data.py
python data_scripts/features.py
```

### 2. Run Test Suite
```powershell
pytest -v
```

### 3. Model Training
```powershell
# Train rubric-compliant fallback models (no GPU required):
python scripts/train_fallback_models.py --samples 50000 --max-iter 30
```

### 4. Search & Metric Evaluation
```powershell
# Run classical and oracle baselines
python scripts/run_baselines.py --split test --output results/evaluation

# Run full neural benchmark, alpha sweep, and cell metrics
python scripts/run_evaluation.py --backend sklearn --split test --output results/evaluation
```

### 5. Figures and LaTeX Tables
```powershell
# Generate all publication figures and run error/bias auditing
python scripts/make_figures.py

# Refresh LaTeX table snippets
python scripts/results_to_latex.py
```

---

## Model Weights & Checkpoints
Trained model checkpoints are hosted on the Hugging Face Hub (per non-negotiable repository policy):
- **Hugging Face Repository:** `https://huggingface.co/x-abdullah-saeed-x/Neural-Heuristic-Research`
- **Fallback Models:** `models/fallback_mlp_regressor.joblib` and `fallback_mlp_classifier.joblib` (reproducible with seed 42).

---

## Contribution Matrix

| Member | GitHub Username | Core Modules | Primary Responsibilities |
|---|---|---|---|
| **Abdullah Zia** | `@24L2507` | `src/search.py`, `src/metrics.py`, `src/evaluate.py`, `src/analysis.py`, `src/sklearn_heuristic.py` | A* search, evaluation harness, baselines, error & bias checks, LaTeX tables, publication figures, report |
| **Muhammad Abdullah Saeed** | `@x-abdullah-saeed-x` | `src/data.py`, `src/train.py`, `src/tune.py`, `src/config.py` | Data loaders, split manifests, hyperparameter tuning, PyTorch training loop, checkpointing |
| **Muhammad Hamid Abad** | `@Hamid4325` | `src/model.py`, `src/features.py`, `src/heuristics.py` | Multi-task architecture, feature refactoring, heuristic wrappers, ablations |

---

## Known Limitations
1. **Synthetic Static Environments:** The current dataset models 2D static grid environments with uniform cost 1.0. Dynamic obstacles or variable terrain friction are not modeled.
2. **Local Feature Receptive Field:** The 5$\times$5 local wall context provides strong local obstacle impedance cues, but long-range topological cul-de-sacs require global distance coordinates ($x/N, y/N$) and goal offsets for orientation.
3. **Overestimation & Admissibility:** Raw unconstrained neural network predictions can overestimate geodesic distances. When absolute shortest-path optimality is strictly required, scaling $\alpha \le 0.8$ or clipping with the Manhattan distance ensures optimal path discovery.
