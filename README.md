# Semantic Resume Matcher

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). The system matches an
English CV to a job description by meaning, not by keywords. It also reports which
requirements in the job description the CV does not support yet. Goals, scope, and
constraints are in [docs/PLAN.md](docs/PLAN.md). Design decisions are in
[docs/decisions.md](docs/decisions.md).

Results: TBD. No experiment has run yet.

## Repository layout

- `src/resume_matcher/`: Python package with all core logic.
- `tests/`: pytest tests.
- `notebooks/colab.ipynb`: Colab bootstrap. It mounts Drive, clones or pulls, installs, and runs commands.
- `docs/`: plan, decisions, and AI usage log.

Data, checkpoints, and run folders are not in this repository. They live on Google Drive
under `MyDrive/resume-matcher/`.

## Run in Colab

1. Open [notebooks/colab.ipynb](https://colab.research.google.com/github/SauHin/semantic-resume-matcher/blob/main/notebooks/colab.ipynb) in Colab.
2. Select a GPU runtime.
3. Run the cells in order.

## Run unit tests locally

If the data folder is not present, the tests that need the real data skip.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest
```

On Linux or macOS, use `.venv/bin/python` instead of `.venv/Scripts/python`.
