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

Data, checkpoints, and run folders are not in this repository. Locally, they live in the
gitignored `data/` and `runs/` folders. In Colab, they live on Google Drive under
`MyDrive/resume-matcher/` (`docs/decisions.md` D12).

## Run locally

1. Save your Kaggle API token in `~/.kaggle/access_token`. Never commit the token.
2. Download the raw data to `data/raw/`. The script writes the SHA-256 hashes to `MANIFEST.json`.
3. Run the EDA. The results go to a new folder in `runs/`.

```bash
.venv/Scripts/python -m resume_matcher.download
```

```bash
.venv/Scripts/python -m resume_matcher.eda
```

## Run GPU jobs in Colab

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
