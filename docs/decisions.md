# Decisions

Implementation decisions made with the owner. `docs/PLAN.md` holds goals, scope, and
constraints. Each entry gives the decision, the reason, and the rejected alternatives.

## D1. Public GitHub repository (2026-09-29, milestone 1)

- Decision: the repository is public at `SauHin/semantic-resume-matcher`.
- Reason: Colab can clone without a token, so no GitHub token is stored in Colab Secrets.
- Consequence: never commit Kaggle data, checkpoints, user testing data, or notebook
  outputs. `.gitignore` blocks `data/`, `runs/`, `logs/`, and checkpoint files.
- Rejected: private repository with a read-only token in Colab Secrets.

## D2. Workflow between git, Colab, and Drive (2026-09-29, milestone 1)

- Decision: edit locally, commit, and push to `main`. In Colab, pull, run
  `pip install -e ".[dev]"`, and start scripts with `!python -m resume_matcher.<module>`.
- The notebook `notebooks/colab.ipynb` only mounts Drive, clones or pulls, installs, and
  runs commands. Core logic stays in `src/`.
- Scripts run in a subprocess (`!python -m`), so a new editable install is visible without a
  kernel restart.
- Work directly on `main` with small commits, so Colab pulls one branch only.

## D3. Storage on Google Drive (2026-09-29, milestone 1)

- Decision: `MyDrive/resume-matcher/` holds `data/raw`, `data/processed`, `runs/`, and `logs/`.
- Scripts read data directly from Drive. The CSV files are small, so a copy to `/content`
  is not necessary.
- Run folder name: `<name>_<UTC YYYYMMDD-HHMMSS>_<git sha>`. Each run folder has `env.json`
  with the GPU, `nvidia-smi` output, Python version, installed packages, git sha, and a
  flag for uncommitted code. A new run never overwrites an existing folder.
- Checkpoints: keep `last` and `best` only. Per-epoch history stays complete in the history
  file.
- Rejected: keep a checkpoint for every epoch. mpnet checkpoints are large and Drive quota
  is limited.

## D4. Result files back to the repository (2026-09-29, milestone 1)

- Decision: copy small result files from Drive to `results/` with Google Drive for Desktop,
  from milestone 4. The owner installs Drive for Desktop before then.
- Reason: the copy is mechanical. No number passes through an AI transcription.
- Rejected: Colab pushes results itself, because it needs a write token and gives commit
  conflicts. Rejected: copying numbers from cell output.

## D5. Dependencies (2026-09-29, milestone 1)

- Decision: add a dependency to `pyproject.toml` at the milestone that needs it. Do not pin
  packages that Colab preinstalls, such as torch and numpy.
- Reason: a reinstall of torch is slow and can break the CUDA match. `env.json` records the
  exact version of every package per run, so each result stays traceable.
- Rejected: a full lock file, because it conflicts with changes to the Colab base image.

## D6. Two test layers (2026-09-29, milestone 1)

- Decision: unit tests without data or GPU (chunking, metrics, perturbations, split logic on
  toy data) run locally and in Colab. Data tests (no ID or text hash in two splits, split
  sizes) run in Colab only. If the data folder is not present, the data tests skip.
- The full `pytest` suite must pass in Colab before an experiment run.
- Local test results are a development aid only. They are not evidence in the report.
- Rejected: Colab only, because each small change then needs a live Colab session.
  Rejected: GitHub Actions, because it adds a workflow file and gives feedback only after a
  push.

## D7. Configuration format (2026-09-29, milestone 1)

- Decision: YAML configuration files, from milestone 4.
- Reason: YAML allows comments, so each hyperparameter can carry its reason for the report.

## D8. Pretraining effect for RQ2 (2026-09-29)

- Problem: BS and B2 differ in pretraining, architecture (BiLSTM vs Transformer), tokenizer
  (word vocabulary vs WordPiece), size, and hyperparameters. A difference between BS and B2
  cannot be attributed to pretraining alone.
- Decision: add a variant with the MiniLM architecture and tokenizer, random initial
  weights, and the same loss and data as B2. Working ID: `B2-rand` (the owner can rename it).
  - `B2-rand` vs B2 isolates pretraining.
  - BS vs `B2-rand` compares BiLSTM and Transformer, both trained from scratch.
  - BS vs B2 stays as the practical comparison: train an own model or use a pretrained one.
- The report also states the BS vs B2 confound as a limitation.
- Priority: above B4. Details (data size, learning rate) are decided at milestone 5.
- Open: the owner adds `B2-rand` to the table in `docs/PLAN.md` §5.
