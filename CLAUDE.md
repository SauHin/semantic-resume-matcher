# CLAUDE.md — Semantic Resume Matcher

Final project for COMP6826001 Deep Learning (BINUS, AY 2026/2027). Owner: Jonathan Davin.
`docs/PLAN.md` holds the project's goals, scope, and constraints. Read it before starting any task.

## What this project is

A system that matches an English CV to a job description by meaning, not keywords, and
reports which requirements in the job description the CV does not support yet.

Single experiment track on semantic matching: B0 TF-IDF (non-DL reference), BS Siamese
BiLSTM from scratch with contrastive loss (DL baseline), B1 MiniLM frozen, B2 MiniLM
fine-tuned with contrastive loss at several training-data sizes, B3 mpnet frozen, B4 mpnet
fine-tuned (optional). Extra evaluations: robustness suite (synonyms, keyword stuffing) and
a requirement-gap evaluation on a small owner-labeled set.

Data: Kaggle "Resume Dataset" (snehaanbhawal) and "Real or Fake Job Posting Prediction"
(shivamb). Ground truth = owner-decided mapping from resume category to job group.

`docs/PLAN.md` fixes goals, scope, and constraints. Implementation details are open:
discuss options with the owner, then record the decision in `docs/decisions.md`.

## Scope rules

- Out of scope: LLM API features, bilingual support, resume classification as a separate
  track or app output, bias audit, cross-encoder reranking.
- If a task seems to need something outside the plan, raise it instead of adding it.

## Integrity rules (non-negotiable)

This project is graded, and fabricated evidence is an academic integrity violation.

- Never write, estimate, or "fill in" any metric, training time, or result. Every number in
  docs, READMEs, figures, or the report must come from a saved result file produced by
  running code. If a result doesn't exist yet, write `TBD` and say so.
- Never create or edit requirement annotation labels, the synonym dictionary's final content, the category mapping's final decisions, or user
  testing responses. You may propose drafts, but mark them clearly as `PROPOSAL` in a
  separate file; the owner decides and writes the final version.
- Never generate fake or synthetic data and present it as real.
- When you help with something substantial, remind the owner to add an entry to
  `docs/ai_usage_log.md`. When asked to draft an entry, follow the rules section in that
  file exactly: never fill in verification, test results, dates, or entry numbers you
  don't know; ask the owner instead.

## Data leakage rules

- Resumes and job postings are split (stratified, fixed seed) **before** any pair is built. Pairs for training come only from train splits.
- Vocabularies, TF-IDF vocab, thresholds, and any fitted statistic are fit on train (or on
  the designated calibration split), never on test.
- Requirement-gap threshold is chosen on the calibration subset only (see PLAN §6).
- Keep a test that fails if any ID appears in more than one split.

## Compute and workflow (full Google Colab)

- All execution (data prep, training, evaluation, tests) runs in Google Colab Pro through the
  Colab MCP server (`googlecolab/colab-mcp`), connected to a Colab notebook open in the
  browser. Claude Code runs locally only to edit code, use git, run unit tests that need no
  data or GPU, and drive the Colab session. Local test results are a development aid only.
  The full test suite must pass in Colab before an experiment run (`docs/decisions.md` D6).
- The git repo is the single source of truth. Never write core logic directly into notebook
  cells. Flow: edit locally → commit and push → in Colab: `git pull` → install the package →
  run the relevant script with its config.
- `/content` is wiped when the runtime resets. Keep raw and processed data, checkpoints, and
  run folders on Google Drive (e.g. `MyDrive/resume-matcher/`). Copy small result files
  (metrics, per-epoch history, config) back into the repo and commit them.
- Long jobs: start them in the background with logs on Drive
  (`nohup python -m ... > <drive>/logs/<run_id>.log 2>&1 &`), then poll the log instead of
  blocking one cell. Save a checkpoint every epoch so a disconnect doesn't lose the run.
- Before running a cell that installs packages, deletes files, or starts a GPU job longer
  than a few minutes, state what will run and wait for approval.
- Record the GPU type (`nvidia-smi`) and library versions in every run folder, because Colab
  can assign different GPUs between sessions.
- Kaggle and GitHub tokens live in Colab Secrets. Never print them or write them to code,
  notebooks, or logs.

## Code conventions (defaults, can be changed by agreement)

- Python package in `src/`, experiments driven by config files, core logic outside notebooks.
- Every run writes a folder with its config, metrics, and (for trained models) per-epoch
  history of train/val loss and train/val metric.
- Fixed seeds; library versions and GPU type recorded per run.
- The app reuses the evaluation pipeline code instead of a copy.
- Tests for data splitting, chunking, metrics, and perturbations.
- Figures are generated by scripts from saved results.

## How to work with me

- Work one milestone of `docs/PLAN.md` at a time. Before writing code, discuss the approach
  and trade-offs, then wait for approval.
- Explain non-obvious design choices so the owner can defend them in the report and
  presentation.
- After finishing, run the tests, list what changed, and note decisions in `docs/decisions.md`.
- Prefer small commits with clear messages.
- If a library API differs from what you expect, check the installed version's docs
  instead of guessing.
