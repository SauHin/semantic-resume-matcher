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
- `docs/PLAN.md` §5 and §9 include `B2-rand` (owner request, 2026-09-29).

## D9. Web UI first, with a fixed API contract (2026-09-30, early work for milestone 8)

- Decision: the owner builds the UI before the models exist. The UI is a static React +
  Vite + TypeScript app in `web/`, for deployment on Vercel (Hobby plan).
- Constraint from the owner: every hosting option must be free and need no credit card.
- The UI depends only on the contract in `web/src/api.ts`: `{cv_text, jd_text}` in,
  `{mock, model, match_level, requirements[{text, supported, evidence}]}` out. Any backend
  can replace another one without a change to the UI. D10 changes this contract.
- Until a backend exists, `match()` returns mock output. The page shows a "Mock results"
  notice for it. Mock output never goes into the report or into screenshots as a result.
- PDF text extraction runs in the browser with pdf.js, so the CV file never leaves the tab.
  If the user uploads a PDF, pdf.js loads at that time, not with the page.
- Backend plan: a Gradio Space on ZeroGPU, called from the UI through its API. ZeroGPU is
  free for accounts older than 30 days with a verified email. Fallbacks, in order:
  1. The model in the browser with transformers.js.
  2. A local backend through Cloudflare Quick Tunnel.
  3. Render free web service. Its RAM limit and card requirement are not checked yet.
  4. A Gradio UI only.
- Open point: PLAN §7 says the app must run on CPU. ZeroGPU runs the model on a GPU. The
  same code still runs on CPU. The owner decides at milestone 8.
- Rejected: Next.js, because the app is one page and needs no server rendering. Rejected:
  a Docker Space, because Hugging Face now needs a paid plan for Gradio and Docker Spaces
  on CPU.

## D10. Batch mode: several job posts per check (2026-09-30)

- Decision: the user adds up to 10 job posts with one CV. The app ranks them by match
  score, best fit first. Each job post shows its match level and its requirement markup.
  This is an owner request and an addition to PLAN §7.
- Contract change in `web/src/api.ts`: `{cv_text, jd_texts[]}` in,
  `{mock, model, results[]}` out. Each result has `score`, `match_level`, and
  `requirements`. `results[i]` belongs to `jd_texts[i]`. A check of one job post is a
  batch of one, so there is only one contract.
- Reason for one request: the backend encodes the CV once, and ZeroGPU takes one queue
  slot instead of one per job post.
- `score` is only for the order of job posts. The UI does not show it, because a raw
  similarity value means little to a job seeker.
- The cap of 10 is an unmeasured first value. Its purpose is to keep one request inside
  the default ZeroGPU time limit of 60 seconds. Set the final value from the latency
  measured at milestone 7.
- Rejected: one request per job post from the UI, because it encodes the CV again for each
  job post.

## D11. Term-mismatch flag: same meaning, different term (2026-10-01, milestone 8)

- Problem: the embedding model can mark the requirement "PowerPoint" as supported because
  the CV says "PPT". A company that filters CVs by exact keywords can still reject that CV.
  Without a flag, the app gives a false sense of safety.
- Decision: for a supported requirement, the app flags the important words of the
  requirement that do not appear anywhere in the CV. Example message: "You already have
  this skill (evidence: '...PPT...'). The job post uses the term 'PowerPoint'."
- Purpose: the step "revise CV outside the app and re-run" in PLAN §7 becomes actionable.
  This is most useful after batch mode (D10) shows the best-fit job posts.
- Rules:
  1. The flag appears only for supported requirements, that is, requirements with an
     evidence sentence. For unsupported requirements, the app suggests no words. Reason:
     the feature must not encourage keyword stuffing, which the robustness suite tests as
     a weakness.
  2. The flag appears for each content word of the requirement that is not in the CV, also
     for a partial match. Example: job post "Microsoft PowerPoint", CV "PowerPoint" gives a
     flag for "Microsoft". If the full job-post term is in the CV, no flag appears. Example:
     job post "PowerPoint", CV "Microsoft PowerPoint".
  3. Common words such as "experience" or "strong" do not trigger a flag. The list of
     ignored words is decided at milestone 8.
- Implementation: reuse the TF-IDF overlap code from the requirement-gap baseline (PLAN §6,
  milestone 6). No new model and no new dependency.
- The note in the app still says that the score measures a match in meaning, not whether
  the CV passes a company's keyword filter.
- Evaluation: one user testing item (PLAN §8) asks whether the flag helped the tester
  revise the CV. A claim in the report that the flag is useful must come from tester
  answers, not from an assumption.
- Open for milestone 8: the ignored-word list, word matching rules (case, plural forms,
  abbreviations), and the language of the app messages.
