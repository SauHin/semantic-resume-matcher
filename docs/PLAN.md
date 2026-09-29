# Project Plan — Semantic Resume Matcher

This plan fixes the **direction, goals, and constraints**. Implementation details (file
layout, exact hyperparameters, library choices, phase boundaries) are open for discussion
and should be decided while building. Record decisions in `docs/decisions.md`.

---

## 1. Goal

Build and evaluate a deep learning system that matches an English CV to a job description
**by meaning, not keywords**, and tells the user which requirements in the job description
their CV does not support yet.

The project must show *why* embedding-based matching behaves differently from keyword
matching, *when* it fails, and whether it is actually suitable for its users — not just
that one model scores higher.

**Target users:** entry-level job seekers, especially students applying for internships.
**Course:** COMP6826001 Deep Learning, BINUS (final project). The guideline is the
authority on what must be delivered.

## 2. Scope boundaries

In scope:
- English CVs and job postings only.
- One experiment track on semantic matching (below).
- Two extra evaluations: robustness suite and requirement-gap evaluation.
- A working app and user testing with 5 real job seekers.

Out of scope (decided, don't reintroduce without discussion):
- LLM API features (e.g. CV rewrite suggestions).
- Bilingual / Indonesian support.
- Resume category classification as a separate track or app output.
- Bias audit, thin-CV scenario, cross-encoder reranking.

## 3. Research questions

1. Do dense embeddings retrieve same-field job postings better than TF-IDF?
2. With a small dataset, how much does pretraining matter? (from scratch vs pretrained vs
   pretrained + fine-tuned)
3. How does contrastive fine-tuning behave during training, and how sensitive is it to the
   amount of training data?
4. What is the accuracy vs compute trade-off between a small and a larger pretrained model?
5. How robust is each approach to synonym substitution and keyword stuffing?
6. How accurately can each approach detect job requirements a CV does not support?

## 4. Data and ground truth

- **Resumes:** Kaggle "Resume Dataset" (snehaanbhawal), 24 categories.
- **Job postings:** Kaggle "Real or Fake Job Posting Prediction" (shivamb). Drop fraudulent
  postings; use `function`/`industry` as proxy labels.
- **Ground truth:** a manual mapping from resume category → job group, decided by the owner
  and documented with reasons. Categories without a reasonable match are excluded from
  matching evaluation. A resume–posting pair is "relevant" if both map to the same group.
- This ground truth measures **field-level** fit, not candidate quality. That limitation
  must be stated explicitly in the report.

Non-negotiable data rules:
- Split resumes and postings (train/val/test) **before** building any pairs.
- Anything fitted (vocabularies, TF-IDF, thresholds) is fitted on train or on a designated
  calibration subset, never on test.
- Long texts exceed model input limits, so chunking + pooling is required and its effect
  should be measured.
- Check for title leakage (first line of a resume often names its category).

## 5. Experiment design (single track)

| ID | Configuration | Role |
|---|---|---|
| B0 | TF-IDF + cosine similarity | Non-DL reference |
| BS | Siamese BiLSTM trained from scratch with contrastive loss | DL baseline |
| B1 | `all-MiniLM-L6-v2`, frozen | Pretrained, no training |
| B2 | MiniLM fine-tuned with contrastive loss, at different training-data sizes | Transfer learning + training-condition variable |
| B3 | `all-mpnet-base-v2`, frozen | Capacity vs compute |
| B4 (optional) | mpnet fine-tuned | Only if time allows |

Key requirements:
- BS and B2 use the same objective so the comparison isolates pretraining.
- Trained models (BS, B2) log per-epoch train/val loss **and** train/val retrieval metric,
  so overfitting and convergence can be analyzed.
- Watch for false negatives with in-batch-negative losses (two items from the same group
  in one batch). How to handle this is an open design decision.
- Sanity-check BS before full runs (loss decreases, beats random baseline).

## 6. Evaluation

**Retrieval (all models):** each test resume queries the test posting pool. Report P@k,
MRR, nDCG, per-category results, and a random baseline. Show match vs non-match score
distributions (histogram + ROC-AUC).

**Robustness suite:**
- Synonym substitution (owner-written dictionary of skill terms): how much do scores and
  ranks change?
- Keyword stuffing (append a posting's top keywords to non-matching CVs, several amounts):
  how far do irrelevant CVs climb? Compare pooling strategies.

**Requirement-gap evaluation:**
- ~20 test-split job descriptions × a few CVs, labeled by the owner (supported / not
  supported + evidence sentence), following a labeling guide written before labeling.
- Threshold chosen on a small calibration subset, evaluated on the rest.
- Precision/recall/F1 for detecting unsupported requirements; TF-IDF overlap as baseline.

**Analysis the guideline expects:** training curves and behavior, model comparison with
compute cost (params, training time, latency, memory), error analysis with patterns,
robustness, scalability (search time as the posting pool grows), limitations, and a
justified choice of deployment model.

## 7. Application

User flow: upload CV (PDF/text) → check and edit extracted text (manual paste fallback) →
paste job description → see results → revise CV outside the app and re-run.

Outputs (all from the project's own models):
- Match level (low/medium/high), calibrated from validation score distributions.
- Each requirement marked supported / not supported, with the supporting CV sentence.
- A short note on what the score does and does not measure.

Constraints: CVs are processed in memory and never stored; the app reuses the same pipeline
code as evaluation; it must run on CPU and be deployed somewhere testers can reach without
a live Colab session (e.g. Hugging Face Spaces). Framework (Streamlit/Gradio) is open.

Optional, only after everything else: "nearest fields" (the 2–3 job groups closest to the
CV using the same encoder).

## 8. User testing

Five real job seekers, with consent. Each rates their own fit for 3 job descriptions they
choose *before* seeing results, then uses the app and fills in SUS + Likert items, plus a
short interview. Results are reported descriptively. Responses are collected and entered by
the owner only.

## 9. Milestones (order, not a schedule)

1. Setup: repo, Colab MCP workflow, Drive persistence.
2. EDA and category mapping (owner decides mapping).
3. Preprocessing, splits, chunking.
4. B0, B1, B3 evaluation harness and results. Start requirement labeling in parallel.
5. BS and B2 training (including data-size variants).
6. Robustness suite, requirement-gap evaluation.
7. Error analysis, compute and scalability measurements, deployment model choice.
8. App and deployment.
9. User testing.
10. Figures, README, report, AI usage log finalized.

## 10. Guideline coverage

| Guideline requirement | Where it's covered |
|---|---|
| Real-world problem, stakeholder | §1 |
| Data description, preparation, bias, limits | §4 |
| DL model construction, pretrained explanation | BS (from scratch), B2 (fine-tuning) |
| Baseline + ≥2 scenarios with clear variables | §5 |
| Metrics, training analysis, comparison, errors | §6 |
| Real-world suitability | §6 robustness/compute/scalability, §8 |
| Working application | §7 |
| AI usage log and integrity | `docs/ai_usage_log.md`, `CLAUDE.md` |
