"""Support threshold per model, without labels, from a null distribution (Djinni val split).

Null pairs: a job post and a CV with different keywords, so their lines should not match.
For each requirement line of the job, take its best similarity to any CV line. The threshold
is the 95th percentile of these values, so about 5% of unrelated requirements pass.
shortcut: label-free; replace with the owner-labeled calibration subset (PLAN §6) when it exists.
Run: python -m resume_matcher.calibrate
"""
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from resume_matcher.djinni import PROCESSED
from resume_matcher.env import new_run_dir
from resume_matcher.match import MODEL_ID, THRESHOLDS, split_lines

MODELS = [MODEL_ID, "models/b2_quick"]
N_PAIRS, REQS_PER_JOB, PERCENTILE, SEED = 300, 5, 95, 42


def make_pairs(jobs, cvs, same, rng):
    """One CV per job: with the same keyword (same=True) or a different one."""
    pairs = []
    for kw, text in zip(jobs.keyword, jobs.text):
        pool = cvs[(cvs.keyword == kw) == same]
        pairs.append((text, pool.text.iloc[rng.integers(len(pool))]))
    return pairs


def best_sims(encode, pairs, rng):
    """Best CV-line similarity for up to REQS_PER_JOB random requirement lines per pair."""
    reqs, cv_lines = [], []
    for job, cv in pairs:
        r, c = split_lines(job), split_lines(cv)
        if r and c:
            reqs.append(list(rng.permutation(r))[:REQS_PER_JOB])
            cv_lines.append(c)
    R = encode([x for r in reqs for x in r])
    C = encode([x for c in cv_lines for x in c])
    out, i, j = [], 0, 0
    for r, c in zip(reqs, cv_lines):
        out.extend((R[i:i + len(r)] @ C[j:j + len(c)].T).max(1))
        i, j = i + len(r), j + len(c)
    return np.array(out)


def main():
    jobs = pd.read_parquet(PROCESSED / "djinni_jobs.parquet").query("split == 'val'")
    cvs = pd.read_parquet(PROCESSED / "djinni_cvs.parquet").query("split == 'val'")
    jobs = jobs.sample(n=N_PAIRS, random_state=SEED)
    thresholds = json.loads(THRESHOLDS.read_text()) if THRESHOLDS.exists() else {}
    run = new_run_dir("calibrate-support")
    report = {}
    for model_id in MODELS:
        model = SentenceTransformer(model_id, device="cpu")
        encode = lambda t: model.encode(t, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
        null = best_sims(encode, make_pairs(jobs, cvs, False, np.random.default_rng(SEED)), np.random.default_rng(SEED))
        same = best_sims(encode, make_pairs(jobs, cvs, True, np.random.default_rng(SEED)), np.random.default_rng(SEED))
        t = float(np.percentile(null, PERCENTILE))
        report[model_id] = {"threshold": t, "percentile": PERCENTILE, "n_null": len(null),
                            "null_pass_rate": float((null >= t).mean()),
                            "same_keyword_pass_rate": float((same >= t).mean())}
        thresholds[model_id] = {"threshold": t, "percentile": PERCENTILE, "run": run.name}
        print(model_id, report[model_id], flush=True)
    (run / "metrics.json").write_text(json.dumps(report, indent=2))
    THRESHOLDS.parent.mkdir(exist_ok=True)
    THRESHOLDS.write_text(json.dumps(thresholds, indent=2) + "\n")
    print(f"run folder: {run}")


if __name__ == "__main__":
    main()
