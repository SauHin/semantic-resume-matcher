"""Quick run on Djinni: B0 TF-IDF, B1 MiniLM frozen, B2-quick MiniLM fine-tuned (contrastive).

Prototype scale on CPU: small samples, 1 epoch, short inputs. Not the final experiment.
Run: python -m resume_matcher.train_quick
"""
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from resume_matcher.djinni import PROCESSED
from resume_matcher.env import ROOT, new_run_dir

CONFIG = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "seed": 42,
    "n_train_pairs": 4000,
    "n_val_cvs": 300, "n_val_jobs": 600,
    "n_test_cvs": 1000, "n_test_jobs": 2000,
    "train_max_seq": 128,  # shortcut: shorter than the model limit (256) to train fast on CPU
    "batch_size": 32, "lr": 2e-5, "epochs": 1, "scale": 20.0,
}


def retrieval_metrics(cv_vecs, job_vecs, cv_kw, job_kw, k=10):
    """Each CV ranks all jobs. Relevant = same keyword. Returns P@k and MRR."""
    scores = cv_vecs @ job_vecs.T
    order = np.argsort(-scores, axis=1)
    rel = np.asarray(job_kw)[order] == np.asarray(cv_kw)[:, None]
    first = rel.argmax(1)
    mrr = np.where(rel.any(1), 1.0 / (first + 1), 0.0)
    return {f"P@{k}": float(rel[:, :k].mean()), "MRR": float(mrr.mean())}


def encode(model, texts, max_seq=256):
    model.max_seq_length = max_seq
    return model.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)


def sample(df, split, n, seed):
    return df[df.split == split].sample(n=n, random_state=seed)


def train_epoch(model, pairs, opt, cfg):
    """In-batch contrastive loss (like MultipleNegativesRankingLoss). Other jobs with the same
    keyword as the CV are masked out, so they do not count as false negatives."""
    model.train()
    model.max_seq_length = cfg["train_max_seq"]
    pairs = pairs.sample(frac=1, random_state=cfg["seed"])
    losses = []
    for i in range(0, len(pairs), cfg["batch_size"]):
        b = pairs.iloc[i:i + cfg["batch_size"]]
        cv = model({k: v for k, v in model.tokenize(list(b.cv_text)).items()})["sentence_embedding"]
        job = model({k: v for k, v in model.tokenize(list(b.job_text)).items()})["sentence_embedding"]
        sims = F.normalize(cv, dim=-1) @ F.normalize(job, dim=-1).T * cfg["scale"]
        kw = np.asarray(b.keyword)
        same = torch.tensor(kw[:, None] == kw[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
        loss = F.cross_entropy(sims.masked_fill(same, -1e4), torch.arange(len(b)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    cfg = CONFIG
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    run = new_run_dir("djinni-quick")
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    cvs = pd.read_parquet(PROCESSED / "djinni_cvs.parquet")
    jobs = pd.read_parquet(PROCESSED / "djinni_jobs.parquet")

    # Training pairs: train CVs, each with a random train job of the same keyword.
    tr_cv = sample(cvs, "train", cfg["n_train_pairs"], cfg["seed"])
    tr_jobs = jobs[jobs.split == "train"].groupby("keyword")
    rng = np.random.default_rng(cfg["seed"])
    job_text = [tr_jobs.get_group(k).text.iloc[rng.integers(len(tr_jobs.get_group(k)))] for k in tr_cv.keyword]
    pairs = pd.DataFrame({"cv_text": tr_cv.text.values, "job_text": job_text, "keyword": tr_cv.keyword.values})

    va_cv, va_job = sample(cvs, "val", cfg["n_val_cvs"], 1), sample(jobs, "val", cfg["n_val_jobs"], 1)
    te_cv, te_job = sample(cvs, "test", cfg["n_test_cvs"], 2), sample(jobs, "test", cfg["n_test_jobs"], 2)

    def evaluate(model, cv_df, job_df):
        return retrieval_metrics(encode(model, cv_df.text), encode(model, job_df.text), cv_df.keyword, job_df.keyword)

    metrics, history = {}, []
    # B0: TF-IDF fitted on train texts only.
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
    tfidf.fit(pd.concat([cvs[cvs.split == "train"].text, jobs[jobs.split == "train"].text]))
    tv = lambda s: tfidf.transform(s).toarray()
    metrics["B0_tfidf"] = retrieval_metrics(tv(te_cv.text), tv(te_job.text), te_cv.keyword, te_job.keyword)

    model = SentenceTransformer(cfg["model"], device="cpu")
    metrics["B1_minilm_frozen"] = evaluate(model, te_cv, te_job)
    history.append({"epoch": 0, "train_loss": None, "val": evaluate(model, va_cv, va_job)})

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        loss = train_epoch(model, pairs, opt, cfg)
        history.append({"epoch": epoch, "train_loss": loss, "seconds": round(time.time() - t0, 1),
                        "val": evaluate(model, va_cv, va_job)})
        print(history[-1], flush=True)
    metrics["B2_quick_minilm_finetuned"] = evaluate(model, te_cv, te_job)

    model.save(str(ROOT / "models" / "b2_quick"))
    (run / "history.json").write_text(json.dumps(history, indent=2))
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"run folder: {run}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
