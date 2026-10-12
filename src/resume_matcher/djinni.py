"""Djinni Recruitment Dataset (English): download, filter, and split CVs and job posts.

Ground truth (D13): a CV and a job post are relevant if their "Primary Keyword" is the same.
Run: python -m resume_matcher.djinni
"""
import json
import urllib.request

import pandas as pd

from resume_matcher.download import sha256
from resume_matcher.env import ROOT

URLS = {
    "jobs": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-job-descriptions-english/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "cvs": "https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
}
RAW = ROOT / "data" / "raw" / "djinni"
PROCESSED = ROOT / "data" / "processed"
SEED = 42
MIN_WORDS = 50  # shortcut: drops very short profiles; check the effect in EDA
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in URLS.items():
        path = RAW / f"{name}.parquet"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
        manifest[name] = {"url": url, "bytes": path.stat().st_size, "sha256": sha256(path)}
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def split_by_keyword(df):
    """Stratified split: shuffle each keyword group with a fixed seed, then cut 70/15/15."""
    parts = []
    for _, g in df.groupby("keyword"):
        g = g.sample(frac=1, random_state=SEED)
        n_train, n_val = round(len(g) * SPLITS["train"]), round(len(g) * SPLITS["val"])
        parts.append(g.assign(split=["train"] * n_train + ["val"] * n_val + ["test"] * (len(g) - n_train - n_val)))
    return pd.concat(parts).sort_index()


def prepare():
    jobs = pd.read_parquet(RAW / "jobs.parquet")
    cvs = pd.read_parquet(RAW / "cvs.parquet")
    jobs = jobs[jobs["Long Description_lang"] == "en"].rename(columns={"Long Description": "text", "Primary Keyword": "keyword"})
    cvs = cvs[cvs["CV_lang"] == "en"].rename(columns={"CV": "text", "Primary Keyword": "keyword"})
    out = {}
    for name, df in (("jobs", jobs), ("cvs", cvs)):
        df = df[df["keyword"].notna() & (df["keyword"] != "Other")]
        df = df[df["text"].fillna("").str.split().str.len() >= MIN_WORDS]
        out[name] = df[["id", "keyword", "text"]].drop_duplicates("id")
    shared = set(out["jobs"]["keyword"]) & set(out["cvs"]["keyword"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df = split_by_keyword(df[df["keyword"].isin(shared)])
        df.to_parquet(PROCESSED / f"djinni_{name}.parquet")
        out[name] = df
    return out


if __name__ == "__main__":
    print(json.dumps(download(), indent=2))
    for name, df in prepare().items():
        print(name, len(df), "rows,", df["keyword"].nunique(), "keywords,", df["split"].value_counts().to_dict())
