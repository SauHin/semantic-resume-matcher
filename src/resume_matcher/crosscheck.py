"""Quick check: does a Djinni-trained model still work on formal CVs (Kaggle resumes)?

Same Djinni test job pool and the same relevance groups for both CV sources, so the gap
between them shows the domain shift. Run: python -m resume_matcher.crosscheck
"""
import json

import pandas as pd
from sentence_transformers import SentenceTransformer

from resume_matcher.data import load_resumes
from resume_matcher.djinni import PROCESSED
from resume_matcher.env import ROOT, new_run_dir
from resume_matcher.train_quick import CONFIG, encode, retrieval_metrics, sample

# shortcut: assumed mapping for this check only, not the owner's ground truth (D13 is open).
GROUPS = {"HR": ["HR", "Recruiter"], "SALES": ["Sales", "Lead Generation"], "DESIGNER": ["Design"]}
KW_TO_GROUP = {kw: g for g, kws in GROUPS.items() for kw in kws}


def main():
    jobs = sample(pd.read_parquet(PROCESSED / "djinni_jobs.parquet"), "test", CONFIG["n_test_jobs"], 2)
    job_group = jobs.keyword.map(KW_TO_GROUP).fillna("other")
    kaggle = load_resumes()
    kaggle = kaggle[kaggle.Category.isin(GROUPS)]
    djinni = pd.read_parquet(PROCESSED / "djinni_cvs.parquet")
    djinni = djinni[(djinni.split == "test") & djinni.keyword.isin(KW_TO_GROUP)]
    djinni = djinni.sample(n=min(len(djinni), len(kaggle)), random_state=3)
    # EDA: most Kaggle resumes in these categories name their field in the first 10 words.
    no_title = kaggle.Resume_str.str.split().str[10:].str.join(" ")
    sources = {"kaggle_formal_cvs": (kaggle.Resume_str, kaggle.Category),
               "kaggle_formal_cvs_no_title": (no_title, kaggle.Category),
               "djinni_profiles": (djinni.text, djinni.keyword.map(KW_TO_GROUP))}

    out = {"groups": GROUPS, "n_jobs": len(jobs),
           "random_P@10": {g: float((job_group == g).mean()) for g in GROUPS}}
    for name, path in [("B1_minilm_frozen", CONFIG["model"]), ("B2_quick", str(ROOT / "models" / "b2_quick"))]:
        model = SentenceTransformer(path, device="cpu")
        job_vecs = encode(model, jobs.text)
        for src, (texts, groups) in sources.items():
            m = retrieval_metrics(encode(model, texts), job_vecs, groups, job_group)
            out[f"{name}/{src}"] = {**m, "n_cvs": len(texts)}
            print(name, src, out[f"{name}/{src}"], flush=True)
    run = new_run_dir("crosscheck-formal-cvs")
    (run / "metrics.json").write_text(json.dumps(out, indent=2))
    print(f"run folder: {run}")


if __name__ == "__main__":
    main()
