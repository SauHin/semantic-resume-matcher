"""Milestone 2 EDA. Writes tables and figures to a new run folder on Drive.

Run in Colab: python -m resume_matcher.eda
"""
import json
import re  # noqa: F401  (for the owner tasks below)

import matplotlib.pyplot as plt
import pandas as pd

from resume_matcher.data import load_jobs, load_resumes
from resume_matcher.env import new_run_dir

JOB_TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]


# ---------------------------------------------------------------------------
# Owner tasks. Make tests/test_eda.py pass, then run the EDA again in Colab.
# ---------------------------------------------------------------------------

def normalize(text):
    """Return text in lowercase, with every run of whitespace replaced by one space,
    and no space at the start or end.

    Example: "  Hello\\n\\tWORLD  " -> "hello world"

    Why: two postings that differ only in spaces or capital letters are the same posting.
    Hint: str.lower(), str.strip(), and re.sub with the pattern r"\\s+".
    """
    raise NotImplementedError("TODO(owner)")


def duplicate_counts(texts):
    """Return a pd.Series that maps each normalized text that occurs more than once to its
    number of occurrences. Ignore missing texts (NaN or None).

    Example: ["Data  analyst", "data analyst", "Chef", None, None] -> {"data analyst": 2}

    Why: if the same text lands in train and in test, the test score is too optimistic
    (data leakage). We must know how many duplicates exist before the split.
    Hint: Series.dropna(), Series.map(normalize), Series.value_counts(), then keep counts > 1.
    """
    raise NotImplementedError("TODO(owner)")


def mentions_category(text, category, first_n_words=10):
    """Return True if the category name occurs as whole words in the first `first_n_words`
    words of the resume text. Ignore case, and read "-" in the category as a space.

    Examples:
        ("HR ADMINISTRATOR with 5 years", "HR") -> True
        ("Information technology manager", "Information-Technology") -> True
        ("Worked THROUGH many projects", "HR") -> False  ("hr" inside "through" does not count)

    Why: if a resume starts with its own category name, a model can score well by reading
    the title alone, without understanding the rest of the CV (title leakage).
    Hint: text.split()[:first_n_words], " ".join(...), re.search with r"\\b" and re.escape().
    """
    raise NotImplementedError("TODO(owner)")


# ---------------------------------------------------------------------------

def word_counts(texts):
    return texts.fillna("").str.split().str.len()


def save_barh(counts, path, title):
    ax = counts.sort_values().plot.barh(figsize=(8, max(3, 0.3 * len(counts))), title=title)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=120)
    plt.close(ax.figure)


def main():
    run = new_run_dir("eda")
    resumes, jobs = load_resumes(), load_jobs()
    real = jobs[jobs["fraudulent"] == 0]  # PLAN §4: fraudulent postings are dropped
    summary = {
        "resumes": len(resumes),
        "jobs": len(jobs),
        "jobs_fraudulent": int(jobs["fraudulent"].sum()),
        "jobs_real": len(real),
    }

    cats = resumes["Category"].value_counts()
    cats.to_csv(run / "resume_category_counts.csv")
    save_barh(cats, run / "resume_category_counts.png", "Resumes per category")

    # Input for the owner's category mapping. NaN rows are counted too.
    for col in ["function", "industry"]:
        real[col].value_counts(dropna=False).to_csv(run / f"job_{col}_counts.csv")

    missing = pd.concat({"resumes": resumes.isna().mean(), "jobs_real": real.isna().mean()})
    missing.rename("missing_fraction").to_csv(run / "missing_values.csv")

    lengths = {"resume": word_counts(resumes["Resume_str"])}
    lengths |= {f"job_{c}": word_counts(real[c]) for c in JOB_TEXT_COLS}
    stats = pd.DataFrame({k: v.describe(percentiles=[0.5, 0.9, 0.99]) for k, v in lengths.items()}).T
    stats.to_csv(run / "text_length_words.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ["resume", "job_description"]):
        lengths[key].plot.hist(bins=50, ax=ax, title=f"{key}: words per text")
    fig.tight_layout()
    fig.savefig(run / "text_length_words.png", dpi=120)
    plt.close(fig)

    try:
        dup = {}
        for name, texts in [("resumes", resumes["Resume_str"]), ("job_descriptions", real["description"])]:
            counts = duplicate_counts(texts)
            dup[name] = {"groups": len(counts), "rows_in_groups": int(counts.sum())}
        summary["duplicates"] = dup
    except NotImplementedError:
        print("Owner task not done yet: duplicate check skipped.")
    try:
        leak = resumes.apply(lambda r: mentions_category(r["Resume_str"], r["Category"]), axis=1)
        leak.groupby(resumes["Category"]).mean().rename("leak_rate").to_csv(run / "title_leak_by_category.csv")
        summary["title_leak_rate"] = float(leak.mean())
    except NotImplementedError:
        print("Owner task not done yet: title leakage check skipped.")

    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"run folder: {run}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
