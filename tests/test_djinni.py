import numpy as np
import pandas as pd

from resume_matcher.djinni import split_by_keyword
from resume_matcher.train_quick import retrieval_metrics


def test_split_puts_each_id_in_one_split_and_keeps_keywords_in_all_splits():
    df = pd.DataFrame({"id": [f"x{i}" for i in range(40)], "keyword": ["Python"] * 20 + ["QA"] * 20})
    out = split_by_keyword(df)
    assert out.groupby("id")["split"].nunique().max() == 1
    assert len(out) == 40
    assert (out.groupby("keyword")["split"].nunique() == 3).all()


def test_retrieval_metrics_by_hand():
    cv = np.array([[1.0, 0.0]])
    jobs = np.array([[0.0, 1.0], [1.0, 0.0], [0.9, 0.1]])  # ranking: job 1, job 2, job 0
    m = retrieval_metrics(cv, jobs, ["Python"], ["Python", "QA", "Python"], k=2)
    assert m == {"P@2": 0.5, "MRR": 0.5}  # top 2 = QA, Python; first relevant at rank 2
