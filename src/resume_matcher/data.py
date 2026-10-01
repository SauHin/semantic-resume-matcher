"""Read the raw Kaggle CSVs straight from their zip files on Drive."""
import zipfile

import pandas as pd

from resume_matcher.download import RAW


def load_resumes():
    # Skip Resume_html: it repeats Resume_str with markup and doubles the memory use.
    with zipfile.ZipFile(RAW / "resume-dataset.zip") as z, z.open("Resume/Resume.csv") as f:
        return pd.read_csv(f, usecols=["ID", "Resume_str", "Category"])


def load_jobs():
    return pd.read_csv(RAW / "real-or-fake-fake-jobposting-prediction.zip")
