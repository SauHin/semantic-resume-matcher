import pandas as pd
import pytest

from resume_matcher.eda import duplicate_counts, mentions_category, normalize


def test_normalize():
    assert normalize("  Hello\n\tWORLD  ") == "hello world"


def test_duplicate_counts_ignores_case_spaces_and_missing():
    texts = pd.Series(["Data  analyst", "data analyst", "Chef", None, None])
    assert duplicate_counts(texts).to_dict() == {"data analyst": 2}


@pytest.mark.parametrize("text, category, expected", [
    ("HR ADMINISTRATOR with 5 years", "HR", True),
    ("Information technology manager", "Information-Technology", True),
    ("INFORMATION-TECHNOLOGY manager", "Information-Technology", True),
    ("Teacher at a public school", "Advocate", False),
    ("Worked THROUGH many projects", "HR", False),
    ("one two three four five six seven eight nine ten Chef", "Chef", False),
])
def test_mentions_category(text, category, expected):
    assert mentions_category(text, category) is expected
