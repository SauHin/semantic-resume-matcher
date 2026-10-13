import numpy as np

from resume_matcher.match import match, split_lines


def test_split_lines_drops_bullets_and_headings():
    assert split_lines("Requirements:\n- Python 3 years\n2) SQL skills. Git basics.") == [
        "Python 3 years", "SQL skills.", "Git basics."]


def test_split_lines_joins_lines_a_pdf_wrapped_mid_sentence():
    text = ("I collected 60,283 tweets, labelled them with rules \nand by hand. Off the shelf and fine-\n"
            "tuned models.\nTools: pandas, PyTorch,\nUMAP, Git\n- Python\n- SQL\n3 years of Go.")
    assert split_lines(text) == [
        "I collected 60,283 tweets, labelled them with rules and by hand.",
        "Off the shelf and fine-tuned models.", "Tools: pandas, PyTorch, UMAP, Git", "Python", "SQL",
        "3 years of Go."]


def test_split_lines_drops_section_headings_of_any_wording_but_keeps_bulleted_skills():
    text = ("WORK EXPERIENCE\nProfessional Experience\nProjects and Research\nTECHNICAL SKILLS\n"
            "Languages: Python, SQL, C\nBuilt a REST API with Flask.\n- Docker\nPeer Mentor, BINUS 2025")
    assert split_lines(text) == [
        "Languages: Python, SQL, C", "Built a REST API with Flask.", "Docker", "Peer Mentor, BINUS 2025"]


def fake_encode(texts):
    # One-hot on the first word: lines that start with the same word count as identical.
    vocab = ["python", "sql", "java"]
    return np.array([[1.0 if t.split()[0].lower() == w else 0.0 for w in vocab] for t in texts])


def test_match_marks_supported_requirements_with_evidence():
    [res] = match("Python developer for 2 years.\nSQL reports weekly.", ["Python required.\nJava required."], fake_encode)
    python, java = res["requirements"]
    assert python == {"text": "Python required.", "supported": True, "evidence": "Python developer for 2 years."}
    assert java["supported"] is False and java["evidence"] is None
    assert res["match_level"] == "medium"  # 1 of 2 supported
