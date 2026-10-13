"""CV-to-job matching for the app. Prototype: support thresholds per model come from
python -m resume_matcher.calibrate (configs/support_thresholds.json).

Output follows the contract in web/src/api.ts (D9, D10): results[i] belongs to jd_texts[i].
"""
import json
import re
from pathlib import Path

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLDS = Path(__file__).resolve().parents[2] / "configs" / "support_thresholds.json"
# shortcut: fallback picked by eye on the made-up UI example, for models without calibration.
SUPPORT_THRESHOLD = 0.4
LEVELS = [(0.6, "high"), (0.35, "medium")]  # fraction of supported requirements -> level


def load_threshold(model_id):
    try:
        return json.loads(THRESHOLDS.read_text())[model_id]["threshold"]
    except (FileNotFoundError, KeyError):
        print(f"No calibrated threshold for {model_id}, using {SUPPORT_THRESHOLD}")
        return SUPPORT_THRESHOLD


def is_heading(line):
    """A section heading, found by shape instead of a word list, so 'EXPERIENCE',
    'Work Experience', and 'Professional Experience' all count: at most 5 words, no digits or
    punctuation, and ALL CAPS or Title Case.
    shortcut: a CV line with only skill names and no commas ('Python', 'Git and GitHub') also
    counts as a heading. Use a layout-aware PDF parser if this loses real evidence."""
    words = line.split()
    if not words or len(words) > 5 or re.search(r"[\d.,;:!?()|@/+#]", line):
        return False
    return line.isupper() or all(w[0].isupper() or w.lower() in {"and", "of", "&"} for w in words)


def split_lines(text):
    """Split text into lines or sentences, without bullets, numbering, headings ending in ':',
    or section headings. A line that starts with a bullet is never a heading."""
    out = []
    for raw in re.split(r"\n|(?<=[.!?])\s+", text):
        line = re.sub(r"^[\s\-*•●·\d.)]+", "", raw).strip()
        bulleted = line != raw.strip()
        if len(line) > 3 and not line.endswith(":") and (bulleted or not is_heading(line)):
            out.append(line)
    return out


def match(cv_text, jd_texts, encode, threshold=SUPPORT_THRESHOLD):
    """encode(list[str]) -> L2-normalized embeddings, one row per string."""
    cv_lines = split_lines(cv_text)
    if not cv_lines:
        raise ValueError("The CV has no text.")
    cv_emb = encode(cv_lines)
    cv_vec = cv_emb.mean(0)
    cv_vec /= np.linalg.norm(cv_vec)
    results = []
    for jd in jd_texts:
        reqs = split_lines(jd)
        if not reqs:
            results.append({"score": 0.0, "match_level": "low", "requirements": []})
            continue
        req_emb = encode(reqs)
        sims = req_emb @ cv_emb.T  # cosine: rows are requirements, columns are CV lines
        best = sims.argmax(1)
        requirements = [
            {"text": r, "supported": bool(sims[i, best[i]] >= threshold),
             "evidence": cv_lines[best[i]] if sims[i, best[i]] >= threshold else None}
            for i, r in enumerate(reqs)
        ]
        jd_vec = req_emb.mean(0)
        score = float(jd_vec @ cv_vec / np.linalg.norm(jd_vec))
        share = sum(r["supported"] for r in requirements) / len(requirements)
        level = next((name for cut, name in LEVELS if share >= cut), "low")
        results.append({"score": score, "match_level": level, "requirements": requirements})
    return results
