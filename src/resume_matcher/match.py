"""CV-to-job matching for the app. Prototype: frozen MiniLM (B1), thresholds not calibrated yet.

Output follows the contract in web/src/api.ts (D9, D10): results[i] belongs to jd_texts[i].
"""
import re

import numpy as np

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
# shortcut: picked by eye on the made-up UI example, not calibrated and not a result.
# PLAN §6/§7: set both from the calibration and validation data later.
SUPPORT_THRESHOLD = 0.4
LEVELS = [(0.6, "high"), (0.35, "medium")]  # fraction of supported requirements -> level


def split_lines(text):
    """Split text into lines or sentences, without bullets, numbering, or headings ending in ':'."""
    parts = re.split(r"\n|(?<=[.!?])\s+", text)
    parts = [re.sub(r"^[\s\-*•·\d.)]+", "", p).strip() for p in parts]
    return [p for p in parts if len(p) > 3 and not p.endswith(":")]


def match(cv_text, jd_texts, encode):
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
            {"text": r, "supported": bool(sims[i, best[i]] >= SUPPORT_THRESHOLD),
             "evidence": cv_lines[best[i]] if sims[i, best[i]] >= SUPPORT_THRESHOLD else None}
            for i, r in enumerate(reqs)
        ]
        jd_vec = req_emb.mean(0)
        score = float(jd_vec @ cv_vec / np.linalg.norm(jd_vec))
        share = sum(r["supported"] for r in requirements) / len(requirements)
        level = next((name for cut, name in LEVELS if share >= cut), "low")
        results.append({"score": score, "match_level": level, "requirements": requirements})
    return results
