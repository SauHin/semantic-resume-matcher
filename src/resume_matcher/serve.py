"""Local backend for the web UI: POST /api/match with {cv_text, jd_texts}.

Run: python -m resume_matcher.serve   (the Vite dev server proxies /api to port 8000)
Set RM_MODEL to a model folder, for example models/b2_quick, to serve a fine-tuned model.
CVs stay in memory only. Nothing from the request body is stored or logged.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer

from resume_matcher.match import MODEL_ID, load_threshold, match

MAX_BODY = 1_000_000
MAX_JOBS = 10  # same cap as web/src/api.ts
MODEL = os.environ.get("RM_MODEL", MODEL_ID)
model = SentenceTransformer(MODEL, device="cpu")
THRESHOLD = load_threshold(MODEL)


def encode(texts):
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/api/match":
            return self.reply(404, {"error": "not found"})
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_BODY:
            return self.reply(413, {"error": "request too large"})
        try:
            req = json.loads(self.rfile.read(size))
            cv, jds = req["cv_text"], req["jd_texts"]
            if not isinstance(cv, str) or not isinstance(jds, list) or not all(isinstance(j, str) for j in jds):
                raise ValueError("cv_text must be text and jd_texts a list of texts.")
            if not 1 <= len(jds) <= MAX_JOBS:
                raise ValueError(f"Send 1 to {MAX_JOBS} job posts.")
            results = match(cv, jds, encode, THRESHOLD)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            return self.reply(400, {"error": str(e)})
        self.reply(200, {"mock": False, "model": f"prototype: {MODEL} (uncalibrated)", "results": results})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print("Backend ready on http://127.0.0.1:8000/api/match")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
