"""Download the two Kaggle datasets as zip files to ROOT/data/raw and write MANIFEST.json.

Needs the KAGGLE_API_TOKEN environment variable. In Colab, set it from Colab Secrets and
never print it. Raw zips are never overwritten: delete a zip by hand to download it again.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone

from resume_matcher.env import ROOT, _run

DATASETS = {
    "resumes": "snehaanbhawal/resume-dataset",
    "jobs": "shivamb/real-or-fake-fake-jobposting-prediction",
}
RAW = ROOT / "data" / "raw"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    cli = _run(["kaggle", "--version"]) or ""
    manifest = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kaggle_cli": cli.splitlines()[-1] if cli else None,  # last line; earlier lines are upgrade warnings
        "datasets": {},
    }
    for name, slug in DATASETS.items():
        zip_path = RAW / f"{slug.split('/')[1]}.zip"
        if not zip_path.exists():
            subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(RAW)], check=True)
        manifest["datasets"][name] = {
            "slug": slug,
            "file": zip_path.name,
            "bytes": zip_path.stat().st_size,
            "sha256": sha256(zip_path),
            # Kaggle sets the file time to the dataset's last update, not the download time.
            "kaggle_updated_utc": datetime.fromtimestamp(zip_path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        }
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
