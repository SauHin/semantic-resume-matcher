"""Storage paths and environment records. Every run folder gets an env.json."""
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

# Project folder on Google Drive. Set RM_ROOT to use another folder.
ROOT = Path(os.environ.get("RM_ROOT", "/content/drive/MyDrive/resume-matcher"))
REPO = Path(__file__).resolve().parents[2]


def _run(cmd):
    try:
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def env_info():
    status = _run(["git", "status", "--porcelain"])
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _run(["git", "rev-parse", "--short", "HEAD"]),
        # True means the run used uncommitted code, so the sha alone does not reproduce it.
        "git_dirty": bool(status) if status is not None else None,
        "gpu": _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()),
    }


def new_run_dir(name):
    """Create ROOT/runs/<name>_<UTC time>_<git sha> and write env.json into it.

    Fails if the folder exists, so a run never overwrites another run.
    """
    info = env_info()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "runs" / f"{name}_{stamp}_{info['git_sha'] or 'nogit'}"
    run_dir.mkdir(parents=True)
    (run_dir / "env.json").write_text(json.dumps(info, indent=2))
    return run_dir


if __name__ == "__main__":
    # Setup check: write env.json to a new run folder on Drive.
    run_dir = new_run_dir("setup-check")
    info = json.loads((run_dir / "env.json").read_text())
    print(f"run folder: {run_dir}")
    for key in ("git_sha", "git_dirty", "gpu", "python"):
        print(f"{key}: {info[key]}")
