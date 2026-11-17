"""Environment manifest for every report (BOOT-004, master prompt section 13)."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from . import __version__


def _pkg(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_id() -> str:
    """git commit if the repo is under git, else 'no-git'."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=os.path.dirname(__file__))
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "no-git"


def manifest() -> dict:
    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hcc_lab": __version__,
        "build_id": build_id(),
        "python": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "packages": {p: _pkg(p) for p in ("numpy", "opencv-contrib-python", "opencv-python", "mediapipe", "jsonschema")},
    }
