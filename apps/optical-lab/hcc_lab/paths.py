"""Repository-relative locations (override with HCC_REPO_ROOT)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(os.environ.get("HCC_REPO_ROOT") or Path(__file__).resolve().parents[3])
PROTOCOL_DIR = REPO_ROOT / "packages" / "protocol"
FIXTURES_DIR = REPO_ROOT / "packages" / "test-fixtures"
PATTERNS_DIR = FIXTURES_DIR / "patterns"
REPORTS_DIR = REPO_ROOT / "reports"
LAB_DIR = REPO_ROOT / "docs" / "lab"
LOCAL_DATA_DIR = REPO_ROOT / "apps" / "optical-lab" / "local"  # gitignored: profiles, session reports
