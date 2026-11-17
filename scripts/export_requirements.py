#!/usr/bin/env python3
"""Extract requirement IDs from the source PRD/TZ into traceability CSV files.

Sources are read-only inputs (docs/source). Statuses come ONLY from
docs/traceability/status-overrides.csv and are validated against evidence:

* VERIFIED_AUTOMATED / VERIFIED_MANUAL  -> evidence_path must exist on disk
* IMPLEMENTED_UNVERIFIED                -> implementation_files must exist
* anything without an override          -> NOT_STARTED

Usage:  python scripts/export_requirements.py [--check]
  --check   do not write files; exit 1 if overrides are invalid or the
            committed CSV is out of date.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "source"
TRACE = ROOT / "docs" / "traceability"
TZ_FILE = SRC / "TZ-HCC-001_v1.0-RC.md"
PRD_FILE = SRC / "PRD-HCC-001_v0.9.md"

STATUSES = {
    "NOT_STARTED", "IN_PROGRESS", "IMPLEMENTED_UNVERIFIED", "VERIFIED_AUTOMATED",
    "VERIFIED_MANUAL", "BLOCKED_HARDWARE", "BLOCKED_EXTERNAL", "DEFERRED",
    "REJECTED_WITH_REASON",
}

# Prefix -> responsible module (initial allocation, see docs/spec/ARCHITECTURE.md)
MODULES = {
    "SYS": "system", "OPT": "hardware/optics+apps/optical-lab", "DSP": "apps/optical-lab+apps/device-runtime",
    "MEC": "hardware/mechanical", "ELE": "hardware/electronics", "AUD": "apps/device-runtime+media",
    "APP": "apps/android", "AI": "portrait-pipeline", "RTC": "rtc", "BCK": "services/backend",
    "IDN": "services/backend+security", "DEV": "apps/device-runtime", "SEC": "security",
    "OPS": "infra/ops", "DAT": "services/backend", "NFR": "qa/measurement", "ERR": "all-clients",
    "ENG": "process/ci", "MFG": "manufacturing", "REG": "regulatory", "PM": "project-management",
    "FR-ON": "services/backend+apps/android", "FR-CL": "services/backend+clients", "FR-VI": "portrait-pipeline",
    "FR-AU": "media/audio", "FR-NW": "rtc", "FR-OP": "hardware/optics", "FR-HW": "hardware",
    "FR-UX": "clients/ux", "FR-AV": "research/avatar",
}

TZ_REQ = re.compile(
    r"^\| \*\*(?P<id>[A-Z]+-\d{3})\*\* \| (?P<prio>P0|P1|P2|EXP) (?P<phase>D[0-3]) \| (?P<text>.*?) \| (?P<check>.*?) \|\s*$"
)
PRD_FR = re.compile(
    r"^\| (?P<id>FR-[A-Z]{2}-\d{3}); (?P<phase>D[0-3]) \| (?P<prio>P0|P1|P2|EXP) \| (?P<text>.*?) \| (?P<check>.*?) \|\s*$"
)
PRD_NFR = re.compile(r"^\| (?P<id>NFR-\d{2}) / (?P<phase>D[0-3]) \| (?P<text>.*?) \| (?P<target>.*?) \| (?P<check>.*?) \|\s*$")
TZ_TEST = re.compile(r"^\| \*\*(?P<id>TC-[A-Z]+-\d{3})\*\* \| (?P<phase>D[0-3]) \| (?P<scenario>.*?) \| (?P<proc>.*?) \| (?P<crit>.*?) \|\s*$")
HEADING = re.compile(r"^(#{1,3}) (?P<title>.+?)\s*$")

REQ_COLUMNS = [
    "requirement_id", "title", "source_doc", "source_section", "priority", "phase", "module",
    "acceptance_method", "status", "implementation_files", "test_ids", "evidence_path", "owner", "notes",
]
TEST_COLUMNS = [
    "test_id", "source_test_id", "requirement_ids", "environment", "setup", "action", "expected",
    "observed", "status", "log_path", "evidence_path", "executed_at", "build_id",
]


@dataclass
class Req:
    rid: str
    title: str
    source_doc: str
    section: str
    priority: str
    phase: str
    acceptance: str
    line: int
    extra: dict = field(default_factory=dict)


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("**", "")).strip()


def module_for(rid: str) -> str:
    if rid.startswith("FR-"):
        return MODULES.get(rid[:5], "unassigned")
    return MODULES.get(rid.split("-")[0], "unassigned")


def parse(path: Path, doc: str, patterns: list[tuple[re.Pattern, str]]) -> tuple[list[Req], list[dict]]:
    reqs: list[Req] = []
    tests: list[dict] = []
    section = ""
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        h = HEADING.match(line)
        if h:
            section = clean(h.group("title"))
            continue
        for pat, kind in patterns:
            m = pat.match(line)
            if not m:
                continue
            g = m.groupdict()
            if kind == "test":
                tests.append({"source_test_id": g["id"], "phase": g["phase"], "scenario": clean(g["scenario"]),
                              "procedure": clean(g["proc"]), "criterion": clean(g["crit"]), "line": n})
            else:
                acceptance = clean(g["check"])
                if kind == "prd_nfr":
                    acceptance = f"Цель: {clean(g['target'])}; метод: {acceptance}"
                reqs.append(Req(g["id"], clean(g["text"]), doc, section, g.get("prio", "NFR-TARGET"),
                                g["phase"], acceptance, n))
            break
    return reqs, tests


def load_overrides(errors: list[str]) -> dict[str, dict]:
    path = TRACE / "status-overrides.csv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=2):
            rid = (row.get("requirement_id") or "").strip()
            if not rid or rid.startswith("#"):
                continue
            status = (row.get("status") or "").strip()
            if status not in STATUSES:
                errors.append(f"status-overrides.csv:{i}: unknown status {status!r} for {rid}")
                continue
            if rid in out:
                errors.append(f"status-overrides.csv:{i}: duplicate override for {rid}")
            ev = [p for p in (row.get("evidence_path") or "").split(";") if p.strip()]
            impl = [p for p in (row.get("implementation_files") or "").split(";") if p.strip()]
            if status.startswith("VERIFIED_"):
                if not ev:
                    errors.append(f"{rid}: {status} without evidence_path")
                for p in ev:
                    if not (ROOT / p.strip()).exists():
                        errors.append(f"{rid}: evidence not found: {p.strip()}")
            if status in {"IMPLEMENTED_UNVERIFIED", "VERIFIED_AUTOMATED", "VERIFIED_MANUAL"}:
                if not impl:
                    errors.append(f"{rid}: {status} without implementation_files")
                for p in impl:
                    if not (ROOT / p.strip()).exists():
                        errors.append(f"{rid}: implementation file not found: {p.strip()}")
            if status.startswith("BLOCKED_") or status in {"DEFERRED", "REJECTED_WITH_REASON"}:
                if not (row.get("notes") or "").strip():
                    errors.append(f"{rid}: {status} requires a reason in notes")
            out[rid] = row
    return out


def render_csv(columns: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    for f in (TZ_FILE, PRD_FILE):
        if not f.exists():
            print(f"MISSING SOURCE: {f.relative_to(ROOT)}", file=sys.stderr)
            return 2

    tz_reqs, tz_tests = parse(TZ_FILE, "TZ-HCC-001 v1.0-RC", [(TZ_REQ, "req"), (TZ_TEST, "test")])
    prd_reqs, _ = parse(PRD_FILE, "PRD-HCC-001 v0.9", [(PRD_FR, "req"), (PRD_NFR, "prd_nfr")])

    issues: list[str] = []
    errors: list[str] = []

    for label, reqs in (("TZ", tz_reqs), ("PRD", prd_reqs)):
        dup = [k for k, v in Counter(r.rid for r in reqs).items() if v > 1]
        for d in dup:
            lines = [str(r.line) for r in reqs if r.rid == d]
            issues.append(f"DUPLICATE {label} requirement ID {d} at lines {', '.join(lines)}")

    tz_ids = {r.rid for r in tz_reqs}
    prd_ids = {r.rid for r in prd_reqs}
    clash = sorted(tz_ids & prd_ids)
    for c in clash:
        issues.append(f"ID {c} present in both PRD and TZ")

    overrides = load_overrides(errors)
    all_ids = tz_ids | prd_ids
    for rid in overrides:
        if rid not in all_ids:
            errors.append(f"override for unknown requirement {rid}")

    rows = []
    for r in tz_reqs + prd_reqs:
        o = overrides.get(r.rid, {})
        rows.append({
            "requirement_id": r.rid, "title": r.title, "source_doc": r.source_doc,
            "source_section": f"{r.section} (line {r.line})", "priority": r.priority, "phase": r.phase,
            "module": module_for(r.rid), "acceptance_method": r.acceptance,
            "status": o.get("status", "NOT_STARTED").strip() or "NOT_STARTED",
            "implementation_files": o.get("implementation_files", ""), "test_ids": o.get("test_ids", ""),
            "evidence_path": o.get("evidence_path", ""), "owner": o.get("owner", "TBD"),
            "notes": o.get("notes", ""),
        })

    # Tests: TZ section 24 reuses IDs (e.g. TC-OPT-001 x6) -> derive unique IDs, keep source ID.
    seen: dict[str, int] = defaultdict(int)
    counts = Counter(t["source_test_id"] for t in tz_tests)
    test_rows = []
    for t in tz_tests:
        sid = t["source_test_id"]
        seen[sid] += 1
        tid = f"{sid}.{seen[sid]}" if counts[sid] > 1 else sid
        test_rows.append({
            "test_id": tid, "source_test_id": sid, "requirement_ids": "", "environment": t["phase"],
            "setup": t["scenario"], "action": t["procedure"], "expected": t["criterion"], "observed": "",
            "status": "NOT_RUN", "log_path": "", "evidence_path": "", "executed_at": "", "build_id": "",
        })
    for sid, n in sorted(counts.items()):
        if n > 1:
            issues.append(f"TZ section 24 reuses test ID {sid} for {n} different tests -> derived IDs {sid}.1..{sid}.{n} (pending QA approval)")

    req_csv = render_csv(REQ_COLUMNS, rows)
    test_csv = render_csv(TEST_COLUMNS, test_rows)

    by = Counter((r["phase"], r["priority"]) for r in rows)
    st = Counter(r["status"] for r in rows)
    summary = [
        f"TZ requirements: {len(tz_reqs)} (unique {len(tz_ids)}); PRD FR/NFR: {len(prd_reqs)} (unique {len(prd_ids)})",
        f"TZ acceptance test rows: {len(tz_tests)} (distinct source IDs {len(counts)})",
        "By phase/priority: " + ", ".join(f"{p} {q}={n}" for (p, q), n in sorted(by.items())),
        "By status: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())),
    ]

    p0 = [r for r in rows if r["priority"] == "P0" and r["phase"] in ("D0", "D1")]
    p0_md = ["# P0 требования этапов D0 и D1 (сгенерировано scripts/export_requirements.py)", "",
             "| ID | Этап | Модуль | Требование | Статус |", "|---|---|---|---|---|"]
    for r in sorted(p0, key=lambda r: (r["phase"], r["requirement_id"])):
        p0_md.append(f"| {r['requirement_id']} | {r['phase']} | {r['module']} | {r['title']} | {r['status']} |")
    p0_text = "\n".join(p0_md) + "\n"

    issues_md = "\n".join([
        "# Повторы ID и неоднозначности (сгенерировано автоматически + ручные заметки ниже)", "",
        *[f"- {i}" for i in issues], "",
        "## Ручные наблюдения (не исправляют источник; решения — в docs/decisions/OPEN-DECISIONS.md)", "",
        "- Коды метрик `NFR-01..NFR-22` (PRD гл. 08 и ТЗ гл. 20.1) пересекаются по форме с требованиями ТЗ `NFR-001..NFR-012`.",
        "- PRD `NFR-03` = AV-синхронизация, а в ТЗ 20.1 `NFR-03` = motion-to-photon p95, AV-синхронизация = `NFR-04`: одинаковый код, разный смысл.",
        "- В трассируемость включены только требования ТЗ (364) и FR/NFR PRD; метрики ТЗ 20.1 описаны в docs/spec/METRICS.md.",
        "",
    ])

    outputs = {
        TRACE / "requirements.csv": req_csv,
        TRACE / "tests.csv": test_csv,
        TRACE / "P0_D0_D1.md": p0_text,
        TRACE / "ID_ISSUES.md": issues_md,
    }

    for line in summary:
        print(line)
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    if args.check:
        stale = [p for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        for p in stale:
            print(f"STALE: {p.relative_to(ROOT)} (run scripts/export_requirements.py)", file=sys.stderr)
        return 1 if errors or stale else 0

    if errors:
        return 1
    TRACE.mkdir(parents=True, exist_ok=True)
    for p, text in outputs.items():
        p.write_text(text, encoding="utf-8", newline="\n")
    print(f"P0 D0+D1: {len(p0)} requirements -> {(TRACE / 'P0_D0_D1.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
