"""Optical measurement report and G0 package (B-011, B-014, B-015, PM-002, TZ 23.2).

Only values typed in by a person are used. Missing values stay "NOT MEASURED";
nothing is interpolated, estimated or synthesized.
"""
from __future__ import annotations

import csv
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import REPO_ROOT

ASSEMBLIES = ("single_reflector", "four_face_pyramid")
LUX_PLAN = (100, 300, 500, 1000)
EYEBOX_PLAN = [(d, x, z) for d in (50, 75, 100) for x in (-20, 0, 20) for z in (-10, 0, 10)]  # cm, TZ 23.2
COLUMNS = [
    "experiment_id", "date_utc", "operator", "assembly", "material_id", "material_lot", "reflector_angle_deg",
    "display_profile_id", "profile_checksum", "pattern_id", "pattern_sha256", "ambient_lux", "light_direction",
    "observer_distance_cm", "observer_x_cm", "observer_z_cm", "camera_model", "iso", "shutter_s",
    "white_balance_k", "visible_height_mm", "ghosting_score_0_3", "contrast_ratio", "recognizable",
    "photo_path", "notes",
]
REQUIRED = ("experiment_id", "date_utc", "operator", "assembly", "pattern_id", "ambient_lux",
            "observer_distance_cm", "photo_path")
NUMERIC = {"reflector_angle_deg": (0, 90), "ambient_lux": (0, 200000), "observer_distance_cm": (1, 1000),
           "observer_x_cm": (-500, 500), "observer_z_cm": (-500, 500), "iso": (1, 409600), "shutter_s": (0, 60),
           "white_balance_k": (1000, 20000), "visible_height_mm": (0, 2000), "ghosting_score_0_3": (0, 3),
           "contrast_ratio": (0, 1e6)}
NOT_MEASURED = "NOT MEASURED"


@dataclass
class Row:
    line: int
    data: dict
    errors: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors


def _num(v: str) -> float | None:
    v = (v or "").strip().replace(",", ".")
    if v == "" or v.upper() == NOT_MEASURED:
        return None
    return float(v)


def load_measurements(paths: list[Path]) -> list[Row]:
    rows: list[Row] = []
    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            unknown = set(reader.fieldnames or []) - set(COLUMNS)
            for n, raw in enumerate(reader, start=2):
                if not any((v or "").strip() for v in raw.values()):
                    continue
                if (raw.get("experiment_id") or "").startswith("#"):
                    continue  # comment / example line
                r = Row(n, {k: (raw.get(k) or "").strip() for k in COLUMNS})
                if unknown:
                    r.errors.append(f"unknown columns {sorted(unknown)}")
                for k in REQUIRED:
                    if not r.data[k]:
                        r.errors.append(f"{k} is required")
                if r.data["assembly"] and r.data["assembly"] not in ASSEMBLIES:
                    r.errors.append(f"assembly must be one of {ASSEMBLIES}")
                if r.data["recognizable"] and r.data["recognizable"] not in ("yes", "no", "unknown"):
                    r.errors.append("recognizable must be yes/no/unknown")
                parsed = {}
                for k, (lo, hi) in NUMERIC.items():
                    try:
                        val = _num(r.data[k])
                    except ValueError:
                        r.errors.append(f"{k}={r.data[k]!r} is not a number")
                        continue
                    if val is None:
                        r.missing.append(k)
                    elif not lo <= val <= hi:
                        r.errors.append(f"{k}={val} outside [{lo}, {hi}]")
                    parsed[k] = val
                r.data["_num"] = parsed
                photo = r.data["photo_path"]
                if photo:
                    p = Path(photo)
                    if not (p if p.is_absolute() else (path.parent / p)).exists() and not (REPO_ROOT / p).exists():
                        r.errors.append(f"photo_path not found: {photo} (raw evidence is mandatory)")
                rows.append(r)
    return rows


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def summarize(rows: list[Row]) -> dict:
    out = {}
    for a in ASSEMBLIES:
        good = [r for r in rows if r.valid and r.data["assembly"] == a]
        lux = sorted({int(r.data["_num"]["ambient_lux"]) for r in good if r.data["_num"].get("ambient_lux") is not None})
        pos = {(r.data["_num"].get("observer_distance_cm"), r.data["_num"].get("observer_x_cm"),
                r.data["_num"].get("observer_z_cm")) for r in good}
        covered = [p for p in EYEBOX_PLAN if p in pos]
        rec = [r.data["recognizable"] for r in good if r.data["recognizable"] in ("yes", "no")]
        out[a] = {
            "valid_rows": len(good),
            "lux_levels_measured": lux,
            "lux_plan_missing": [x for x in LUX_PLAN if x not in lux],
            "eyebox_positions_covered": len(covered),
            "eyebox_positions_planned": len(EYEBOX_PLAN),
            "visible_height_mm_median": _median(r.data["_num"].get("visible_height_mm") for r in good),
            "contrast_ratio_median": _median(r.data["_num"].get("contrast_ratio") for r in good),
            "ghosting_score_max": max((r.data["_num"]["ghosting_score_0_3"] for r in good
                                       if r.data["_num"].get("ghosting_score_0_3") is not None), default=None),
            "recognizable_fraction": (rec.count("yes") / len(rec)) if rec else None,
        }
    return out


def evaluate_g0(rows: list[Row], thresholds: dict | None) -> dict:
    """Recommendation for the human G0 decision (PO + optics engineer sign it, PRD 20.1)."""
    summary = summarize(rows)
    reasons = []
    invalid = [r for r in rows if not r.valid]
    if not rows:
        reasons.append("G0 BLOCKED: hardware evidence missing (no measurement rows)")
    for a, s in summary.items():
        if s["valid_rows"] == 0:
            reasons.append(f"no valid measurements for {a}")
        else:
            if s["lux_plan_missing"]:
                reasons.append(f"{a}: lux levels not measured: {s['lux_plan_missing']}")
            if s["eyebox_positions_covered"] < s["eyebox_positions_planned"]:
                reasons.append(f"{a}: eye-box grid {s['eyebox_positions_covered']}/{s['eyebox_positions_planned']} positions")
    if thresholds is None:
        reasons.append("G0 thresholds are TBD (NFR-16): PO + optics engineer must approve docs/lab/g0-thresholds.json")
    if reasons:
        return {"recommendation": "NOT EVALUABLE", "reasons": reasons, "summary": summary,
                "invalid_rows": [{"line": r.line, "errors": r.errors} for r in invalid]}
    results = {}
    for a, s in summary.items():
        checks = {
            "contrast": s["contrast_ratio_median"] is not None and s["contrast_ratio_median"] >= thresholds["min_contrast_ratio"],
            "height": s["visible_height_mm_median"] is not None and s["visible_height_mm_median"] >= thresholds["min_visible_height_mm"],
            "ghosting": s["ghosting_score_max"] is not None and s["ghosting_score_max"] <= thresholds["max_ghosting_score"],
            "recognizable": s["recognizable_fraction"] is not None and s["recognizable_fraction"] >= thresholds["min_recognizable_fraction"],
        }
        results[a] = checks
    passing = [a for a, c in results.items() if all(c.values())]
    if passing:
        rec = "GO"
    elif any(sum(c.values()) >= 2 for c in results.values()):
        rec = "REDESIGN"
    else:
        rec = "STOP"
    return {"recommendation": rec, "passing_assemblies": passing, "checks": results, "summary": summary,
            "invalid_rows": [{"line": r.line, "errors": r.errors} for r in invalid],
            "note": "Recommendation only. G0 is decided and signed by PO + optics engineer (PRD 20.1, TZ 29.1)."}


def render_markdown(result: dict, sources: list[Path]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"# G0 optical evidence report — {ts}", "",
             f"**Recommendation:** `{result['recommendation']}`", "",
             "Sources: " + (", ".join(f"`{p}`" for p in sources) or "none"), ""]
    if result.get("reasons"):
        lines += ["## Why not evaluable", ""] + [f"- {r}" for r in result["reasons"]] + [""]
    lines += ["## Summary per assembly (only measured values)", "",
              "| Assembly | valid rows | lux measured | eye-box | height mm (median) | contrast (median) | ghosting max | recognizable |",
              "|---|---|---|---|---|---|---|---|"]
    for a, s in result["summary"].items():
        f = lambda v: NOT_MEASURED if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {a} | {s['valid_rows']} | {s['lux_levels_measured'] or NOT_MEASURED} | "
                     f"{s['eyebox_positions_covered']}/{s['eyebox_positions_planned']} | {f(s['visible_height_mm_median'])} | "
                     f"{f(s['contrast_ratio_median'])} | {f(s['ghosting_score_max'])} | {f(s['recognizable_fraction'])} |")
    if result.get("invalid_rows"):
        lines += ["", "## Rejected rows", ""] + [f"- line {r['line']}: {'; '.join(r['errors'])}" for r in result["invalid_rows"]]
    lines += ["", "> Screenshots/simulations are not optical evidence (master prompt rule 16).", ""]
    return "\n".join(lines)


def write_template(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        w.writerow(["# example (ignored): OPT-LAB-001", "2026-11-04T09:00:00Z", "name", "single_reflector", "glass_A",
                    "lot_1", "45", "win:DISPLAY1:1920x1080", "sha256:...", "HCC-PAT-01", "<sha256>", "300",
                    "ceiling, behind observer", "75", "0", "0", "<camera>", "200", "0.0167", "5000", "",
                    "", "", "unknown", "docs/lab/photos/OPT-LAB-001.jpg", "fill real values only"])
    return path


def load_thresholds(path: Path | None) -> dict | None:
    if path is None or not path.exists():
        return None
    t = json.loads(path.read_text(encoding="utf-8"))
    required = {"min_contrast_ratio", "min_visible_height_mm", "max_ghosting_score", "min_recognizable_fraction",
                "approved_by", "approved_at"}
    missing = required - set(t)
    if missing:
        raise ValueError(f"thresholds file misses {sorted(missing)}")
    return t
