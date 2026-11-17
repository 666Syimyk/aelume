"""Keyboard calibration edits, G0 report, CLI (C-014, B-014, B-015, PM-002, T-QA-02)."""
import csv
import json

import pytest

from hcc_lab import calibration as cal
from hcc_lab.app import apply_edit
from hcc_lab.cli import main as cli_main
from hcc_lab.report import COLUMNS, EYEBOX_PLAN, LUX_PLAN, evaluate_g0, load_measurements, write_template


# ------------------------------------------------------------------ calibration edits (C-014)
def test_edits_stay_inside_safe_bounds():
    p = cal.safe_default_profile()
    for _ in range(200):
        p, _ = apply_edit(p, "scale_up")
        p, _ = apply_edit(p, "offset_right")
        p, _ = apply_edit(p, "brightness_up")
        p, _ = apply_edit(p, "viewport_grow")
    assert p["scale"] == 4.0 and p["offset_normalized"]["x"] == 1.0 and p["content_brightness"] == 1.0
    for _ in range(400):
        p, _ = apply_edit(p, "scale_down")
        p, _ = apply_edit(p, "brightness_down")
        p, _ = apply_edit(p, "viewport_shrink")
    assert p["scale"] == 0.1 and p["content_brightness"] == 0.05 and p["viewport_normalized"]["w"] == 0.05


def test_edited_profile_validates_after_resign():
    p = cal.safe_default_profile()
    for a in ("mirror_x", "rotate", "rotate", "mode", "offset_up", "viewport_shrink", "brightness_down"):
        p, _ = apply_edit(p, a, fine=True)
    assert cal.validation_errors(cal.resign(p)) == []
    assert p["rotation_deg"] == 180 and p["mirror_x"] and p["mode"] == "four_face_pyramid"


def test_edit_does_not_mutate_input():
    p = cal.safe_default_profile()
    before = json.dumps(p, sort_keys=True)
    apply_edit(p, "offset_left")
    assert json.dumps(p, sort_keys=True) == before


def test_unknown_edit_rejected():
    with pytest.raises(ValueError):
        apply_edit(cal.safe_default_profile(), "teleport")


# ------------------------------------------------------------------ G0 report (B-014, B-015)
def _write_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})


def _row(assembly="single_reflector", photo="photo.jpg", **kw):
    base = {"experiment_id": "OPT-LAB-001", "date_utc": "2026-11-04T09:00:00Z", "operator": "tester",
            "assembly": assembly, "pattern_id": "HCC-PAT-01", "ambient_lux": "300", "observer_distance_cm": "75",
            "observer_x_cm": "0", "observer_z_cm": "0", "photo_path": photo}
    base.update(kw)
    return base


def test_no_data_is_not_evaluable(tmp_path):
    f = tmp_path / "m.csv"
    _write_rows(f, [])
    res = evaluate_g0(load_measurements([f]), None)
    assert res["recommendation"] == "NOT EVALUABLE"
    assert any("hardware evidence missing" in r for r in res["reasons"])


def test_template_example_row_is_ignored(tmp_path):
    t = write_template(tmp_path / "t.csv")
    assert load_measurements([t]) == []


def test_invalid_rows_reported_never_fixed(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"x")
    f = tmp_path / "m.csv"
    _write_rows(f, [_row(ambient_lux="bright"), _row(assembly="hologram"), _row(photo="missing.jpg"),
                    _row(ghosting_score_0_3="7")])
    rows = load_measurements([f])
    assert all(not r.valid for r in rows)
    errs = " ".join(" ".join(r.errors) for r in rows)
    assert "not a number" in errs and "assembly" in errs and "photo_path not found" in errs and "outside" in errs


def test_empty_measurement_stays_not_measured(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"x")
    f = tmp_path / "m.csv"
    _write_rows(f, [_row()])
    rows = load_measurements([f])
    assert rows[0].valid and "contrast_ratio" in rows[0].missing
    res = evaluate_g0(rows, None)
    assert res["summary"]["single_reflector"]["contrast_ratio_median"] is None
    assert res["recommendation"] == "NOT EVALUABLE"


def test_complete_grid_without_thresholds_is_still_not_evaluable(tmp_path):
    (tmp_path / "photo.jpg").write_bytes(b"x")
    rows = []
    for a in ("single_reflector", "four_face_pyramid"):
        for lux in LUX_PLAN:
            for d, x, z in EYEBOX_PLAN:
                rows.append(_row(a, ambient_lux=str(lux), observer_distance_cm=str(d), observer_x_cm=str(x),
                                 observer_z_cm=str(z), contrast_ratio="12", visible_height_mm="80",
                                 ghosting_score_0_3="1", recognizable="yes"))
    f = tmp_path / "m.csv"
    _write_rows(f, rows)
    loaded = load_measurements([f])
    res = evaluate_g0(loaded, None)
    assert res["recommendation"] == "NOT EVALUABLE" and any("TBD" in r for r in res["reasons"])
    thresholds = {"min_contrast_ratio": 10, "min_visible_height_mm": 60, "max_ghosting_score": 1,
                  "min_recognizable_fraction": 0.8, "approved_by": "PO+optics", "approved_at": "2026-11-05"}
    res = evaluate_g0(loaded, thresholds)
    assert res["recommendation"] == "GO" and "signed" in res["note"]


# ------------------------------------------------------------------ CLI
def test_cli_validate_profile_ok_and_bad(tmp_path, capsys):
    good = tmp_path / "good.json"
    cal.save_profile_atomic(good, cal.safe_default_profile())
    assert cli_main(["validate-profile", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1, "scale": NaN}', encoding="utf-8")
    assert cli_main(["validate-profile", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().out


def test_cli_report_missing_file_is_clear_error(tmp_path, capsys):
    assert cli_main(["report", "--measurements", str(tmp_path / "nope.csv"), "--out", str(tmp_path)]) == 2
    assert "not found" in capsys.readouterr().out
