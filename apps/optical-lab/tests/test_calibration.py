"""Calibration profile v1 (C-015, D-009, D-017, D-019, DSP-009, DSP-010, DEV-010, TC-DSP-001.1)."""
import json
import math

import pytest

from hcc_lab import calibration as cal
from hcc_lab.paths import PROTOCOL_DIR

EXAMPLE = PROTOCOL_DIR / "examples" / "optical-calibration.v1.example.json"
CHECKSUM_FIXTURE = PROTOCOL_DIR / "fixtures" / "calibration-checksum.v1.json"


def good_profile(**over):
    p = cal.safe_default_profile("dev_test_01", "display_test")
    p.update(over)
    p["checksum"] = cal.compute_checksum(p)
    return p


def test_safe_default_is_valid_dim_and_identity():
    p = cal.safe_default_profile()
    assert cal.validation_errors(p) == []
    assert p["content_brightness"] < 1.0
    assert (p["mirror_x"], p["mirror_y"], p["rotation_deg"], p["scale"]) == (False, False, 0, 1.0)


def test_example_from_protocol_package_is_valid():
    profile = cal.parse_profile_text(EXAMPLE.read_text(encoding="utf-8"))
    assert cal.validation_errors(profile) == []


def test_checksum_matches_cross_language_fixture():
    # Same fixture is checked by the TypeScript device runtime (contract test).
    fx = json.loads(CHECKSUM_FIXTURE.read_text(encoding="utf-8"))
    assert cal.canonical_json(fx["profile_without_checksum"]) == fx["canonical_json"]
    assert cal.compute_checksum(fx["profile_without_checksum"]) == fx["checksum"]


def test_canonical_json_number_forms_match_ecmascript():
    assert cal.canonical_json({"b": 1.0, "a": 0.05, "c": [True, None], "d": 1e-7, "e": 1e21}) == \
        '{"a":0.05,"b":1,"c":[true,null],"d":1e-7,"e":1e+21}'


def test_edit_without_resign_is_detected():
    p = good_profile()
    p["scale"] = 2.0
    assert any("checksum" in e for e in cal.validation_errors(p))
    assert cal.validation_errors(cal.resign(p)) == []


@pytest.mark.parametrize("field,value", [
    ("rotation_deg", 45), ("scale", 0), ("scale", 9.0), ("mode", "hologram_360"),
    ("mirror_x", "yes"), ("schema_version", 2), ("device_id", ""), ("unknown_field", 1),
])
def test_invalid_values_rejected(field, value):
    p = good_profile(**{field: value})
    assert cal.validation_errors(p), f"{field}={value!r} should be rejected"


def test_viewport_outside_unit_square_rejected():
    p = good_profile(viewport_normalized={"x": 0.5, "y": 0.0, "w": 0.6, "h": 1.0})
    assert any("x + w" in e for e in cal.validation_errors(p))


def test_zero_size_viewport_rejected():
    p = good_profile(viewport_normalized={"x": 0.0, "y": 0.0, "w": 0.0, "h": 1.0})
    assert cal.validation_errors(p)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nan_and_infinity_rejected_at_parse(token):
    text = json.dumps(good_profile()).replace('"scale": 1.0', f'"scale": {token}')
    with pytest.raises(cal.ProfileError):
        cal.parse_profile_text(text)


def test_nan_inside_python_object_rejected():
    p = good_profile()
    p["scale"] = math.nan
    assert cal.validation_errors(p)


def test_unsupported_schema_version_rejected():
    with pytest.raises(cal.ProfileError, match="unsupported schema_version"):
        cal.parse_profile_text(json.dumps({"schema_version": 7}))


def test_v0_prd_draft_is_migrated():
    v0 = {"schema_version": 0, "device_id": "dev_a", "optical_revision": "opt_rev_A",
          "display_width": 1920, "display_height": 1080, "region_x": 192, "region_y": 108,
          "region_w": 960, "region_h": 540, "horizontal_flip": True, "vertical_flip": False,
          "rotation_deg": 0, "scale": 1.0, "offset_x": 96, "offset_y": 0, "brightness_limit": 0.7,
          "last_calibrated_at": "2026-10-31T06:00:00Z"}
    p = cal.parse_profile_text(json.dumps(v0))
    assert cal.validation_errors(p) == []
    assert p["mirror_x"] is True
    assert p["viewport_normalized"] == {"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5}
    assert p["offset_normalized"]["x"] == pytest.approx(0.1)


def test_atomic_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "optical-calibration.v1.json"
    p = good_profile(mirror_x=True)
    cal.save_profile_atomic(path, p)
    r = cal.load_profile(path)
    assert r.source == "primary" and r.profile == p and not r.degraded
    assert not list(tmp_path.glob(".*.tmp-*")), "temp file must not remain"


def test_save_refuses_invalid_profile(tmp_path):
    p = good_profile()
    p["scale"] = -1
    with pytest.raises(cal.ProfileError):
        cal.save_profile_atomic(tmp_path / "p.json", p)
    assert not (tmp_path / "p.json").exists()


def test_corrupted_primary_falls_back_to_last_valid(tmp_path):
    # Simulates power loss that left a truncated primary file (DSP-009, TC-DSP-001.1).
    path = tmp_path / "optical-calibration.v1.json"
    first = good_profile(scale=1.5)
    cal.save_profile_atomic(path, first)
    second = cal.resign(first)
    cal.save_profile_atomic(path, second)  # first becomes *.last-valid.json
    path.write_text(path.read_text(encoding="utf-8")[:40], encoding="utf-8")
    r = cal.load_profile(path)
    assert r.source == "last_valid" and r.error_code == cal.ERROR_CODE
    assert r.profile["scale"] == 1.5


def test_wrong_checksum_and_no_backup_gives_safe_default(tmp_path):
    path = tmp_path / "p.json"
    p = good_profile()
    p["checksum"] = "sha256:" + "0" * 64
    path.write_text(json.dumps(p), encoding="utf-8")
    r = cal.load_profile(path, device_id="dev_x")
    assert r.source == "safe_default" and r.error_code == cal.ERROR_CODE
    assert r.profile["content_brightness"] < 1.0  # never a full-brightness white screen
    assert any("checksum" in w for w in r.warnings)


def test_missing_profile_uses_safe_default_without_error_code(tmp_path):
    r = cal.load_profile(tmp_path / "absent.json")
    assert r.source == "safe_default" and r.error_code is None


def test_corrupted_primary_is_never_promoted_to_last_valid(tmp_path):
    path = tmp_path / "p.json"
    path.write_text("{not json", encoding="utf-8")
    cal.save_profile_atomic(path, good_profile())
    assert not cal.last_valid_path(path).exists()


def test_interrupted_temp_write_does_not_touch_primary(tmp_path):
    path = tmp_path / "p.json"
    p = good_profile(scale=2.0)
    cal.save_profile_atomic(path, p)
    (tmp_path / ".p.json.tmp-999").write_text('{"schema_version": 1, "trunc', encoding="utf-8")
    assert cal.load_profile(path).profile == p


def test_binding_warnings_for_other_display():
    p = good_profile(display_profile_id="display_proto_01")
    assert cal.binding_warnings(p, display_profile_id="display_other")
    assert cal.binding_warnings(p, display_profile_id="display_proto_01") == []


def test_profile_to_geometry():
    g = cal.profile_to_geometry(good_profile(mirror_y=True, rotation_deg=90))
    assert g.mirror_y and g.rotation_deg == 90 and g.viewport.w == 1.0
