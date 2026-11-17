"""Optical calibration profile v1: validate, checksum, atomic save, safe fallback.

Requirements: TZ 7.2, 22.4, DSP-002/004/009/010, DEV-003/010, C-015, D-009, D-017, D-019.
Schema (single source of truth): packages/protocol/schemas/optical-calibration.v1.schema.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .geometry import GeometryParams, Offset, Viewport
from .paths import PROTOCOL_DIR

SCHEMA_PATH = PROTOCOL_DIR / "schemas" / "optical-calibration.v1.schema.json"
CURRENT_SCHEMA_VERSION = 1
ERROR_CODE = "OPT_PROFILE_INVALID"
_EPS = 1e-9


class ProfileError(ValueError):
    """Profile rejected; ``errors`` lists every problem found."""

    code = ERROR_CODE

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class LoadResult:
    profile: dict[str, Any]
    source: str  # "primary" | "last_valid" | "safe_default"
    warnings: list[str] = field(default_factory=list)
    error_code: str | None = None

    @property
    def degraded(self) -> bool:
        return self.source != "primary"


# ---------------------------------------------------------------- canonical JSON
def _canonical_number(x: float | int) -> str:
    if isinstance(x, bool):  # bool is an int subclass; handled by caller
        raise TypeError("bool is not a number")
    if isinstance(x, int):
        return str(x)
    if not math.isfinite(x):
        raise ProfileError([f"non-finite number {x!r}"])
    if x.is_integer() and abs(x) < 1e21:
        return str(int(x))
    r = repr(x)
    if "e" in r:  # match ECMAScript Number#toString exponent form: 1e-7, 1e+21
        mant, exp = r.split("e")
        e = int(exp)
        r = f"{mant}e{'+' if e > 0 else '-'}{abs(e)}"
    return r


def canonical_json(obj: Any) -> str:
    """Deterministic JSON (sorted keys, no whitespace, JS-compatible numbers).

    The TypeScript runtime implements the same rules so both sides compute the
    same checksum (contract fixture: packages/protocol/fixtures/calibration-checksum.v1.json).
    """
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, (int, float)):
        return _canonical_number(obj)
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(canonical_json(v) for v in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0])
        return "{" + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{canonical_json(v)}" for k, v in items) + "}"
    raise TypeError(f"unsupported type {type(obj).__name__}")


def compute_checksum(profile: dict[str, Any]) -> str:
    body = {k: v for k, v in profile.items() if k != "checksum"}
    return "sha256:" + hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- validation
_validator: Draft202012Validator | None = None


def _schema_validator() -> Draft202012Validator:
    global _validator
    if _validator is None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        _validator = Draft202012Validator(schema)
    return _validator


def _walk_numbers(obj: Any, path: str = "$"):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_numbers(v, f"{path}[{i}]")


def validation_errors(profile: Any, *, verify_checksum: bool = True) -> list[str]:
    if not isinstance(profile, dict):
        return ["profile must be a JSON object"]
    errors = [f"{p}: non-finite number" for p, v in _walk_numbers(profile) if isinstance(v, float) and not math.isfinite(v)]
    if errors:
        return errors  # jsonschema behaviour with NaN is undefined; stop here
    for e in sorted(_schema_validator().iter_errors(profile), key=lambda e: list(e.path)):
        loc = "$" + "".join(f".{p}" if isinstance(p, str) else f"[{p}]" for p in e.path)
        errors.append(f"{loc}: {e.message}")
    vp = profile.get("viewport_normalized")
    if isinstance(vp, dict) and all(isinstance(vp.get(k), (int, float)) for k in "xywh"):
        if vp["x"] + vp["w"] > 1 + _EPS:
            errors.append("$.viewport_normalized: x + w must be <= 1")
        if vp["y"] + vp["h"] > 1 + _EPS:
            errors.append("$.viewport_normalized: y + h must be <= 1")
    if verify_checksum and not errors:
        expected = compute_checksum(profile)
        if profile.get("checksum") != expected:
            errors.append("$.checksum: mismatch (profile corrupted or edited without re-signing)")
    return errors


def validate_profile(profile: Any) -> dict[str, Any]:
    errors = validation_errors(profile)
    if errors:
        raise ProfileError(errors)
    return profile


# ---------------------------------------------------------------- parsing / migration
def _reject_constant(name: str):
    raise ProfileError([f"non-standard JSON constant {name} (NaN/Infinity are not allowed)"])


def parse_profile_text(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise ProfileError([f"invalid JSON: {exc.msg} at line {exc.lineno} col {exc.colno}"]) from exc
    if not isinstance(data, dict):
        raise ProfileError(["profile must be a JSON object"])
    return migrate(data)


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade known older formats; reject unknown versions (D-019)."""
    version = data.get("schema_version")
    if version == CURRENT_SCHEMA_VERSION:
        return data
    if version == 0:
        return _migrate_v0_prd_draft(data)
    raise ProfileError([f"unsupported schema_version {version!r} (supported: 0 -> migrated, {CURRENT_SCHEMA_VERSION})"])


def _migrate_v0_prd_draft(d: dict[str, Any]) -> dict[str, Any]:
    """v0 = field names of PRD-HCC-001 18.3 (pixel regions). See OPEN-DECISIONS OD-003."""
    try:
        dw, dh = float(d["display_width"]), float(d["display_height"])
        rx, ry, rw, rh = (float(d[k]) for k in ("region_x", "region_y", "region_w", "region_h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError([f"v0 profile missing/invalid field: {exc}"]) from exc
    if dw <= 0 or dh <= 0 or rw <= 0 or rh <= 0:
        raise ProfileError(["v0 profile has non-positive display/region size"])
    ts = d.get("last_calibrated_at") or utc_now()
    v1 = {
        "schema_version": 1,
        "device_id": d.get("device_id", "dev_unknown"),
        "optical_assembly_id": d.get("optical_revision", "opt_unknown"),
        "display_profile_id": d.get("display_profile_id", f"display_{int(dw)}x{int(dh)}"),
        "mode": d.get("mode", "single_reflector"),
        "mirror_x": bool(d.get("horizontal_flip", False)),
        "mirror_y": bool(d.get("vertical_flip", False)),
        "rotation_deg": d.get("rotation_deg", 0),
        "scale": d.get("scale", 1.0),
        "offset_normalized": {"x": float(d.get("offset_x", 0)) / rw, "y": float(d.get("offset_y", 0)) / rh},
        "viewport_normalized": {"x": rx / dw, "y": ry / dh, "w": rw / dw, "h": rh / dh},
        "content_brightness": d.get("brightness_limit", 0.8),
        "max_luminance_profile": d.get("max_luminance_profile", "lab_safe_01"),
        "profile_revision": 1,
        "created_at": ts,
        "updated_at": ts,
        "software_version": "migrated-from-v0",
    }
    v1["checksum"] = compute_checksum(v1)
    return v1


# ---------------------------------------------------------------- defaults / helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_default_profile(device_id: str = "dev_lab_pc", display_profile_id: str = "display_unknown") -> dict[str, Any]:
    """Dim, identity geometry. Used when no valid profile exists (TZ 7.2, D-009)."""
    ts = utc_now()
    p = {
        "schema_version": 1,
        "device_id": device_id,
        "optical_assembly_id": "opt_unknown",
        "display_profile_id": display_profile_id,
        "mode": "single_reflector",
        "mirror_x": False,
        "mirror_y": False,
        "rotation_deg": 0,
        "scale": 1.0,
        "offset_normalized": {"x": 0.0, "y": 0.0},
        "viewport_normalized": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        "content_brightness": 0.6,
        "max_luminance_profile": "lab_safe_01",
        "profile_revision": 1,
        "created_at": ts,
        "updated_at": ts,
        "software_version": _software_version(),
    }
    p["checksum"] = compute_checksum(p)
    return p


def _software_version() -> str:
    from . import __version__

    return f"hcc-lab/{__version__}"


def resign(profile: dict[str, Any], *, bump_revision: bool = True) -> dict[str, Any]:
    """Return a copy with updated_at/revision/checksum refreshed (after an edit)."""
    p = copy.deepcopy(profile)
    p["updated_at"] = utc_now()
    if bump_revision:
        p["profile_revision"] = int(p.get("profile_revision", 0)) + 1
    p["software_version"] = _software_version()
    p["checksum"] = compute_checksum(p)
    return p


def profile_to_geometry(profile: dict[str, Any]) -> GeometryParams:
    vp = profile["viewport_normalized"]
    off = profile["offset_normalized"]
    return GeometryParams(
        mirror_x=bool(profile["mirror_x"]),
        mirror_y=bool(profile["mirror_y"]),
        rotation_deg=int(profile["rotation_deg"]),
        scale=float(profile["scale"]),
        offset=Offset(float(off["x"]), float(off["y"])),
        viewport=Viewport(float(vp["x"]), float(vp["y"]), float(vp["w"]), float(vp["h"])),
    )


def binding_warnings(profile: dict[str, Any], *, display_profile_id: str | None = None,
                     optical_assembly_id: str | None = None) -> list[str]:
    """DSP-010 / D-015: profile bound to another display or optics revision."""
    out = []
    if display_profile_id and profile.get("display_profile_id") not in (display_profile_id, "display_unknown"):
        out.append(f"profile is for display {profile.get('display_profile_id')!r}, current display is {display_profile_id!r}")
    if optical_assembly_id and profile.get("optical_assembly_id") not in (optical_assembly_id, "opt_unknown"):
        out.append(f"profile is for optics {profile.get('optical_assembly_id')!r}, configured optics is {optical_assembly_id!r}")
    return out


# ---------------------------------------------------------------- persistence
def last_valid_path(path: Path) -> Path:
    return path.with_name(path.stem + ".last-valid.json")


def _read_valid(path: Path) -> dict[str, Any]:
    profile = parse_profile_text(path.read_text(encoding="utf-8"))
    return validate_profile(profile)


def load_profile(path: str | os.PathLike, *, device_id: str = "dev_lab_pc",
                 display_profile_id: str = "display_unknown") -> LoadResult:
    """Never raises for bad data: falls back to last valid, then to the safe default."""
    path = Path(path)
    warnings: list[str] = []
    if path.exists():
        try:
            return LoadResult(_read_valid(path), "primary")
        except (ProfileError, OSError, UnicodeDecodeError) as exc:
            warnings.append(f"{ERROR_CODE}: {path.name}: {exc}")
    else:
        warnings.append(f"profile {path.name} not found")
    backup = last_valid_path(path)
    if backup.exists():
        try:
            return LoadResult(_read_valid(backup), "last_valid", warnings + [f"using last valid profile {backup.name}"], ERROR_CODE)
        except (ProfileError, OSError, UnicodeDecodeError) as exc:
            warnings.append(f"{ERROR_CODE}: {backup.name}: {exc}")
    code = ERROR_CODE if path.exists() or backup.exists() else None
    return LoadResult(safe_default_profile(device_id, display_profile_id), "safe_default",
                      warnings + ["using SAFE DEFAULT profile (dim, identity geometry)"], code)


def save_profile_atomic(path: str | os.PathLike, profile: dict[str, Any]) -> Path:
    """Validate, keep the previous valid file as *.last-valid.json, write via temp + fsync + replace."""
    path = Path(path)
    validate_profile(profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            _read_valid(path)
            shutil.copyfile(path, last_valid_path(path))
        except (ProfileError, OSError, UnicodeDecodeError):
            pass  # never promote a corrupted file to "last valid"
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path
