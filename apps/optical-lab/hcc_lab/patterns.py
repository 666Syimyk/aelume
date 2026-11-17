"""Original calibration patterns (B-008, B-009, DSP-001, OPT-008, OPT-010).

All images are generated from code (no photos of people). Every file is listed in
packages/test-fixtures/patterns/manifest.json with its SHA-256; ``check_suite``
regenerates the suite and compares hashes (reproducibility, NFR-001).
"""
from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .paths import PATTERNS_DIR

GENERATOR_VERSION = "1"
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
GRAY = (160, 160, 160)


def _blank(w: int, h: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _text(img, s, org, scale, color=WHITE, thick=2, center=False):
    if center:
        (tw, th), _ = cv2.getTextSize(s, FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def _u(w, h):  # layout unit
    return min(w, h) / 100.0


def asym_lr_arrows(w: int, h: int) -> np.ndarray:
    """Disambiguates mirror_x, mirror_y and every rotation: L/R arrows, TOP marker, 'F' glyph, red corner."""
    img, u = _blank(w, h), _u(w, h)
    cy = h // 2
    cv2.arrowedLine(img, (int(w * 0.42), cy), (int(w * 0.08), cy), WHITE, max(2, int(u * 1.2)), cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(img, (int(w * 0.58), cy), (int(w * 0.92), cy), WHITE, max(2, int(u * 1.2)), cv2.LINE_AA, tipLength=0.25)
    _text(img, "L", (w * 0.22, cy - u * 12), u * 0.25, WHITE, max(2, int(u * 0.8)), center=True)
    _text(img, "R", (w * 0.78, cy - u * 12), u * 0.25, WHITE, max(2, int(u * 0.8)), center=True)
    cv2.arrowedLine(img, (w // 2, int(h * 0.30)), (w // 2, int(h * 0.06)), WHITE, max(2, int(u)), cv2.LINE_AA, tipLength=0.3)
    _text(img, "TOP", (w / 2 + u * 9, h * 0.13), u * 0.09, WHITE, max(1, int(u * 0.4)), center=True)
    # 'F' glyph: asymmetric under every flip/rotation
    gx, gy, s = int(w / 2 - u * 6), int(cy - u * 12), u
    t = max(2, int(u * 1.6))
    cv2.line(img, (gx, gy), (gx, int(gy + s * 24)), WHITE, t)
    cv2.line(img, (gx, gy), (int(gx + s * 13), gy), WHITE, t)
    cv2.line(img, (gx, int(gy + s * 11)), (int(gx + s * 9), int(gy + s * 11)), WHITE, t)
    cv2.rectangle(img, (0, 0), (int(u * 8), int(u * 8)), (0, 0, 255), -1)  # red = top-left corner
    _text(img, "HCC-PAT-01 v1 top-left=RED", (u * 2, h - u * 3), u * 0.05, GRAY, 1)
    return img


def grid(w: int, h: int) -> np.ndarray:
    img, u = _blank(w, h), _u(w, h)
    for i in range(0, 21):
        x, y = int(round(i * (w - 1) / 20)), int(round(i * (h - 1) / 20))
        major = i % 5 == 0
        col = WHITE if major else (90, 90, 90)
        cv2.line(img, (x, 0), (x, h - 1), col, 2 if major else 1)
        cv2.line(img, (0, y), (w - 1, y), col, 2 if major else 1)
        if major:
            _text(img, f"{i * 5}%", (x + u, u * 4), u * 0.06, GRAY, 1)
            _text(img, f"{i * 5}%", (u, y - u), u * 0.06, GRAY, 1)
    _text(img, "HCC-PAT-02 grid 5%", (u * 2, h - u * 3), u * 0.05, GRAY, 1)
    return img


def gray_ramp(w: int, h: int, steps: int = 16) -> np.ndarray:
    img, u = _blank(w, h), _u(w, h)
    top = int(h * 0.62)
    for i in range(steps):
        v = int(round(i * 255 / (steps - 1)))
        x0, x1 = int(i * w / steps), int((i + 1) * w / steps)
        img[:top, x0:x1] = v
        _text(img, str(v), (x0 + u, top + u * 6), u * 0.06, GRAY, 1)
    ramp = np.linspace(0, 255, w, dtype=np.float64).round().astype(np.uint8)
    img[int(h * 0.75):int(h * 0.92), :] = ramp[None, :, None]
    _text(img, "HCC-PAT-03 gray 16 steps + ramp", (u * 2, h - u * 2), u * 0.05, GRAY, 1)
    return img


RGB_BARS = [("R", (0, 0, 255)), ("G", (0, 255, 0)), ("B", (255, 0, 0)), ("C", (255, 255, 0)),
            ("M", (255, 0, 255)), ("Y", (0, 255, 255)), ("W", (255, 255, 255)), ("K", (0, 0, 0))]


def rgb_bars(w: int, h: int) -> np.ndarray:
    """Channel-order check: the bar labelled R must look red on the optical display."""
    img, u = _blank(w, h), _u(w, h)
    n = len(RGB_BARS)
    for i, (name, bgr) in enumerate(RGB_BARS):
        x0, x1 = int(i * w / n), int((i + 1) * w / n)
        img[: int(h * 0.85), x0:x1] = bgr
        _text(img, name, ((x0 + x1) / 2, h * 0.92), u * 0.12, WHITE, 2, center=True)
    return img


def neutral_tones(w: int, h: int) -> np.ndarray:
    """Generic warm-neutral tonal ramp T1..T8 (NOT calibrated skin references, no faces)."""
    img, u = _blank(w, h), _u(w, h)
    light, dark = np.array([180, 200, 232], float), np.array([30, 40, 60], float)  # BGR
    for i in range(8):
        c = light + (dark - light) * i / 7
        x0, x1 = int(i * w / 8), int((i + 1) * w / 8)
        img[int(h * 0.1): int(h * 0.8), x0 + 4:x1 - 4] = c.round().astype(np.uint8)
        _text(img, f"T{i + 1}", ((x0 + x1) / 2, h * 0.88), u * 0.1, WHITE, 2, center=True)
    _text(img, "HCC-PAT-05 neutral tones (generic, not skin-calibrated)", (u * 2, h - u * 2), u * 0.05, GRAY, 1)
    return img


def checkerboard(w: int, h: int, squares: int = 16) -> np.ndarray:
    s = max(1, min(w, h) // squares)
    yy, xx = np.mgrid[0:h, 0:w]
    board = (((xx // s) + (yy // s)) % 2 * 255).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


def centering_target(w: int, h: int) -> np.ndarray:
    img, u = _blank(w, h), _u(w, h)
    c = (w // 2, h // 2)
    for r in range(1, 6):
        cv2.circle(img, c, int(r * u * 8), WHITE if r % 2 else GRAY, 2, cv2.LINE_AA)
    cv2.line(img, (c[0], 0), (c[0], h - 1), GRAY, 1)
    cv2.line(img, (0, c[1]), (w - 1, c[1]), GRAY, 1)
    cv2.circle(img, c, max(2, int(u * 0.6)), WHITE, -1)
    L = int(u * 8)
    for (x, y, dx, dy) in ((0, 0, 1, 1), (w - 1, 0, -1, 1), (0, h - 1, 1, -1), (w - 1, h - 1, -1, -1)):
        cv2.line(img, (x, y), (x + dx * L, y), WHITE, 3)
        cv2.line(img, (x, y), (x, y + dy * L), WHITE, 3)
    return img


def ghost_line(w: int, h: int) -> np.ndarray:
    """High-contrast thin lines for double-image / ghosting measurement (OPT-008, TC-OPT-001.2)."""
    img, u = _blank(w, h), _u(w, h)
    cv2.line(img, (int(w * 0.1), h // 2), (int(w * 0.9), h // 2), WHITE, 2)
    cv2.line(img, (w // 2, int(h * 0.1)), (w // 2, int(h * 0.9)), WHITE, 2)
    cv2.line(img, (int(w * 0.1), int(h * 0.2)), (int(w * 0.9), int(h * 0.2)), WHITE, 1)
    _text(img, "HCC-PAT-08 ghost lines 2px/1px", (u * 2, h - u * 2), u * 0.05, GRAY, 1)
    return img


def size_marks(w: int, h: int, bar_px: int = 200) -> np.ndarray:
    """Corner marks at known pixel positions + scale bar, to measure the virtual image size (DSP-002)."""
    img, u = _blank(w, h), _u(w, h)
    inset = int(u * 10)
    for (x, y) in ((inset, inset), (w - inset, inset), (inset, h - inset), (w - inset, h - inset)):
        cv2.drawMarker(img, (x, y), WHITE, cv2.MARKER_CROSS, int(u * 6), 2)
    cv2.drawMarker(img, (w // 2, h // 2), WHITE, cv2.MARKER_CROSS, int(u * 10), 2)
    y = h - inset - int(u * 8)
    cv2.line(img, (w // 2 - bar_px // 2, y), (w // 2 + bar_px // 2, y), WHITE, 3)
    _text(img, f"{bar_px} px; marks inset {inset} px; canvas {w}x{h}", (w / 2, y - u * 4), u * 0.06, GRAY, 1, center=True)
    return img


@dataclass(frozen=True)
class PatternSpec:
    pid: str
    name: str
    fn: Callable[[int, int], np.ndarray]
    purpose: str


PATTERNS = [
    PatternSpec("HCC-PAT-01", "asym_lr_arrows", asym_lr_arrows, "mirror/rotation orientation (OPT-010, B-010)"),
    PatternSpec("HCC-PAT-02", "grid", grid, "geometry, keystone, scale"),
    PatternSpec("HCC-PAT-03", "gray_ramp", gray_ramp, "black level, gamma, contrast steps"),
    PatternSpec("HCC-PAT-04", "rgb_bars", rgb_bars, "channel order (C-008)"),
    PatternSpec("HCC-PAT-05", "neutral_tones", neutral_tones, "tonal rendering, generic patches"),
    PatternSpec("HCC-PAT-06", "checkerboard", checkerboard, "focus/sharpness, distortion"),
    PatternSpec("HCC-PAT-07", "centering_target", centering_target, "centre alignment"),
    PatternSpec("HCC-PAT-08", "ghost_line", ghost_line, "double reflection / ghosting (OPT-008)"),
    PatternSpec("HCC-PAT-09", "size_marks", size_marks, "virtual image size measurement (DSP-002, OPT-007)"),
]
PATTERN_BY_NAME = {p.name: p for p in PATTERNS}
SUITE_SIZES = [(1920, 1080), (1080, 1080)]
VIDEO_SPEC = {"name": "timer_counter", "width": 1280, "height": 720, "fps": 30, "frames": 150}


def render(name: str, w: int, h: int) -> np.ndarray:
    if w <= 0 or h <= 0:
        raise ValueError("pattern size must be positive")
    return PATTERN_BY_NAME[name].fn(w, h)


def timer_frame(i: int, w: int, h: int, fps: int) -> np.ndarray:
    """Frame i of the animated pattern: counter, time, sweep bar, 16-bit code, 1 Hz flash."""
    img, u = _blank(w, h), _u(w, h)
    _text(img, f"frame {i:05d}", (w / 2, h * 0.42), u * 0.3, WHITE, 3, center=True)
    _text(img, f"t = {i / fps:7.3f} s @ {fps} fps", (w / 2, h * 0.6), u * 0.12, GRAY, 2, center=True)
    x = int((i % fps) / fps * (w - 1))
    cv2.line(img, (x, int(h * 0.75)), (x, int(h * 0.9)), WHITE, 4)
    s = int(u * 3)
    for b in range(16):  # binary frame code, MSB first, for camera-based frame identification
        on = (i >> (15 - b)) & 1
        cv2.rectangle(img, (int(u * 2 + b * s), int(u * 2)), (int(u * 2 + (b + 1) * s - 2), int(u * 2 + s)),
                      WHITE if on else (40, 40, 40), -1)
    if i % fps == 0:
        cv2.rectangle(img, (w - int(u * 12), int(u * 2)), (w - int(u * 2), int(u * 12)), WHITE, -1)
    return img


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_suite(out_dir: Path = PATTERNS_DIR, *, with_video: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for (w, h) in SUITE_SIZES:
        for spec in PATTERNS:
            fname = f"{spec.name}_{w}x{h}.png"
            path = out_dir / fname
            if not cv2.imwrite(str(path), render(spec.name, w, h), [cv2.IMWRITE_PNG_COMPRESSION, 9]):
                raise OSError(f"failed to write {path}")
            entries.append({"id": spec.pid, "name": spec.name, "file": fname, "width": w, "height": h,
                            "bytes": path.stat().st_size, "sha256": _sha256(path), "purpose": spec.purpose})
    videos = []
    if with_video:
        v = VIDEO_SPEC
        fname = f"{v['name']}_{v['width']}x{v['height']}_{v['fps']}fps.mp4"
        path = out_dir / fname
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), v["fps"], (v["width"], v["height"]))
        if not wr.isOpened():
            raise OSError("cv2.VideoWriter could not open mp4v writer")
        for i in range(v["frames"]):
            wr.write(timer_frame(i, v["width"], v["height"], v["fps"]))
        wr.release()
        videos.append({"id": "HCC-VID-01", "file": fname, "fps": v["fps"], "frames": v["frames"],
                       "width": v["width"], "height": v["height"], "codec": "mp4v",
                       "bytes": path.stat().st_size, "sha256": _sha256(path),
                       "purpose": "frame identification / timing (B-009); frame number encoded as 16-bit squares"})
    manifest = {
        "generator": "hcc_lab.patterns", "generator_version": GENERATOR_VERSION,
        "opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "note": "Hashes depend on OpenCV/zlib versions; regenerate with the pinned lockfile.",
        "patterns": entries, "videos": videos,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def check_suite(manifest_path: Path = PATTERNS_DIR / "manifest.json", *, with_video: bool = True) -> list[str]:
    """Regenerate into a temp dir; return a list of mismatches (empty = reproducible)."""
    ref = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    problems = []
    if ref.get("opencv") != cv2.__version__:
        problems.append(f"OpenCV version differs: manifest {ref.get('opencv')} vs installed {cv2.__version__}")
    with tempfile.TemporaryDirectory() as tmp:
        new = generate_suite(Path(tmp), with_video=with_video)
        for kind in ("patterns", "videos") if with_video else ("patterns",):
            old = {e["file"]: e["sha256"] for e in ref.get(kind, [])}
            for e in new[kind]:
                if old.get(e["file"]) != e["sha256"]:
                    problems.append(f"{e['file']}: sha256 {e['sha256'][:12]} != manifest {str(old.get(e['file']))[:12]}")
    for e in ref.get("patterns", []) + ref.get("videos", []):
        on_disk = Path(manifest_path).parent / e["file"]
        if not on_disk.exists():
            problems.append(f"missing file {e['file']}")
        elif _sha256(on_disk) != e["sha256"]:
            problems.append(f"{e['file']} on disk does not match manifest")
    return problems
