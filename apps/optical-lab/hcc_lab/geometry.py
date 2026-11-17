"""Optical compositor geometry, independent of UI (D-001..D-004).

Coordinate spaces (docs/spec/COORDINATES.md):
* source / display pixels: origin top-left, x right, y down, continuous coords
  (pixel i covers [i, i+1), its centre is i + 0.5);
* normalized rects: fractions of the display size, in [0, 1];
* 3D optics (TZ 6.1): right-handed, metres, mirror plane given by point + normal.

Fixed order of operations: rotate (clockwise) -> mirror -> crop -> scale -> translate -> display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class Viewport:
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0

    def validate(self, name: str = "viewport") -> None:
        vals = (self.x, self.y, self.w, self.h)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            raise ValueError(f"{name}: values must be finite numbers")
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"{name}: w and h must be > 0")
        if self.x < 0 or self.y < 0 or self.x + self.w > 1 + 1e-9 or self.y + self.h > 1 + 1e-9:
            raise ValueError(f"{name}: must lie inside [0, 1]")

    def to_pixels(self, width: float, height: float) -> tuple[float, float, float, float]:
        return self.x * width, self.y * height, self.w * width, self.h * height


@dataclass(frozen=True)
class Offset:
    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class GeometryParams:
    mirror_x: bool = False
    mirror_y: bool = False
    rotation_deg: int = 0
    scale: float = 1.0
    offset: Offset = field(default_factory=Offset)
    viewport: Viewport = field(default_factory=Viewport)
    crop: Viewport = field(default_factory=Viewport)

    def validate(self) -> None:
        if self.rotation_deg not in ROTATIONS or isinstance(self.rotation_deg, bool):
            raise ValueError(f"rotation_deg must be one of {ROTATIONS}, got {self.rotation_deg!r}")
        if not isinstance(self.scale, (int, float)) or not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be a finite number > 0, got {self.scale!r}")
        if not all(math.isfinite(v) for v in (self.offset.x, self.offset.y)):
            raise ValueError("offset must be finite")
        self.viewport.validate("viewport")
        self.crop.validate("crop")


class Transform:
    """Affine map in continuous pixel coordinates (3x3 homogeneous matrix)."""

    __slots__ = ("m",)

    def __init__(self, m: np.ndarray):
        self.m = np.asarray(m, dtype=np.float64).reshape(3, 3)

    def apply_point(self, x: float, y: float) -> tuple[float, float]:
        r = self.m @ np.array([x, y, 1.0])
        return float(r[0]), float(r[1])

    def then(self, other: "Transform") -> "Transform":
        """Apply self first, then other."""
        return Transform(other.m @ self.m)

    def inverse(self) -> "Transform":
        return Transform(np.linalg.inv(self.m))

    def cv2_matrix(self) -> np.ndarray:
        """2x3 matrix for cv2.warpAffine, whose pixel centres sit at integers."""
        to_cont = _translate(0.5, 0.5)
        to_index = _translate(-0.5, -0.5)
        return (to_index @ self.m @ to_cont)[:2, :]

    @property
    def scale_factor(self) -> float:
        return float(math.hypot(self.m[0, 0], self.m[1, 0]))


def _translate(tx: float, ty: float) -> np.ndarray:
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)


def _check_size(name: str, w: float, h: float) -> None:
    for v in (w, h):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{name} size must be positive and finite, got {w!r}x{h!r}")


def rotation_matrix(rotation_deg: int, w: float, h: float) -> tuple[np.ndarray, float, float]:
    """Clockwise rotation of a w x h frame; returns (matrix, new_w, new_h)."""
    if rotation_deg == 0:
        return np.eye(3), w, h
    if rotation_deg == 90:  # (u, v) -> (h - v, u)
        return np.array([[0, -1, h], [1, 0, 0], [0, 0, 1]], dtype=np.float64), h, w
    if rotation_deg == 180:  # (u, v) -> (w - u, h - v)
        return np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float64), w, h
    if rotation_deg == 270:  # (u, v) -> (v, w - u)
        return np.array([[0, 1, 0], [-1, 0, w], [0, 0, 1]], dtype=np.float64), h, w
    raise ValueError(f"rotation_deg must be one of {ROTATIONS}")


def build_transform(params: GeometryParams, src_w: float, src_h: float, dst_w: float, dst_h: float) -> Transform:
    """Source frame -> display framebuffer, aspect ratio always preserved (DSP-005)."""
    _check_size("source", src_w, src_h)
    _check_size("display", dst_w, dst_h)
    params.validate()

    rot, w1, h1 = rotation_matrix(params.rotation_deg, float(src_w), float(src_h))
    mirror = np.eye(3)
    if params.mirror_x:
        mirror = np.array([[-1, 0, w1], [0, 1, 0], [0, 0, 1]], dtype=np.float64) @ mirror
    if params.mirror_y:
        mirror = np.array([[1, 0, 0], [0, -1, h1], [0, 0, 1]], dtype=np.float64) @ mirror

    cx, cy, cw, ch = params.crop.to_pixels(w1, h1)
    crop = _translate(-cx, -cy)

    vx, vy, vw, vh = params.viewport.to_pixels(float(dst_w), float(dst_h))
    f = min(vw / cw, vh / ch) * float(params.scale)
    scale = np.diag([f, f, 1.0])

    tx = vx + (vw - f * cw) / 2 + params.offset.x * vw
    ty = vy + (vh - f * ch) / 2 + params.offset.y * vh
    return Transform(_translate(tx, ty) @ scale @ crop @ mirror @ rot)


def warp(image: np.ndarray, transform: Transform, dst_w: int, dst_h: int, *,
         interpolation: int = cv2.INTER_LINEAR, clip: Viewport | None = None,
         out: np.ndarray | None = None) -> np.ndarray:
    """Render ``image`` onto a black framebuffer. ``clip`` blanks everything outside a normalized rect."""
    dst = cv2.warpAffine(image, transform.cv2_matrix(), (int(dst_w), int(dst_h)), flags=interpolation,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    if clip is not None:
        x0, y0, w, h = clip.to_pixels(dst_w, dst_h)
        xa, ya = max(0, int(round(x0))), max(0, int(round(y0)))
        xb, yb = min(int(dst_w), int(round(x0 + w))), min(int(dst_h), int(round(y0 + h)))
        mask = np.zeros(dst.shape[:2], dtype=bool)
        mask[ya:yb, xa:xb] = True
        dst[~mask] = 0
    if out is not None:
        np.maximum(out, dst, out=out)
        return out
    return dst


# ------------------------------------------------------------------ optical layouts
@dataclass(frozen=True)
class Zone:
    face_id: str
    label: str
    viewport: Viewport
    base_rotation_deg: int


def layout_zones(mode: str, viewport: Viewport, dst_w: int, dst_h: int) -> list[Zone]:
    """Display regions per optical mode.

    four_face_pyramid uses the classic 3x3 cross inside the largest centred square:
    each face shows the SAME video (no fake 3D views, D-006), rotated so image-up points
    outward. This is a test layout: correct orientation must be confirmed with the
    asymmetric pattern through real optics (TZ OPT-010), not assumed.
    """
    viewport.validate()
    if mode == "single_reflector":
        return [Zone("S1", "single", viewport, 0)]
    if mode != "four_face_pyramid":
        raise ValueError(f"unknown optical mode {mode!r}")
    vx, vy, vw, vh = viewport.to_pixels(dst_w, dst_h)
    side = min(vw, vh)
    sx, sy = vx + (vw - side) / 2, vy + (vh - side) / 2
    c = side / 3

    def rect(col: int, row: int) -> Viewport:
        return Viewport((sx + col * c) / dst_w, (sy + row * c) / dst_h, c / dst_w, c / dst_h)

    return [
        Zone("F1", "front", rect(1, 2), 180),
        Zone("F2", "right", rect(2, 1), 90),
        Zone("F3", "back", rect(1, 0), 0),
        Zone("F4", "left", rect(0, 1), 270),
    ]


def zone_params(params: GeometryParams, zone: Zone) -> GeometryParams:
    """Geometry for one zone: zone rotation composed with the profile rotation."""
    rot = (zone.base_rotation_deg + params.rotation_deg) % 360
    return GeometryParams(mirror_x=params.mirror_x, mirror_y=params.mirror_y, rotation_deg=rot,
                          scale=params.scale, offset=params.offset, viewport=zone.viewport, crop=params.crop)


# ------------------------------------------------------------------ 3D plane mirror (TZ 6.1)
def _unit(n) -> np.ndarray:
    v = np.asarray(n, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("plane normal must be a non-zero finite vector")
    return v / norm


def reflect_point(p, p0, n) -> tuple[float, float, float]:
    """p' = p - 2 n (n . (p - p0)) for an ideal plane mirror through p0 with normal n."""
    nn = _unit(n)
    pp = np.asarray(p, dtype=np.float64)
    r = pp - 2.0 * nn * float(nn @ (pp - np.asarray(p0, dtype=np.float64)))
    return float(r[0]), float(r[1]), float(r[2])


def plate_hit_point(eye, virtual_point, p0, n) -> tuple[float, float, float] | None:
    """Where the eye->virtual-image ray crosses the reflector plane (None if parallel/behind)."""
    nn = _unit(n)
    e = np.asarray(eye, dtype=np.float64)
    d = np.asarray(virtual_point, dtype=np.float64) - e
    denom = float(nn @ d)
    if abs(denom) < 1e-12:
        return None
    t = float(nn @ (np.asarray(p0, dtype=np.float64) - e)) / denom
    if not 0.0 <= t <= 1.0:
        return None
    hit = e + t * d
    return float(hit[0]), float(hit[1]), float(hit[2])
