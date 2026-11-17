"""Video frame and mask contracts (TZ 12.1, master prompt 06.7, C-002, C-006).

Timestamps are ``time.monotonic()`` seconds of THIS process. They must never be
subtracted from another device's clock.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

SOURCE_KINDS = ("camera", "video_file", "synthetic", "pattern")


@dataclass(frozen=True)
class Frame:
    frame_id: int
    capture_time_monotonic: float
    image: np.ndarray  # H x W x 3, uint8, BGR
    source_kind: str
    camera_facing: str = "external"  # front | back | external | none
    orientation_deg: int = 0  # clockwise rotation that makes the image upright
    mirrored: bool = False  # True if the source already delivers a mirror image (selfie preview)
    pixel_format: str = "BGR24"
    color_space: str = "sRGB"
    crop: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels, None = full frame
    meta: dict = field(default_factory=dict)

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def validate(self) -> None:
        if self.source_kind not in SOURCE_KINDS:
            raise ValueError(f"unknown source_kind {self.source_kind!r}")
        if self.image.ndim != 3 or self.image.shape[2] != 3 or self.image.dtype != np.uint8:
            raise ValueError(f"frame image must be HxWx3 uint8, got {self.image.shape} {self.image.dtype}")
        if self.orientation_deg not in (0, 90, 180, 270):
            raise ValueError("orientation_deg must be 0/90/180/270")
        if not math.isfinite(self.capture_time_monotonic):
            raise ValueError("capture_time_monotonic must be finite")


@dataclass(frozen=True)
class Mask:
    frame_id: int
    alpha: np.ndarray  # H x W float32 in [0, 1]
    model_version: str
    inference_ms: float
    confidence: float  # 0..1, mean certainty of the person/background decision
    coverage: float  # 0..1, fraction of pixels with alpha > 0.5


class MaskRejected(ValueError):
    pass


def mask_stats(alpha: np.ndarray) -> tuple[float, float]:
    """(confidence, coverage). confidence = mean |2a-1|: 1 = crisp, 0 = undecided everywhere.

    Aggregates only, so large masks are measured on a 1/4 area downscale (profiling: 7.5 ms -> <1 ms).
    """
    if alpha.size == 0:
        return 0.0, 0.0
    a = alpha
    if a.size > 120_000:
        a = cv2.resize(a, (max(1, a.shape[1] // 2), max(1, a.shape[0] // 2)), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(2.0 * a - 1.0))), float(np.count_nonzero(a > 0.5) / a.size)


def validate_mask(mask: Mask, frame: Frame, *, now: float | None = None, max_age_s: float | None = None) -> Mask:
    """Reject masks that would corrupt the framebuffer (C-006, AI-006). Returns a clipped copy."""
    a = mask.alpha
    if not isinstance(a, np.ndarray):
        raise MaskRejected("alpha is not an ndarray")
    if mask.frame_id != frame.frame_id:
        raise MaskRejected(f"mask frame_id {mask.frame_id} != frame {frame.frame_id}")
    if a.ndim != 2 or a.shape != (frame.height, frame.width):
        raise MaskRejected(f"mask shape {a.shape} != frame {(frame.height, frame.width)}")
    if a.dtype != np.float32:
        raise MaskRejected(f"mask dtype {a.dtype} != float32")
    ok, _ = cv2.checkRange(a, True, -1e-3, 1 + 1e-3)  # one fast pass: NaN/Inf/out-of-range
    if not ok:
        if not np.isfinite(a).all():
            raise MaskRejected("mask contains NaN/Inf")
        raise MaskRejected(f"alpha out of range [{a.min():.3f}, {a.max():.3f}]")
    if max_age_s is not None and now is not None and now - frame.capture_time_monotonic > max_age_s:
        raise MaskRejected(f"mask too old: {(now - frame.capture_time_monotonic) * 1000:.0f} ms > {max_age_s * 1000:.0f} ms")
    if not (math.isfinite(mask.inference_ms) and mask.inference_ms >= 0):
        raise MaskRejected("inference_ms invalid")
    lo, hi, _, _ = cv2.minMaxLoc(a)
    clipped = np.clip(a, 0.0, 1.0) if (lo < 0 or hi > 1) else a
    return Mask(mask.frame_id, clipped, mask.model_version, mask.inference_ms, mask.confidence, mask.coverage)
