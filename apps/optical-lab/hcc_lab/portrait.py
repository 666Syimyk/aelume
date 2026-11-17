"""Live Portrait: segmentation -> matte -> black-background composite -> stabilized framing.

TZ 12.2: C_out = a * C_rgb + (1 - a) * (0, 0, 0). The output is ALWAYS derived from the
current frame: an old frame is never re-shown as live (AI-007, FR-VI-007).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable

import cv2
import numpy as np

from .frames import Frame, MaskRejected, validate_mask
from .segmentation import Segmenter


class PortraitStatus(str, Enum):
    LIVE_PORTRAIT = "LIVE_PORTRAIT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # AI works but is unsure: suggest better light / Original
    ORIGINAL = "ORIGINAL"  # AI switched off by the user (bypass)
    AI_UNAVAILABLE = "AI_UNAVAILABLE"  # AI failed: honest fallback, audio unaffected
    PERSON_LOST = "PERSON_LOST"  # nobody in frame longer than the timeout: black, no frozen face


USER_MESSAGES_RU = {  # PRD 12.3 copy; codes are stable, text is localizable
    PortraitStatus.LOW_CONFIDENCE: "Сегментация неуверенна: добавьте света или включите обычное видео",
    PortraitStatus.AI_UNAVAILABLE: "Портрет временно недоступен — показываем обычное видео",
    PortraitStatus.PERSON_LOST: "Камера не видит человека. Вернитесь в кадр или включите обычное видео",
}
LOW_LIGHT_MESSAGE_RU = "Слишком темно: добавьте света перед лицом"


def mean_luma(image_bgr: np.ndarray) -> float:
    """Cheap scene brightness (0..255) on a 1/8 downscale; used for the low-light hint (T-UX-04)."""
    small = cv2.resize(image_bgr, (max(1, image_bgr.shape[1] // 8), max(1, image_bgr.shape[0] // 8)),
                       interpolation=cv2.INTER_AREA)
    return float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())


# ------------------------------------------------------------------ matte
@dataclass(frozen=True)
class MatteParams:
    edge_softness_px: float = 1.5
    low: float = 0.2
    high: float = 0.8
    temporal: float = 0.3  # weight of the previous mask (0 = off)

    def clamped(self) -> "MatteParams":
        """Bounded tuning so edge processing cannot erase hair or dark clothing (C-007)."""
        low = min(max(self.low, 0.0), 0.45)
        high = max(min(self.high, 1.0), 0.55)
        return MatteParams(min(max(self.edge_softness_px, 0.0), 8.0), low, high, min(max(self.temporal, 0.0), 0.8))


def smoothstep(low: float, high: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def refine_alpha(alpha: np.ndarray, prev: np.ndarray | None, params: MatteParams) -> np.ndarray:
    p = params.clamped()
    a = alpha.astype(np.float32, copy=False)
    if p.temporal > 0 and prev is not None and prev.shape == a.shape:
        a = p.temporal * prev + (1.0 - p.temporal) * a
    if p.edge_softness_px > 0:
        k = int(2 * math.ceil(2 * p.edge_softness_px) + 1)
        a = cv2.GaussianBlur(a, (k, k), p.edge_softness_px)
    return smoothstep(p.low, p.high, a).astype(np.float32)


def composite_on_black(image_bgr: np.ndarray, alpha: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """C_out = a * C (black background). Bit-exact for a in {0, 1}, at most 1 LSB off in between
    (alpha quantized to 1/255). Channel order is preserved (BGR in, BGR out).

    OpenCV integer path: 2.5 ms vs 31.9 ms for float numpy on 800x600 (profiled 2026-11-03).
    """
    if image_bgr.shape[:2] != alpha.shape:
        raise ValueError(f"alpha {alpha.shape} does not match image {image_bgr.shape[:2]}")
    a8 = cv2.convertScaleAbs(alpha, alpha=255.0 * float(gain))  # rounds + saturates to 0..255
    return cv2.multiply(image_bgr, cv2.merge((a8, a8, a8)), scale=1.0 / 255.0)


# ------------------------------------------------------------------ framing
@dataclass(frozen=True)
class RoiParams:
    aspect: float = 3 / 4  # output w/h
    head_room: float = 0.10  # of person height, above the head
    margin: float = 0.20  # of person width, on each side
    deadzone_pos: float = 0.06  # centre moves smaller than this fraction of ROI height are ignored
    deadzone_scale: float = 0.08  # relative size changes smaller than this are ignored
    settle: float = 0.01  # hysteresis: once moving, continue until this close
    max_speed: float = 1.0  # ROI heights per second
    max_zoom_rate: float = 0.6  # relative size change per second


@dataclass
class Roi:
    cx: float
    cy: float
    h: float  # width = h * aspect

    def rect(self, aspect: float) -> tuple[float, float, float, float]:
        w = self.h * aspect
        return self.cx - w / 2, self.cy - self.h / 2, w, self.h


def person_bbox(alpha: np.ndarray, thr: float = 0.5) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(alpha > thr)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


class RoiStabilizer:
    """Deadzone + hysteresis + rate limits: small head motion does not move or zoom the frame (AI-008)."""

    def __init__(self, params: RoiParams, frame_w: int, frame_h: int):
        self.p, self.fw, self.fh = params, frame_w, frame_h
        self.roi: Roi | None = None
        self.moving = False

    def target(self, bbox: tuple[int, int, int, int]) -> Roi:
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        h = bh * (1 + self.p.head_room)
        h = max(h, bw * (1 + 2 * self.p.margin) / self.p.aspect)
        h = min(h, self.fh, self.fw / self.p.aspect)
        top = y0 - bh * self.p.head_room
        return Roi((x0 + x1) / 2, top + h / 2, h)

    def update(self, alpha: np.ndarray | None, dt: float, scale: tuple[float, float] = (1.0, 1.0)) -> Roi:
        """``scale`` maps alpha pixels to frame pixels when alpha is a downscaled mask."""
        bbox = person_bbox(alpha) if alpha is not None else None
        if bbox is not None and scale != (1.0, 1.0):
            sx, sy = scale
            bbox = (int(bbox[0] * sx), int(bbox[1] * sy), int(math.ceil(bbox[2] * sx)), int(math.ceil(bbox[3] * sy)))
        if bbox is None:
            if self.roi is None:
                h = min(self.fh, self.fw / self.p.aspect)
                self.roi = Roi(self.fw / 2, self.fh / 2, h)
            return self._clamped(self.roi)
        tgt = self.target(bbox)
        if self.roi is None:
            self.roi = tgt
            return self._clamped(self.roi)
        cur = self.roi
        dpos = math.hypot(tgt.cx - cur.cx, tgt.cy - cur.cy) / cur.h
        dscale = abs(tgt.h - cur.h) / cur.h
        if not self.moving and dpos < self.p.deadzone_pos and dscale < self.p.deadzone_scale:
            return self._clamped(cur)
        self.moving = True
        step = self.p.max_speed * cur.h * max(dt, 0.0)
        dist = math.hypot(tgt.cx - cur.cx, tgt.cy - cur.cy)
        k = 1.0 if dist <= step or dist == 0 else step / dist
        cx, cy = cur.cx + (tgt.cx - cur.cx) * k, cur.cy + (tgt.cy - cur.cy) * k
        max_dh = cur.h * self.p.max_zoom_rate * max(dt, 0.0)
        h = cur.h + max(-max_dh, min(max_dh, tgt.h - cur.h))
        self.roi = Roi(cx, cy, h)
        if dpos < self.p.settle and dscale < self.p.settle:
            self.moving = False
        return self._clamped(self.roi)

    def _clamped(self, r: Roi) -> Roi:
        h = min(r.h, self.fh, self.fw / self.p.aspect)
        w = h * self.p.aspect
        cx = min(max(r.cx, w / 2), self.fw - w / 2)
        cy = min(max(r.cy, h / 2), self.fh - h / 2)
        return Roi(cx, cy, h)


def crop_rect_to_output(image: np.ndarray, rect: tuple[float, float, float, float], out_w: int, out_h: int) -> np.ndarray:
    """Crop a (continuous) rect and scale it to out_w x out_h; outside the image is black/zero."""
    x, y, w, h = rect
    M = np.array([[out_w / w, 0, -x * out_w / w], [0, out_h / h, -y * out_h / h]], dtype=np.float64)
    M[:, 2] += 0.5 * (M[:, 0] + M[:, 1]) - 0.5  # continuous -> pixel-index convention for warpAffine
    return cv2.warpAffine(image, M, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_to_output(image: np.ndarray, roi: Roi, aspect: float, out_w: int, out_h: int) -> np.ndarray:
    return crop_rect_to_output(image, roi.rect(aspect), out_w, out_h)


def letterbox(image: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Fit without distortion onto a black (zero) canvas; works for HxWx3 images and HxW masks."""
    h, w = image.shape[:2]
    f = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    out = np.zeros((out_h, out_w) + image.shape[2:], image.dtype)
    x0, y0 = (out_w - nw) // 2, (out_h - nh) // 2
    if (nw, nh) == (w, h):
        out[y0:y0 + nh, x0:x0 + nw] = image
    else:
        out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    return out


# ------------------------------------------------------------------ processor
@dataclass
class PortraitConfig:
    ai_enabled: bool = True
    output_size: tuple[int, int] = (600, 800)  # w, h (3:4)
    mask_long_side: int = 320  # matte work resolution; the model itself outputs 256x256 detail
    stabilize: bool = True
    mask_hold_s: float = 0.3  # reuse the last good mask for brief model dropouts only
    person_lost_timeout_s: float = 1.0
    min_coverage: float = 0.01
    low_confidence: float = 0.55
    max_mask_age_s: float = 0.5  # AI-006: frame/mask timing mismatch -> mask rejected
    ai_failure_threshold: int = 5
    ai_retry_s: float = 5.0
    fallback: str = "original"  # original | black
    low_light_luma: float = 20.0  # mean frame luma below this -> low_light hint


@dataclass
class PortraitResult:
    frame_id: int
    capture_time_monotonic: float
    image: np.ndarray
    status: PortraitStatus
    model_version: str | None = None
    inference_ms: float | None = None
    confidence: float | None = None
    coverage: float | None = None
    held_mask: bool = False
    low_light: bool = False
    luma: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def user_message(self) -> str | None:
        if self.low_light and self.status in (PortraitStatus.PERSON_LOST, PortraitStatus.LOW_CONFIDENCE):
            return LOW_LIGHT_MESSAGE_RU
        return USER_MESSAGES_RU.get(self.status)


class PortraitProcessor:
    def __init__(self, segmenter: Segmenter | None, config: PortraitConfig | None = None,
                 matte: MatteParams | None = None, roi: RoiParams | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.segmenter = segmenter
        self.cfg = config or PortraitConfig()
        self.matte = matte or MatteParams()
        self.roi_params = replace(roi or RoiParams(), aspect=self.cfg.output_size[0] / self.cfg.output_size[1])
        self.clock = clock
        self._stab: RoiStabilizer | None = None
        self._prev_alpha: np.ndarray | None = None
        self._last_good: tuple[float, np.ndarray] | None = None
        self._no_person_since: float | None = None
        self._failures = 0
        self._ai_down_until: float | None = None
        self._last_t: float | None = None
        self.counters = {"inference_errors": 0, "masks_rejected": 0, "fallback_frames": 0, "person_lost_frames": 0}

    def set_ai_enabled(self, enabled: bool) -> None:
        self.cfg.ai_enabled = enabled
        self._failures, self._ai_down_until = 0, None

    def _out(self, frame: Frame, img: np.ndarray, status: PortraitStatus, **kw) -> PortraitResult:
        luma = mean_luma(frame.image)
        return PortraitResult(frame.frame_id, frame.capture_time_monotonic, img, status,
                              low_light=luma < self.cfg.low_light_luma, luma=round(luma, 1), **kw)

    def _fallback(self, frame: Frame, reason: str) -> PortraitResult:
        self.counters["fallback_frames"] += 1
        w, h = self.cfg.output_size
        img = letterbox(frame.image, w, h) if self.cfg.fallback == "original" else np.zeros((h, w, 3), np.uint8)
        return self._out(frame, img, PortraitStatus.AI_UNAVAILABLE, notes=[reason])

    def process(self, frame: Frame) -> PortraitResult:
        frame.validate()
        now = self.clock()
        dt = 0.0 if self._last_t is None else now - self._last_t
        self._last_t = now
        w, h = self.cfg.output_size
        if not self.cfg.ai_enabled or self.segmenter is None:
            return self._out(frame, letterbox(frame.image, w, h), PortraitStatus.ORIGINAL)
        if self._ai_down_until is not None and now < self._ai_down_until:
            return self._fallback(frame, "AI_UNAVAILABLE: cooling down after repeated failures")

        alpha, info, held, error = None, {}, False, None
        try:
            mask = self.segmenter.segment(frame)
            mask = validate_mask(mask, frame, now=self.clock(), max_age_s=self.cfg.max_mask_age_s)
            alpha = mask.alpha
            info = {"model_version": mask.model_version, "inference_ms": mask.inference_ms,
                    "confidence": mask.confidence, "coverage": mask.coverage}
            self._failures, self._ai_down_until = 0, None
        except MaskRejected as exc:
            self.counters["masks_rejected"] += 1
            error = f"mask rejected: {exc}"
        except Exception as exc:  # inference crash must not take down the pipeline (SYS-005)
            self.counters["inference_errors"] += 1
            error = f"inference error: {type(exc).__name__}: {exc}"

        if alpha is None:
            self._failures += 1
            if self._failures >= self.cfg.ai_failure_threshold:
                self._ai_down_until = now + self.cfg.ai_retry_s
            if self._last_good and now - self._last_good[0] <= self.cfg.mask_hold_s \
                    and self._last_good[1].shape == frame.image.shape[:2]:
                alpha, held = self._last_good[1], True
            else:
                return self._fallback(frame, error or "no mask")

        coverage = cv2.countNonZero(cv2.threshold(alpha, 0.5, 1.0, cv2.THRESH_BINARY)[1]) / alpha.size
        if coverage < self.cfg.min_coverage:
            if self._no_person_since is None:
                self._no_person_since = now
            if self._last_good and now - self._last_good[0] <= self.cfg.mask_hold_s \
                    and self._last_good[1].shape == alpha.shape:
                alpha, held = self._last_good[1], True  # brief dropout: keep the person visible
            elif now - self._no_person_since >= self.cfg.person_lost_timeout_s:
                self.counters["person_lost_frames"] += 1
                self._prev_alpha = None
                return self._out(frame, np.zeros((h, w, 3), np.uint8), PortraitStatus.PERSON_LOST, **info)
        else:
            self._no_person_since = None
            if not held:
                self._last_good = (now, alpha)

        # Matte work happens at reduced resolution; compositing only inside the output crop (C-016 profiling).
        k = max(1, int(round(max(frame.width, frame.height) / max(16, self.cfg.mask_long_side))))
        if k > 1:
            small = cv2.resize(alpha, (max(1, frame.width // k), max(1, frame.height // k)), interpolation=cv2.INTER_AREA)
            matte = replace(self.matte, edge_softness_px=self.matte.edge_softness_px / k)
        else:
            small, matte = alpha, self.matte
        sx, sy = small.shape[1] / frame.width, small.shape[0] / frame.height
        if self._prev_alpha is not None and self._prev_alpha.shape != small.shape:
            self._prev_alpha = None
        refined = refine_alpha(small, self._prev_alpha, matte)
        self._prev_alpha = refined
        if self.cfg.stabilize:
            if self._stab is None or (self._stab.fw, self._stab.fh) != (frame.width, frame.height):
                self._stab = RoiStabilizer(self.roi_params, frame.width, frame.height)
            roi = self._stab.update(small, dt, scale=(1 / sx, 1 / sy))
            x, y, rw, rh = roi.rect(self.roi_params.aspect)
            img_out = crop_rect_to_output(frame.image, (x, y, rw, rh), w, h)
            a_out = crop_rect_to_output(refined, (x * sx, y * sy, rw * sx, rh * sy), w, h)
        else:
            img_out = letterbox(frame.image, w, h)
            a_out = letterbox(refined if k == 1 else cv2.resize(refined, (frame.width, frame.height)), w, h)
        out = composite_on_black(img_out, a_out)
        status = PortraitStatus.LIVE_PORTRAIT
        if info.get("confidence") is not None and info["confidence"] < self.cfg.low_confidence:
            status = PortraitStatus.LOW_CONFIDENCE
        return self._out(frame, out, status, held_mask=held, **info)
