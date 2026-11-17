"""Person segmentation backends (C-005, AI-002, ADR-001).

MediaPipeSelfieSegmenter uses the model bundled inside the pinned ``mediapipe``
wheel (legacy Solutions API, Apache-2.0). Its model file SHA-256 is part of
``model_version`` so every session report names the exact model.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .frames import Frame, Mask, mask_stats


class Segmenter(Protocol):
    model_version: str

    def segment(self, frame: Frame) -> Mask: ...
    def close(self) -> None: ...


class SegmenterUnavailable(RuntimeError):
    code = "AI_UNAVAILABLE"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MediaPipeSelfieSegmenter:
    """model_selection 0 = general (256x256), 1 = landscape (144x256, faster)."""

    def __init__(self, model_selection: int = 0):
        try:
            import mediapipe as mp
            from mediapipe.python.solutions import selfie_segmentation
        except Exception as exc:  # ImportError, DLL load errors on Windows, ...
            raise SegmenterUnavailable(f"AI_UNAVAILABLE: mediapipe import failed: {exc}") from exc
        name = "selfie_segmentation_landscape.tflite" if model_selection == 1 else "selfie_segmentation.tflite"
        model = Path(mp.__file__).parent / "modules" / "selfie_segmentation" / name
        digest = _file_sha256(model) if model.exists() else "unknown"
        self.model_version = f"mediapipe-{mp.__version__}/{name}/sha256:{digest}"
        self.model_sha256 = digest
        self._seg = selfie_segmentation.SelfieSegmentation(model_selection=model_selection)

    def segment(self, frame: Frame) -> Mask:
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        result = self._seg.process(rgb)
        dt = (time.perf_counter() - t0) * 1000.0
        alpha = result.segmentation_mask
        if alpha is None:
            raise SegmenterUnavailable("AI_UNAVAILABLE: model returned no mask")
        alpha = np.asarray(alpha, dtype=np.float32)
        conf, cov = mask_stats(alpha)
        return Mask(frame.frame_id, alpha, self.model_version, dt, conf, cov)

    def close(self) -> None:
        self._seg.close()


class GroundTruthSegmenter:
    """TEST DOUBLE: returns the exact alpha shipped with synthetic frames. Not AI."""

    model_version = "synthetic-ground-truth (test double, not AI)"

    def segment(self, frame: Frame) -> Mask:
        alpha = frame.meta.get("gt_alpha")
        if alpha is None:
            raise SegmenterUnavailable("ground truth only exists for synthetic frames")
        conf, cov = mask_stats(alpha)
        return Mask(frame.frame_id, alpha.astype(np.float32, copy=False), self.model_version, 0.0, conf, cov)

    def close(self) -> None:
        pass


class ScriptedSegmenter:
    """TEST DOUBLE: delays / failures / bad masks on demand (fault injection, C-017, AI-012)."""

    model_version = "scripted (test double, not AI)"

    def __init__(self, base: Segmenter | None = None, *, delay_s: float = 0.0, fail: bool = False,
                 bad_shape: bool = False, nan: bool = False):
        self.base = base or GroundTruthSegmenter()
        self.delay_s, self.fail, self.bad_shape, self.nan = delay_s, fail, bad_shape, nan
        self.calls = 0

    def segment(self, frame: Frame) -> Mask:
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise SegmenterUnavailable("injected inference failure")
        m = self.base.segment(frame)
        alpha = m.alpha
        if self.bad_shape:
            alpha = alpha[: alpha.shape[0] // 2]
        if self.nan:
            alpha = alpha.copy()
            alpha[0, 0] = np.nan
        return Mask(m.frame_id, alpha, self.model_version, self.delay_s * 1000, m.confidence, m.coverage)

    def close(self) -> None:
        pass
