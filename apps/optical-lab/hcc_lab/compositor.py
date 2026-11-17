"""Optical compositor: content -> black display framebuffer using a calibration profile.

Nothing but content is drawn in the optical scene unless DEBUG is on, and DEBUG is
always labelled (C-012, FR-VI-005, DSP-003). Screenshots of this framebuffer do NOT
prove an optical effect (D-013).
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibration import profile_to_geometry
from .geometry import Transform, Zone, build_transform, layout_zones, warp, zone_params

DEBUG_BANNER = "DEBUG - engineering overlay, not an optical result"


class OpticalCompositor:
    def __init__(self, width: int, height: int):
        if width <= 0 or height <= 0:
            raise ValueError("display size must be positive")
        self.w, self.h = int(width), int(height)
        self._cache_key = None
        self._cache: list[tuple[Zone, Transform]] = []

    def plan(self, profile: dict, content_w: int, content_h: int) -> list[tuple[Zone, Transform]]:
        key = (profile.get("checksum"), profile.get("mode"), repr(profile_to_geometry(profile)), content_w, content_h)
        if key != self._cache_key:
            params = profile_to_geometry(profile)
            zones = layout_zones(profile["mode"], params.viewport, self.w, self.h)
            self._cache = [(z, build_transform(zone_params(params, z), content_w, content_h, self.w, self.h)) for z in zones]
            self._cache_key = key
        return self._cache

    def render(self, content: np.ndarray, profile: dict, *, face_labels: bool = False,
               debug_lines: list[str] | None = None) -> np.ndarray:
        ch, cw = content.shape[:2]
        fb = np.zeros((self.h, self.w, 3), np.uint8)
        for zone, t in self.plan(profile, cw, ch):
            warp(content, t, self.w, self.h, clip=zone.viewport, out=fb)
            if face_labels and profile["mode"] == "four_face_pyramid":
                x, y, zw, zh = zone.viewport.to_pixels(self.w, self.h)
                cv2.putText(fb, zone.face_id, (int(x + zw * 0.04), int(y + zh * 0.12)), cv2.FONT_HERSHEY_SIMPLEX,
                            max(0.5, zh / 300), (0, 200, 255), 2, cv2.LINE_AA)
        gain = float(profile.get("content_brightness", 1.0))
        if gain < 0.999:
            fb = cv2.convertScaleAbs(fb, alpha=gain, beta=0)
        if debug_lines:
            self._debug_overlay(fb, debug_lines)
        return fb

    def _debug_overlay(self, fb: np.ndarray, lines: list[str]) -> None:
        scale = max(0.45, self.h / 1800)
        lh = int(26 * scale / 0.6)
        cv2.rectangle(fb, (0, 0), (int(self.w * 0.55), lh * (len(lines) + 2)), (25, 25, 25), -1)
        cv2.putText(fb, DEBUG_BANNER, (10, lh), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 165, 255), 1, cv2.LINE_AA)
        for i, s in enumerate(lines):
            cv2.putText(fb, s, (10, lh * (i + 2)), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 200), 1, cv2.LINE_AA)
