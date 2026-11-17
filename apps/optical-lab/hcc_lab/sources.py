"""Frame sources: camera, video file, synthetic, pattern (C-001..C-003, C-017, C-018, C-021).

Every live source runs its producer in one thread and hands frames over through a
single-slot mailbox: a slow consumer causes counted drops, never a growing queue.
"""
from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import cv2
import numpy as np

from .frames import Frame


class SourceError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class LatestSlot:
    """Single-slot mailbox. put() overwrites an unconsumed item and reports the drop."""

    def __init__(self):
        self._cond = threading.Condition()
        self._item = None
        self._fresh = False
        self.closed = False

    def put(self, item) -> bool:
        with self._cond:
            dropped = self._fresh
            self._item, self._fresh = item, True
            self._cond.notify_all()
            return dropped

    def get(self, timeout: float | None = None):
        with self._cond:
            if not self._fresh and not self.closed:
                self._cond.wait(timeout)
            if not self._fresh:
                return None
            self._fresh = False
            return self._item

    def close(self) -> None:
        with self._cond:
            self.closed = True
            self._cond.notify_all()

    @property
    def depth(self) -> int:
        return 1 if self._fresh else 0


class Producer(Protocol):
    kind: str

    def open(self) -> None: ...
    def produce(self) -> Frame | None: ...  # blocks until the next frame; None = end of stream
    def close(self) -> None: ...
    def info(self) -> dict: ...


class ThreadedSource:
    def __init__(self, producer: Producer, *, max_consecutive_failures: int = 30,
                 on_frame: Callable[[Frame], None] | None = None):
        self.producer = producer
        self.slot = LatestSlot()
        self.max_failures = max_consecutive_failures
        self.on_frame = on_frame
        self.dropped = 0
        self.captured = 0
        self.error: str | None = None
        self.eos = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def open(self) -> "ThreadedSource":
        self.producer.open()
        self._thread = threading.Thread(target=self._run, name=f"source-{self.producer.kind}", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        failures = 0
        try:
            while not self._stop.is_set():
                try:
                    frame = self.producer.produce()
                except SourceError as exc:
                    failures += 1
                    if failures >= self.max_failures:  # controlled stop, no infinite retry loop (AUD-013)
                        self.error = str(exc)
                        break
                    time.sleep(0.02)
                    continue
                if frame is None:
                    self.eos = True
                    break
                failures = 0
                self.captured += 1
                if self.on_frame:
                    self.on_frame(frame)
                if self.slot.put(frame):
                    self.dropped += 1
        finally:
            self.slot.close()

    def read(self, timeout: float = 1.0) -> Frame | None:
        return self.slot.get(timeout)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.producer.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ camera
def default_camera_api() -> int:
    # Windows: Media Foundation. Measured 2026-11-03 on the lab laptop (USB2.0 HD UVC WebCam):
    # MSMF 1280x720 = 29.8 fps, DirectShow = 9.9 fps at any size (stuck on YUY2). See docs/BASELINE_AUDIT.md.
    return cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY


class CameraProducer:
    kind = "camera"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 api: int | None = None, facing: str = "front"):
        self.index, self.req = index, (width, height, fps)
        self.api = default_camera_api() if api is None else api
        self.facing = facing
        self.cap: cv2.VideoCapture | None = None
        self.frame_id = 0
        self.actual: tuple[int, int, float] | None = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.index, self.api)
        if not cap.isOpened():
            cap.release()
            raise SourceError("MEDIA_CAMERA_UNAVAILABLE", f"camera index {self.index} could not be opened "
                              "(missing device, in use by another app, or OS privacy setting denies camera access)")
        w, h, fps = self.req
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, img = cap.read()
        if not ok or img is None:
            cap.release()
            raise SourceError("MEDIA_CAMERA_UNAVAILABLE", f"camera {self.index} opened but returned no frame")
        self.actual = (img.shape[1], img.shape[0], float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        self.cap = cap

    def produce(self) -> Frame | None:
        if self.cap is None:
            raise SourceError("MEDIA_CAMERA_UNAVAILABLE", "camera not open")
        ok, img = self.cap.read()
        t = time.monotonic()  # time the frame reached this process (not sensor exposure time)
        if not ok or img is None:
            raise SourceError("MEDIA_CAMERA_READ_FAILED", f"camera {self.index} read failed")
        self.frame_id += 1
        return Frame(self.frame_id, t, img, "camera", camera_facing=self.facing)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self) -> dict:
        names = {cv2.CAP_DSHOW: "DSHOW", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "ANY"}
        return {"kind": self.kind, "index": self.index, "requested": self.req, "actual": self.actual,
                "api": names.get(self.api, str(self.api))}


def list_cameras(max_index: int = 4, api: int | None = None) -> list[dict]:
    api = default_camera_api() if api is None else api
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i, api)
        try:
            if cap.isOpened():
                ok, img = cap.read()
                found.append({"index": i, "readable": bool(ok),
                              "resolution": f"{img.shape[1]}x{img.shape[0]}" if ok else None})
        finally:
            cap.release()
    return found


# ------------------------------------------------------------------ video file
class VideoFileProducer:
    kind = "video_file"

    def __init__(self, path: str | Path, *, loop: bool = True, realtime: bool = True):
        self.path, self.loop, self.realtime = Path(path), loop, realtime
        self.cap: cv2.VideoCapture | None = None
        self.fps = 30.0
        self.frame_id = 0
        self._next_t = 0.0

    def open(self) -> None:
        if not self.path.exists():
            raise SourceError("MEDIA_FILE_NOT_FOUND", str(self.path))
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise SourceError("MEDIA_FILE_UNREADABLE", str(self.path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and math.isfinite(fps) and fps > 0 else 30.0
        self.cap = cap
        self._next_t = time.monotonic()

    def produce(self) -> Frame | None:
        assert self.cap is not None
        ok, img = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
            if not ok:
                return None
        if self.realtime:
            self._next_t += 1.0 / self.fps
            delay = self._next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_t = time.monotonic()
        self.frame_id += 1
        return Frame(self.frame_id, time.monotonic(), img, "video_file", camera_facing="none")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def info(self) -> dict:
        return {"kind": self.kind, "path": str(self.path), "fps": self.fps}


# ------------------------------------------------------------------ synthetic (no real people)
@dataclass
class SyntheticScenario:
    name: str = "steady"  # steady | move | leave_frame | dark_clothing | absent
    jitter_px: float = 1.5  # small head motion amplitude
    drift_px: float = 0.0  # large horizontal motion amplitude
    leave_at_s: float | None = None
    clothing_bgr: tuple[int, int, int] = (90, 120, 160)

    @classmethod
    def named(cls, name: str) -> "SyntheticScenario":
        presets = {
            "steady": cls("steady"),
            "move": cls("move", drift_px=180.0),
            "leave_frame": cls("leave_frame", leave_at_s=2.0),
            "dark_clothing": cls("dark_clothing", clothing_bgr=(12, 12, 14)),
            "absent": cls("absent", leave_at_s=0.0),
        }
        if name not in presets:
            raise ValueError(f"unknown synthetic scenario {name!r}; choose from {sorted(presets)}")
        return presets[name]


class SyntheticProducer:
    """Procedural person-like silhouette on a textured background + exact ground-truth alpha.

    Frame i has virtual time i/fps (deterministic). ``realtime`` paces output like a camera.
    """

    kind = "synthetic"

    def __init__(self, width: int = 640, height: int = 480, fps: float = 30.0,
                 scenario: SyntheticScenario | str = "steady", *, realtime: bool = False,
                 clock: Callable[[], float] = time.monotonic, max_frames: int | None = None):
        self.w, self.h, self.fps = width, height, fps
        self.scenario = SyntheticScenario.named(scenario) if isinstance(scenario, str) else scenario
        self.realtime, self.clock, self.max_frames = realtime, clock, max_frames
        self.frame_id = 0
        self._t0 = 0.0
        yy, xx = np.mgrid[0:height, 0:width]
        bg = np.zeros((height, width, 3), np.float32)
        bg[..., 0] = 60 + 50 * (xx / width)
        bg[..., 1] = 90 + 40 * (yy / height)
        bg[..., 2] = 70 + 30 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
        stripes = ((xx // 40 + yy // 40) % 2) * 25
        self._bg = np.clip(bg + stripes[..., None], 0, 255).astype(np.uint8)

    def open(self) -> None:
        self.frame_id = 0
        self._t0 = self.clock()

    def person_center(self, t: float) -> tuple[float, float]:
        s = self.scenario
        cx = self.w / 2 + s.drift_px * math.sin(2 * math.pi * t / 6.0) + s.jitter_px * math.sin(2 * math.pi * 1.7 * t)
        cy = self.h * 0.42 + s.jitter_px * math.cos(2 * math.pi * 1.3 * t)
        return cx, cy

    def present(self, t: float) -> bool:
        return self.scenario.leave_at_s is None or t < self.scenario.leave_at_s

    def render(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        img = self._bg.copy()
        alpha = np.zeros((self.h, self.w), np.float32)
        if self.present(t):
            cx, cy = self.person_center(t)
            u = self.h / 480.0
            m = np.zeros((self.h, self.w), np.uint8)
            cv2.ellipse(m, (int(cx), int(cy - 60 * u)), (int(52 * u), int(66 * u)), 0, 0, 360, 255, -1)  # head
            cv2.rectangle(m, (int(cx - 22 * u), int(cy)), (int(cx + 22 * u), int(cy + 30 * u)), 255, -1)  # neck
            cv2.ellipse(m, (int(cx), int(cy + 150 * u)), (int(150 * u), int(125 * u)), 0, 180, 360, 255, -1)  # shoulders
            cv2.rectangle(m, (int(cx - 150 * u), int(cy + 150 * u)), (int(cx + 150 * u), self.h), 255, -1)  # torso
            sel = m > 0
            skin = np.array([150, 170, 200], np.uint8)  # generic flat tone, not a real face
            img[sel] = skin
            body = np.zeros_like(m)
            cv2.ellipse(body, (int(cx), int(cy + 150 * u)), (int(150 * u), int(125 * u)), 0, 180, 360, 255, -1)
            cv2.rectangle(body, (int(cx - 150 * u), int(cy + 150 * u)), (int(cx + 150 * u), self.h), 255, -1)
            img[body > 0] = self.scenario.clothing_bgr
            alpha = sel.astype(np.float32)
        return img, alpha

    def produce(self) -> Frame | None:
        if self.max_frames is not None and self.frame_id >= self.max_frames:
            return None
        t_virtual = self.frame_id / self.fps
        if self.realtime:
            delay = self._t0 + t_virtual - self.clock()
            if delay > 0:
                time.sleep(delay)
        img, alpha = self.render(t_virtual)
        self.frame_id += 1
        return Frame(self.frame_id, self.clock(), img, "synthetic", camera_facing="none",
                     meta={"gt_alpha": alpha, "t_virtual": t_virtual, "scenario": self.scenario.name})

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {"kind": self.kind, "size": f"{self.w}x{self.h}", "fps": self.fps, "scenario": self.scenario.name}


class PatternProducer:
    kind = "pattern"

    def __init__(self, image: np.ndarray, fps: float = 30.0, *, realtime: bool = True):
        self.image, self.fps, self.realtime = image, fps, realtime
        self.frame_id = 0
        self._next_t = 0.0

    def open(self) -> None:
        self._next_t = time.monotonic()

    def produce(self) -> Frame | None:
        if self.realtime:
            self._next_t += 1.0 / self.fps
            delay = self._next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.frame_id += 1
        return Frame(self.frame_id, time.monotonic(), self.image, "pattern", camera_facing="none")

    def close(self) -> None:
        pass

    def info(self) -> dict:
        return {"kind": self.kind, "size": f"{self.image.shape[1]}x{self.image.shape[0]}", "fps": self.fps}
