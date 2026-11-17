"""Rolling per-stage statistics (C-016, NFR-003). Real samples only; nothing is estimated."""
from __future__ import annotations

import threading
import time
from collections import deque


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


class Series:
    def __init__(self, window: int = 600):
        self.values: deque[float] = deque(maxlen=window)
        self.total = 0

    def add(self, v: float) -> None:
        self.values.append(float(v))
        self.total += 1

    def summary(self) -> dict:
        s = sorted(self.values)
        return {"n_window": len(s), "n_total": self.total,
                "p50": percentile(s, 0.5), "p95": percentile(s, 0.95), "max": s[-1] if s else None}


class RateMeter:
    """Events per second over a sliding time window."""

    def __init__(self, window_s: float = 2.0):
        self.window_s = window_s
        self.times: deque[float] = deque()
        self.total = 0

    def tick(self, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        self.times.append(t)
        self.total += 1
        while self.times and t - self.times[0] > self.window_s:
            self.times.popleft()

    def rate(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        while self.times and now - self.times[0] > self.window_s:
            self.times.popleft()
        if len(self.times) < 2:
            return 0.0
        span = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / span if span > 0 else 0.0


class PipelineStats:
    """Thread-safe counters for capture / inference / compose / render stages."""

    STAGES = ("capture", "processed", "rendered")

    def __init__(self):
        self.lock = threading.Lock()
        self.rates = {s: RateMeter() for s in self.STAGES}
        self.series = {k: Series() for k in ("inference_ms", "compose_ms", "capture_to_render_ms", "capture_to_processed_ms")}
        self.counters = {"dropped_capture_slot": 0, "dropped_processing_slot": 0, "masks_rejected": 0,
                         "inference_errors": 0, "fallback_frames": 0, "person_lost_frames": 0}

    def tick(self, stage: str, t: float | None = None) -> None:
        with self.lock:
            self.rates[stage].tick(t)

    def add(self, name: str, value: float) -> None:
        with self.lock:
            self.series[name].add(value)

    def inc(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] = self.counters.get(name, 0) + n

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "fps": {s: round(m.rate(), 2) for s, m in self.rates.items()},
                "frames_total": {s: m.total for s, m in self.rates.items()},
                "latency": {k: s.summary() for k, s in self.series.items()},
                "counters": dict(self.counters),
                "method": ("time.monotonic() in one process; capture_to_render = present call minus capture "
                           "callback time. Software-internal only, NOT motion-to-photon (TZ 20.2)."),
            }
