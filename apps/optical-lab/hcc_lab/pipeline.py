"""Bounded realtime pipeline: source thread -> 1-slot -> processing thread -> 1-slot -> renderer.

Queues never exceed one item per stage; a slow model drops frames (counted) instead of
adding latency (C-017, AI-005, AI-011, RTC-008).
"""
from __future__ import annotations

import threading
import time

from .portrait import PortraitProcessor, PortraitResult
from .sources import LatestSlot, ThreadedSource
from .stats import PipelineStats


class PortraitPipeline:
    def __init__(self, source: ThreadedSource, processor: PortraitProcessor, stats: PipelineStats | None = None):
        self.source = source
        self.processor = processor
        self.stats = stats or PipelineStats()
        self.out = LatestSlot()
        self.max_queue_depth_seen = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> "PortraitPipeline":
        self.source.on_frame = lambda f: self.stats.tick("capture", f.capture_time_monotonic)
        self.source.open()
        self._thread = threading.Thread(target=self._run, name="portrait-processing", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        last_dropped = 0
        while not self._stop.is_set():
            frame = self.source.read(timeout=0.2)
            if frame is None:
                if self.source.error:
                    self.error = self.source.error
                    break
                if self.source.eos:
                    break
                continue
            if self.source.dropped != last_dropped:
                self.stats.inc("dropped_capture_slot", self.source.dropped - last_dropped)
                last_dropped = self.source.dropped
            t0 = time.perf_counter()
            result = self.processor.process(frame)
            self.stats.add("compose_ms", (time.perf_counter() - t0) * 1000 - (result.inference_ms or 0.0))
            if result.inference_ms is not None:
                self.stats.add("inference_ms", result.inference_ms)
            self.stats.add("capture_to_processed_ms", (time.monotonic() - frame.capture_time_monotonic) * 1000)
            self.stats.tick("processed")
            self.max_queue_depth_seen = max(self.max_queue_depth_seen, self.out.depth + 1)
            if self.out.put(result):
                self.stats.inc("dropped_processing_slot")
        self.out.close()

    def latest(self, timeout: float = 0.05) -> PortraitResult | None:
        return self.out.get(timeout)

    def mark_rendered(self, result: PortraitResult) -> None:
        self.stats.tick("rendered")
        self.stats.add("capture_to_render_ms", (time.monotonic() - result.capture_time_monotonic) * 1000)

    def stop(self) -> None:
        """Release camera/model/threads; safe to call twice (C-018)."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.source.close()
        seg = self.processor.segmenter
        if seg is not None:
            try:
                seg.close()
            except Exception:
                pass
            self.processor.segmenter = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
