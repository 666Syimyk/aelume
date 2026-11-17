"""Sources, bounded pipeline, patterns (C-001, C-002, C-017, C-018, C-020, B-008, B-009, AI-001, AI-005)."""
import json
import os
import time

import numpy as np
import pytest

from hcc_lab import patterns
from hcc_lab.portrait import PortraitConfig, PortraitProcessor
from hcc_lab.segmentation import GroundTruthSegmenter, ScriptedSegmenter
from hcc_lab.sources import (
    CameraProducer, LatestSlot, SourceError, SyntheticProducer, ThreadedSource,
)
from hcc_lab.pipeline import PortraitPipeline


# ------------------------------------------------------------------ slot / sources
def test_latest_slot_overwrites_and_counts_drop():
    s = LatestSlot()
    assert s.put(1) is False
    assert s.put(2) is True  # 1 was never consumed
    assert s.get(0) == 2 and s.get(0) is None


def test_synthetic_frames_have_increasing_ids_and_monotonic_time():
    p = SyntheticProducer(160, 120, 30, "steady")
    p.open()
    frames = [p.produce() for _ in range(20)]
    ids = [f.frame_id for f in frames]
    ts = [f.capture_time_monotonic for f in frames]
    assert ids == list(range(1, 21))
    assert all(b >= a for a, b in zip(ts, ts[1:]))
    assert all(f.meta["gt_alpha"].shape == (120, 160) for f in frames)


def test_threaded_source_never_queues_more_than_one_frame():
    src = ThreadedSource(SyntheticProducer(160, 120, 200, "steady", realtime=True, max_frames=60))
    with src:
        time.sleep(0.25)  # consumer is idle -> producer keeps overwriting
        f = src.read(1.0)
        assert f is not None
        assert src.slot.depth <= 1
        time.sleep(0.2)
    assert src.dropped > 10


class _FailingProducer:
    kind = "synthetic"

    def __init__(self):
        self.calls = 0

    def open(self):
        pass

    def produce(self):
        self.calls += 1
        raise SourceError("MEDIA_CAMERA_READ_FAILED", "unplugged")

    def close(self):
        pass

    def info(self):
        return {}


def test_repeated_capture_failures_stop_without_infinite_loop():
    prod = _FailingProducer()
    src = ThreadedSource(prod, max_consecutive_failures=5)
    src.open()
    time.sleep(0.5)
    assert not src.alive and src.error and "MEDIA_CAMERA_READ_FAILED" in src.error
    assert prod.calls == 5
    src.close()


def test_missing_camera_reports_clear_error():
    with pytest.raises(SourceError) as e:
        CameraProducer(index=97).open()
    assert e.value.code == "MEDIA_CAMERA_UNAVAILABLE"


# ------------------------------------------------------------------ pipeline (C-017, AI-005)
def test_slow_inference_drops_frames_instead_of_growing_latency():
    # 30 fps source, 100 ms model: without dropping, latency would grow ~70 ms per frame.
    src = ThreadedSource(SyntheticProducer(320, 240, 30, "steady", realtime=True))
    proc = PortraitProcessor(ScriptedSegmenter(delay_s=0.1), PortraitConfig(output_size=(90, 120)))
    pipe = PortraitPipeline(src, proc)
    latencies = []
    with pipe:
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end:
            r = pipe.latest(0.2)
            if r is not None:
                pipe.mark_rendered(r)
                latencies.append((time.monotonic() - r.capture_time_monotonic) * 1000)
    snap = pipe.stats.snapshot()
    assert len(latencies) >= 8
    assert max(latencies[-5:]) < 400, latencies  # bounded: ~ one inference + one frame
    assert snap["counters"]["dropped_capture_slot"] > 10
    assert pipe.max_queue_depth_seen <= 2


def test_pipeline_stop_releases_and_can_restart():
    for _ in range(3):  # C-018: repeated start/stop works
        src = ThreadedSource(SyntheticProducer(160, 120, 60, "steady", realtime=True))
        pipe = PortraitPipeline(src, PortraitProcessor(GroundTruthSegmenter(), PortraitConfig(output_size=(60, 80))))
        pipe.start()
        assert pipe.latest(1.0) is not None
        pipe.stop()
        pipe.stop()  # idempotent
        assert not src.alive


def test_pipeline_works_offline_without_camera_or_model():
    # C-020: synthetic source + no segmenter (Original mode) needs no camera, network or account
    src = ThreadedSource(SyntheticProducer(160, 120, 60, "steady", realtime=True))
    with PortraitPipeline(src, PortraitProcessor(None, PortraitConfig(output_size=(60, 80)))) as pipe:
        r = pipe.latest(1.0)
        assert r is not None and r.status.value == "ORIGINAL"


# ------------------------------------------------------------------ patterns (B-008, B-009)
def test_every_pattern_renders_at_two_sizes():
    for spec in patterns.PATTERNS:
        for w, h in ((640, 360), (300, 300)):
            img = patterns.render(spec.name, w, h)
            assert img.shape == (h, w, 3) and img.dtype == np.uint8 and img.max() > 0


def test_asymmetric_pattern_is_not_symmetric_under_any_flip_or_rotation():
    img = patterns.render("asym_lr_arrows", 400, 400)
    variants = [img[:, ::-1], img[::-1, :], np.rot90(img, 1), np.rot90(img, 2), np.rot90(img, 3)]
    assert all(not np.array_equal(img, v) for v in variants)
    assert img[2, 2].tolist() == [0, 0, 255]  # red marks the top-left corner


def test_rgb_bar_labelled_r_is_red_in_bgr():
    img = patterns.render("rgb_bars", 800, 100)
    assert img[10, 50].tolist() == [0, 0, 255]


def test_timer_frames_encode_frame_number():
    a, b = patterns.timer_frame(1, 640, 360, 30), patterns.timer_frame(2, 640, 360, 30)
    assert not np.array_equal(a, b)


def test_committed_pattern_suite_matches_manifest():
    manifest = patterns.PATTERNS_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip("pattern suite not generated yet (python -m hcc_lab patterns)")
    problems = patterns.check_suite(manifest, with_video=os.environ.get("HCC_CHECK_VIDEO") == "1")
    assert problems == [], problems


def test_manifest_lists_hashes_for_every_file():
    manifest = patterns.PATTERNS_DIR / "manifest.json"
    if not manifest.exists():
        pytest.skip("pattern suite not generated yet")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(m["patterns"]) == len(patterns.PATTERNS) * len(patterns.SUITE_SIZES)
    assert all(len(e["sha256"]) == 64 for e in m["patterns"] + m["videos"])
