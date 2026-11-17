"""Live Portrait + compositor (C-006..C-010, C-019, AI-003, AI-006..AI-008, AI-012, D-006, D-007, DSP-008)."""
import numpy as np
import pytest

from hcc_lab import calibration as cal
from hcc_lab.compositor import OpticalCompositor
from hcc_lab.frames import Frame, Mask, MaskRejected, validate_mask
from hcc_lab.geometry import Viewport, layout_zones
from hcc_lab.portrait import (
    MatteParams, PortraitConfig, PortraitProcessor, PortraitStatus, RoiParams, RoiStabilizer,
    composite_on_black, refine_alpha,
)
from hcc_lab.segmentation import GroundTruthSegmenter, ScriptedSegmenter
from hcc_lab.sources import SyntheticProducer


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def synthetic_frames(scenario="steady", n=10, fps=30.0, clock=None, size=(320, 240)):
    p = SyntheticProducer(size[0], size[1], fps, scenario, clock=clock or FakeClock())
    p.open()
    return p, [p.produce() for _ in range(n)]


# ------------------------------------------------------------------ composite (C-008, AI-003)
def test_composite_exact_for_full_alpha_and_zero_alpha():
    img = np.random.default_rng(1).integers(0, 256, (6, 5, 3), dtype=np.uint8)
    assert np.array_equal(composite_on_black(img, np.ones((6, 5), np.float32)), img)
    assert composite_on_black(img, np.zeros((6, 5), np.float32)).max() == 0


def test_composite_half_alpha_rounds_and_keeps_channel_order():
    img = np.zeros((1, 1, 3), np.uint8)
    img[0, 0] = (255, 100, 1)  # B, G, R
    out = composite_on_black(img, np.full((1, 1), 0.5, np.float32))
    assert out[0, 0].tolist() == [128, 50, 1]  # B stays first


def test_composite_rejects_mismatched_alpha():
    with pytest.raises(ValueError):
        composite_on_black(np.zeros((4, 4, 3), np.uint8), np.zeros((3, 4), np.float32))


def test_matte_params_are_bounded():
    p = MatteParams(edge_softness_px=99, low=0.9, high=0.1, temporal=5).clamped()
    assert p.edge_softness_px == 8 and p.low <= 0.45 and p.high >= 0.55 and p.temporal <= 0.8


def test_refine_alpha_keeps_solid_interior_and_background():
    a = np.zeros((40, 40), np.float32)
    a[10:30, 10:30] = 1.0
    r = refine_alpha(a, None, MatteParams())
    assert r[20, 20] == pytest.approx(1.0) and r[0, 0] == pytest.approx(0.0)
    assert r.dtype == np.float32 and 0.0 <= r.min() and r.max() <= 1.0


# ------------------------------------------------------------------ mask contract (C-006, AI-006)
def _frame(fid=1, h=4, w=5, t=10.0):
    return Frame(fid, t, np.zeros((h, w, 3), np.uint8), "synthetic")


@pytest.mark.parametrize("alpha,fid,why", [
    (np.zeros((4, 4), np.float32), 1, "shape"),
    (np.zeros((4, 5), np.float64), 1, "dtype"),
    (np.full((4, 5), 2.0, np.float32), 1, "range"),
    (np.full((4, 5), np.nan, np.float32), 1, "NaN"),
    (np.zeros((4, 5), np.float32), 2, "frame_id"),
])
def test_bad_masks_rejected(alpha, fid, why):
    with pytest.raises(MaskRejected):
        validate_mask(Mask(fid, alpha, "m", 1.0, 1.0, 0.0), _frame())


def test_stale_mask_rejected():
    with pytest.raises(MaskRejected, match="too old"):
        validate_mask(Mask(1, np.zeros((4, 5), np.float32), "m", 1.0, 1.0, 0.0), _frame(t=10.0),
                      now=11.0, max_age_s=0.5)


def test_bad_mask_does_not_corrupt_output_and_falls_back():
    clock = FakeClock()
    _, frames = synthetic_frames(n=8, clock=clock)
    proc = PortraitProcessor(ScriptedSegmenter(nan=True), PortraitConfig(output_size=(90, 120)), clock=clock)
    r = proc.process(frames[0])
    assert r.status == PortraitStatus.AI_UNAVAILABLE and proc.counters["masks_rejected"] == 1
    assert r.image.shape == (120, 90, 3) and r.image.dtype == np.uint8


# ------------------------------------------------------------------ processor behaviour
def test_live_portrait_background_is_black():
    clock = FakeClock()
    _, frames = synthetic_frames(n=2, clock=clock)
    proc = PortraitProcessor(GroundTruthSegmenter(), PortraitConfig(output_size=(320, 240), stabilize=False), clock=clock)
    r = proc.process(frames[0])
    assert r.status == PortraitStatus.LIVE_PORTRAIT
    assert frames[0].image[0:5, 0:5].min() > 40  # the source background is NOT black ...
    assert r.image[0:5, 0:5].max() == 0  # ... but the composited background is
    assert r.image.max() > 100  # the person is visible


def test_dark_clothing_is_kept_when_mask_covers_it():
    import cv2

    clock = FakeClock()
    prod, frames = synthetic_frames("dark_clothing", n=1, clock=clock, size=(320, 240))
    proc = PortraitProcessor(GroundTruthSegmenter(), PortraitConfig(output_size=(320, 240), stabilize=False), clock=clock)
    r = proc.process(frames[0])
    gt = (frames[0].meta["gt_alpha"] > 0.5).astype(np.uint8)
    interior = cv2.erode(gt, np.ones((7, 7), np.uint8)) > 0
    # near-black clothing must survive compositing unchanged (no colour keying, C-007)
    assert np.abs(r.image[interior].astype(int) - frames[0].image[interior].astype(int)).max() <= 1


def test_ai_toggle_original_bypasses_segmentation():
    clock = FakeClock()
    _, frames = synthetic_frames(n=1, clock=clock)
    seg = ScriptedSegmenter()
    proc = PortraitProcessor(seg, PortraitConfig(output_size=(320, 240)), clock=clock)
    proc.set_ai_enabled(False)
    r = proc.process(frames[0])
    assert r.status == PortraitStatus.ORIGINAL and seg.calls == 0
    assert r.image[0:5, 0:5].max() > 0  # background visible in Original mode


def test_person_leaves_frame_goes_black_after_timeout_not_frozen():
    clock = FakeClock()
    prod = SyntheticProducer(320, 240, 30, "leave_frame", clock=clock)
    prod.open()
    cfg = PortraitConfig(output_size=(90, 120), person_lost_timeout_s=1.0, mask_hold_s=0.3)
    proc = PortraitProcessor(GroundTruthSegmenter(), cfg, clock=clock)
    statuses = []
    for i in range(int(30 * 3.5)):  # person present 0..2 s, then gone
        clock.t = 100.0 + i / 30
        r = proc.process(prod.produce())
        statuses.append((i / 30, r.status, int(r.image.max())))
    before = [s for t, s, _ in statuses if t < 1.9]
    assert all(s == PortraitStatus.LIVE_PORTRAIT for s in before)
    late = [(s, m) for t, s, m in statuses if t > 3.1]
    assert late and all(s == PortraitStatus.PERSON_LOST and m == 0 for s, m in late)
    # inside the hold window (<= 0.3 s after leaving) the person may still be shown
    assert all(s != PortraitStatus.PERSON_LOST for t, s, _ in statuses if 2.0 <= t < 2.9)


def test_ai_failure_falls_back_then_recovers():
    clock = FakeClock()
    prod, _ = synthetic_frames(n=0, clock=clock)
    seg = ScriptedSegmenter(fail=True)
    cfg = PortraitConfig(output_size=(90, 120), ai_failure_threshold=3, ai_retry_s=2.0, mask_hold_s=0.0)
    proc = PortraitProcessor(seg, cfg, clock=clock)
    out = []
    for i in range(5):
        clock.t += 1 / 30
        out.append(proc.process(prod.produce()))
    assert all(r.status == PortraitStatus.AI_UNAVAILABLE for r in out)
    calls_when_down = seg.calls
    clock.t += 0.5
    proc.process(prod.produce())
    assert seg.calls == calls_when_down, "no retry during cool-down"
    seg.fail = False
    clock.t += 2.0
    r = proc.process(prod.produce())
    assert r.status == PortraitStatus.LIVE_PORTRAIT


def test_dark_scene_gives_low_light_hint_not_false_success():
    # T-UX-04: a near-black camera image (measured 5.5/255 on the lab webcam) must say "too dark"
    clock = FakeClock()
    frame = Frame(1, clock(), np.full((120, 160, 3), 5, np.uint8), "camera")
    proc = PortraitProcessor(ScriptedSegmenter(base=_EmptySegmenter()), PortraitConfig(output_size=(60, 80),
                             person_lost_timeout_s=0.0, mask_hold_s=0.0), clock=clock)
    r = proc.process(frame)
    assert r.low_light and r.status == PortraitStatus.PERSON_LOST
    assert r.user_message.startswith("Слишком темно")


class _EmptySegmenter:
    model_version = "empty (test double)"

    def segment(self, frame):
        a = np.zeros((frame.height, frame.width), np.float32)
        return Mask(frame.frame_id, a, self.model_version, 1.0, 1.0, 0.0)

    def close(self):
        pass


def test_brief_model_dropout_uses_last_good_mask():
    clock = FakeClock()
    prod, _ = synthetic_frames(n=0, clock=clock)
    seg = ScriptedSegmenter()
    proc = PortraitProcessor(seg, PortraitConfig(output_size=(90, 120), mask_hold_s=0.3), clock=clock)
    clock.t += 1 / 30
    proc.process(prod.produce())
    seg.fail = True
    clock.t += 1 / 30
    r = proc.process(prod.produce())
    assert r.held_mask and r.status == PortraitStatus.LIVE_PORTRAIT


def test_roi_does_not_jitter_with_small_head_motion():
    clock = FakeClock()
    prod = SyntheticProducer(640, 480, 30, "steady", clock=clock)  # +-1.5 px motion
    prod.open()
    stab = RoiStabilizer(RoiParams(), 640, 480)
    hs, cxs = [], []
    for i in range(90):
        f = prod.produce()
        roi = stab.update(f.meta["gt_alpha"], 1 / 30)
        hs.append(roi.h)
        cxs.append(roi.cx)
    assert max(hs) - min(hs) < 1e-6 and max(cxs) - min(cxs) < 1e-6  # zero scale/position jitter


def test_roi_follows_large_motion_with_rate_limit():
    clock = FakeClock()
    prod = SyntheticProducer(640, 480, 30, "move", clock=clock)  # +-180 px drift over 6 s
    prod.open()
    p = RoiParams(max_speed=1.0)
    stab = RoiStabilizer(p, 640, 480)
    prev = None
    moved = 0.0
    for i in range(180):
        f = prod.produce()
        roi = stab.update(f.meta["gt_alpha"], 1 / 30)
        if prev is not None:
            step = abs(roi.cx - prev.cx)
            assert step <= p.max_speed * prev.h / 30 + 1e-6
            moved += step
        prev = roi
    assert moved > 100  # it does follow


# ------------------------------------------------------------------ compositor (D-005..D-012)
def test_compositor_single_reflector_outside_viewport_is_black():
    profile = cal.safe_default_profile()
    profile["viewport_normalized"] = {"x": 0.25, "y": 0.25, "w": 0.5, "h": 0.5}
    profile["content_brightness"] = 1.0
    profile = cal.resign(profile)
    fb = OpticalCompositor(200, 100).render(np.full((50, 50, 3), 255, np.uint8), profile)
    assert fb[:20].max() == 0 and fb[80:].max() == 0
    assert fb[50, 100].tolist() == [255, 255, 255]


def test_brightness_changes_level_not_geometry():
    content = np.full((50, 50, 3), 200, np.uint8)
    p1 = cal.safe_default_profile()
    p1["content_brightness"] = 1.0
    p2 = dict(p1, content_brightness=0.5)
    c = OpticalCompositor(120, 80)
    a, b = c.render(content, cal.resign(p1)), c.render(content, cal.resign(p2))
    assert np.array_equal(a > 0, b > 0)  # same lit area
    assert b.max() == 100 and a.max() == 200


def test_four_face_layout_has_four_disjoint_zones_inside_square():
    zones = layout_zones("four_face_pyramid", Viewport(), 1920, 1080)
    assert [z.face_id for z in zones] == ["F1", "F2", "F3", "F4"]
    assert sorted(z.base_rotation_deg for z in zones) == [0, 90, 180, 270]
    rects = [z.viewport.to_pixels(1920, 1080) for z in zones]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            overlap_x = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
            overlap_y = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
            assert overlap_x <= 1e-6 or overlap_y <= 1e-6
    # 3x3 cross in the centred 1080x1080 square: cell = 360 px
    front = rects[0]
    assert front == pytest.approx((420 + 360, 720, 360, 360))


def test_four_face_render_puts_same_content_in_each_zone_and_labels_faces():
    p = cal.safe_default_profile()
    p.update(mode="four_face_pyramid", content_brightness=1.0)
    p = cal.resign(p)
    content = np.zeros((40, 30, 3), np.uint8)
    content[:5, :, 2] = 255  # red band at the TOP of the content
    c = OpticalCompositor(300, 300)
    fb = c.render(content, p)
    zones = layout_zones("four_face_pyramid", Viewport(), 300, 300)
    lit = [fb[int(z.viewport.y * 300):int((z.viewport.y + z.viewport.h) * 300),
              int(z.viewport.x * 300):int((z.viewport.x + z.viewport.w) * 300)].max() for z in zones]
    assert all(v == 255 for v in lit)
    # back zone F3 has rotation 0: the red band sits at the top of that zone
    f3 = zones[2].viewport
    y0, x_mid = int(f3.y * 300), int((f3.x + f3.w / 2) * 300)
    assert fb[y0 + 1:y0 + 15, x_mid, 2].max() == 255
    labelled = c.render(content, p, face_labels=True)
    assert not np.array_equal(labelled, fb)


def test_debug_overlay_only_when_requested():
    p = cal.safe_default_profile()
    c = OpticalCompositor(400, 300)
    black = np.zeros((10, 10, 3), np.uint8)
    assert c.render(black, p).max() == 0
    assert c.render(black, p, debug_lines=["fps 30"]).max() > 0
