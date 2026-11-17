"""Interactive D0 optical bench (C-012..C-014, C-024, D-010, D-011, D-014).

A fullscreen black scene on the chosen display; calibration is edited live with the
keyboard and saved atomically with S. The optical window shows content only; every
engineering overlay is labelled DEBUG.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import cv2
import numpy as np

from .calibration import (ERROR_CODE, binding_warnings, load_profile, profile_to_geometry, resign,
                          safe_default_profile, save_profile_atomic)
from .compositor import OpticalCompositor
from .displays import list_displays, set_cursor_visible
from .geometry import layout_zones
from .patterns import PATTERNS, render
from .pipeline import PortraitPipeline
from .portrait import PortraitConfig, PortraitProcessor
from .segmentation import GroundTruthSegmenter, MediaPipeSelfieSegmenter, SegmenterUnavailable
from .session import SessionReport
from .sources import CameraProducer, SourceError, SyntheticProducer, ThreadedSource, VideoFileProducer
from .stats import PipelineStats

WIN = "HoloCall Optical Output"
STALE_S = 0.5  # no new processed frame for this long -> black + NO VIDEO (SYS-006)
ARROWS = {2424832: "offset_left", 2555904: "offset_right", 2490368: "offset_up", 2621440: "offset_down",
          65361: "offset_left", 65363: "offset_right", 65362: "offset_up", 65364: "offset_down"}
KEYMAP = {ord("+"): "scale_up", ord("="): "scale_up", ord("-"): "scale_down", ord("x"): "mirror_x",
          ord("y"): "mirror_y", ord("r"): "rotate", ord("]"): "brightness_up", ord("["): "brightness_down",
          ord("."): "viewport_grow", ord(","): "viewport_shrink", ord("m"): "mode"}
HELP = """keys: arrows=offset  +/-=scale  x/y=mirror  r=rotate 90  [ ]=brightness  , .=viewport size
      m=single/four-face  f=fine/coarse step  a=AI on/off  p=next pattern  l=live source  d=DEBUG overlay
      s=SAVE profile  0=reset to safe default (unsaved)  h=help  q/Esc=quit"""


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def apply_edit(profile: dict, action: str, *, fine: bool = False) -> tuple[dict, str]:
    """Pure calibration edit with safe bounds; returns (new_profile, message). Not re-signed (S saves)."""
    p = copy.deepcopy(profile)
    step = 0.005 if fine else 0.02
    off, vp = p["offset_normalized"], p["viewport_normalized"]
    if action.startswith("offset_"):
        dx = {"offset_left": -step, "offset_right": step}.get(action, 0.0)
        dy = {"offset_up": -step, "offset_down": step}.get(action, 0.0)
        off["x"], off["y"] = round(_clamp(off["x"] + dx, -1, 1), 4), round(_clamp(off["y"] + dy, -1, 1), 4)
        msg = f"offset ({off['x']:+.3f}, {off['y']:+.3f})"
    elif action in ("scale_up", "scale_down"):
        f = 1.01 if fine else 1.05
        p["scale"] = round(_clamp(p["scale"] * (f if action == "scale_up" else 1 / f), 0.1, 4.0), 4)
        msg = f"scale {p['scale']:.3f}"
    elif action in ("mirror_x", "mirror_y"):
        p[action] = not p[action]
        msg = f"{action} = {p[action]}"
    elif action == "rotate":
        p["rotation_deg"] = (p["rotation_deg"] + 90) % 360
        msg = f"rotation {p['rotation_deg']} deg"
    elif action in ("brightness_up", "brightness_down"):
        b = p.get("content_brightness", 1.0) + (0.05 if action == "brightness_up" else -0.05)
        p["content_brightness"] = round(_clamp(b, 0.05, 1.0), 2)
        msg = f"content brightness {p['content_brightness']:.2f} (software gain, not backlight)"
    elif action in ("viewport_grow", "viewport_shrink"):
        d = step if action == "viewport_grow" else -step
        cx, cy = vp["x"] + vp["w"] / 2, vp["y"] + vp["h"] / 2
        w, h = _clamp(vp["w"] + d, 0.05, 1.0), _clamp(vp["h"] + d, 0.05, 1.0)
        x, y = _clamp(cx - w / 2, 0.0, 1.0 - w), _clamp(cy - h / 2, 0.0, 1.0 - h)
        p["viewport_normalized"] = {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}
        msg = f"viewport {p['viewport_normalized']}"
    elif action == "mode":
        p["mode"] = "four_face_pyramid" if p["mode"] == "single_reflector" else "single_reflector"
        msg = f"mode {p['mode']} (four-face = same video on 4 faces, not 3D views)"
    else:
        raise ValueError(f"unknown edit action {action!r}")
    profile_to_geometry(p).validate()  # never produce an unrenderable profile
    return p, msg


def build_pipeline(args, stats: PipelineStats) -> tuple[PortraitPipeline, dict]:
    if args.source == "camera":
        producer = CameraProducer(args.camera, args.width, args.height, args.fps)
    elif args.source == "video":
        if not args.video:
            raise SourceError("MEDIA_FILE_NOT_FOUND", "--video PATH is required for --source video")
        producer = VideoFileProducer(args.video)
    elif args.source == "synthetic":
        producer = SyntheticProducer(args.width, args.height, args.fps, args.scenario, realtime=True)
    else:
        raise ValueError(f"no pipeline for source {args.source}")
    segmenter, model = None, None
    if not args.no_ai:
        if args.source == "synthetic" and args.segmenter == "ground-truth":
            segmenter = GroundTruthSegmenter()
        else:
            try:
                segmenter = MediaPipeSelfieSegmenter(args.model)
            except SegmenterUnavailable as exc:
                print(f"WARNING: {exc} -> running in ORIGINAL mode")
        model = getattr(segmenter, "model_version", None)
    proc = PortraitProcessor(segmenter, PortraitConfig(ai_enabled=segmenter is not None))
    pipe = PortraitPipeline(ThreadedSource(producer), proc, stats)
    pipe.start()
    return pipe, {"source": producer.info(), "model_version": model}


def _content_size(profile: dict, w: int, h: int) -> tuple[int, int]:
    params = profile_to_geometry(profile)
    z = layout_zones(profile["mode"], params.viewport, w, h)[0]
    _, _, zw, zh = z.viewport.to_pixels(w, h)
    rot = (z.base_rotation_deg + params.rotation_deg) % 360
    cw, ch = (zh, zw) if rot in (90, 270) else (zw, zh)
    return max(16, int(round(cw))), max(16, int(round(ch)))


def run(args) -> int:
    report = SessionReport("run", vars(args))
    stats = PipelineStats()
    displays = list_displays()
    if not 0 <= args.display < len(displays):
        print(f"ERROR: display {args.display} not found; available: {[(d.index, d.name, d.width, d.height) for d in displays]}")
        return 2
    disp = displays[args.display]
    out_w, out_h = (args.window_w, args.window_h) if args.windowed else (disp.width, disp.height)
    profile_path = Path(args.profile)
    lr = load_profile(profile_path, device_id=args.device_id, display_profile_id=disp.profile_id)
    for w in lr.warnings + binding_warnings(lr.profile, display_profile_id=disp.profile_id):
        print(f"WARNING: {w}")
    profile, dirty = lr.profile, False
    service = f"{ERROR_CODE}: safe profile + diagnostic pattern" if lr.error_code else None
    mode = "pattern" if args.source == "pattern" or lr.error_code else "live"
    pattern_idx, debug, fine = 0, args.debug, False
    pipe, info, last, last_t = None, {}, None, 0.0
    exit_code = 0
    if args.source != "pattern":
        try:
            pipe, info = build_pipeline(args, stats)
        except SourceError as exc:
            print(f"ERROR: {exc}. Showing patterns instead; fix the source and press L to retry.")
            service, mode = exc.code, "pattern"
    report.data.update(info)
    comp = OpticalCompositor(out_w, out_h)
    pattern_cache: dict = {}
    print(f"Display {disp.index} {disp.name} {disp.width}x{disp.height} at ({disp.x},{disp.y}); profile "
          f"{profile_path} [{lr.source}] rev {profile.get('profile_revision')}\n{HELP}")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if args.windowed:
        cv2.resizeWindow(WIN, out_w, out_h)
    else:
        cv2.moveWindow(WIN, disp.x, disp.y)
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        set_cursor_visible(False)
    t_end = time.monotonic() + args.duration if args.duration else None
    try:
        while True:
            now = time.monotonic()
            if t_end and now >= t_end:
                break
            shown = None
            if mode == "live" and pipe is not None:
                r = pipe.latest(0.01)
                if r is not None:
                    last, last_t, shown = r, now, r
                    report.count_status(r.status.value + ("+LOW_LIGHT" if r.low_light else ""))
                if pipe.error:
                    service = f"{pipe.error} - press L to retry"
                content = last.image if last is not None and now - last_t <= STALE_S else None
            else:
                name = PATTERNS[pattern_idx % len(PATTERNS)].name
                size = _content_size(profile, out_w, out_h)
                key = (name, size)
                if key not in pattern_cache:
                    pattern_cache = {key: render(name, *size)}
                content = pattern_cache[key]
            dbg = None
            if debug:
                s = stats.snapshot()
                lat = s["latency"]
                f = lambda d: "n/a" if d["p50"] is None else f"{d['p50']:.0f}/{d['p95']:.0f}"  # noqa: E731
                dbg = [f"mode {mode} | source {args.source} | AI {'on' if pipe and pipe.processor.cfg.ai_enabled else 'off'}"
                       f" | status {last.status.value if last else '-'}",
                       f"fps capture {s['fps']['capture']} processed {s['fps']['processed']} rendered {s['fps']['rendered']}",
                       f"inference p50/p95 {f(lat['inference_ms'])} ms | capture->render {f(lat['capture_to_render_ms'])} ms",
                       f"drops capture {s['counters']['dropped_capture_slot']} processing {s['counters']['dropped_processing_slot']}",
                       f"profile rev {profile['profile_revision']} [{lr.source}]{' UNSAVED' if dirty else ''} {profile['mode']}"
                       f" mx={profile['mirror_x']} my={profile['mirror_y']} rot={profile['rotation_deg']}"
                       f" scale={profile['scale']:.3f} off=({profile['offset_normalized']['x']:+.3f},"
                       f"{profile['offset_normalized']['y']:+.3f}) gain={profile.get('content_brightness', 1):.2f}",
                       f"display {disp.profile_id} | step {'fine' if fine else 'coarse'}"]
            black = np.zeros((16, 16, 3), np.uint8)
            fb = comp.render(content if content is not None else black, profile,
                             face_labels=(mode == "pattern"), debug_lines=dbg)
            if mode == "live" and content is None:
                cv2.putText(fb, "NO VIDEO", (20, out_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 1, cv2.LINE_AA)
            if service:
                cv2.putText(fb, service[:110], (20, out_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1, cv2.LINE_AA)
            cv2.imshow(WIN, fb)
            if shown is not None:
                pipe.mark_rendered(shown)
            k = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == -1:
                continue
            ch = k & 0xFF if k < 0x10000 else -1
            action = ARROWS.get(k) or KEYMAP.get(ch)
            if ch in (27, ord("q")):
                break
            if action:
                try:
                    profile, msg = apply_edit(profile, action, fine=fine)
                    dirty = True
                    print(msg)
                except ValueError as exc:
                    print(f"edit rejected: {exc}")
            elif ch == ord("f"):
                fine = not fine
            elif ch == ord("d"):
                debug = not debug
            elif ch == ord("h"):
                print(HELP)
            elif ch == ord("p"):
                pattern_idx = pattern_idx + 1 if mode == "pattern" else pattern_idx
                mode = "pattern"
                print(f"pattern {PATTERNS[pattern_idx % len(PATTERNS)].pid} {PATTERNS[pattern_idx % len(PATTERNS)].name}")
            elif ch == ord("l"):
                if pipe is None or pipe.error:
                    if pipe:
                        pipe.stop()
                    try:
                        pipe, info = build_pipeline(args, stats) if args.source != "pattern" else (None, {})
                        report.data.update(info)
                        service = None if pipe else service
                    except SourceError as exc:
                        service = exc.code
                        print(f"ERROR: {exc}")
                mode = "live" if pipe else "pattern"
            elif ch == ord("a") and pipe is not None:
                pipe.processor.set_ai_enabled(not pipe.processor.cfg.ai_enabled and pipe.processor.segmenter is not None)
                print(f"AI {'ON (Live Portrait)' if pipe.processor.cfg.ai_enabled else 'OFF (Original video)'}")
            elif ch == ord("s"):
                profile = resign(profile)
                save_profile_atomic(profile_path, profile)
                dirty, service = False, None
                print(f"saved {profile_path} rev {profile['profile_revision']} {profile['checksum'][:19]}")
            elif ch == ord("0"):
                profile, dirty = safe_default_profile(args.device_id, disp.profile_id), True
                print("reset to SAFE DEFAULT (not saved; press S to save)")
    except KeyboardInterrupt:
        exit_code = 130
    finally:
        if pipe is not None:
            pipe.stop()
        cv2.destroyAllWindows()
        set_cursor_visible(True)
        if dirty:
            print("NOTE: unsaved calibration changes were discarded (press S to save next time)")
        out = report.finish(exit_code, stats.snapshot(), model_version=report.data.get("model_version"),
                            source_info=report.data.get("source"),
                            profile={"path": str(profile_path), "loaded_from": lr.source,
                                     "revision": profile.get("profile_revision"), "unsaved_changes": dirty})
        print(f"session report: {out}")
    return exit_code
