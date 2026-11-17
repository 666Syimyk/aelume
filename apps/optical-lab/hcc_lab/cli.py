"""hcc-lab command line (python -m hcc_lab <command>). Exit codes: 0 ok, 1 check failed, 2 usage/environment."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .paths import LAB_DIR, LOCAL_DATA_DIR, PATTERNS_DIR, REPORTS_DIR

DEFAULT_PROFILE = LOCAL_DATA_DIR / "optical-calibration.v1.json"


def _source_args(p: argparse.ArgumentParser, default_source: str) -> None:
    p.add_argument("--source", choices=["camera", "video", "synthetic", "pattern"], default=default_source)
    p.add_argument("--camera", type=int, default=0, help="camera index (see list-cameras)")
    p.add_argument("--video", type=Path, help="video file for --source video (must be licensed/consented)")
    p.add_argument("--scenario", default="steady", help="synthetic scenario: steady|move|leave_frame|dark_clothing|absent")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-ai", action="store_true", help="Original video, segmentation off")
    p.add_argument("--segmenter", choices=["mediapipe", "ground-truth"], default="mediapipe",
                   help="ground-truth = synthetic test double (synthetic source only)")
    p.add_argument("--model", type=int, choices=[0, 1], default=0, help="MediaPipe model: 0 general, 1 landscape")


def cmd_env(args) -> int:
    from .displays import list_displays
    from .envinfo import manifest
    from .sources import list_cameras

    m = manifest()
    m["displays"] = [d.__dict__ | {"profile_id": d.profile_id} for d in list_displays()]
    m["cameras"] = list_cameras(args.max_index) if args.probe_cameras else "not probed (use --probe-cameras)"
    try:
        from .segmentation import MediaPipeSelfieSegmenter

        seg = MediaPipeSelfieSegmenter(0)
        m["segmentation_model"] = seg.model_version
        seg.close()
    except Exception as exc:
        m["segmentation_model"] = f"UNAVAILABLE: {exc}"
    print(json.dumps(m, indent=2, ensure_ascii=False))
    return 0


def cmd_list_cameras(args) -> int:
    from .sources import list_cameras

    cams = list_cameras(args.max_index)
    if not cams:
        print("MEDIA_CAMERA_UNAVAILABLE: no camera could be opened (device missing, busy, or blocked by OS privacy settings)")
        return 1
    for c in cams:
        print(f"camera {c['index']}: readable={c['readable']} resolution={c['resolution']}")
    return 0


def cmd_list_displays(args) -> int:
    from .displays import list_displays

    for d in list_displays():
        print(f"display {d.index}: {d.name} {d.width}x{d.height} at ({d.x},{d.y}) primary={d.primary} id={d.profile_id}")
    return 0


def cmd_patterns(args) -> int:
    from . import patterns

    if args.check:
        problems = patterns.check_suite(Path(args.out) / "manifest.json", with_video=not args.no_video)
        for p in problems:
            print(f"MISMATCH: {p}")
        print("pattern suite reproducible" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    m = patterns.generate_suite(Path(args.out), with_video=not args.no_video)
    for e in m["patterns"] + m["videos"]:
        print(f"{e['sha256'][:16]}  {e['bytes']:>8}  {e['file']}")
    print(f"manifest: {Path(args.out) / 'manifest.json'}")
    return 0


def cmd_validate_profile(args) -> int:
    from .calibration import ProfileError, parse_profile_text, validation_errors

    try:
        profile = parse_profile_text(Path(args.path).read_text(encoding="utf-8"))
    except (OSError, ProfileError) as exc:
        print(f"INVALID: {exc}")
        return 1
    errors = validation_errors(profile)
    for e in errors:
        print(f"INVALID: {e}")
    if not errors:
        print(f"VALID: rev {profile['profile_revision']} {profile['mode']} {profile['checksum']}")
    return 1 if errors else 0


def cmd_run(args) -> int:
    from .app import run

    return run(args)


def cmd_bench(args) -> int:
    """Headless run for N seconds: real per-stage timings on this machine (C-016). No frames are stored."""
    from .app import build_pipeline
    from .calibration import resign, safe_default_profile
    from .compositor import OpticalCompositor
    from .session import SessionReport
    from .sources import SourceError
    from .stats import PipelineStats

    report = SessionReport("bench", vars(args))
    stats = PipelineStats()
    dw, dh = (int(v) for v in args.display_size.lower().split("x"))
    profile = safe_default_profile()
    profile["mode"] = args.mode
    profile = resign(profile)
    try:
        pipe, info = build_pipeline(args, stats)
    except SourceError as exc:
        print(f"ERROR: {exc}")
        report.notes.append(str(exc))
        report.finish(2, None)
        return 2
    comp = OpticalCompositor(dw, dh)
    t_end = time.monotonic() + args.seconds
    error = None
    try:
        while time.monotonic() < t_end:
            r = pipe.latest(0.2)
            if r is not None:
                comp.render(r.image, profile)
                pipe.mark_rendered(r)
                report.count_status(r.status.value + ("+LOW_LIGHT" if r.low_light else ""))
            if pipe.error:
                error = pipe.error
                break
    finally:
        pipe.stop()
    snap = stats.snapshot()
    if error:
        report.notes.append(error)
    report.notes.append(f"render = OpticalCompositor.render to an off-screen {dw}x{dh} framebuffer (no display scan-out)")
    out = report.finish(1 if error else 0, snap, model_version=info.get("model_version"), source_info=info.get("source"))
    print(json.dumps({"fps": snap["fps"], "latency": snap["latency"], "counters": snap["counters"],
                      "status_histogram": dict(report.status_hist)}, indent=2))
    print(f"report: {out}")
    return 1 if error else 0


def cmd_template(args) -> int:
    from .report import write_template

    print(f"template written: {write_template(Path(args.out))}")
    return 0


def cmd_report(args) -> int:
    from datetime import datetime, timezone

    from .report import evaluate_g0, load_measurements, load_thresholds, render_markdown

    paths = [Path(p) for p in args.measurements]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"ERROR: measurement file(s) not found: {missing}")
        return 2
    rows = load_measurements(paths)
    result = evaluate_g0(rows, load_thresholds(Path(args.thresholds) if args.thresholds else None))
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"G0_{stamp}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    md = render_markdown(result, paths)
    (out / f"G0_{stamp}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hcc-lab", description="HoloCall Cube D0 optical lab")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("env", help="environment manifest")
    p.add_argument("--probe-cameras", action="store_true")
    p.add_argument("--max-index", type=int, default=3)
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("list-cameras")
    p.add_argument("--max-index", type=int, default=4)
    p.set_defaults(func=cmd_list_cameras)

    sub.add_parser("list-displays").set_defaults(func=cmd_list_displays)

    p = sub.add_parser("patterns", help="generate or --check the calibration pattern suite")
    p.add_argument("--out", default=str(PATTERNS_DIR))
    p.add_argument("--check", action="store_true")
    p.add_argument("--no-video", action="store_true")
    p.set_defaults(func=cmd_patterns)

    p = sub.add_parser("validate-profile")
    p.add_argument("path")
    p.set_defaults(func=cmd_validate_profile)

    p = sub.add_parser("run", help="interactive fullscreen optical bench")
    _source_args(p, "camera")
    p.add_argument("--display", type=int, default=0)
    p.add_argument("--windowed", action="store_true", help="window instead of fullscreen (development)")
    p.add_argument("--window-w", type=int, default=1280)
    p.add_argument("--window-h", type=int, default=720)
    p.add_argument("--profile", default=str(DEFAULT_PROFILE))
    p.add_argument("--device-id", default="dev_lab_pc")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--duration", type=float, default=0, help="auto-exit after N seconds (0 = until q/Esc)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench", help="headless timing run (no window, nothing stored but numbers)")
    _source_args(p, "synthetic")
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--display-size", default="1920x1080")
    p.add_argument("--mode", choices=["single_reflector", "four_face_pyramid"], default="single_reflector")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("template", help="write the measurement CSV template")
    p.add_argument("--out", default=str(LAB_DIR / "templates" / "optical_measurements_template.csv"))
    p.set_defaults(func=cmd_template)

    p = sub.add_parser("report", help="G0 evidence report from human-entered measurements")
    p.add_argument("--measurements", nargs="+", required=True)
    p.add_argument("--thresholds", help="approved G0 thresholds JSON (absent -> NOT EVALUABLE)")
    p.add_argument("--out", default=str(REPORTS_DIR / "gates"))
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
