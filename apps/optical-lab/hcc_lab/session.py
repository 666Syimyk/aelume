"""Run reports: reports/runs/<utc>_<name>/session.json + SUMMARY.md (A-015, master prompt 13).

Reports contain measurements and metadata only: never camera frames, masks or audio.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .envinfo import manifest
from .paths import REPORTS_DIR


class SessionReport:
    def __init__(self, name: str, args: dict, *, out_root: Path | None = None):
        self.name = name
        self.args = {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items() if k != "func"}
        self.started_utc = datetime.now(timezone.utc)
        self.t0 = time.monotonic()
        self.status_hist: Counter[str] = Counter()
        self.notes: list[str] = []
        self.out_root = out_root or REPORTS_DIR / "runs"
        self.data: dict = {}

    def count_status(self, status: str) -> None:
        self.status_hist[status] += 1

    def finish(self, exit_code: int, stats: dict | None = None, **extra) -> Path:
        stamp = self.started_utc.strftime("%Y-%m-%dT%H-%M-%SZ")
        out = self.out_root / f"{stamp}_lab-{self.name}"
        out.mkdir(parents=True, exist_ok=True)
        body = {
            "report": f"hcc_lab {self.name}", "started_utc": self.started_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": round(time.monotonic() - self.t0, 3), "exit_code": exit_code, "args": self.args,
            "environment": manifest(), "stats": stats, "status_histogram": dict(self.status_hist),
            "notes": self.notes, **extra,
        }
        (out / "session.json").write_text(json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
                                          encoding="utf-8")
        (out / "SUMMARY.md").write_text(self._summary(body), encoding="utf-8")
        return out

    @staticmethod
    def _summary(b: dict) -> str:
        lines = [f"# {b['report']} — {b['started_utc']}", "",
                 f"- exit code: `{b['exit_code']}`, duration {b['duration_s']} s, build `{b['environment']['build_id']}`",
                 f"- args: `{json.dumps(b['args'], ensure_ascii=False, default=str)}`"]
        s = b.get("stats") or {}
        if s:
            lat = s.get("latency", {})
            fmt = lambda d: "n/a" if not d or d.get("p50") is None else f"p50 {d['p50']:.1f} / p95 {d['p95']:.1f} ms (n={d['n_window']})"  # noqa: E731
            lines += [f"- fps: {s.get('fps')}", f"- frames total: {s.get('frames_total')}",
                      f"- inference: {fmt(lat.get('inference_ms'))}", f"- compose: {fmt(lat.get('compose_ms'))}",
                      f"- capture->processed: {fmt(lat.get('capture_to_processed_ms'))}",
                      f"- capture->render: {fmt(lat.get('capture_to_render_ms'))}",
                      f"- counters: {s.get('counters')}", f"- method: {s.get('method')}"]
        if b.get("status_histogram"):
            lines.append(f"- portrait status histogram: {b['status_histogram']}")
        for k in ("model_version", "source_info", "profile"):
            if b.get(k):
                lines.append(f"- {k}: `{json.dumps(b[k], ensure_ascii=False, default=str)}`")
        lines += [f"- note: {n}" for n in b.get("notes", [])]
        lines += ["", "Software-internal timings only; not motion-to-photon and not an optical result.", ""]
        return "\n".join(lines)
