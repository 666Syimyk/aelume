"""Physical display enumeration (C-013, D-010). Windows: Win32 API; elsewhere: one fallback entry."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Display:
    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    primary: bool

    @property
    def profile_id(self) -> str:
        """Stable-ish id used to bind a calibration profile to a display (DSP-010)."""
        safe = re.sub(r"[^A-Za-z0-9_.:-]", "", self.name) or f"display{self.index}"
        return f"win:{safe}:{self.width}x{self.height}"[:64]


def _windows_displays() -> list[Display]:
    import ctypes
    from ctypes import wintypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # physical pixels, not scaled
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD), ("szDevice", wintypes.WCHAR * 32)]

    out: list[Display] = []
    proc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def cb(hmon, _hdc, _rect, _lp):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        if ctypes.windll.user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
            r = info.rcMonitor
            out.append(Display(len(out), info.szDevice, r.left, r.top, r.right - r.left, r.bottom - r.top,
                               bool(info.dwFlags & 1)))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(None, None, proc(cb), 0)
    return out


def list_displays() -> list[Display]:
    if sys.platform == "win32":
        try:
            found = _windows_displays()
            if found:
                return found
        except Exception:
            pass
    return [Display(0, "default", 0, 0, 1920, 1080, True)]


def set_cursor_visible(visible: bool) -> None:
    """Hide the mouse cursor over our windows in show mode (D-011). Windows only; no-op elsewhere."""
    if sys.platform != "win32":
        return
    import ctypes

    user32 = ctypes.windll.user32
    if visible:
        while user32.ShowCursor(True) < 0:
            pass
    else:
        while user32.ShowCursor(False) >= 0:
            pass
