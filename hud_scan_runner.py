"""Standalone runner — called as a subprocess by Daily.
Usage: python hud_scan_runner.py <duration_seconds>
Prints "ADMIN_FIRST" immediately on detection, then "ADMIN" or "NONE" at exit.
"""

import sys
from machine_vision import run_hud_scan


def _on_admin():
    print("ADMIN_FIRST", flush=True)


if __name__ == "__main__":
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    result = run_hud_scan(duration_seconds=duration, on_admin=_on_admin, voice_delay=0)
    print("ADMIN" if result else "NONE", flush=True)
