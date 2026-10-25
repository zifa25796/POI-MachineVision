# The Machine — Surveillance Interface

Person of Interest HUD simulator. Real-time face detection + identification
with a surveillance-style heads-up display inspired by *Person of Interest*'s
"The Machine".

## Features

- Real-time face detection via webcam (face_recognition / dlib HOG)
- Identity matching against a local photo folder — labels faces as `ADMIN` or `UNKNOWN`
- Segmented tracking boxes: solid L-corners, dashed edges, perpendicular centre ticks
- Smooth box tracking with lerp interpolation
- Decorative HUD: corner brackets, timestamp, surveillance header — all in Machine gold

## Requirements

- Python 3.10+
- A webcam

```bash
uv sync
```

## Setup

1. Clone the repo
2. Put 3–5 photos of yourself in `known_faces/` (jpg or png, front-facing, different lighting)
3. Run:

```bash
uv run python machine_vision.py
```

Or double-click `run.bat` on Windows.

Press **Q** or close the window to exit.

## Configuration

All tunable parameters are at the top of `machine_vision.py`:

| Parameter | Default | Description |
|---|---|---|
| `TOLERANCE` | 0.5 | Face match strictness |
| `RECOGNITION_INTERVAL` | 30 | Frames between re-identification |
| `LERP_FACTOR` | 0.15 | Box movement smoothing |
| `PROCESS_SCALE` | 0.25 | Detection downscale (speed vs accuracy) |
| `BOX_SCALE` | 1.20 | Face box size multiplier |
| `BOX_OFFSET_Y` | -15 | Vertical box offset (negative = up) |
| `HUD_COLOR` | (0, 215, 255) | Primary accent (#FFD700 gold) |
| `CORNER_LEN` | 20 | Corner arm length |
| `DASH_LEN` / `GAP_LEN` | 8 / 8 | Dash pattern |
| `TICK_HALF` | 6 | Centre tick half-length |

## File Structure

```
CameraView/
├── machine_vision.py    # Main program
├── hud_scan_runner.py   # One-shot subprocess adapter used by Daily
├── run.bat              # Windows launcher
├── known_faces/         # Your photos (not tracked in git)
└── pyproject.toml       # Standalone uv dependencies
```

## Running inside POI

From the POI root, use `uv run python CameraView/machine_vision.py`. `run.bat` selects the parent POI environment when `.gitmodules` is present; standalone checkouts use this project's environment. Python 3.10 is recommended for the native dependencies.

Face photos and encoding caches are local files excluded from Git; only `known_faces/.gitkeep` is tracked. Build with PyInstaller from this directory using `pyinstaller pyinstaller.spec`.

Daily launches `hud_scan_runner.py <duration_seconds>` with its own interpreter.
The runner writes `ADMIN_FIRST` on first detection, then `ADMIN` or `NONE` at
exit. These stdout lines are protocol messages, not formatted console logs.
