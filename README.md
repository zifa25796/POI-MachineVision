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
pip install face_recognition opencv-python numpy Pillow
```

## Setup

1. Clone the repo
2. Put 3–5 photos of yourself in `known_faces/` (jpg or png, front-facing, different lighting)
3. Run:

```bash
python machine_vision.py
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
├── run.bat              # Windows launcher
├── known_faces/         # Your photos (not tracked in git)
└── memory/              # Claude Code session memory
```
