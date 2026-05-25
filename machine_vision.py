"""
machine_vision.py — Person of Interest "The Machine" HUD Simulator

Dual-engine architecture:
  - MediaPipe FaceDetection  →  fast bounding-box detection every frame
  - face_recognition         →  identity confirmation on-demand only
    (triggered when a new face appears, or every RECOGNITION_INTERVAL frames)

Usage:
    pip install face_recognition opencv-python numpy Pillow mediapipe
    python machine_vision.py
"""

import cv2
import numpy as np
import face_recognition
import mediapipe as mp
from datetime import datetime
import os
import sys
import time

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.5                       # Face match tolerance (0 = strict, 1 = loose)
RECOGNITION_INTERVAL = 30             # Re-confirm identity every N frames per tracked face
LERP_FACTOR = 0.15                    # Box position smoothing (0 = frozen, 1 = instant)
CAMERA_INDEX = 0
MIRROR_MODE = True
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 1280                   # Native capture resolution
CAMERA_HEIGHT = 720
WARMUP_FRAMES = 30                    # Discard N frames while AE / AWB / AF settle

# MediaPipe
MP_DETECTION_CONFIDENCE = 0.6
MP_MODEL_SELECTION = 0                # 0 = short-range (< 2 m), 1 = long-range

# HUD palette  (BGR byte order — #FFD700 = (0, 215, 255))
HUD_COLOR = (0, 215, 255)
ADMIN_COLOR = (0, 215, 255)
UNKNOWN_COLOR = (0, 165, 195)
HUD_ALPHA = 0.55

# Corner brackets
BRACKET_ARM = 24
BRACKET_WEIGHT = 2
BRACKET_MARGIN = 28

# Typography
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
FONT_WEIGHT = 1
LABEL_SCALE = 0.55
LABEL_WEIGHT = 2

# Tracker
TRACK_TTL = 5                         # Drop stale tracks after N missed frames

# ============================================================================
# GLOBAL STATE
# ============================================================================

_known_encodings = None
_tracked_faces = {}                   # face_id → {box, name, age, last_rec_frame, needs_recognition}
_next_face_id = 0
_mp_face_detection = None

# ============================================================================
# FACE DATABASE  (load once at startup)
# ============================================================================

def load_known_faces(directory):
    """Scan *directory* for images, extract face encodings. Exits on failure."""
    global _known_encodings

    if not os.path.isdir(directory):
        print(f"[ERROR]  Folder not found: '{directory}'")
        print("Create a 'known_faces/' folder and put your photos inside (jpg / png).")
        sys.exit(1)

    files = [f for f in os.listdir(directory)
             if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        print(f"[ERROR]  No images in '{directory}/'.")
        sys.exit(1)

    encodings = []
    for fn in files:
        path = os.path.join(directory, fn)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            encodings.append(encs[0])
            print(f"  [OK]   {fn}")
        else:
            print(f"  [SKIP] {fn}  — no face detected")

    if not encodings:
        print("[ERROR]  No faces found in any of the provided images.")
        sys.exit(1)

    print(f"  -> {len(encodings)} encoding(s) loaded for ADMIN.")
    _known_encodings = encodings

# ============================================================================
# DETECTION  (MediaPipe — fast, runs every frame)
# ============================================================================

def detect_faces_mediapipe(rgb_frame):
    """
    Run MediaPipe FaceDetection on *rgb_frame*.
    Returns a list of (top, right, bottom, left) boxes in pixel coords.
    """
    h, w = rgb_frame.shape[:2]
    results = _mp_face_detection.process(rgb_frame)
    boxes = []
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            left   = max(0,           int(bbox.xmin * w))
            top    = max(0,           int(bbox.ymin * h))
            right  = min(w, int((bbox.xmin + bbox.width)  * w))
            bottom = min(h, int((bbox.ymin + bbox.height) * h))
            boxes.append((top, right, bottom, left))
    return boxes

# ============================================================================
# RECOGNITION  (face_recognition — on-demand only)
# ============================================================================

def run_recognition_on_faces(rgb_frame, tracked_faces, frame_no):
    """
    Call face_recognition on every tracked face whose *needs_recognition*
    flag is set.  Updates 'name' and 'last_rec_frame' in-place.
    """
    rec_fids = []
    rec_boxes = []

    for fid, data in tracked_faces.items():
        if data.get("needs_recognition", True):
            rec_fids.append(fid)
            rec_boxes.append(data["box"])

    if not rec_boxes:
        return

    encodings = face_recognition.face_encodings(
        rgb_frame, known_face_locations=rec_boxes
    )
    for fid, enc in zip(rec_fids, encodings):
        matches = face_recognition.compare_faces(_known_encodings, enc, tolerance=TOLERANCE)
        tracked_faces[fid]["name"] = "ADMIN" if any(matches) else "UNKNOWN"
        tracked_faces[fid]["last_rec_frame"] = frame_no
        tracked_faces[fid]["needs_recognition"] = False

# ============================================================================
# FACE TRACKING  (centre-proximity matching + lerp smoothing)
# ============================================================================

def _box_centre(box):
    """(top, right, bottom, left) -> (cx, cy)."""
    t, r, b, l = box
    return ((l + r) / 2.0, (t + b) / 2.0)


def update_tracker(detected_boxes, frame_no):
    """
    Match incoming detections to existing tracked faces by closest centre.
    Lerp matched boxes toward new positions; instantiate new faces for
    unmatched detections.  Age and cull stale tracks.
    """
    global _tracked_faces, _next_face_id

    # Age all existing tracks
    for data in _tracked_faces.values():
        data["age"] += 1

    if not detected_boxes:
        _tracked_faces = {
            fid: d for fid, d in _tracked_faces.items()
            if d["age"] < TRACK_TTL
        }
        return _tracked_faces

    det_centres = [_box_centre(b) for b in detected_boxes]
    track_items = list(_tracked_faces.items())
    track_centres = [_box_centre(d["box"]) for _, d in track_items]

    used = set()
    new_tracks = {}

    for i, dc in enumerate(det_centres):
        # Find closest unmatched tracked face
        best_j = None
        best_dist = float("inf")
        for j, tc in enumerate(track_centres):
            if j in used:
                continue
            dist = (dc[0] - tc[0]) ** 2 + (dc[1] - tc[1]) ** 2
            if dist < best_dist:
                best_dist = dist
                best_j = j

        if best_j is not None and best_dist < 80 * 80:
            # Existing face — lerp box, carry forward identity
            fid = track_items[best_j][0]
            used.add(best_j)
            old = _tracked_faces[fid]
            old_box = old["box"]
            new_box = detected_boxes[i]
            lerped = tuple(
                int(o + (n - o) * LERP_FACTOR) for o, n in zip(old_box, new_box)
            )
            needs_rec = (
                old.get("needs_recognition", False) or
                (frame_no - old.get("last_rec_frame", -RECOGNITION_INTERVAL)) >= RECOGNITION_INTERVAL
            )
            new_tracks[fid] = {
                "box":               lerped,
                "name":              old.get("name", "UNKNOWN"),
                "age":               0,
                "last_rec_frame":    old.get("last_rec_frame", -1),
                "needs_recognition": needs_rec,
            }
        else:
            # New face — trigger immediate recognition
            fid = _next_face_id
            _next_face_id += 1
            new_tracks[fid] = {
                "box":               detected_boxes[i],
                "name":              "UNKNOWN",
                "age":               0,
                "last_rec_frame":    -1,
                "needs_recognition": True,
            }

    _tracked_faces = new_tracks
    return _tracked_faces

# ============================================================================
# HUD RENDERING
# ============================================================================

def _draw_label(frame, text, x, y, colour):
    """Black-background pill with yellow text."""
    (tw, th), _ = cv2.getTextSize(text, FONT, LABEL_SCALE, LABEL_WEIGHT)
    pad = 4
    bx = max(0, x)
    by = max(0, y - th - pad)
    bw, bh = tw + pad * 2, th + pad * 2
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (bx + pad, by + th + pad),
                FONT, LABEL_SCALE, colour, LABEL_WEIGHT)


def _draw_corner_brackets(layer):
    """Four L-shaped decorative brackets in the screen corners."""
    h, w = layer.shape[:2]
    m = BRACKET_MARGIN
    a = BRACKET_ARM
    t = BRACKET_WEIGHT
    c = HUD_COLOR

    # Top-left
    cv2.line(layer, (m, m + a), (m, m), c, t)
    cv2.line(layer, (m, m), (m + a, m), c, t)
    # Top-right
    cv2.line(layer, (w - m - a, m), (w - m, m), c, t)
    cv2.line(layer, (w - m, m), (w - m, m + a), c, t)
    # Bottom-right  — vertical bar extends upward from corner
    cv2.line(layer, (w - m, h - m - a), (w - m, h - m), c, t)
    cv2.line(layer, (w - m, h - m), (w - m - a, h - m), c, t)
    # Bottom-left
    cv2.line(layer, (m, h - m - a), (m, h - m), c, t)
    cv2.line(layer, (m, h - m), (m + a, h - m), c, t)


def draw_hud(frame, tracked_faces):
    """
    Composite the full Machine HUD onto *frame* in-place.
    Face boxes + labels are opaque; decorative elements are alpha-blended.
    """
    h, w = frame.shape[:2]

    # --- face boxes & labels (opaque) ---
    for data in tracked_faces.values():
        t, r, b, l = data["box"]
        name = data.get("name", "UNKNOWN")
        colour = ADMIN_COLOR if name == "ADMIN" else UNKNOWN_COLOR

        cv2.rectangle(frame, (l, t), (r, b), colour, 2)

        label_x = l
        label_y = t - 6
        if label_y < 22:               # near top edge -> put label below the box
            label_y = b + 22
        _draw_label(frame, name, label_x, label_y, colour)

    # --- decorative HUD overlay (semi-transparent) ---
    hud_layer = np.zeros_like(frame)

    _draw_corner_brackets(hud_layer)

    # Top-left header
    cv2.putText(hud_layer, "THE MACHINE — SURVEILLANCE FEED",
                (BRACKET_MARGIN + BRACKET_ARM + 10, BRACKET_MARGIN + 6),
                FONT, FONT_SCALE, HUD_COLOR, FONT_WEIGHT)

    # Bottom-right timestamp
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    (tw, th), _ = cv2.getTextSize(ts, FONT, FONT_SCALE, FONT_WEIGHT)
    cv2.putText(hud_layer, ts,
                (w - tw - BRACKET_MARGIN - BRACKET_ARM - 10, h - BRACKET_MARGIN + 6),
                FONT, FONT_SCALE, HUD_COLOR, FONT_WEIGHT)

    blended = cv2.addWeighted(hud_layer, HUD_ALPHA, frame, 1.0, 0)
    frame[:] = blended

# ============================================================================
# CAMERA
# ============================================================================

def open_camera():
    """Open webcam, set resolution + autofocus, discard warmup frames."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR]  Cannot access camera. Check CAMERA_INDEX.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Discard frames while auto-exposure, white-balance, and focus settle
    for _ in range(WARMUP_FRAMES):
        cap.read()

    return cap

# ============================================================================
# MAIN
# ============================================================================

def main():
    global _tracked_faces, _next_face_id, _mp_face_detection

    print("=" * 58)
    print("  THE  MACHINE  —  Surveillance Interface")
    print("  Person of Interest  .  HUD Simulator")
    print("=" * 58)

    # --- init MediaPipe ---
    print("\n[INIT]  Starting MediaPipe face detection ...")
    mp_face_detection = mp.solutions.face_detection
    _mp_face_detection = mp_face_detection.FaceDetection(
        model_selection=MP_MODEL_SELECTION,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
    )

    # --- load known faces ---
    print("[LOAD]  Scanning known_faces/ ...")
    load_known_faces(KNOWN_FACES_DIR)

    # --- camera ---
    print("[CAM]   Opening webcam ...")
    cap = open_camera()

    cv2.namedWindow("THE MACHINE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("THE MACHINE", WINDOW_WIDTH, WINDOW_HEIGHT)

    frame_no = 0
    fps_clock = time.time()

    print("[RUN]   Press Q or close the window to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR]  Frame read failed.")
            break

        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- MediaPipe detection (every frame, fast) ----
        boxes = detect_faces_mediapipe(rgb)

        # ---- tracking + lerp smoothing ----
        tracked = update_tracker(boxes, frame_no)

        # ---- on-demand face_recognition ----
        run_recognition_on_faces(rgb, tracked, frame_no)

        # ---- render HUD ----
        draw_hud(frame, tracked)

        # ---- FPS counter ----
        now = time.time()
        fps = 1.0 / (now - fps_clock + 0.0001)
        fps_clock = now
        cv2.putText(frame, f"{fps:.0f} fps",
                    (WINDOW_WIDTH - 90, 22), FONT, 0.4, HUD_COLOR, 1)

        cv2.imshow("THE MACHINE", frame)

        # Quit on Q key or window close (X button)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty("THE MACHINE", cv2.WND_PROP_VISIBLE) < 1:
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
    print("\n[EXIT]  Surveillance terminated.")


if __name__ == "__main__":
    main()
