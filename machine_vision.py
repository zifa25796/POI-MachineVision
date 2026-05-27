"""
machine_vision.py — Person of Interest "The Machine" HUD Simulator

Real-time face detection + identification with POI-style HUD overlay.
Detection runs on a down-scaled frame for speed; recognition is throttled
per tracked face (every N frames or when a new face appears).

Usage:
    pip install face_recognition opencv-python numpy Pillow
    python machine_vision.py
"""

import cv2
import numpy as np
import face_recognition
from datetime import datetime
import os
import pickle
import sys
import time

# ============================================================================
# CONFIGURABLE PARAMETERS
# ============================================================================

KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "known_faces")
TOLERANCE = 0.5                       # Face match tolerance (0 = strict, 1 = loose)
RECOGNITION_INTERVAL = 30             # Re-confirm identity every N frames per tracked face
PROCESS_SCALE = 0.25                  # Downscale factor for detection (0.25 = 1/4 size)
LERP_FACTOR = 0.15                    # Box position smoothing (0 = frozen, 1 = instant)
CAMERA_INDEX = 0
MIRROR_MODE = True
WINDOW_WIDTH = 720
WINDOW_HEIGHT = 720
CAMERA_WIDTH = 1280                   # Native capture resolution
CAMERA_HEIGHT = 720
WARMUP_FRAMES = 10                    # Discard N frames while AE / AWB / AF settle

# HUD palette  (BGR byte order -- #FFD700 = (0, 215, 255))
HUD_COLOR = (0, 215, 255)
ADMIN_COLOR = (0, 215, 255)
UNKNOWN_COLOR = (0, 165, 195)
HUD_ALPHA = 0.55

# Corner brackets (screen decoration)
BRACKET_ARM = 24
BRACKET_WEIGHT = 2
BRACKET_MARGIN = 28

# Face box style (segmented: solid corners + dashed edges + perpendicular centre tick)
CORNER_LEN = 20                      # Length of each solid corner arm (px)
TICK_HALF = 6                        # Half-length of centre cross-tick (12 px total)
DASH_LEN = 8                         # Dash segment length (px)
GAP_LEN = 8                          # Gap between dashes (px)
DASH_COLOR = (0, 0, 0)               # Black for dashed edges
BOX_WEIGHT = 2                       # Line thickness for face boxes
BOX_SCALE = 1.20                     # Scale face boxes (1.0 = exact, >1 = larger)
BOX_OFFSET_Y = -15                   # Vertical shift in px (negative = up)

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
_tracked_faces = {}                   # face_id -> {box, name, age, last_rec_frame, needs_recognition}
_next_face_id = 0

# ============================================================================
# FACE DATABASE  (load once at startup)
# ============================================================================

def load_known_faces(directory):
    """Scan *directory* for images, extract face encodings.  Uses pickle cache."""
    global _known_encodings

    if not os.path.isdir(directory):
        print(f"[ERROR]  Folder not found: '{directory}'")
        print("Create a 'known_faces/' folder and put your photos inside (jpg / png).")
        sys.exit(1)

    files = sorted(f for f in os.listdir(directory)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if not files:
        print(f"[ERROR]  No images in '{directory}/'.")
        sys.exit(1)

    cache_path = os.path.join(directory, ".encodings_cache.pkl")

    # Check if cache is fresh (same files, same mtimes)
    cache_valid = False
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                cache = pickle.load(fh)
            cached_sig = cache.get("sig", [])
            cur_sig = sorted((fn, os.path.getmtime(os.path.join(directory, fn)))
                             for fn in files)
            if cached_sig == cur_sig:
                _known_encodings = cache["encodings"]
                cache_valid = True
        except Exception:
            pass

    if cache_valid:
        print(f"  -> {len(_known_encodings)} encoding(s) loaded from cache.")
        return

    # Compute encodings from scratch
    encodings = []
    for fn in files:
        path = os.path.join(directory, fn)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            encodings.append(encs[0])
            print(f"  [OK]   {fn}")
        else:
            print(f"  [SKIP] {fn}  -- no face detected")

    if not encodings:
        print("[ERROR]  No faces found in any of the provided images.")
        sys.exit(1)

    # Save cache
    try:
        sig = sorted((fn, os.path.getmtime(os.path.join(directory, fn)))
                     for fn in files)
        with open(cache_path, "wb") as fh:
            pickle.dump({"sig": sig, "encodings": encodings}, fh)
    except Exception:
        pass

    print(f"  -> {len(encodings)} encoding(s) computed and cached for ADMIN.")
    _known_encodings = encodings

# ============================================================================
# DETECTION  (face_recognition HOG on down-scaled frame -- fast enough for CPU)
# ============================================================================

def detect_faces(rgb_small, scale):
    """
    Run face_recognition face_locations on a down-scaled RGB image.
    Returns a list of (top, right, bottom, left) boxes in original coords,
    with BOX_SCALE and BOX_OFFSET_Y applied.
    """
    locations_small = face_recognition.face_locations(rgb_small, model="hog")
    boxes = []
    for t_s, r_s, b_s, l_s in locations_small:
        t = int(t_s / scale)
        r = int(r_s / scale)
        b = int(b_s / scale)
        l = int(l_s / scale)
        # Expand from centre
        cw = (r - l) * (BOX_SCALE - 1.0) / 2.0
        ch = (b - t) * (BOX_SCALE - 1.0) / 2.0
        boxes.append((
            int(t - ch + BOX_OFFSET_Y),
            int(r + cw),
            int(b + ch + BOX_OFFSET_Y),
            int(l - cw),
        ))
    return boxes

# ============================================================================
# RECOGNITION  (face_recognition -- on-demand only)
# ============================================================================

def run_recognition_on_faces(rgb_full, tracked_faces, frame_no):
    """
    Call face_recognition on every tracked face whose *needs_recognition*
    flag is set.  Updates 'name' and 'last_rec_frame' in-place.
    Accepts full-resolution rgb frame for accurate encoding.
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
        rgb_full, known_face_locations=rec_boxes
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
            # Existing face -- lerp box, carry forward identity
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
            # New face -- trigger immediate recognition
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

def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len, gap_len):
    """Draw a dashed line from *pt1* to *pt2* by painting short segments."""
    x1, y1 = pt1
    x2, y2 = pt2
    total = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    if total == 0:
        return
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    step = dash_len + gap_len
    pos = 0
    while pos < total:
        end = min(pos + dash_len, total)
        cv2.line(img,
                 (int(x1 + dx * pos), int(y1 + dy * pos)),
                 (int(x1 + dx * end), int(y1 + dy * end)),
                 color, thickness, cv2.LINE_4)
        pos += step


def _draw_segmented_box(frame, left, top, right, bottom):
    """Draw a single segmented face box: solid L corners + dashed edges."""
    l, t, r, b = left, top, right, bottom
    cl = CORNER_LEN

    # --- solid L-shaped corners (HUD_COLOR) ---
    # Top-left
    cv2.line(frame, (l, t), (l, t + cl), HUD_COLOR, BOX_WEIGHT)
    cv2.line(frame, (l, t), (l + cl, t), HUD_COLOR, BOX_WEIGHT)
    # Top-right
    cv2.line(frame, (r - cl, t), (r, t), HUD_COLOR, BOX_WEIGHT)
    cv2.line(frame, (r, t), (r, t + cl), HUD_COLOR, BOX_WEIGHT)
    # Bottom-right
    cv2.line(frame, (r, b - cl), (r, b), HUD_COLOR, BOX_WEIGHT)
    cv2.line(frame, (r - cl, b), (r, b), HUD_COLOR, BOX_WEIGHT)
    # Bottom-left
    cv2.line(frame, (l, b - cl), (l, b), HUD_COLOR, BOX_WEIGHT)
    cv2.line(frame, (l, b), (l + cl, b), HUD_COLOR, BOX_WEIGHT)

    # --- dashed edges with perpendicular centre tick ---
    mx = (l + r) // 2                                      # horizontal midpoint
    my = (t + b) // 2                                      # vertical midpoint
    th = TICK_HALF

    # Top edge:  corner ─┄┄ dashes ┄┄ ─|─ ┄┄ dashes ┄┄─ corner
    _draw_dashed_line(frame, (l + cl, t), (mx, t),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)
    cv2.line(frame, (mx, t - th), (mx, t + th), HUD_COLOR, BOX_WEIGHT)
    _draw_dashed_line(frame, (mx, t), (r - cl, t),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)

    # Right edge
    _draw_dashed_line(frame, (r, t + cl), (r, my),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)
    cv2.line(frame, (r - th, my), (r + th, my), HUD_COLOR, BOX_WEIGHT)
    _draw_dashed_line(frame, (r, my), (r, b - cl),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)

    # Bottom edge
    _draw_dashed_line(frame, (r - cl, b), (mx, b),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)
    cv2.line(frame, (mx, b - th), (mx, b + th), HUD_COLOR, BOX_WEIGHT)
    _draw_dashed_line(frame, (mx, b), (l + cl, b),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)

    # Left edge
    _draw_dashed_line(frame, (l, b - cl), (l, my),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)
    cv2.line(frame, (l - th, my), (l + th, my), HUD_COLOR, BOX_WEIGHT)
    _draw_dashed_line(frame, (l, my), (l, t + cl),
                      DASH_COLOR, BOX_WEIGHT, DASH_LEN, GAP_LEN)


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
    # Bottom-right  -- vertical bar extends upward from corner
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
        _draw_segmented_box(frame, l, t, r, b)

        label_x = l
        label_y = t - 6
        if label_y < 22:
            label_y = b + 22
        _draw_label(frame, name, label_x, label_y, colour)

    # --- decorative HUD overlay (semi-transparent) ---
    hud_layer = np.zeros_like(frame)
    _draw_corner_brackets(hud_layer)

    cv2.putText(hud_layer, "THE MACHINE -- SURVEILLANCE FEED",
                (BRACKET_MARGIN + BRACKET_ARM + 10, BRACKET_MARGIN + 6),
                FONT, FONT_SCALE, HUD_COLOR, FONT_WEIGHT)

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

    for _ in range(WARMUP_FRAMES):
        cap.read()

    return cap

def _crop_centre_to_aspect(frame, target_w, target_h):
    """Centre-crop *frame* to match target aspect ratio, avoiding stretch."""
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    src_ratio = w / h
    if src_ratio > target_ratio:
        new_w = int(h * target_ratio)
        offset = (w - new_w) // 2
        frame = frame[:, offset:offset + new_w]
    elif src_ratio < target_ratio:
        new_h = int(w / target_ratio)
        offset = (h - new_h) // 2
        frame = frame[offset:offset + new_h, :]
    return frame


# ============================================================================
# ONE-SHOT IDENTITY CHECK  (for external callers, e.g. Daily companion)
# ============================================================================

def quick_identity_check(known_faces_dir=None,
                         camera_index: int = 0,
                         capture_warmup: int = 15) -> bool:
    """Open camera, take a frame, check if ADMIN is visible. Returns True/False."""
    directory = known_faces_dir or KNOWN_FACES_DIR

    # ensure known faces are loaded
    global _known_encodings
    if _known_encodings is None:
        if not os.path.isdir(directory):
            return False
        load_known_faces(directory)

    # open camera
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    try:
        for _ in range(capture_warmup):
            cap.read()

        ok, frame = cap.read()
        if not ok:
            return False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        boxes = detect_faces(small, PROCESS_SCALE)

        if not boxes:
            return False

        encodings = face_recognition.face_encodings(rgb, known_face_locations=boxes)
        for enc in encodings:
            matches = face_recognition.compare_faces(_known_encodings, enc, tolerance=TOLERANCE)
            if any(matches):
                return True
        return False
    finally:
        cap.release()


def run_hud_scan(duration_seconds: float = 5.0,
                 known_faces_dir=None,
                 camera_index: int = 0,
                 on_admin=None,
                 voice_delay: float = 1.0) -> bool:
    """Launch the full Machine HUD window for *duration_seconds*.
    Calls *on_admin* after *voice_delay* seconds from first ADMIN detection.
    Returns True if ADMIN was detected at any point during the scan."""

    global _tracked_faces, _next_face_id, _known_encodings

    directory = known_faces_dir or KNOWN_FACES_DIR
    if not os.path.isdir(directory):
        return False

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    for _ in range(WARMUP_FRAMES):
        cap.read()

    cv2.namedWindow("THE MACHINE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("THE MACHINE", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setWindowProperty("THE MACHINE", cv2.WND_PROP_TOPMOST, 1)
    cv2.waitKey(1)  # ensure window is rendered

    # Show camera feed while loading faces (avoids blank-window delay)
    loading_start = time.time()
    faces_loaded = _known_encodings is not None
    while not faces_loaded:
        ok, frame = cap.read()
        if not ok:
            break
        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        frame = _crop_centre_to_aspect(frame, WINDOW_WIDTH, WINDOW_HEIGHT).copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        clr = (0, 215, 255) if int(time.time() * 2) % 2 else (0, 140, 180)
        cv2.putText(frame, "INITIALIZING...", (220, WINDOW_HEIGHT // 2),
                    FONT, 1.0, clr, 2)
        cv2.imshow("THE MACHINE", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return False
        if cv2.getWindowProperty("THE MACHINE", cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            return False
        if _known_encodings is not None:
            faces_loaded = True
        elif time.time() - loading_start > 0.5:
            # Start loading without blocking the display thread
            if not os.path.isdir(directory):
                cap.release()
                cv2.destroyAllWindows()
                return False
            load_known_faces(directory)
        # Brief sleep to keep loading screen responsive
        time.sleep(0.03)

    _tracked_faces = {}
    _next_face_id = 0

    frame_no = 0
    fps_clock = time.time()
    start_time = time.time()
    admin_seen = False
    admin_first_at = None
    admin_notified = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if MIRROR_MODE:
            frame = cv2.flip(frame, 1)
        frame = _crop_centre_to_aspect(frame, WINDOW_WIDTH, WINDOW_HEIGHT).copy()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        small = cv2.resize(rgb, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        boxes = detect_faces(small, PROCESS_SCALE)
        tracked = update_tracker(boxes, frame_no)

        # Render FIRST then recognise — avoids video freeze during face encoding
        for data in tracked.values():
            if data.get("name") == "ADMIN":
                if not admin_seen:
                    admin_first_at = time.time()
                admin_seen = True

        if (on_admin is not None
                and admin_seen
                and not admin_notified
                and admin_first_at is not None
                and time.time() - admin_first_at >= voice_delay):
            on_admin()
            admin_notified = True

        draw_hud(frame, tracked)
        now_f = time.time()
        fps = 1.0 / (now_f - fps_clock + 0.0001)
        fps_clock = now_f
        cv2.putText(frame, f"{fps:.0f} fps",
                    (WINDOW_WIDTH - 90, 22), FONT, 0.4, HUD_COLOR, 1)

        cv2.imshow("THE MACHINE", frame)

        # Recognition runs after imshow so the video never freezes
        run_recognition_on_faces(rgb, tracked, frame_no)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if cv2.getWindowProperty("THE MACHINE", cv2.WND_PROP_VISIBLE) < 1:
            break
        if time.time() - start_time >= duration_seconds:
            break

        frame_no += 1

    cap.release()
    cv2.destroyAllWindows()
    return admin_seen


# ============================================================================
# MAIN
# ============================================================================

def main():
    global _tracked_faces, _next_face_id

    print("=" * 58)
    print("  THE  MACHINE  --  Surveillance Interface")
    print("  Person of Interest  .  HUD Simulator")
    print("=" * 58)

    # --- load known faces ---
    print("\n[LOAD]  Scanning known_faces/ ...")
    load_known_faces(KNOWN_FACES_DIR)

    # --- camera ---
    print("[CAM]   Opening webcam ...")
    cap = open_camera()

    cv2.namedWindow("THE MACHINE", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("THE MACHINE", WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setWindowProperty("THE MACHINE", cv2.WND_PROP_TOPMOST, 1)

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
        frame = _crop_centre_to_aspect(frame, WINDOW_WIDTH, WINDOW_HEIGHT)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ---- detection on down-scaled image (fast) ----
        small = cv2.resize(rgb, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        boxes = detect_faces(small, PROCESS_SCALE)

        # ---- tracking + lerp smoothing ----
        tracked = update_tracker(boxes, frame_no)

        # ---- on-demand recognition (full-res for accuracy) ----
        # ---- render HUD ----
        draw_hud(frame, tracked)

        # ---- FPS counter ----
        now = time.time()
        fps = 1.0 / (now - fps_clock + 0.0001)
        fps_clock = now
        cv2.putText(frame, f"{fps:.0f} fps",
                    (WINDOW_WIDTH - 90, 22), FONT, 0.4, HUD_COLOR, 1)

        cv2.imshow("THE MACHINE", frame)

        # Recognition after imshow — video never freezes
        run_recognition_on_faces(rgb, tracked, frame_no)

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
