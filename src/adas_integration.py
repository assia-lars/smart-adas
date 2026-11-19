
import cv2
import numpy as np
from ultralytics import YOLO

print("\n--- SMART ADAS: LANE DETECTION + VEHICLE TRACKING ---")

# ==================================================
# 1. CONFIGURATION
# ==================================================

VIDEO_PATH = "data/videos/road.mp4"
MODEL_PATH = "yolo11n.pt"
TRACKER_PATH = "bytetrack.yaml"

CONFIDENCE = 0.15
IOU = 0.65
IMAGE_SIZE = 960

VEHICLE_CLASSES = [2, 3, 5, 7]

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# Temporal smoothing
ALPHA = 0.75
MAX_MISSING_FRAMES = 12

# Lane geometry
Y_LEVELS = np.array([205, 230, 260, 290, 320, 350])

MIN_BOTTOM_WIDTH = 180
MAX_BOTTOM_WIDTH = 590
MIN_TOP_WIDTH = 35
MAX_TOP_WIDTH = 330

# Colors (BGR)
LANE_COLOR = (0, 145, 0)
CENTER_COLOR = (180, 75, 0)
LANE_FILL = (0, 110, 0)

VIDEO_DELAY = 100

# ==================================================
# 2. INITIALIZATION
# ==================================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    raise SystemExit

previous_left = None
previous_right = None
missing_frames = 0
frame_count = 0

print("Video started. Press Q to quit.")


# ==================================================
# 3. LANE GEOMETRY VALIDATION
# ==================================================

def validate_lanes(left, right):

    if left is None or right is None:
        return False

    widths = right - left

    if np.any(widths <= 0):
        return False

    bottom_width = widths[-1]
    top_width = widths[0]

    if not MIN_BOTTOM_WIDTH <= bottom_width <= MAX_BOTTOM_WIDTH:
        return False

    if not MIN_TOP_WIDTH <= top_width <= MAX_TOP_WIDTH:
        return False

    if np.any(np.abs(np.diff(widths)) > 100):
        return False

    return True


# ==================================================
# 4. FIT LANE BOUNDARIES
# ==================================================

def fit_lane(points, side):

    if len(points) < 4:
        return None

    points = np.array(points, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]

    try:

        # Quadratic fitting
        coefficients = np.polyfit(y, x, 2)

        fitted_x = np.polyval(coefficients, Y_LEVELS)

        if np.any(fitted_x < 0) or np.any(fitted_x > FRAME_WIDTH):
            return None

        if side == "left":

            if np.any(fitted_x > FRAME_WIDTH * 0.62):
                return None

        else:

            if np.any(fitted_x < FRAME_WIDTH * 0.38):
                return None

        return fitted_x.astype(np.float32)

    except (ValueError, np.linalg.LinAlgError):

        return None


# ==================================================
# 5. LANE DETECTION
# ==================================================

def detect_lanes(frame):

    global previous_left, previous_right, missing_frames

    height, width = frame.shape[:2]

    small = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    # White markings
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 165]),
        np.array([180, 75, 255])
    )

    # Yellow markings
    yellow_mask = cv2.inRange(
        hsv,
        np.array([12, 55, 75]),
        np.array([42, 255, 255])
    )

    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    edges = cv2.Canny(mask, 50, 150)

    # Region of interest
    roi = np.zeros_like(edges)

    polygon = np.array([[
        (25, 350),
        (215, 195),
        (425, 195),
        (615, 350)
    ]], dtype=np.int32)

    cv2.fillPoly(roi, polygon, 255)

    cropped_edges = cv2.bitwise_and(edges, roi)

    # Hough transform
    lines = cv2.HoughLinesP(
        cropped_edges,
        1,
        np.pi / 180,
        threshold=35,
        minLineLength=30,
        maxLineGap=60
    )

    left_points = []
    right_points = []

    # --------------------------------------------------
    # SAFE HOUGH LINE EXTRACTION
    # --------------------------------------------------

    if lines is not None:

        for line in lines:

            values = np.asarray(line).reshape(-1)

            if values.size != 4:
                continue

            x1, y1, x2, y2 = map(int, values)

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 5:
                continue

            slope = dy / dx

            if abs(slope) < 0.35:
                continue

            midpoint_x = (x1 + x2) / 2

            # Left boundary
            if slope < -0.35 and midpoint_x < 320:

                left_points.extend([
                    (x1, y1),
                    (x2, y2)
                ])

            # Right boundary
            elif slope > 0.35 and midpoint_x > 320:

                right_points.extend([
                    (x1, y1),
                    (x2, y2)
                ])

    # Fit both boundaries
    new_left = fit_lane(left_points, "left")
    new_right = fit_lane(right_points, "right")

    # --------------------------------------------------
    # TEMPORAL SMOOTHING
    # --------------------------------------------------

    if validate_lanes(new_left, new_right):

        if previous_left is None or previous_right is None:

            previous_left = new_left.copy()
            previous_right = new_right.copy()

        else:

            candidate_left = (
                ALPHA * previous_left
                + (1 - ALPHA) * new_left
            )

            candidate_right = (
                ALPHA * previous_right
                + (1 - ALPHA) * new_right
            )

            if validate_lanes(candidate_left, candidate_right):

                previous_left = candidate_left
                previous_right = candidate_right

            else:

                # Reject invalid update
                missing_frames += 1

        if missing_frames == 0 or validate_lanes(
            previous_left, previous_right
        ):
            missing_frames = 0

    else:

        missing_frames += 1

        if missing_frames > MAX_MISSING_FRAMES:

            previous_left = None
            previous_right = None

    # ==================================================
    # DRAW LANES
    # ==================================================

    lane_info = None

    if validate_lanes(previous_left, previous_right):

        scale_x = width / FRAME_WIDTH
        scale_y = height / FRAME_HEIGHT

        left_draw = np.array([
            [
                int(x * scale_x),
                int(y * scale_y)
            ]
            for x, y in zip(previous_left, Y_LEVELS)
        ], dtype=np.int32)

        right_draw = np.array([
            [
                int(x * scale_x),
                int(y * scale_y)
            ]
            for x, y in zip(previous_right, Y_LEVELS)
        ], dtype=np.int32)

        # Lane area
        lane_polygon = np.concatenate([
            left_draw,
            right_draw[::-1]
        ])

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [lane_polygon],
            LANE_FILL
        )

        cv2.addWeighted(
            overlay,
            0.18,
            frame,
            0.82,
            0,
            frame
        )

        # Left boundary
        cv2.polylines(
            frame,
            [left_draw],
            False,
            LANE_COLOR,
            3,
            cv2.LINE_AA
        )

        # Right boundary
        cv2.polylines(
            frame,
            [right_draw],
            False,
            LANE_COLOR,
            3,
            cv2.LINE_AA
        )

        # Center line
        center_draw = (
            left_draw.astype(np.float32)
            + right_draw.astype(np.float32)
        ) / 2

        center_draw = center_draw.astype(np.int32)

        cv2.polylines(
            frame,
            [center_draw],
            False,
            CENTER_COLOR,
            2,
            cv2.LINE_AA
        )

        lane_info = (left_draw, right_draw)

    return frame, lane_info


# ==================================================
# 6. VEHICLE LANE CLASSIFICATION
# ==================================================

def classify_vehicle(x, y, lane_info):

    if lane_info is None:
        return "LANE UNKNOWN"

    left_points, right_points = lane_info

    top_y = left_points[0][1]
    bottom_y = left_points[-1][1]

    target_y = max(top_y, min(y, bottom_y))

    left_x = np.interp(
        target_y,
        left_points[:, 1],
        left_points[:, 0]
    )

    right_x = np.interp(
        target_y,
        right_points[:, 1],
        right_points[:, 0]
    )

    if left_x >= right_x:
        return "LANE UNKNOWN"

    if left_x <= x <= right_x:
        return "OUR LANE"

    return "OTHER LANE"


# ==================================================
# 7. MAIN LOOP
# ==================================================

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # Lane detection
        frame, lane_info = detect_lanes(frame)

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_PATH,
            conf=CONFIDENCE,
            iou=IOU,
            imgsz=IMAGE_SIZE,
            max_det=300,
            classes=VEHICLE_CLASSES,
            verbose=False
        )

        result = results[0]

        detected_count = 0
        tracked_count = 0

        if result.boxes is not None:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                if class_id not in VEHICLE_CLASSES:
                    continue

                detected_count += 1

                vehicle_name = model.names[class_id]

                track_id = None

                if box.id is not None:

                    track_id = int(box.id[0])
                    tracked_count += 1

                center_x = (x1 + x2) // 2
                bottom_y = y2

                lane_position = classify_vehicle(
                    center_x,
                    bottom_y,
                    lane_info
                )

                # Approximate distance
                box_width = max(1, x2 - x1)

                distance = (1.8 * 900) / box_width

                # Vehicle colors
                if lane_position == "OTHER LANE":

                    color = (255, 0, 0)

                elif lane_position == "LANE UNKNOWN":

                    color = (0, 165, 255)

                elif distance < 10:

                    color = (0, 0, 220)

                elif distance < 20:

                    color = (0, 145, 220)

                else:

                    color = (0, 190, 0)

                # Bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                # Vehicle ID
                if track_id is not None:

                    label = f"ID {track_id} | {vehicle_name}"

                else:

                    label = f"{vehicle_name} | No ID"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA
                )

                # Lane + distance
                info = f"{lane_position} | {distance:.1f} m"

                cv2.putText(
                    frame,
                    info,
                    (x1, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA
                )

        # ==================================================
        # 8. INFORMATION PANEL
        # ==================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (350, 115),
            (30, 30, 30),
            -1
        )

        cv2.putText(
            frame,
            "SMART ADAS",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2
        )

        cv2.putText(
            frame,
            f"Vehicles detected: {detected_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 200),
            1
        )

        cv2.putText(
            frame,
            f"Vehicles tracked: {tracked_count}",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 200),
            1
        )

        lane_status = (
            "LANES DETECTED"
            if lane_info is not None
            else "LANES NOT DETECTED"
        )

        cv2.putText(
            frame,
            lane_status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (170, 200, 170),
            1
        )

        cv2.imshow(
            "SMART ADAS - Lane Detection + Vehicle Tracking",
            frame
        )

        if cv2.waitKey(VIDEO_DELAY) & 0xFF == ord('q'):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()

    print("\nADAS stopped.")
