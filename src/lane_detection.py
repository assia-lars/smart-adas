
import cv2
import numpy as np

print("\n--- ADAS Lane Detection ---")

# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/road2.mp4"

MAX_MISSING_FRAMES = 10
ALPHA = 0.80

# ==========================================
# 2. LOAD VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video:", VIDEO_PATH)
    exit()

print("Starting lane detection...")
print("Press Q to quit.")

# ==========================================
# 3. TRACKING VARIABLES
# ==========================================

previous_left_lane = None
previous_right_lane = None

missing_frames = 0


# ==========================================
# 4. LANE FITTING FUNCTION
# ==========================================

def get_lane_points(points, height, width):

    if len(points) < 2:
        return None

    points = np.array(points, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        points,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).flatten()

    if abs(vy) < 0.0001:
        return None

    y_bottom = int(height * 0.98)
    y_top = int(height * 0.65)

    x_bottom = int(x0 + (y_bottom - y0) * vx / vy)
    x_top = int(x0 + (y_top - y0) * vx / vy)

    if not (
        0 <= x_bottom < width
        and 0 <= x_top < width
    ):
        return None

    return (
        (x_bottom, y_bottom),
        (x_top, y_top)
    )


# ==========================================
# 5. VALIDATE LANE PAIR
# ==========================================

def valid_lane_pair(left, right, width):

    if left is None or right is None:
        return False

    left_bottom, left_top = left
    right_bottom, right_top = right

    top_width = right_top[0] - left_top[0]
    bottom_width = right_bottom[0] - left_bottom[0]

    return (
        width * 0.10 < top_width < width * 0.80
        and width * 0.15 < bottom_width < width * 0.80
    )


# ==========================================
# 6. MOVE A LANE
# ==========================================

def shift_lane(lane, dx_bottom, dx_top, width):

    if lane is None:
        return None

    bottom, top = lane

    new_bottom = (
        int(np.clip(bottom[0] + dx_bottom, 0, width - 1)),
        bottom[1]
    )

    new_top = (
        int(np.clip(top[0] + dx_top, 0, width - 1)),
        top[1]
    )

    return new_bottom, new_top


# ==========================================
# 7. SMOOTH LANES
# ==========================================

def smooth_lane(new_lane, old_lane, alpha):

    if old_lane is None:
        return new_lane

    result = []

    for new_point, old_point in zip(new_lane, old_lane):

        point = (
            int(alpha * new_point[0] + (1 - alpha) * old_point[0]),
            int(alpha * new_point[1] + (1 - alpha) * old_point[1])
        )

        result.append(point)

    return tuple(result)


# ==========================================
# 8. MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # ======================================
    # STEP 1: HSV COLOR DETECTION
    # ======================================

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 180]),
        np.array([180, 60, 255])
    )

    yellow_mask = cv2.inRange(
        hsv,
        np.array([15, 70, 100]),
        np.array([40, 255, 255])
    )

    lane_mask = cv2.bitwise_or(
        white_mask,
        yellow_mask
    )

    # ======================================
    # STEP 2: EDGE DETECTION
    # ======================================

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    combined = cv2.bitwise_and(edges, lane_mask)

    # ======================================
    # STEP 3: REGION OF INTEREST
    # ======================================

    roi_mask = np.zeros_like(combined)

    polygon = np.array([[
        (int(width * 0.05), height),
        (int(width * 0.40), int(height * 0.55)),
        (int(width * 0.60), int(height * 0.55)),
        (int(width * 0.95), height)
    ]], dtype=np.int32)

    cv2.fillPoly(roi_mask, polygon, 255)

    road_edges = cv2.bitwise_and(combined, roi_mask)

    # ======================================
    # STEP 4: HOUGH LINE DETECTION
    # ======================================

    lines = cv2.HoughLinesP(
        road_edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=100
    )

    left_points = []
    right_points = []

    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line.reshape(-1)[:4]

            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            slope = dy / dx

            if abs(slope) < 0.4:
                continue

            midpoint_x = (x1 + x2) / 2

            # Left lane
            if slope < 0 and midpoint_x < width * 0.65:

                left_points.extend([
                    (x1, y1),
                    (x2, y2)
                ])

            # Right lane
            elif slope > 0 and midpoint_x > width * 0.35:

                right_points.extend([
                    (x1, y1),
                    (x2, y2)
                ])

    # ======================================
    # STEP 5: FIT LANE LINES
    # ======================================

    detected_left = get_lane_points(
        left_points, height, width
    )

    detected_right = get_lane_points(
        right_points, height, width
    )

    # ======================================
    # STEP 6: TEMPORAL TRACKING
    # ======================================

    left_lane = None
    right_lane = None

    current_valid = valid_lane_pair(
        detected_left,
        detected_right,
        width
    )

    # Both lines detected
    if current_valid:

        left_lane = detected_left
        right_lane = detected_right

        # Light smoothing
        if previous_left_lane is not None:

            left_lane = smooth_lane(
                left_lane,
                previous_left_lane,
                ALPHA
            )

            right_lane = smooth_lane(
                right_lane,
                previous_right_lane,
                ALPHA
            )

        # Check again after smoothing
        if valid_lane_pair(left_lane, right_lane, width):

            missing_frames = 0

        else:

            left_lane = None
            right_lane = None

    # Only left line detected
    elif (
        detected_left is not None
        and detected_right is None
        and previous_left_lane is not None
        and previous_right_lane is not None
    ):

        old_bottom = previous_left_lane[0]
        old_top = previous_left_lane[1]

        new_bottom = detected_left[0]
        new_top = detected_left[1]

        dx_bottom = new_bottom[0] - old_bottom[0]
        dx_top = new_top[0] - old_top[0]

        estimated_right = shift_lane(
            previous_right_lane,
            dx_bottom,
            dx_top,
            width
        )

        if valid_lane_pair(detected_left, estimated_right, width):

            left_lane = detected_left
            right_lane = estimated_right

            missing_frames = 0

    # Only right line detected
    elif (
        detected_right is not None
        and detected_left is None
        and previous_left_lane is not None
        and previous_right_lane is not None
    ):

        old_bottom = previous_right_lane[0]
        old_top = previous_right_lane[1]

        new_bottom = detected_right[0]
        new_top = detected_right[1]

        dx_bottom = new_bottom[0] - old_bottom[0]
        dx_top = new_top[0] - old_top[0]

        estimated_left = shift_lane(
            previous_left_lane,
            dx_bottom,
            dx_top,
            width
        )

        if valid_lane_pair(estimated_left, detected_right, width):

            left_lane = estimated_left
            right_lane = detected_right

            missing_frames = 0

    # No reliable pair: use previous positions temporarily
    if left_lane is None or right_lane is None:

        missing_frames += 1

        if (
            previous_left_lane is not None
            and previous_right_lane is not None
            and missing_frames <= MAX_MISSING_FRAMES
        ):

            left_lane = previous_left_lane
            right_lane = previous_right_lane

    # ======================================
    # STEP 7: FINAL VALIDATION
    # ======================================

    output = frame.copy()

    valid_lanes = valid_lane_pair(
        left_lane,
        right_lane,
        width
    )

    if valid_lanes:

        previous_left_lane = left_lane
        previous_right_lane = right_lane

    # ======================================
    # STEP 8: DRAW GREEN AREA
    # ======================================

    if valid_lanes:

        left_bottom, left_top = left_lane
        right_bottom, right_top = right_lane

        lane_polygon = np.array([
            left_bottom,
            left_top,
            right_top,
            right_bottom
        ], dtype=np.int32)

        overlay = output.copy()

        cv2.fillPoly(
            overlay,
            [lane_polygon],
            (0, 255, 0)
        )

        output = cv2.addWeighted(
            overlay,
            0.25,
            output,
            0.75,
            0
        )

        # ==================================
        # STEP 9: DRAW GREEN LINES
        # ==================================

        cv2.line(
            output,
            left_bottom,
            left_top,
            (0, 255, 0),
            5
        )

        cv2.line(
            output,
            right_bottom,
            right_top,
            (0, 255, 0),
            5
        )

        # ==================================
        # STEP 10: BLUE CENTER LINE
        # ==================================

        center_bottom = (
            (left_bottom[0] + right_bottom[0]) // 2,
            (left_bottom[1] + right_bottom[1]) // 2
        )

        center_top = (
            (left_top[0] + right_top[0]) // 2,
            (left_top[1] + right_top[1]) // 2
        )

        cv2.line(
            output,
            center_bottom,
            center_top,
            (255, 0, 0),
            3
        )

        status = "LANE DETECTION: ACTIVE"
        color = (0, 255, 0)

    else:

        status = "LANE DETECTION: SEARCHING"
        color = (0, 165, 255)

    # ======================================
    # STEP 11: DISPLAY STATUS
    # ======================================

    cv2.putText(
        output,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    cv2.imshow("ADAS Lane Detection", output)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# ==========================================
# 12. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Lane detection completed!")
