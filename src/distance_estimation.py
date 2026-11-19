
import cv2
from ultralytics import YOLO

print("\n--- ADAS Vehicle Distance Estimation ---")

# ==========================================
# 1. CONFIGURATION
# ==========================================

VIDEO_PATH = "data/videos/road2.mp4"
MODEL_PATH = "yolo11n.pt"

# Approximate real vehicle width (meters)
REAL_VEHICLE_WIDTH = 1.8

# Approximate camera focal length in pixels
FOCAL_LENGTH = 900

CONFIDENCE = 0.40

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ==========================================
# 2. LOAD MODEL AND VIDEO
# ==========================================

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Cannot open video:", VIDEO_PATH)
    exit()

print("Starting distance estimation...")
print("Press Q to quit.")

# ==========================================
# 3. MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ======================================
    # STEP 1: VEHICLE DETECTION
    # ======================================

    results = model(
        frame,
        conf=CONFIDENCE,
        verbose=False
    )

    # ======================================
    # STEP 2: PROCESS DETECTIONS
    # ======================================

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            if class_id not in VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            box_width = x2 - x1

            if box_width <= 0:
                continue

            # ==================================
            # STEP 3: DISTANCE ESTIMATION
            # ==================================

            distance = (
                REAL_VEHICLE_WIDTH * FOCAL_LENGTH
            ) / box_width

            vehicle_name = VEHICLE_CLASSES[class_id]

            # ==================================
            # STEP 4: WARNING LEVEL
            # ==================================

            if distance < 10:
                color = (0, 0, 255)
                warning = "DANGER"

            elif distance < 20:
                color = (0, 165, 255)
                warning = "CAUTION"

            else:
                color = (0, 255, 0)
                warning = "SAFE"

            # ==================================
            # STEP 5: DRAW RESULTS
            # ==================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            label = f"{vehicle_name}: {distance:.1f} m"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 25, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            cv2.putText(
                frame,
                warning,
                (x1, min(y2 + 25, frame.shape[0] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ======================================
    # STEP 6: DISPLAY
    # ======================================

    cv2.imshow(
        "ADAS Vehicle Distance Estimation",
        frame
    )

    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

# ==========================================
# 7. RELEASE RESOURCES
# ==========================================

cap.release()
cv2.destroyAllWindows()

print("Distance estimation completed!")
