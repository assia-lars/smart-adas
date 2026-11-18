
from ultralytics import YOLO
import cv2
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "data" / "videos" / "trafic.mp4"
OUTPUT_PATH = BASE_DIR / "results" / "vehicle_detection.mp4"

# Load YOLO11n model
model = YOLO(str(MODEL_PATH))

# COCO classes: person, bicycle, car, motorcycle, bus, truck
vehicle_classes = [0, 1, 2, 3, 5, 7]

# Open video
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Error: Cannot open video:", VIDEO_PATH)
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 25

# Create output directory
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Cannot create output video.")
    cap.release()
    exit()

print("--- Smart ADAS Video Detection ---")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # YOLO detection on each frame
    results = model.predict(
        source=frame,
        conf=0.40,
        classes=vehicle_classes,
        imgsz=640,
        device=0,
        verbose=False
    )

    # Draw bounding boxes
    annotated_frame = results[0].plot()

    # Display video
    cv2.imshow("Smart ADAS - Video Detection", annotated_frame)

    # Save processed frame
    out.write(annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("Detection completed!")
print("Saved video:", OUTPUT_PATH)
