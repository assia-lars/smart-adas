
from ultralytics import YOLO
import cv2
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
IMAGE_PATH = BASE_DIR / "bus.jpg"
OUTPUT_PATH = BASE_DIR / "results" / "vehicle_detection.jpg"

# Load YOLO11n model
model = YOLO(str(MODEL_PATH))

# Classes to detect (COCO dataset)
# 0: person, 1: bicycle, 2: car
# 3: motorcycle, 5: bus, 7: truck
vehicle_classes = [0, 1, 2, 3, 5, 7]

# Run detection using NVIDIA GPU
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.40,
    classes=vehicle_classes,
    imgsz=640,
    device=0,
    verbose=False
)

# Draw bounding boxes
result = results[0]
annotated_image = result.plot()

# Save result
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), annotated_image)

# Display detection information
print("\n--- ADAS Vehicle Detection ---")

for box in result.boxes:
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    class_name = model.names[class_id]

    print(f"{class_name}: {confidence:.2f}")

print(f"\nTotal detections: {len(result.boxes)}")
print(f"Saved image: {OUTPUT_PATH}")
