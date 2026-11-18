
from ultralytics import YOLO
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "yolo11n.pt"
VIDEO_PATH = BASE_DIR / "data" / "videos" / "trafic.mp4"
OUTPUT_DIR = BASE_DIR / "results"

# Load YOLO11 model
model = YOLO(str(MODEL_PATH))

# Classes: person, bicycle, car, motorcycle, bus, truck
vehicle_classes = [0, 1, 2, 3, 5, 7]

print("\n--- ADAS Vehicle Tracking ---")
print("Starting tracking...")

# Run YOLO + ByteTrack
results = model.track(
    source=str(VIDEO_PATH),
    tracker="bytetrack.yaml",
    conf=0.40,
    classes=vehicle_classes,
    imgsz=640,
    device=0,
    save=True,
    project=str(OUTPUT_DIR),
    name="tracking",
    exist_ok=True,
    show=True,
    verbose=False
)

print("\nTracking completed!")
print("Check the results folder:", OUTPUT_DIR / "tracking")
