import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best.pt")  # pastikan path-nya benar

def detect_helmet_from_frame(frame):
    results = model(frame, stream=True)
    for r in results:
        annotated_frame = r.plot()
        return annotated_frame
