"""
detector.py — Vehicle detection engine for Traffic Lens AI.

Uses Ultralytics YOLOv8 (COCO-pretrained) to detect and count vehicles in
images and videos, then writes an annotated copy with bounding boxes.

COCO vehicle class IDs:
    1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
"""
import os
import cv2
from collections import Counter

from ultralytics import YOLO

VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Box colors in BGR (OpenCV order)
# amber  #FFB627 -> (39,182,255) | cyan #36E2C4 -> (196,226,54)
COLOR_AMBER = (39, 182, 255)
COLOR_CYAN = (196, 226, 54)
CLASS_COLOR = {
    "car": COLOR_CYAN,
    "motorcycle": COLOR_CYAN,
    "bicycle": COLOR_CYAN,
    "bus": COLOR_AMBER,
    "truck": COLOR_AMBER,
}


class VehicleDetector:
    """Loads a YOLO model once and reuses it for every request."""

    def __init__(self, model_path="yolov8n.pt", conf=0.35):
        # Model weights auto-download on first use if not present locally.
        self.model = YOLO(model_path)
        self.conf = conf
        self.vehicle_ids = list(VEHICLE_CLASSES.keys())

    # ---------- drawing ----------
    def _draw_box(self, frame, xyxy, label, conf):
        x1, y1, x2, y2 = (int(v) for v in xyxy)
        color = CLASS_COLOR.get(label, COLOR_CYAN)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, text, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 23, 28), 1, cv2.LINE_AA)

    def _annotate(self, frame, result):
        """Draw all vehicle boxes on a frame, return a Counter of this frame."""
        counts = Counter()
        if result.boxes is None:
            return counts
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label = VEHICLE_CLASSES[cls_id]
            conf = float(box.conf[0])
            self._draw_box(frame, box.xyxy[0].tolist(), label, conf)
            counts[label] += 1
        return counts

    # ---------- public API ----------
    def detect_image(self, input_path, output_path):
        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError("Could not read image file.")

        result = self.model.predict(frame, conf=self.conf,
                                    classes=self.vehicle_ids, verbose=False)[0]
        counts = self._annotate(frame, result)
        cv2.imwrite(output_path, frame)

        total = sum(counts.values())
        return {
            "media_type": "image",
            "total_vehicles": total,
            "counts": dict(counts),
            "frames_processed": 1,
        }

    def detect_video(self, input_path, output_path, stride=3):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps / max(stride, 1),
                                 (width, height))

        peak = Counter()       # max simultaneous count per class (best estimate)
        frames_done = 0
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                result = self.model.predict(frame, conf=self.conf,
                                            classes=self.vehicle_ids,
                                            verbose=False)[0]
                counts = self._annotate(frame, result)
                for k, v in counts.items():
                    peak[k] = max(peak[k], v)
                writer.write(frame)
                frames_done += 1
            idx += 1

        cap.release()
        writer.release()

        total = sum(peak.values())
        return {
            "media_type": "video",
            "total_vehicles": total,          # peak simultaneous vehicles
            "counts": dict(peak),
            "frames_processed": frames_done,
        }