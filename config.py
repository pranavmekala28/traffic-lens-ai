"""Central configuration for Traffic Lens AI."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # Folders
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    RESULTS_FOLDER = os.path.join(BASE_DIR, "static", "results")
    DATA_FOLDER = os.path.join(BASE_DIR, "data")
    HISTORY_FILE = os.path.join(DATA_FOLDER, "history.json")

    # Uploads
    MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB
    ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "bmp", "webp"}
    ALLOWED_VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}

    # Detection model
    MODEL_PATH = os.environ.get("TLA_MODEL", "yolov8n.pt")  # auto-downloads on first run
    CONFIDENCE = float(os.environ.get("TLA_CONF", "0.35"))

    # Process every Nth frame for videos (speed vs. accuracy)
    VIDEO_FRAME_STRIDE = int(os.environ.get("TLA_STRIDE", "30"))

    @staticmethod
    def ensure_dirs():
        for folder in (Config.UPLOAD_FOLDER, Config.RESULTS_FOLDER, Config.DATA_FOLDER):
            os.makedirs(folder, exist_ok=True)