"""
app.py — Traffic Lens AI (Flask backend)

Routes
------
  GET  /            Home
  GET  /upload      Upload page
  GET  /dashboard   Dashboard page
  GET  /about       About page

API
---
  POST /api/detect  Upload an image/video, run detection, return JSON
  GET  /api/history Recent detection runs
  GET  /api/stats   Aggregated totals for the dashboard
"""
import json
import os
import uuid
from datetime import datetime

from flask import (Flask, jsonify, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from config import Config
from detector import VehicleDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
Config.ensure_dirs()

# Load the model once at startup (weights auto-download on first run).
print("[Traffic Lens AI] Loading detection model:", Config.MODEL_PATH)
detector = VehicleDetector(model_path=Config.MODEL_PATH, conf=Config.CONFIDENCE)
print("[Traffic Lens AI] Model ready.")


# ----------------------------- helpers --------------------------------
def _ext(filename):
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _media_kind(filename):
    e = _ext(filename)
    if e in Config.ALLOWED_IMAGE_EXT:
        return "image"
    if e in Config.ALLOWED_VIDEO_EXT:
        return "video"
    return None


def _load_history():
    if not os.path.exists(Config.HISTORY_FILE):
        return []
    try:
        with open(Config.HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_record(record):
    history = _load_history()
    history.insert(0, record)
    history = history[:200]  # keep last 200 runs
    with open(Config.HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


# ------------------------------ pages ---------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


# ------------------------------- API ----------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    kind = _media_kind(file.filename)
    if kind is None:
        return jsonify({"error": "Unsupported file type. Use an image or video."}), 400

    # Save upload with a unique name.
    uid = uuid.uuid4().hex[:12]
    safe = secure_filename(file.filename)
    in_name = f"{uid}_{safe}"
    in_path = os.path.join(Config.UPLOAD_FOLDER, in_name)
    file.save(in_path)

    # Result filename (always one of these extensions for the browser).
    out_ext = "jpg" if kind == "image" else "mp4"
    out_name = f"{uid}_result.{out_ext}"
    out_path = os.path.join(Config.RESULTS_FOLDER, out_name)

    try:
        if kind == "image":
            stats = detector.detect_image(in_path, out_path)
        else:
            stats = detector.detect_video(in_path, out_path,
                                          stride=Config.VIDEO_FRAME_STRIDE)
    except Exception as exc:  # noqa: BLE001 - surface a clean message
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    record = {
        "id": uid,
        "filename": safe,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result_url": url_for("static", filename=f"results/{out_name}"),
        **stats,
    }
    _save_record(record)
    return jsonify(record)


@app.route("/api/history")
def api_history():
    return jsonify(_load_history())


@app.route("/api/stats")
def api_stats():
    history = _load_history()
    totals = {"car": 0, "truck": 0, "bus": 0, "motorcycle": 0, "bicycle": 0}
    total_vehicles = 0
    for rec in history:
        total_vehicles += rec.get("total_vehicles", 0)
        for k, v in rec.get("counts", {}).items():
            totals[k] = totals.get(k, 0) + v
    return jsonify({
        "runs": len(history),
        "total_vehicles": total_vehicles,
        "by_class": totals,
        "recent": history[:8],
    })


@app.route("/uploads/<path:name>")
def serve_upload(name):
    return send_from_directory(Config.UPLOAD_FOLDER, name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)