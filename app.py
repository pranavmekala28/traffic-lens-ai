"""
app.py — Traffic Lens AI (Flask backend)

Routes
------
  GET  /            Home
  GET  /upload      Upload page
  GET  /dashboard   Dashboard page
  GET  /about       About page
  GET  /collisions  Jockey collision timeline (reads traffic_output/collision_report.txt)

API
---
  POST /api/detect     Upload an image/video, run detection, return JSON
  GET  /api/history    Recent detection runs
  GET  /api/stats      Aggregated totals for the dashboard
  GET  /api/collisions Parsed collision incidents as JSON
"""
import json
import os
import re
import uuid
from datetime import datetime
from flask_cors import CORS

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory, url_for)
from werkzeug.utils import secure_filename

from config import Config
from detector import VehicleDetector

app = Flask(__name__)
CORS(app)
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


# ============================ collisions ==============================
#  Reads the saved Jockey output (traffic_output/collision_report.txt),
#  parses it into a timeline of incidents with timestamps, and serves the
#  annotated video so the page can seek to each collision.
# ----------------------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_output")
COLLISION_REPORT = os.path.join(OUTPUT_DIR, "collision_report.txt")

# Annotated videos to look for, in priority order. First one found is embedded.
_VIDEO_CANDIDATES = [
    "accidents_small.mp4",
    "traffic_video_annotated.mp4",
    "new_traffic_video_annotated.mp4",
    "new_traffic_video_trucks_only.mp4",
]

# Catches mm:ss, hh:mm:ss, and bare seconds like "12s" / "12.5 sec" / "12 seconds"
_TIME_RE = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})"            # 0:03  or 1:02:33
    r"|(?P<sec>\d+(?:\.\d+)?)\s*(?:s\b|sec\b|secs\b|seconds?\b)"  # 12s / 12.5 sec
)

# "Heavy" keywords tint a card amber; everything else stays cyan.
_HEAVY_RE = re.compile(
    r"\b(truck|bus|semi|lorry|head-?on|rollover|pile-?up|fatal|severe|major)\b", re.I
)


def _ts_to_seconds(match):
    """Convert a _TIME_RE match into float seconds."""
    if match.group("sec") is not None:
        return float(match.group("sec"))
    h = int(match.group("h")) if match.group("h") else 0
    m = int(match.group("m"))
    s = int(match.group("s"))
    return h * 3600 + m * 60 + s


def _fmt_seconds(total):
    """Seconds -> 'm:ss' (or 'h:mm:ss') for display."""
    total = int(round(total))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _clean(text):
    """Strip markdown + Jockey <vref> tags so descriptions read clean in HTML."""
    text = re.sub(r"<vref[^>]*>|</vref>", "", text)          # drop <vref ...> / </vref>
    text = re.sub(r"[*_`#>]+", "", text)                     # markdown emphasis
    text = re.sub(r"^\s*[-•\d.]+\s*", "", text)              # leading bullets / "1."
    text = re.sub(r"\s{2,}", " ", text)                      # collapse double spaces
    return text.strip()


def _split_blocks(report):
    """
    Break the freeform report into per-incident blocks. Tries, in order:
      1) blank-line separated paragraphs
      2) lines that start a new numbered / 'Collision N' / bullet item
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", report) if b.strip()]
    if len(blocks) > 1:
        return blocks
    # Fallback: one wall of text — split on item markers at line starts.
    parts = re.split(
        r"\n(?=\s*(?:\d+[.)]\s|[-•]\s|collision\b|incident\b))", report, flags=re.I
    )
    return [p.strip() for p in parts if p.strip()]


def parse_collision_report(report):
    """
    Returns a list of incidents:
        {"time_label", "start_seconds", "end_seconds", "description", "heavy"}
    Incidents without any timestamp are kept (start_seconds=None) so nothing
    silently disappears.
    """
    incidents = []
    for block in _split_blocks(report):
        times = list(_TIME_RE.finditer(block))
        start = end = None
        label = ""
        if times:
            start = _ts_to_seconds(times[0])
            if len(times) > 1:
                second = _ts_to_seconds(times[1])
                if second >= start:  # only treat as a range if 2nd time is later
                    end = second
            label = _fmt_seconds(start) + (
                f" – {_fmt_seconds(end)}" if end is not None else ""
            )
        incidents.append({
            "time_label": label or "—",
            "start_seconds": start,
            "end_seconds": end,
            "description": _clean(block),
            "heavy": bool(_HEAVY_RE.search(block)),
        })
    # Sort timed incidents chronologically; keep untimed ones at the end.
    incidents.sort(key=lambda i: (i["start_seconds"] is None, i["start_seconds"] or 0))
    return incidents


def _load_report():
    if not os.path.exists(COLLISION_REPORT):
        return ""
    with open(COLLISION_REPORT, encoding="utf-8") as f:
        return f.read()


def _find_video():
    for name in _VIDEO_CANDIDATES:
        if os.path.exists(os.path.join(OUTPUT_DIR, name)):
            return name
    return None


@app.route("/collisions")
def collisions():
    report = _load_report()
    incidents = parse_collision_report(report) if report else []
    return render_template(
        "collisions.html",
        incidents=incidents,
        raw_report=report,
        video_name=_find_video(),
        has_report=bool(report),
    )


@app.route("/api/collisions")
def api_collisions():
    report = _load_report()
    return jsonify({
        "has_report": bool(report),
        "video": _find_video(),
        "incidents": parse_collision_report(report) if report else [],
    })


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve files (annotated video, charts, etc.) from traffic_output/."""
    safe = os.path.normpath(filename)
    if safe.startswith("..") or os.path.isabs(safe):
        abort(404)
    return send_from_directory(OUTPUT_DIR, safe)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)