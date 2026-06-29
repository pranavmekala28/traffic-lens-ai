import json
import os
import re
import uuid
from datetime import datetime

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from detector import VehicleDetector

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
Config.ensure_dirs()

print("[Traffic Lens AI] Loading detection model:", Config.MODEL_PATH)
detector = VehicleDetector(model_path=Config.MODEL_PATH, conf=Config.CONFIDENCE)
print("[Traffic Lens AI] Model ready.")


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
    except Exception:
        return []


def _save_record(record):
    history = _load_history()
    history.insert(0, record)
    history = history[:200]
    with open(Config.HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


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


@app.route("/api/upload", methods=["POST", "OPTIONS"])
@app.route("/api/detect", methods=["POST", "OPTIONS"])
def api_upload_detect():
    if request.method == "OPTIONS":
        return "", 200

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file or file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    kind = _media_kind(file.filename)
    if kind is None:
        return jsonify({"error": "Unsupported file type. Use image or video."}), 400

    uid = uuid.uuid4().hex[:12]
    safe = secure_filename(file.filename)
    in_name = f"{uid}_{safe}"
    in_path = os.path.join(Config.UPLOAD_FOLDER, in_name)
    file.save(in_path)

    out_ext = "jpg" if kind == "image" else "mp4"
    out_name = f"{uid}_result.{out_ext}"
    out_path = os.path.join(Config.RESULTS_FOLDER, out_name)

    try:
        if kind == "image":
            stats = detector.detect_image(in_path, out_path)
        else:
            stats = detector.detect_video(
                in_path,
                out_path,
                stride=Config.VIDEO_FRAME_STRIDE
            )
    except Exception as exc:
        return jsonify({"error": f"Detection failed: {exc}"}), 500

    record = {
        "id": uid,
        "filename": safe,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "result_url": url_for("static", filename=f"results/{out_name}"),
        "download_url": url_for("static", filename=f"results/{out_name}"),
        **stats,
    }

    _save_record(record)
    return jsonify(record), 200


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


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_output")
COLLISION_REPORT = os.path.join(OUTPUT_DIR, "collision_report.txt")

_VIDEO_CANDIDATES = [
    "accidents_small.mp4",
    "traffic_video_annotated.mp4",
    "new_traffic_video_annotated.mp4",
    "new_traffic_video_trucks_only.mp4",
]

_TIME_RE = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})"
    r"|(?P<sec>\d+(?:\.\d+)?)\s*(?:s\b|sec\b|secs\b|seconds?\b)"
)

_HEAVY_RE = re.compile(
    r"\b(truck|bus|semi|lorry|head-?on|rollover|pile-?up|fatal|severe|major)\b",
    re.I,
)


def _ts_to_seconds(match):
    if match.group("sec") is not None:
        return float(match.group("sec"))
    h = int(match.group("h")) if match.group("h") else 0
    m = int(match.group("m"))
    s = int(match.group("s"))
    return h * 3600 + m * 60 + s


def _fmt_seconds(total):
    total = int(round(total))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _clean(text):
    text = re.sub(r"<vref[^>]*>|</vref>", "", text)
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"^\s*[-•\d.]+\s*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _split_blocks(report):
    blocks = [b.strip() for b in re.split(r"\n\s*\n", report) if b.strip()]
    if len(blocks) > 1:
        return blocks

    parts = re.split(
        r"\n(?=\s*(?:\d+[.)]\s|[-•]\s|collision\b|incident\b))",
        report,
        flags=re.I,
    )
    return [p.strip() for p in parts if p.strip()]


def parse_collision_report(report):
    incidents = []

    for block in _split_blocks(report):
        times = list(_TIME_RE.finditer(block))
        start = end = None
        label = ""

        if times:
            start = _ts_to_seconds(times[0])
            if len(times) > 1:
                second = _ts_to_seconds(times[1])
                if second >= start:
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
    safe = os.path.normpath(filename)
    if safe.startswith("..") or os.path.isabs(safe):
        abort(404)
    return send_from_directory(OUTPUT_DIR, safe)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)