# Traffic Lens AI

Detect and count vehicles in traffic **images and videos** with computer vision.
Upload a file, get it back with a box around every car, truck, bus, motorcycle and
bicycle — plus a per-class count and a dashboard of your totals.

Built with **Flask**, **Ultralytics YOLOv8**, **OpenCV**, and **Chart.js**.
Everything runs locally; no footage leaves your machine.

## Quick start

> Requires **Python 3.9–3.12**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open **http://localhost:5000**. On the first run, Ultralytics auto-downloads the
`yolov8n.pt` weights (~6 MB); after that it runs offline.

## Configuration (optional env vars)

| Variable     | Default      | Meaning                                  |
|--------------|--------------|------------------------------------------|
| `TLA_MODEL`  | `yolov8n.pt` | YOLO weights (try `yolov8s.pt`)          |
| `TLA_CONF`   | `0.35`       | Confidence threshold (0–1)               |
| `TLA_STRIDE` | `3`          | Process every Nth video frame            |

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Traffic Lens AI"
git branch -M main
git remote add origin https://github.com/<you>/traffic-lens-ai.git
git push -u origin main
```

## Notes

- Video counts report the **peak simultaneous** vehicles in any scanned frame;
  unique counting across frames needs object tracking (e.g. ByteTrack) — a good next step.
- For `.mp4` playback issues, install `ffmpeg` and transcode, or change the codec in `detector.py`.