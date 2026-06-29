# Traffic Lens AI

An AI system that analyzes traffic camera footage and produces exact vehicle counts, congestion assessment, incident detection, an annotated video, charts, and a full HTML report.

It combines two models, each doing what it's genuinely good at:

- **YOLO (yolov8n)** — precise per-frame object detection. Draws bounding boxes, classifies vehicles (car / truck / bus / motorcycle), and gives **exact counts**.
- **NVIDIA Nemotron 3 Nano Omni** — scene-level reasoning a detector can't do: congestion level, incidents (stopped vehicles, wrong-way, debris), and visibility.

The key design idea: a vision-language model is great at *judgment* but unreliable at *counting*, while an object detector is the opposite. Splitting the work lets each cover the other's weakness.

## What it produces

For any input video, the pipeline generates four artifacts in a `traffic_output/` folder:

| Output | Description |
|--------|-------------|
| `*_annotated.mp4` | The video with bounding boxes and a live count overlay |
| `*_timeline.csv` | Exact vehicle counts over time, with timestamps |
| `*_charts.png` | Vehicles over time, type breakdown, and congestion timeline |
| `*_report.html` | A self-contained report: summary stats, charts, incidents, full timeline |
<img width="1097" height="1207" alt="image" src="https://github.com/user-attachments/assets/cb27d1b3-a91e-419e-9df9-5c6203695744" />

## How it works

1. **OpenCV** reads the video frame by frame
2. **YOLO** detects, counts, and annotates every frame
3. **Nemotron** adds congestion/incident reasoning, sampled at intervals to stay efficient
4. Results are logged to a timeline, then rendered into charts and an HTML report

## Setup

```bash
python -m venv env
env\Scripts\activate            # Windows
pip install ultralytics matplotlib opencv-python openai
```

Set your NVIDIA API key (get one free at https://build.nvidia.com):

```bash
# Windows PowerShell
$env:NVIDIA_API_KEY="your-key-here"
```

## Usage

Full analysis (YOLO counting + Nemotron reasoning):

```bash
python traffic_report.py "path/to/video.mp4"
```

YOLO only, no API needed (fast, free):

```bash
python traffic_report.py "path/to/video.mp4" --no-vlm
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | 0.35 | YOLO confidence threshold (lower = catches more) |
| `--record-every` | 1.0 | Seconds between recorded data points |
| `--vlm-every` | 3.0 | Seconds between Nemotron reasoning calls |
| `--no-vlm` | off | Skip Nemotron; YOLO-only run |

## Scripts

- `traffic_report.py` — the full pipeline (annotated video + charts + report)
- `analyze_video.py` — lightweight video timeline (Nemotron only)
- `traffic_lens.py` — single-image analysis

## Tech stack

Python · YOLOv8 (Ultralytics) · NVIDIA Nemotron 3 Nano Omni · OpenCV · matplotlib

## Notes

- The model file (`yolov8n.pt`) downloads automatically on first run.
- Nemotron outputs are for analysis, not safety-certified decision-making.

---

Built as a study in multi-model architecture — matching each model to its strength rather than forcing one model to do everything.
