"""
traffic_report.py — full traffic analysis pipeline.

Produces, in a `traffic_output/` folder next to where you run it:
  1. <name>_annotated.mp4   - video with YOLO boxes + live count overlay
  2. <name>_timeline.csv    - per-timestamp exact counts + scene reasoning
  3. <name>_charts.png      - vehicles over time, type breakdown, congestion
  4. <name>_report.html     - self-contained report (open in any browser)

YOLO does the exact counting + drawing. Nemotron (NVIDIA) adds congestion
and incident reasoning. You can run YOLO-only with --no-vlm (no API needed).

Setup (one time, in your project's env):
    env2\\Scripts\\python.exe -m pip install ultralytics matplotlib opencv-python openai

Run:
    env2\\Scripts\\python.exe traffic_report.py "C:\\Users\\prana\\OneDrive\\Desktop\\traffic_video.mp4"
    ...add --no-vlm to skip the API and just do YOLO counting + charts.

Needs yolov8n.pt in the project folder (you already have it).
"""

import os
import sys
import csv
import json
import base64
import argparse
from collections import defaultdict

try:
    import cv2
except ImportError:
    print("OpenCV missing. Run: pip install opencv-python"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib missing. Run: pip install matplotlib"); sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics (YOLO) missing. Run: pip install ultralytics"); sys.exit(1)

# COCO class ids YOLO uses for vehicles, mapped to friendly names.
VEHICLE_CLASSES = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BOX_COLOR = {"car": (0, 200, 0), "truck": (0, 140, 255), "bus": (255, 80, 0),
             "motorcycle": (200, 0, 200), "bicycle": (0, 220, 220)}

CONGESTION_RANK = {"free_flow": 0, "light": 1, "moderate": 2, "heavy": 3, "gridlock": 4}

# ----- Optional Nemotron scene reasoning -----------------------------------
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE_URL = "https://integrate.api.nvidia.com/v1"
VLM_PROMPT = """Look at this traffic camera frame. Ignore exact vehicle counts.
Return ONLY this JSON:
{"congestion_level":"free_flow|light|moderate|heavy|gridlock","incidents":["stopped vehicle, collision, wrong-way, debris, etc. Empty list if none."],"visibility":"clear|rain|fog|snow|night|low"}
JSON only, no markdown."""


def vlm_scene(client, frame):
    ok, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf).decode("utf-8")
    try:
        comp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": VLM_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            temperature=0.6, top_p=0.95, max_tokens=1024,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        raw = comp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)
    except Exception:
        return {"congestion_level": "", "incidents": [], "visibility": ""}


def main():
    ap = argparse.ArgumentParser(description="Full traffic video analysis + report.")
    ap.add_argument("video", help="Path to the video.")
    ap.add_argument("--weights", default="yolov8n.pt", help="YOLO weights (default yolov8n.pt).")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    ap.add_argument("--record-every", type=float, default=1.0, help="Seconds between data points.")
    ap.add_argument("--vlm-every", type=float, default=3.0, help="Seconds between Nemotron calls.")
    ap.add_argument("--no-vlm", action="store_true", help="Skip Nemotron; YOLO only.")
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}"); sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"YOLO weights not found: {args.weights}"); sys.exit(1)

    use_vlm = not args.no_vlm
    client = None
    if use_vlm:
        key = os.getenv("NVIDIA_API_KEY")
        if not key:
            print("No NVIDIA_API_KEY set; continuing YOLO-only. (Use --no-vlm to silence this.)")
            use_vlm = False
        else:
            from openai import OpenAI
            client = OpenAI(base_url=BASE_URL, api_key=key)

    out_dir = "traffic_output"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.video))[0]
    annotated_path = os.path.join(out_dir, f"{base}_annotated.mp4")
    csv_path = os.path.join(out_dir, f"{base}_timeline.csv")
    chart_path = os.path.join(out_dir, f"{base}_charts.png")
    report_path = os.path.join(out_dir, f"{base}_report.html")

    print("Loading YOLO model...")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}"); sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total / fps if total else 0
    print(f"Video: {dur:.1f}s, {w}x{h}, {fps:.0f} fps, {total} frames.\n")

    writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows = []
    last_record_t = -999.0
    last_vlm_t = -999.0
    last_scene = {"congestion_level": "", "incidents": [], "visibility": ""}
    fidx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = fidx / fps

        # YOLO detection on this frame
        res = model(frame, conf=args.conf, verbose=False)[0]
        counts = defaultdict(int)
        for b in res.boxes:
            cls = int(b.cls[0])
            if cls in VEHICLE_CLASSES:
                name = VEHICLE_CLASSES[cls]
                counts[name] += 1
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                color = BOX_COLOR.get(name, (0, 200, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        total_v = sum(counts.values())

        # overlay summary banner
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        overlay = f"t={t:5.1f}s  Total:{total_v}  car:{counts['car']} truck:{counts['truck']} bus:{counts['bus']} moto:{counts['motorcycle']}"
        cv2.putText(frame, overlay, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

        # record a data point at the chosen cadence
        if t - last_record_t >= args.record_every:
            last_record_t = t
            scene = last_scene
            if use_vlm and (t - last_vlm_t >= args.vlm_every):
                last_vlm_t = t
                scene = vlm_scene(client, frame)
                last_scene = scene
            row = {"time_s": round(t, 1), "total": total_v,
                   "car": counts["car"], "truck": counts["truck"],
                   "bus": counts["bus"], "motorcycle": counts["motorcycle"],
                   "congestion": scene.get("congestion_level", ""),
                   "incidents": "; ".join(scene.get("incidents", [])) or "none",
                   "visibility": scene.get("visibility", "")}
            rows.append(row)
            print(f"[{t:6.1f}s] total {total_v:>2} | car {counts['car']} truck {counts['truck']} "
                  f"bus {counts['bus']} | {row['congestion'] or '-'} | {row['incidents']}")
        fidx += 1

    cap.release()
    writer.release()

    if not rows:
        print("No data recorded."); return

    # ----- CSV -----
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)

    # ----- Charts -----
    times = [r["time_s"] for r in rows]
    totals = [r["total"] for r in rows]
    fig, ax = plt.subplots(3, 1, figsize=(10, 11))

    ax[0].plot(times, totals, marker="o", color="#2563eb")
    ax[0].fill_between(times, totals, alpha=0.15, color="#2563eb")
    ax[0].set_title("Total vehicles over time (YOLO exact count)")
    ax[0].set_xlabel("seconds"); ax[0].set_ylabel("vehicles"); ax[0].grid(alpha=0.3)

    for vt, col in [("car", "#16a34a"), ("truck", "#f59e0b"),
                    ("bus", "#ef4444"), ("motorcycle", "#a855f7")]:
        ax[1].plot(times, [r[vt] for r in rows], marker=".", label=vt, color=col)
    ax[1].set_title("Vehicle types over time")
    ax[1].set_xlabel("seconds"); ax[1].set_ylabel("count"); ax[1].legend(); ax[1].grid(alpha=0.3)

    cong = [CONGESTION_RANK.get(r["congestion"], -1) for r in rows]
    ax[2].step(times, cong, where="mid", color="#dc2626")
    ax[2].set_yticks(list(CONGESTION_RANK.values()))
    ax[2].set_yticklabels(list(CONGESTION_RANK.keys()))
    ax[2].set_title("Congestion level over time (Nemotron)")
    ax[2].set_xlabel("seconds"); ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(chart_path, dpi=110)
    plt.close()

    # ----- Summary stats -----
    peak = max(rows, key=lambda r: r["total"])
    avg = sum(totals) / len(totals)
    type_peaks = {vt: max(r[vt] for r in rows) for vt in ["car", "truck", "bus", "motorcycle"]}
    cong_dist = defaultdict(int)
    for r in rows:
        if r["congestion"]:
            cong_dist[r["congestion"]] += 1
    cong_total = sum(cong_dist.values()) or 1
    incidents = sorted({(r["time_s"], r["incidents"]) for r in rows if r["incidents"] != "none"})

    # ----- HTML report -----
    with open(chart_path, "rb") as f:
        chart_b64 = base64.b64encode(f.read()).decode("utf-8")

    rows_html = "\n".join(
        f"<tr><td>{r['time_s']}</td><td>{r['total']}</td><td>{r['car']}</td>"
        f"<td>{r['truck']}</td><td>{r['bus']}</td><td>{r['motorcycle']}</td>"
        f"<td>{r['congestion'] or '-'}</td><td>{r['incidents']}</td></tr>"
        for r in rows)
    cong_html = "".join(
        f"<li>{lvl}: {n/cong_total*100:.0f}% of sampled time</li>"
        for lvl, n in sorted(cong_dist.items(), key=lambda x: CONGESTION_RANK.get(x[0], 0)))
    inc_html = ("".join(f"<li>{t}s — {txt}</li>" for t, txt in incidents)
                if incidents else "<li>None detected</li>")

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Traffic Report — {base}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;max-width:980px;margin:30px auto;padding:0 16px;color:#1e293b}}
h1{{margin-bottom:4px}} .sub{{color:#64748b;margin-top:0}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:20px 0}}
.card{{flex:1;min-width:150px;background:#f1f5f9;border-radius:10px;padding:16px}}
.card .n{{font-size:28px;font-weight:700;color:#2563eb}}
.card .l{{font-size:13px;color:#64748b}}
img{{width:100%;border:1px solid #e2e8f0;border-radius:10px;margin:14px 0}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{border:1px solid #e2e8f0;padding:6px 8px;text-align:center}}
th{{background:#f8fafc}} ul{{line-height:1.7}}
</style></head><body>
<h1>Traffic Analysis Report</h1>
<p class="sub">Source: {base} &middot; duration {dur:.1f}s &middot; {fps:.0f} fps</p>
<div class="cards">
  <div class="card"><div class="n">{peak['total']}</div><div class="l">peak vehicles (at {peak['time_s']}s)</div></div>
  <div class="card"><div class="n">{avg:.1f}</div><div class="l">average vehicles</div></div>
  <div class="card"><div class="n">{type_peaks['car']}</div><div class="l">peak cars</div></div>
  <div class="card"><div class="n">{type_peaks['truck']}</div><div class="l">peak trucks</div></div>
  <div class="card"><div class="n">{len(rows)}</div><div class="l">data points</div></div>
</div>
<h2>Charts</h2>
<img src="data:image/png;base64,{chart_b64}" alt="charts">
<h2>Congestion breakdown</h2><ul>{cong_html or '<li>No reasoning data (YOLO-only run)</li>'}</ul>
<h2>Incidents</h2><ul>{inc_html}</ul>
<h2>Full timeline</h2>
<table><tr><th>t (s)</th><th>total</th><th>car</th><th>truck</th><th>bus</th><th>moto</th><th>congestion</th><th>incidents</th></tr>
{rows_html}</table>
<p class="sub" style="margin-top:24px">Counts by YOLO ({os.path.basename(args.weights)}). Scene reasoning by NVIDIA Nemotron. Generated by traffic_report.py.</p>
</body></html>"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nDone. {len(rows)} data points across {dur:.1f}s.")
    print("Outputs in 'traffic_output/':")
    print(f"  - {os.path.basename(annotated_path)}  (annotated video)")
    print(f"  - {os.path.basename(csv_path)}  (timeline CSV)")
    print(f"  - {os.path.basename(chart_path)}  (charts image)")
    print(f"  - {os.path.basename(report_path)}  (open this in a browser)")


if __name__ == "__main__":
    main()