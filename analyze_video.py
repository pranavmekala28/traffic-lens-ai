import os, sys, csv, json, base64, argparse, tempfile
try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip install opencv-python"); sys.exit(1)
from openai import OpenAI

MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE_URL = "https://integrate.api.nvidia.com/v1"
SYSTEM_PROMPT = "You are a traffic monitoring vision system analyzing a single frame from a fixed traffic camera. Report only what is visibly supported by the image. Do not guess. Respond with a single JSON object and nothing else."
USER_PROMPT = """Analyze this traffic camera frame and return ONLY this JSON:
{"vehicle_counts":{"car":0,"truck":0,"bus":0,"motorcycle":0,"bicycle":0,"other":0},"total_vehicles":0,"congestion_level":"free_flow|light|moderate|heavy|gridlock","congestion_reason":"one short sentence","incidents":["list stopped vehicles, collisions, etc. Empty list if none."],"traffic_signal_state":"red|yellow|green|not_visible","visibility":"clear|rain|fog|snow|night|low","confidence":"high|medium|low"}
Return valid JSON only. No markdown."""

def analyze_frame(client, image_path):
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role":"system","content":SYSTEM_PROMPT},
                  {"role":"user","content":[{"type":"text","text":USER_PROMPT},
                  {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}]}],
        temperature=0.6, top_p=0.95, max_tokens=2048,
        extra_body={"chat_template_kwargs":{"enable_thinking":False}})
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_parse_error": True}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--every", type=float, default=2.0)
    args = p.parse_args()
    if not os.path.exists(args.video):
        print(f"Video not found: {args.video}"); sys.exit(1)
    key = os.getenv("NVIDIA_API_KEY")
    if not key:
        print('Set key: $env:NVIDIA_API_KEY="nvapi-xxxx"'); sys.exit(1)
    client = OpenAI(base_url=BASE_URL, api_key=key)
    out_path = args.video.rsplit(".",1)[0] + "_timeline.csv"
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = total/fps if total else 0
    print(f"Video: {dur:.1f}s. Sampling every {args.every}s.\n")
    rows = []; t = 0.0
    while t <= dur:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
        ok, frame = cap.read()
        if not ok: break
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False); tmp.close()
        cv2.imwrite(tmp.name, frame)
        r = analyze_frame(client, tmp.name)
        os.unlink(tmp.name)
        if not r.get("_parse_error"):
            c = r.get("vehicle_counts", {})
            row = {"time_s":round(t,1),"total_vehicles":r.get("total_vehicles",""),
                   "cars":c.get("car",""),"trucks":c.get("truck",""),
                   "congestion":r.get("congestion_level",""),
                   "incidents":"; ".join(r.get("incidents",[])) or "none"}
            rows.append(row)
            print(f"[{t:6.1f}s] {row['total_vehicles']} vehicles | {row['congestion']} | {row['incidents']}")
        t += args.every
    cap.release()
    if rows:
        with open(out_path,"w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nDone. {len(rows)} frames. Saved: {out_path}")

if __name__ == "__main__":
    main()