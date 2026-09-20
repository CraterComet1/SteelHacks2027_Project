"""
Live object detection on an Intel RealSense color stream using
Nemotron 3 Nano Omni served through an OpenAI-compatible endpoint
(vLLM, llama.cpp server, LM Studio, OpenRouter, NIM, ...).

pip install pyrealsense2 opencv-python numpy openai

Env vars:
  NEMOTRON_BASE_URL  e.g. http://localhost:8000/v1  (vLLM)  or https://openrouter.ai/api/v1
  NEMOTRON_API_KEY   any string for local servers
  NEMOTRON_MODEL     model name as your server exposes it
  BBOX_MODE          pixel | norm1 | norm1000   (how the model reports box coords; TEST THIS)
"""
import base64, json, os, re, threading, time

import cv2
import numpy as np
import pyrealsense2 as rs
from openai import OpenAI

BASE_URL = os.getenv("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("NEMOTRON_API_KEY", "none")
MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")
BBOX_MODE = os.getenv("BBOX_MODE", "pixel")
W, H, FPS = 640, 480, 30

TARGET = os.getenv("TARGET", "spotted lanternfly")  # what to box; set TARGET=... to change
INTERVAL_S = float(os.getenv("INTERVAL_S", "4"))     # seconds between requests

PROMPT = (
    f"Find every {TARGET} in this image. Respond with ONLY a JSON array, "
    'no prose, no markdown. Each item: {"label": str, "bbox": [x1, y1, x2, y2]} '
    "where bbox is the top-left and bottom-right corner in pixel coordinates of "
    f"an image that is {W} pixels wide and {H} pixels tall. "
    "If there is none, respond with []."
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def to_pixels(box):
    x1, y1, x2, y2 = box
    if BBOX_MODE == "norm1":
        return int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
    if BBOX_MODE == "norm1000":
        return int(x1 / 1000 * W), int(y1 / 1000 * H), int(x2 / 1000 * W), int(y2 / 1000 * H)
    return int(x1), int(y1), int(x2), int(y2)


def detect(bgr):
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(jpg.tobytes()).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=1024,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    )
    text = resp.choices[0].message.content or ""
    print("model said:", text.strip()[:300])
    m = re.search(r"\[.*\]", text, re.S)  # tolerate stray text / code fences
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        try:
            out.append({"label": str(it["label"]), "box": to_pixels(it["bbox"])})
        except (KeyError, ValueError, TypeError):
            continue
    return out


class Worker(threading.Thread):
    """Runs inference on the latest frame without blocking the video loop."""

    def __init__(self):
        super().__init__(daemon=True)
        self.frame = None
        self.result = []          # list of dicts, shown on every frame until updated
        self.latency = 0.0
        self.lock = threading.Lock()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def run(self):
        while True:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            try:
                self.result = detect(frame)
            except Exception as e:
                print("inference error:", e)
                time.sleep(8)  # back off on rate-limit / overload errors
            self.latency = time.time() - t0
            time.sleep(max(0.0, INTERVAL_S - self.latency))


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    pipe.start(cfg)
    align = rs.align(rs.stream.color)

    worker = Worker()
    worker.start()

    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            color_f, depth_f = frames.get_color_frame(), frames.get_depth_frame()
            if not color_f or not depth_f:
                continue
            frame = np.asanyarray(color_f.get_data())
            worker.submit(frame.copy())  # worker grabs the newest frame when it is free

            for d in worker.result:
                x1, y1, x2, y2 = d["box"]
                x1, x2 = sorted((min(max(x1, 0), W - 1), min(max(x2, 0), W - 1)))
                y1, y2 = sorted((min(max(y1, 0), H - 1), min(max(y2, 0), H - 1)))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
                dist = depth_f.get_distance(cx, cy)  # meters, 0 = no depth reading
                tag = d["label"] + (f" {dist:.2f}m" if dist > 0 else "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, tag, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            status = f"{TARGET}: {len(worker.result)} found" if worker.result else f"{TARGET}: none found yet"
            cv2.putText(frame, status, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"model latency: {worker.latency:.1f}s", (8, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("RealSense + Nemotron 3 Nano Omni", frame)
            if worker.result:
                cv2.imwrite("static/LanternFly_Snapshot.png")
                image = cv2.imread("static/LanternFly_Snapshot.png")

                # Draw rectangle
                cv2.rectangle(
                    image,
                    (x1, y1),       # top-left corner
                    (x2, y2),       # bottom-right corner
                    (0, 255, 0),    # color: green (BGR)
                    2               # thickness
                )

                cv2.imwrite("static/LanternFly_Snapshot_Box.png", image)


            if cv2.waitKey(1) & 0xFF == 27:  # Esc
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()