"""
Classify a live RealSense color stream with Nemotron 3 Nano Omni
via an OpenAI-compatible endpoint (vLLM, llama.cpp, LM Studio, OpenRouter).

pip install pyrealsense2 opencv-python numpy openai

Env vars: NEMOTRON_BASE_URL, NEMOTRON_API_KEY, NEMOTRON_MODEL
Edit LABELS below to your own classes.
"""
import base64, json, os, re, threading, time

import cv2
import numpy as np
import pyrealsense2 as rs
from openai import OpenAI

BASE_URL = os.getenv("NEMOTRON_BASE_URL", "http://localhost:8000/v1")
API_KEY = os.getenv("NEMOTRON_API_KEY", "none")
MODEL = os.getenv("NEMOTRON_MODEL", "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16")

LABELS = ["spotted lanternfly", "other insect", "no insect"]  # <- your classes
INTERVAL_S = 4.0  # how often to send a frame (raise if you get 429/503 errors)

PROMPT = (
    "Classify this image into exactly one of these categories: "
    f"{', '.join(LABELS)}. Respond with ONLY JSON, no prose, in the form "
    '{"label": "<category>", "confidence": <0-1>}.'
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def classify(bgr):
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(jpg.tobytes()).decode()
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        max_tokens=512,
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
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return "unparsed", 0.0
    try:
        d = json.loads(m.group(0))
        label = d.get("label", "unparsed")
        if label not in LABELS:
            label = "unparsed"
        return label, float(d.get("confidence", 0.0))
    except (json.JSONDecodeError, ValueError):
        return "unparsed", 0.0


class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.frame, self.lock = None, threading.Lock()
        self.label, self.conf, self.latency = "...", 0.0, 0.0

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
                self.label, self.conf = classify(frame)
                print(f"{time.strftime('%H:%M:%S')}  {self.label}  ({self.conf:.2f})")
            except Exception as e:
                print("inference error:", e)
                time.sleep(8)  # back off on rate-limit / overload errors
            self.latency = time.time() - t0


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    worker = Worker()
    worker.start()
    last = 0.0
    try:
        while True:
            color = pipe.wait_for_frames().get_color_frame()
            if not color:
                continue
            frame = np.asanyarray(color.get_data())
            if time.time() - last >= INTERVAL_S:
                worker.submit(frame.copy())
                last = time.time()
            cv2.putText(frame, f"{worker.label} {worker.conf:.2f}  ({worker.latency:.1f}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("RealSense classify", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Esc
                break
    finally:
        pipe.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()