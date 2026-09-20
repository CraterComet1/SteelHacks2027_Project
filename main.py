import base64, json, re, os, glob, time
import cv2
from openai import OpenAI

# NVIDIA build.nvidia.com hosted endpoint -- no GPU/VRAM needed.
# Get a free key at https://build.nvidia.com/settings/api-keys (starts with "nvapi-")
# and export it: export NVIDIA_API_KEY="nvapi-..."
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

# Same folder frame_capture.py (running on Windows) writes into.
# On WSL this is the /mnt/c/... path; adjust if this script lives elsewhere.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(SCRIPT_DIR, "frames")


def get_buffer_frames(max_frames=None):
    """Read whatever's currently in the rolling buffer, oldest -> newest."""
    files = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.jpg")))
    if max_frames:
        files = files[-max_frames:]

    frames = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:          # skip if caught mid-write/prune
            frames.append(img)
    return frames


def to_data_url(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def locate_object(frames, target):
    """
    frames: list of BGR numpy arrays, oldest -> newest.
    Sends all of them in one message so the model can use recent motion/context,
    but still asks it to answer relative to the most recent frame.
    """
    if not frames:
        raise RuntimeError("no frames available in buffer")

    h, w = frames[-1].shape[:2]
    prompt = (
        f"You are given a short sequence of frames from a live camera feed, oldest to newest, "
        f"each of size {w}x{h}. Locate the object '{target}' in the MOST RECENT (last) frame. "
        'Reply with JSON only: {"found": true/false, "x": int, "y": int, '
        '"bbox": [x1, y1, x2, y2]} where x,y is the object center in pixel coordinates '
        "of the last frame."
    )

    content = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append({"type": "image_url", "image_url": {"url": to_data_url(frame)}})

    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.2,
        top_p=1,
        extra_body={
            "top_k": 1,
            "chat_template_kwargs": {"enable_thinking": False},  # skip reasoning trace, keep output clean JSON
        },
    )
    m = re.search(r"\{.*\}", r.choices[0].message.content, re.S)

    if m:
        return json.loads(m.group(0))

    return {"found": False}


def run_forever(target, poll_interval=2.0, max_frames=8):
    """
    Continuously reads the latest buffer contents and queries the VLM,
    since frame_capture.py keeps updating ./frames/ in real time.

    poll_interval: seconds to wait between VLM calls. Keep this >= a couple
    seconds -- each call sends up to `max_frames` images, so hammering this
    in a tight loop burns through API rate limits fast for no real benefit
    (the buffer doesn't change meaningfully every few hundred ms).
    """
    print(f"Watching {FRAME_DIR} for frames. Press Ctrl+C to stop.")
    last_seen_frame = None
    backoff = poll_interval

    while True:
        frames = get_buffer_frames(max_frames=max_frames)

        if not frames:
            print("Waiting for frames...")
            time.sleep(1)
            continue

        # Skip re-querying the VLM if the buffer hasn't actually advanced
        newest_path = sorted(glob.glob(os.path.join(FRAME_DIR, "frame_*.jpg")))[-1]
        if newest_path == last_seen_frame:
            time.sleep(poll_interval)
            continue
        last_seen_frame = newest_path

        try:
            detected = locate_object(frames, target)
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "ResourceExhausted" in err_str or "rate" in err_str.lower():
                backoff = min(backoff * 2, 60)  # exponential backoff, capped at 60s
                print(f"[{time.strftime('%H:%M:%S')}] Rate limited, backing off for {backoff:.0f}s...")
                time.sleep(backoff)
            else:
                print(f"[{time.strftime('%H:%M:%S')}] Error querying VLM: {e}")
                time.sleep(poll_interval)
            continue
        else:
            backoff = poll_interval  # reset backoff after a successful call

        if detected.get("found"):
            latest = frames[-1].copy()
            cv2.circle(latest, (detected["x"], detected["y"]), 8, (0, 255, 0), 2)
            cv2.imwrite("check.jpg", latest)
            print(f"[{time.strftime('%H:%M:%S')}] Found:", detected)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] Object not found. Checking again in {poll_interval:.0f}s...")

        time.sleep(poll_interval)


if __name__ == "__main__":
    try:
        run_forever("Spotted Lantern Fly", poll_interval=10.0, max_frames=4)
    except KeyboardInterrupt:
        print("\nStopped.")