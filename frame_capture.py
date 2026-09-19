"""
Rolling 30-frame buffer capture from a laptop webcam using OpenCV.

- Shows a live preview window of the camera feed.
- Continuously writes the last BUFFER_SIZE frames to ./frames/ as JPEGs,
  named by timestamp, pruning older ones so only the newest 30 remain.
- Writes are atomic (write to .tmp then rename) so a reader (e.g. WSL)
  never sees a half-written file.

Run with: python frame_capture.py
Press 'q' in the preview window to quit cleanly.
"""

import cv2
import os
import time

# ---- Config ----
CAMERA_INDEX = 0          # laptop webcam is almost always 0; run find_camera.py to confirm
BUFFER_SIZE = 30
TARGET_FPS = 10           # rate frames get saved to disk; preview runs as fast as the camera allows

# Resolve frames/ relative to this script's location, works regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "frames")
os.makedirs(OUT_DIR, exist_ok=True)


def atomic_write(path, img):
    """Write to a temp file then rename, so readers never see a half-written frame."""
    root, ext = os.path.splitext(path)
    tmp_path = root + ".tmp" + ext   # keep the real extension (.jpg) so imwrite can detect format
    cv2.imwrite(tmp_path, img)
    os.replace(tmp_path, path)


def prune_buffer():
    """Keep only the newest BUFFER_SIZE frames on disk."""
    files = sorted(f for f in os.listdir(OUT_DIR)
                    if f.startswith("frame_") and f.endswith(".jpg"))
    for old in files[:-BUFFER_SIZE]:
        try:
            os.remove(os.path.join(OUT_DIR, old))
        except FileNotFoundError:
            pass  # already removed by a race, harmless


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)  # CAP_DSHOW is more reliable on Windows
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            f"Run find_camera.py to confirm the right index."
        )

    print(f"Capturing from camera index {CAMERA_INDEX}")
    print(f"Writing rolling buffer of {BUFFER_SIZE} frames to: {OUT_DIR}")
    print("Press 'q' in the preview window to quit.")

    save_interval = 1.0 / TARGET_FPS
    last_save = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed, retrying...")
                continue

            cv2.imshow("Live Feed - press q to quit", frame)

            now = time.time()
            if now - last_save >= save_interval:
                fname = os.path.join(OUT_DIR, f"frame_{int(now * 1000)}.jpg")
                atomic_write(fname, frame)
                prune_buffer()
                last_save = now

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()