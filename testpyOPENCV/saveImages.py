import cv2
import numpy as np
import pyrealsense2 as rs
import time
import os

# --------------------
# Settings
# --------------------
INTERVAL = 5  # seconds between captures

RGB_FOLDER = "dataset/rgb"
DEPTH_FOLDER = "dataset/depth"

os.makedirs(RGB_FOLDER, exist_ok=True)
os.makedirs(DEPTH_FOLDER, exist_ok=True)

# --------------------
# Start RealSense
# --------------------
pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(
    rs.stream.color,
    640, 480,
    rs.format.bgr8,
    30
)

config.enable_stream(
    rs.stream.depth,
    640, 480,
    rs.format.z16,
    30
)

pipeline.start(config)

# Give camera time to start producing frames
for _ in range(30):
    pipeline.wait_for_frames()

last_saved = time.time()
image_number = 0

try:
    while True:

        # Get synchronized frames
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert to NumPy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Display RGB
        cv2.imshow("RGB", color_image)

        # Display depth for visualization
        depth_display = cv2.convertScaleAbs(
            depth_image,
            alpha=0.03
        )

        cv2.imshow("Depth", depth_display)

        # --------------------
        # Save images
        # --------------------
        current_time = time.time()

        if current_time - last_saved >= INTERVAL:

            filename = f"{image_number:06d}"

            rgb_path = os.path.join(
                RGB_FOLDER,
                filename + ".jpg"
            )

            depth_path = os.path.join(
                DEPTH_FOLDER,
                filename + ".png"
            )

            # Save RGB
            cv2.imwrite(rgb_path, color_image)

            # Save raw depth
            cv2.imwrite(depth_path, depth_image)

            print(f"Saved {filename}")

            image_number += 1
            last_saved = current_time

        # Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()