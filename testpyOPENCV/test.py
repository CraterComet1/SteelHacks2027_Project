import pyrealsense2 as rs
import cv2
from yolo import detect
import numpy as np

pipeline = rs.pipeline()

pipeline.start()

try:
    while True:
        frames = pipeline.wait_for_frames()

        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if not color or not depth:
            continue

        print("Got color and depth frames!")

        frame = np.asanyarray(
                    color.get_data()
                )

        # -----------------------------
        # YOLO inference
        # -----------------------------

        results = detect(frame)

        # Draw YOLO detections
        annotated_frame = frame.copy()
        for det in results:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
            cv2.putText(
                annotated_frame,
                f'{det["class"]} {det["confidence"]:.2f}',
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # Display
        cv2.imshow(
            "RealSense + YOLO",
            annotated_frame
        )

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    pipeline.stop()