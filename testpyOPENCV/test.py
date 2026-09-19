import pyrealsense2 as rs

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

finally:
    pipeline.stop()