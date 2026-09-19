import pyrealsense2 as rs

# Create pipeline
pipeline = rs.pipeline()

# Configure the streams
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

# Start the camera
profile = pipeline.start(config)

try:
    # -------------------------
    # Depth camera intrinsics
    # -------------------------
    depth_profile = profile.get_stream(
        rs.stream.depth
    ).as_video_stream_profile()

    depth_intrinsics = depth_profile.get_intrinsics()

    print("DEPTH INTRINSICS")
    print("Width:", depth_intrinsics.width)
    print("Height:", depth_intrinsics.height)
    print("fx:", depth_intrinsics.fx)
    print("fy:", depth_intrinsics.fy)
    print("cx:", depth_intrinsics.ppx)
    print("cy:", depth_intrinsics.ppy)
    print("Distortion model:", depth_intrinsics.model)
    print("Distortion coefficients:", depth_intrinsics.coeffs)

    # -------------------------
    # RGB camera intrinsics
    # -------------------------
    color_profile = profile.get_stream(
        rs.stream.color
    ).as_video_stream_profile()

    color_intrinsics = color_profile.get_intrinsics()

    print("\nRGB INTRINSICS")
    print("Width:", color_intrinsics.width)
    print("Height:", color_intrinsics.height)
    print("fx:", color_intrinsics.fx)
    print("fy:", color_intrinsics.fy)
    print("cx:", color_intrinsics.ppx)
    print("cy:", color_intrinsics.ppy)
    print("Distortion model:", color_intrinsics.model)
    print("Distortion coefficients:", color_intrinsics.coeffs)

finally:
    pipeline.stop()