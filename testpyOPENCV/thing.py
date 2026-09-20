import pyrealsense2 as rs

ctx = rs.context()
device = ctx.query_devices()[0]

print("Name:", device.get_info(rs.camera_info.name))
print("USB type:", device.get_info(rs.camera_info.usb_type_descriptor))
