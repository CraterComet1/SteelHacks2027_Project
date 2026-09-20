import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.model.kinematics import RobotKinematics

URDF = "/home/peggy/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
PORT = "/dev/ttyACM0"

robot = SO101Follower(SO101FollowerConfig(port=PORT, id="my_follower_arm", use_degrees=True))
robot.connect()
names = list(robot.bus.motors.keys())
kin = RobotKinematics(urdf_path=URDF, target_frame_name="gripper_frame_link", joint_names=names)

try:
    obs = robot.get_observation()
    q = np.array([obs[f"{n}.pos"] for n in names])
    T = kin.forward_kinematics(q)
    x, y, z = T[:3, 3]
    roll, pitch, yaw = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
    print("Joints (deg):", dict(zip(names, np.round(q, 1))))
    print(f"EE position (m): x={x:.3f} y={y:.3f} z={z:.3f}")
    print(f"EE orientation (deg): roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")
finally:
    robot.disconnect()