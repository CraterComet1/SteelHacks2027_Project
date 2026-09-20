import numpy as np
from scipy.spatial.transform import Rotation as R
from lerobot.model.kinematics import RobotKinematics

URDF = "/home/peggy/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"  # use your path from step 1
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

kin = RobotKinematics(
    urdf_path=URDF,
    target_frame_name="gripper_frame_link",
    joint_names=names,
)

pose = {'shoulder_pan': 10.021978021978022, 'shoulder_lift': -32.13186813186813, 'elbow_flex': 94.41758241758242, 'wrist_flex': -9.45054945054945, 'wrist_roll': -5.934065934065934, 'gripper': 2.064220183486239}
#pose = {'shoulder_pan': 11.08, 'shoulder_lift': 64.40, 'elbow_flex': -23.21, 'wrist_flex': -1.63, 'wrist_roll': -6.20, 'gripper': 26.15}

q = np.array([pose[n] for n in names])      # degrees
T = kin.forward_kinematics(q)               # 4x4 matrix

x, y, z = T[:3, 3]
roll, pitch, yaw = R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
print(f"EE position (m): x={x:.3f} y={y:.3f} z={z:.3f}")
print(f"EE orientation (deg): roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")

# Ask IK to move the gripper 2 cm up from the pose above, then check the result
T_target = T.copy()
T_target[2, 3] += 0.02

q_new = kin.inverse_kinematics(q, T_target)
T_new = kin.forward_kinematics(q_new)

print("Joint change (deg):", np.round(q_new - q, 2))
print("Position error (mm):", np.linalg.norm(T_new[:3, 3] - T_target[:3, 3]) * 1000)