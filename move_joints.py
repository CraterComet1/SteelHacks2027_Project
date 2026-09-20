import time
import numpy as np
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import sys

PORT = "COM3" if sys.platform == "win32" else "/dev/ttyACM0"
ROBOT_ID = "my_follower_arm"  # must match your calibration id

robot = SO101Follower(SO101FollowerConfig(
    port=PORT,
    id=ROBOT_ID,
    use_degrees=True,
    max_relative_target=10.0,   # MUST BE A FLOAT/ safety cap per command; delete this line if your version rejects it
))
robot.connect()
names = list(robot.bus.motors.keys())
# ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def read_q():
    obs = robot.get_observation()
    return {n: obs[f"{n}.pos"] for n in names}


def move_to(target, steps=60, dt=0.05):
    """Move smoothly to a target pose. `target` is a dict of joint -> value.
    Joints you leave out stay where they are."""
    start = read_q()
    goal = {n: target.get(n, start[n]) for n in names}
    for a in np.linspace(0, 1, steps):
        action = {f"{n}.pos": float(start[n] + a * (goal[n] - start[n])) for n in names}
        robot.send_action(action)
        time.sleep(dt)


try:
    print(read_q()) #Reads the current SO-101 position

    #Target position (Enter the output from the read_q() and use as the move_to argument to move the SO-101 to a indicated spot)
    move_to({'shoulder_pan': 15.208791208791208, 'shoulder_lift': 61.142857142857146, 'elbow_flex': -54.76923076923077, 'wrist_flex': -0.13186813186813187, 'wrist_roll': -5.934065934065934, 'gripper': 2.064220183486239})
    time.sleep(1)

    #move's duration = steps × dt
    #Open Claw
    move_to({'shoulder_pan': 11.076923076923077, 'shoulder_lift': 64.3956043956044, 'elbow_flex': -23.208791208791208, 'wrist_flex': -1.6263736263736264, 'wrist_roll': -6.197802197802198, 'gripper': 26.146788990825687}, steps=40, dt=0.03)
    time.sleep(1)

    #Close Claw
    #move_to({'shoulder_pan': 11.076923076923077, 'shoulder_lift': 64.3956043956044, 'elbow_flex': -23.208791208791208, 'wrist_flex': -1.6263736263736264, 'wrist_roll': -6.197802197802198, 'gripper': 1.529051987767584}, steps=40, dt=0.03)
    time.sleep(1)

    #Default position 
    #move_to({'shoulder_pan': 10.021978021978022, 'shoulder_lift': -32.13186813186813, 'elbow_flex': 94.41758241758242, 'wrist_flex': -9.45054945054945, 'wrist_roll': -5.934065934065934, 'gripper': 2.064220183486239})
    time.sleep(1)
    
    #-------
    #print("Current pose:", {k: round(v, 1) for k, v in read_q().items()})

    #input("Press Enter to move the base +45 degrees (Ctrl+C to abort)...")
    #move_to({"shoulder_pan": read_q()["shoulder_pan"] + 45})

    time.sleep(1)
    #input("Press Enter to return...")
    #move_to({"shoulder_pan": read_q()["shoulder_pan"] - 45})

    #print("Final pose:", {k: round(v, 1) for k, v in read_q().items()})

finally:
    robot.disconnect()