import argparse
import time
import numpy as np
import placo
from placo_utils.visualization import robot_viz


argparser = argparse.ArgumentParser(description="Read and plot logs")
argparser.add_argument("--log_path", type=str, help="Path to the log file.", required=True)
args = argparser.parse_args()

histories = placo.HistoryCollection()
histories.loadReplays(args.log_path)

dt = 0.005
ts = np.arange(histories.smallestTimestamp(), histories.biggestTimestamp(), dt)

kp = histories.number(f"kp:Left_Shoulder_Pitch", histories.smallestTimestamp() + histories.biggestTimestamp() / 2)
print("Kp = ", kp)

robot = placo.RobotWrapper("../humanoid_model/k1/robot.xml", placo.Flags.mjcf)

viz = robot_viz(robot)
viz.display(robot.state.q)
time.sleep(3)

for t in ts:
    for joint in robot.joint_names():
        robot.set_joint(joint, histories.number(f"q:{joint}", t))

    T = histories.pose("T_world_trunk", t)
    robot.set_T_world_fbase(T)
    robot.update_kinematics()

    viz.display(robot.state.q)
    
    time.sleep(dt)