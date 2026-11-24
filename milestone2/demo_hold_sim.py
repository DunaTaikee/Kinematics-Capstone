import numpy as np

from xarm7_lib import Robot, HOME_POSE

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import move_cartesian
from milestone2.demo_hold import LIFT_M, OFFSETS, hold_blade
from milestone2.tip_transform import calibrate_tip_transform


HANDLE_IN_EE = np.array([0.02, 0.0, 0.0])
TIP_IN_EE = np.array([0.275, 0.0, 0.0])


class SimDetector:
    def __init__(self, robot, noise=0.003):
        self.robot = robot
        self.noise = noise
        self.rng = np.random.default_rng(0)

    def query(self, color):
        T = forward_kinematics(self.robot.joint_values)
        tip = T[:3, :3] @ TIP_IN_EE + T[:3, 3]
        handle = T[:3, :3] @ HANDLE_IN_EE + T[:3, 3]
        ends = np.array([tip, handle])
        return ends + self.rng.normal(0.0, self.noise, ends.shape)


def main():
    with Robot(q0=np.asarray(HOME_POSE, dtype=float)) as robot:
        detector = SimDetector(robot)
        limits = robot.robot.position_limit + np.array([0.1, -0.1])
        print("Start ee position:", np.round(forward_kinematics(robot.joint_values)[:3, 3], 3))
        robot.grip(0.0)
        move_cartesian(robot, np.array([0, 0, LIFT_M, 0, 0, 0]))
        T_ee_tip = calibrate_tip_transform(robot, detector, "red", OFFSETS)
        print("T_ee_tip =")
        print(np.round(T_ee_tip, 4))
        print("tip position error vs truth (m):", np.linalg.norm(T_ee_tip[:3, 3] - TIP_IN_EE))
        print("\nHorizontal")
        hold_blade(robot, T_ee_tip, [0, 1, 0], limits)
        input("Press Enter for vertical...")
        print("\nVertical")
        hold_blade(robot, T_ee_tip, [0, 0, 1], limits)
        T = forward_kinematics(robot.joint_values)
        blade = T[:3, :3] @ (TIP_IN_EE - HANDLE_IN_EE)
        print("blade direction now:", np.round(blade / np.linalg.norm(blade), 3))
        input("Press Enter to finish...")


if __name__ == "__main__":
    main()