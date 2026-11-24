import numpy as np

from xarm7_lib import Robot
from libcapstone import LightsaberDetector

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import move_cartesian
from milestone1.demo_grasp import POSE_FILE, DOWN_DISTANCE_M
from milestone2.demo_ik_poses import move_to_pose
from milestone2.inverse_kinematics import inverse_kinematics
from milestone2.tip_transform import calibrate_tip_transform, ee_target_for_blade_direction, skew


SABER_COLOR = "green"
LIFT_M = 0.15
HOLD_POSITION = np.array([0.45, 0.0, 0.35])
OFFSETS = [
    np.zeros(7),
    [0, 0, 0, 0, 0, 0, 0.3],
    [0, 0, 0, 0, 0, 0, -0.3],
    [0, 0, 0, 0, 0, 0.2, 0],
    [0, 0, 0, 0, 0.3, 0, 0],
]


def grasp(robot):
    q_above = np.load(POSE_FILE)
    violation = robot.check_safety(q_above)
    if violation is not None:
        raise RuntimeError(f"Saved grasp pose is not safe: {violation}")
    input("Stand clear. Press Enter to grasp...")
    robot.set_joint_targets(q_above, speed=0.3, wait=True)
    robot.grip(1.0)
    move_cartesian(robot, np.array([0, 0, -DOWN_DISTANCE_M, 0, 0, 0]))
    robot.grip(0.0, speed=0.3)
    move_cartesian(robot, np.array([0, 0, DOWN_DISTANCE_M, 0, 0, 0]))


def hold_blade(robot, T_ee_tip, direction, limits):
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    q_now = np.asarray(robot.joint_values, dtype=float)
    T_base = ee_target_for_blade_direction(forward_kinematics(q_now), T_ee_tip, direction)
    T_base[:3, 3] = HOLD_POSITION
    K = skew(direction)
    best = None
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        R_roll = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K
        T_target = T_base.copy()
        T_target[:3, :3] = R_roll @ T_base[:3, :3]
        q, ok, _ = inverse_kinematics(T_target, q_now, limits)
        if ok and robot.check_safety(q) is None:
            cost = np.linalg.norm(q - q_now)
            if best is None or cost < best[0]:
                best = (cost, T_target)
    if best is None:
        raise RuntimeError(f"No reachable roll for blade direction {direction}")
    move_to_pose(robot, best[1], limits, speed=0.2)


def main():
    detector = LightsaberDetector()
    with Robot() as robot:
        robot.register_visualizer(detector)
        limits = robot.robot.position_limit + np.array([0.1, -0.1])
        grasp(robot)
        input("Check the jaws are holding the saber, then press Enter...")
        move_cartesian(robot, np.array([0, 0, LIFT_M, 0, 0, 0]))
        T_ee_tip = calibrate_tip_transform(robot, detector, SABER_COLOR, OFFSETS)
        print("T_ee_tip =")
        print(np.round(T_ee_tip, 4))
        print("\nHorizontal")
        hold_blade(robot, T_ee_tip, [0, 1, 0], limits)
        input("Press Enter for vertical...")
        print("\nVertical")
        hold_blade(robot, T_ee_tip, [0, 0, 1], limits)
        input("Press Enter to finish...")


if __name__ == "__main__":
    main()