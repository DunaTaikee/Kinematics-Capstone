import numpy as np

from xarm7_lib import Robot

from milestone1.kinematics import forward_kinematics
from milestone2.inverse_kinematics import pose_error, solve_ik


def make_pose(position, rotation):
    T = np.eye(4)
    T[:3, :3] = np.array(rotation, dtype=float)
    T[:3, 3] = position
    return T


TARGETS = [
    make_pose([0.5, 0.0, 0.0], [[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
    make_pose([0.6, 0.0, 0.3], [[0, 0, 1], [0, -1, 0], [1, 0, 0]]),
    make_pose([0.4, 0.2, 0.25], [[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
]


def move_to_pose(robot, T_target, limits, speed=0.3):
    q_start = np.asarray(robot.joint_values, dtype=float)
    q, ok = solve_ik(T_target, q_start, limits)
    e = pose_error(forward_kinematics(q), T_target)
    print("q =", np.round(q, 4))
    print("position error (m):", np.linalg.norm(e[:3]))
    print("rotation error (rad):", np.linalg.norm(e[3:]))
    if not ok or np.linalg.norm(e[:3]) > 1e-3 or np.linalg.norm(e[3:]) > 1e-2:
        raise RuntimeError("IK did not reach the target pose")
    violation = robot.check_safety(q)
    if violation is not None:
        raise RuntimeError(f"IK solution is not safe: {violation}")
    input("Stand clear. Press Enter to move...")
    robot.set_joint_targets(q, speed=speed, wait=True)


def main():
    with Robot() as robot:
        limits = robot.robot.position_limit + np.array([0.1, -0.1])
        for i, T_target in enumerate(TARGETS, start=1):
            print(f"\nPose {i}")
            move_to_pose(robot, T_target, limits)
            input("Take the photo, then press Enter...")


if __name__ == "__main__":
    main()