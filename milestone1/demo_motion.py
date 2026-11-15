import numpy as np

from xarm7_lib import Robot, HOME_POSE

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import move_cartesian


START_Q = np.asarray(HOME_POSE, dtype=float)


def print_position_change(H_start, H_end):
    p_start = H_start[:3, 3]
    p_end = H_end[:3, 3]

    delta = p_end - p_start

    print("Position change:")
    print(delta)

    return delta


def main():
    with Robot(q0=START_Q) as robot:

        print("Starting joint angles:")
        print(robot.joint_values)

        print("\nSafety check:")
        print(robot.check_safety(robot.joint_values))

        print("\nStarting end-effector position:")
        print(forward_kinematics(robot.joint_values)[:3, 3])

        input("\nPress Enter to begin the motion demo...")

        # --------------------------------------------------
        # 1. Move +x by 10 cm
        # --------------------------------------------------

        q_start = robot.joint_values.copy()
        H_start = forward_kinematics(q_start)

        move_cartesian(
            robot,
            np.array([
                0.10,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
        )

        q_end = robot.joint_values.copy()
        H_end = forward_kinematics(q_end)

        print("\n+x motion")
        print_position_change(H_start, H_end)

        # --------------------------------------------------
        # 2. Move +y by 10 cm
        # --------------------------------------------------

        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.10,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
        )

        q_end = robot.joint_values.copy()
        H_end = forward_kinematics(q_end)

        print("\n+y motion")
        print_position_change(H_start, H_end)

        # --------------------------------------------------
        # 3. Move +z by 10 cm
        # --------------------------------------------------

        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                0.10,
                0.0,
                0.0,
                0.0,
            ]),
        )

        q_end = robot.joint_values.copy()
        H_end = forward_kinematics(q_end)

        print("\n+z motion")
        print_position_change(H_start, H_end)

        # --------------------------------------------------
        # 4. Rotate about base-frame z by pi/3
        # --------------------------------------------------

        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                np.pi / 3,
            ]),
        )

        q_end = robot.joint_values.copy()
        H_end = forward_kinematics(q_end)

        R_start = H_start[:3, :3]
        R_end = H_end[:3, :3]

        # orientation change expressed in the base frame
        R_change = R_end @ R_start.T

        z_angle = np.arctan2(
            R_change[1, 0],
            R_change[0, 0],
        )

        print("\nz rotation")
        print("Measured rotation:", z_angle)
        print("Expected rotation:", np.pi / 3)

        print("\nFinal end-effector position:")
        print(H_end[:3, 3])

        input("\nMotion demo finished. Press Enter to close the simulator...")


if __name__ == "__main__":
    main()