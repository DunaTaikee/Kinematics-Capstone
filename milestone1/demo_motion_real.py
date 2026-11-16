import numpy as np

from xarm7_lib import Robot

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import move_cartesian


# First REL test:
# use 0.2 -> 2 cm translations and pi/15 rotation.
#
# Once everything behaves correctly, change this to 1.0
# for the required 10 cm / pi/3 video.
SCALE = 1.0


def print_position_change(H_start, H_end):
    delta = H_end[:3, 3] - H_start[:3, 3]
    print("Position change:")
    print(delta)


def main():
    with Robot() as robot:

        print("Current joint angles:")
        print(robot.joint_values)

        print("\nFree-drive the arm to a comfortable starting pose.")
        print("Choose a pose with room to move in +x, +y, and +z.")

        robot.free_drive()

        input("\nPress Enter when the starting pose is ready...")

        robot.end_free_drive()

        q_start = robot.joint_values.copy()

        violation = robot.check_safety(q_start)

        print("\nStarting configuration:")
        print(q_start)

        print("\nSafety check:")
        print(violation)

        if violation is not None:
            raise RuntimeError(
                f"Starting pose is not safe: {violation}"
            )

        print("\nStarting end-effector position:")
        print(forward_kinematics(q_start)[:3, 3])

        input("\nStand clear. Press Enter to begin programmed motion...")

        # +x
        H_start = forward_kinematics(robot.joint_values)

        move_cartesian(
            robot,
            np.array([
                0.10 * SCALE,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
        )

        H_end = forward_kinematics(robot.joint_values)

        print("\n+x motion")
        print_position_change(H_start, H_end)

        # +y
        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.10 * SCALE,
                0.0,
                0.0,
                0.0,
                0.0,
            ]),
        )

        H_end = forward_kinematics(robot.joint_values)

        print("\n+y motion")
        print_position_change(H_start, H_end)

        # +z
        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                0.10 * SCALE,
                0.0,
                0.0,
                0.0,
            ]),
        )

        H_end = forward_kinematics(robot.joint_values)

        print("\n+z motion")
        print_position_change(H_start, H_end)

        # rotate about base z
        H_start = H_end

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                (np.pi / 3) * SCALE,
            ]),
        )

        H_end = forward_kinematics(robot.joint_values)

        print("\nz rotation")

        print("Position change during rotation:")
        print(
            H_end[:3, 3]
            - H_start[:3, 3]
        )

        R_start = H_start[:3, :3]
        R_end = H_end[:3, :3]

        R_change = R_end @ R_start.T

        z_angle = np.arctan2(
            R_change[1, 0],
            R_change[0, 0],
        )

        print("Measured rotation:", z_angle)
        print(
            "Expected rotation:",
            (np.pi / 3) * SCALE,
        )

        print("\nMotion demo finished.")


if __name__ == "__main__":
    main()