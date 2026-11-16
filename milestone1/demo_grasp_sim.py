import numpy as np

from xarm7_lib import Robot, HOME_POSE

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import move_cartesian


# Small test distance for simulation only
DOWN_DISTANCE_M = 0.03


def main():
    start_q = np.asarray(HOME_POSE, dtype=float)

    with Robot(q0=start_q) as robot:

        print("Starting joint angles:")
        print(robot.joint_values)

        print("\nSafety check:")
        print(robot.check_safety(robot.joint_values))

        print("\nStarting end-effector position:")
        print(forward_kinematics(robot.joint_values)[:3, 3])

        input("\nPress Enter to begin simulated grasp sequence...")

        # --------------------------------------------------
        # 1. Move to saved/start pose
        # --------------------------------------------------

        print("\nMoving to starting pose...")

        robot.set_joint_targets(
            start_q,
            speed=0.3,
            wait=True,
        )

        # --------------------------------------------------
        # 2. Open gripper
        # --------------------------------------------------

        print("Opening gripper...")

        robot.grip(1.0)

        print("Gripper opening:")
        print(robot.gripper_opening)

        # --------------------------------------------------
        # 3. Move downward using differential IK
        # --------------------------------------------------

        H_before_down = forward_kinematics(robot.joint_values)

        print("Moving down...")

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                -DOWN_DISTANCE_M,
                0.0,
                0.0,
                0.0,
            ]),
        )

        H_after_down = forward_kinematics(robot.joint_values)

        print("Position change after moving down:")
        print(
            H_after_down[:3, 3]
            - H_before_down[:3, 3]
        )

        # --------------------------------------------------
        # 4. Close gripper
        # --------------------------------------------------

        print("Closing gripper...")

        robot.grip(
            0.0,
            speed=0.3,
        )

        print("Gripper opening after closing:")
        print(robot.gripper_opening)

        # --------------------------------------------------
        # 5. Move back upward
        # --------------------------------------------------

        H_before_up = forward_kinematics(robot.joint_values)

        print("Moving up...")

        move_cartesian(
            robot,
            np.array([
                0.0,
                0.0,
                DOWN_DISTANCE_M,
                0.0,
                0.0,
                0.0,
            ]),
        )

        H_after_up = forward_kinematics(robot.joint_values)

        print("Position change after moving up:")
        print(
            H_after_up[:3, 3]
            - H_before_up[:3, 3]
        )

        print("\nFinal end-effector position:")
        print(H_after_up[:3, 3])

        print("\nSimulated grasp sequence complete.")

        input("\nPress Enter to close simulator...")


if __name__ == "__main__":
    main()