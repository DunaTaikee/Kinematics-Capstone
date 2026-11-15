from pathlib import Path

import numpy as np

from xarm7_lib import Robot

from milestone1.differential_ik import move_cartesian


POSE_FILE = Path(__file__).with_name("grasp_pose.npy")

# Fill this in after physically measuring the distance.
# Example: 45 mm -> 0.045 m
DOWN_DISTANCE_M = None


def main():
    if not POSE_FILE.exists():
        raise RuntimeError(
            "No saved grasp pose found. "
            "Run milestone1.save_grasp_pose on the real robot first."
        )

    if DOWN_DISTANCE_M is None:
        raise RuntimeError(
            "Measure the downward grasp distance and set DOWN_DISTANCE_M."
        )

    q_above = np.load(POSE_FILE)

    if q_above.shape != (7,):
        raise ValueError("Saved grasp pose must contain 7 joint angles.")

    with Robot() as robot:

        # Check the saved configuration before moving.
        violation = robot.check_safety(q_above)

        if violation is not None:
            raise RuntimeError(
                f"Saved grasp pose is not safe: {violation}"
            )

        print("Saved grasp pose:")
        print(q_above)

        print("\nDownward distance:")
        print(DOWN_DISTANCE_M, "m")

        input(
            "\nMake sure the area is clear and the lightsaber "
            "has not moved. Press Enter to start..."
        )

        # --------------------------------------------------
        # 1. Move to saved pose above the lightsaber
        # --------------------------------------------------

        print("\nMoving to saved pose...")

        robot.set_joint_targets(
            q_above,
            speed=0.3,
            wait=True,
        )

        # --------------------------------------------------
        # 2. Open gripper
        # --------------------------------------------------

        print("Opening gripper...")

        robot.grip(1.0)

        # --------------------------------------------------
        # 3. Move downward using differential IK
        # --------------------------------------------------

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

        # --------------------------------------------------
        # 4. Close gripper
        # --------------------------------------------------

        print("Closing gripper...")

        robot.grip(
            0.0,
            speed=0.3,
        )

        print("Gripper opening after grasp:")
        print(robot.gripper_opening)

        # --------------------------------------------------
        # 5. Move back upward
        # --------------------------------------------------

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

        print("\nGrasp sequence complete.")


if __name__ == "__main__":
    main()