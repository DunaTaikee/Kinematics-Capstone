import os
from pathlib import Path

import numpy as np

from xarm7_lib import Robot


POSE_FILE = Path(__file__).with_name("grasp_pose.npy")


def main():
    if not os.environ.get("ROBOT_IP"):
        raise RuntimeError(
            "This script requires the real robot because free-drive "
            "is not available in simulation."
        )

    with Robot() as robot:
        # The handout wants the gripper open and positioned
        # slightly above/around the lightsaber handle.
        robot.grip(1.0)

        print("Free-drive the gripper to the pose slightly above the lightsaber.")
        print("Place the open gripper around the desired grasp location.")

        robot.free_drive()

        input("\nPress Enter when the arm is in the desired starting pose...")

        robot.end_free_drive()

        q_above = robot.joint_values.copy()

        print("\nSaved joint angles:")
        print(q_above)

        np.save(POSE_FILE, q_above)

        print(f"\nPose saved to:\n{POSE_FILE}")


if __name__ == "__main__":
    main()