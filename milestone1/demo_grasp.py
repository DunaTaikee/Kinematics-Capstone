from xarm7_lib import Robot

from milestone1.differential_ik import move_cartesian


def main():
    with Robot() as robot:
        # 1. Move to saved pose above the lightsaber
        # 2. Open gripper
        # 3. Move down using differential IK
        # 4. Close gripper
        # 5. Move back up
        pass


if __name__ == "__main__":
    main()