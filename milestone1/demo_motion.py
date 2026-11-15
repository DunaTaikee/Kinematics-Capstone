from xarm7_lib import Robot

from milestone1.differential_ik import move_cartesian


def main():
    with Robot() as robot:
        # 1. Start from a suitable pose
        # 2. Move +x by 10 cm
        # 3. Move +y by 10 cm
        # 4. Move +z by 10 cm
        # 5. Rotate about z by pi/3
        pass


if __name__ == "__main__":
    main()