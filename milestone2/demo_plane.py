import numpy as np

from xarm7_lib import Robot
from libcapstone import LightsaberDetector

from milestone2.plane import fit_plane, record_tip_points


SABER_COLOR = "blue"


def main():
    detector = LightsaberDetector()
    with Robot() as robot:
        robot.register_visualizer(detector)
        input("Press Enter, then sweep the saber for 3 seconds...")
        points = record_tip_points(detector, SABER_COLOR, duration=3.0)
        print(f"Recorded {len(points)} tip points")
        n, d = fit_plane(points)
        print(f"n = {np.round(n, 4)}")
        print(f"d = {d:.4f}")
        print(f"Plane: {n[0]:.4f} x + {n[1]:.4f} y + {n[2]:.4f} z = {d:.4f}")
        detector.draw_plane((n, d))
        input("Take the screenshot, then press Enter to quit...")


if __name__ == "__main__":
    main()