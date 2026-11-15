import numpy as np

from milestone1.kinematics import forward_kinematics
from milestone1.differential_ik import differential_ik_step


def test_motion(q, twist, dt=0.01):
    H_before = forward_kinematics(q)

    q_next = differential_ik_step(q, twist, dt)

    H_after = forward_kinematics(q_next)

    # actual translation caused by one small IK step
    dp = H_after[:3, 3] - H_before[:3, 3]

    # expected translation
    expected_dp = twist[:3] * dt

    print("Expected translation:")
    print(expected_dp)

    print("Actual translation:")
    print(dp)

    print("Translation error:")
    print(dp - expected_dp)


def main():
    # use a nontrivial configuration
    q = np.array([
        0.5,
        -0.5,
        0.3,
        1.0,
        -0.4,
        0.8,
        0.2,
    ])

    print("TEST: +x velocity")
    test_motion(
        q,
        np.array([0.02, 0, 0, 0, 0, 0]),
    )


if __name__ == "__main__":
    main()