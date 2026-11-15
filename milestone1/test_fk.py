import numpy as np

from milestone1.kinematics import forward_kinematics


TEST_CASES = [
    (
        np.array([0, 0, 0, 0, 0, 0, 0], dtype=float),
        np.array([
            [1.0, 0.0,  0.0,  0.20600],
            [0.0, -1.0, 0.0,  0.00000],
            [0.0, 0.0, -1.0, -0.05150],
            [0.0, 0.0,  0.0,  1.0],
        ]),
    ),

    (
        np.array([0, 0.5, 0, 1.2, 0, 0.7, 0], dtype=float),
        np.array([
            [1.0, 0.0,  0.0, 0.54246],
            [0.0, -1.0, 0.0, 0.00000],
            [0.0, 0.0, -1.0, 0.01793],
            [0.0, 0.0,  0.0, 1.0],
        ]),
    ),

    (
        np.array([0.5, -0.5, 0.3, 1.0, -0.4, 0.8, 0.2]),
        np.array([
            [0.4311,  0.5438,  0.7200, 0.37201],
            [0.6535, -0.7384,  0.1665, 0.31780],
            [0.6222,  0.3988, -0.6737, 0.45329],
            [0.0,     0.0,     0.0,    1.0],
        ]),
    ),

    (
        np.array([-0.8, 0.6, 1.0, 1.5, 0.5, 1.0, -1.2]),
        np.array([
            [0.4827,  0.8699,  0.1014, 0.60371],
            [0.4848, -0.3618,  0.7963, 0.16821],
            [0.7294, -0.3352, -0.5963, 0.26665],
            [0.0,     0.0,     0.0,    1.0],
        ]),
    ),
]


def main():
    for i, (q, expected) in enumerate(TEST_CASES, start=1):
        actual = forward_kinematics(q)

        error = np.max(np.abs(actual - expected))

        print(f"Configuration {i}")
        print("max error:", error)

        if np.allclose(actual, expected, atol=1e-4):
            print("PASS")
        else:
            print("FAIL")
            print("Actual:")
            print(actual)
            print("Expected:")
            print(expected)

        print()


if __name__ == "__main__":
    main()