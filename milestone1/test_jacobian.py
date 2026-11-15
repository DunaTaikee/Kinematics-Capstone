import numpy as np

from milestone1.kinematics import forward_kinematics, jacobian


def finite_difference_jacobian(q, epsilon=1e-6):
    q = np.asarray(q, dtype=float)

    H = forward_kinematics(q)
    R = H[:3, :3]

    J = np.zeros((6, 7))

    for i in range(7):
        q_plus = q.copy()
        q_minus = q.copy()

        q_plus[i] += epsilon
        q_minus[i] -= epsilon

        H_plus = forward_kinematics(q_plus)
        H_minus = forward_kinematics(q_minus)

        # Linear velocity
        p_plus = H_plus[:3, 3]
        p_minus = H_minus[:3, 3]

        J[:3, i] = (p_plus - p_minus) / (2 * epsilon)

        # Angular velocity
        R_plus = H_plus[:3, :3]
        R_minus = H_minus[:3, :3]

        R_dot = (R_plus - R_minus) / (2 * epsilon)

        omega_hat = R_dot @ R.T
        omega_hat = 0.5 * (omega_hat - omega_hat.T)

        J[3:, i] = np.array([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0],
        ])

    return J


def main():
    q = np.array([
        0.5,
        -0.5,
        0.3,
        1.0,
        -0.4,
        0.8,
        0.2,
    ])

    J = jacobian(q)
    J_fd = finite_difference_jacobian(q)

    error = np.max(np.abs(J - J_fd))

    print("Analytical Jacobian:")
    print(J)

    print("\nFinite-difference Jacobian:")
    print(J_fd)

    print("\nMaximum error:", error)

    if np.allclose(J, J_fd, atol=1e-5):
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()