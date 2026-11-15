import numpy as np


# xArm7 modified Craig-DH parameters:
# [a_(i-1) (m), alpha_(i-1) (rad), d_i (m), theta_offset (rad)]
DH_PARAMS = np.array([
    [0.0,      0.0,        0.267,  0.0],
    [0.0,     -np.pi / 2,  0.0,    0.0],
    [0.0,      np.pi / 2,  0.293,  0.0],
    [0.0525,   np.pi / 2,  0.0,    0.0],
    [0.0775,   np.pi / 2,  0.3425, 0.0],
    [0.0,      np.pi / 2,  0.0,    0.0],
    [0.076,   -np.pi / 2,  0.269,  0.0],
])


def rot_x(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ], dtype=float)


def rot_z(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=float)


def trans_x(distance):
    H = np.eye(4)
    H[0, 3] = distance
    return H


def trans_z(distance):
    H = np.eye(4)
    H[2, 3] = distance
    return H


def dh_transform(a, alpha, d, theta):
    """
    Modified Craig-DH transform for one joint.
    """
    return (
        rot_x(alpha)
        @ trans_x(a)
        @ rot_z(theta)
        @ trans_z(d)
    )


def all_frames(q):
    """
    Compute the base-to-frame transform for every joint.

    Returns
    -------
    list of 8 transforms:
        frames[0] = identity base frame
        frames[1] ... frames[7] = joint frames
    """
    q = np.asarray(q, dtype=float)

    if q.shape != (7,):
        raise ValueError("q must have shape (7,)")

    frames = [np.eye(4)]
    H = np.eye(4)

    for i in range(7):
        a, alpha, d, theta_offset = DH_PARAMS[i]

        H_i = dh_transform(
            a,
            alpha,
            d,
            q[i] + theta_offset,
        )

        H = H @ H_i
        frames.append(H.copy())

    return frames


def forward_kinematics(q):
    """
    Return the gripper pose in the robot base frame.

    Parameters
    ----------
    q : array, shape (7,)
        Joint angles in radians.

    Returns
    -------
    H : array, shape (4, 4)
        Base-to-gripper homogeneous transform.
    """
    return all_frames(q)[-1]


def jacobian(q):
    """
    Compute the 6x7 geometric Jacobian.

    Row order:
        [vx, vy, vz, wx, wy, wz]
    """
    frames = all_frames(q)

    p_ee = frames[-1][:3, 3]

    J = np.zeros((6, 7))

    for i in range(7):
        # Joint i+1 rotates about the z-axis of its joint frame.
        H_i = frames[i + 1]

        z_i = H_i[:3, 2]
        p_i = H_i[:3, 3]

        # Linear velocity contribution
        J[:3, i] = np.cross(z_i, p_ee - p_i)

        # Angular velocity contribution
        J[3:, i] = z_i

    return J