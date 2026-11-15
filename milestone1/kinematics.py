import numpy as np


def forward_kinematics(q):
    """
    Compute the end-effector pose from the 7 joint angles.

    Parameters
    ----------
    q : np.ndarray, shape (7,)

    Returns
    -------
    np.ndarray, shape (4, 4)
        Homogeneous transform of the gripper in the base frame.
    """
    pass


def jacobian(q):
    """
    Compute the 6x7 end-effector Jacobian.

    Row order:
    [vx, vy, vz, wx, wy, wz]

    Parameters
    ----------
    q : np.ndarray, shape (7,)

    Returns
    -------
    np.ndarray, shape (6, 7)
    """
    pass