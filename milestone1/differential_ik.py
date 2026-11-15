import numpy as np

from milestone1.kinematics import forward_kinematics, jacobian


def differential_ik_step(q, desired_twist):
    """
    Compute a small joint-space update for a desired
    end-effector motion.

    Parameters
    ----------
    q : np.ndarray, shape (7,)
        Current joint angles.

    desired_twist : np.ndarray, shape (6,)
        Desired Cartesian motion:
        [vx, vy, vz, wx, wy, wz]

    Returns
    -------
    np.ndarray, shape (7,)
        Joint-space update.
    """
    pass


def move_cartesian(robot, displacement):
    """
    Move the end effector by a desired Cartesian displacement
    using inverse differential kinematics.
    """
    pass