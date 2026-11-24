import numpy as np
from milestone1.kinematics import forward_kinematics, jacobian

def rotation_error(R_current, R_target):
    R_err = R_target @ R_current.T
    cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-4:
        A = (R_err + np.eye(3)) / 2.0
        k = np.argmax(np.diag(A))
        axis = A[:, k] / np.sqrt(A[k, k])
        return theta * axis
    axis = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ]) / (2.0 * np.sin(theta))
    return theta * axis


def pose_error(T_current, T_target):
    e_pos = T_target[:3, 3] - T_current[:3, 3]
    e_rot = rotation_error(T_current[:3, :3], T_target[:3, :3])
    return np.concatenate([e_pos, e_rot])


def inverse_kinematics(T_target, q_init, limits=None, max_iters=10000,
                       rot_weight=0.5, pos_tol=1e-4, rot_tol=1e-3,
                       max_step=0.1):
    q = np.asarray(q_init, dtype=float).copy()
    W = np.diag([1, 1, 1, rot_weight, rot_weight, rot_weight])
    for i in range(max_iters):
        e = pose_error(forward_kinematics(q), T_target)
        if np.linalg.norm(e[:3]) < pos_tol and np.linalg.norm(e[3:]) < rot_tol:
            return q, True, i
        J = jacobian(q)
        g = J.T @ W @ e
        Jg = J @ g
        denom = Jg @ W @ Jg
        if denom < 1e-12:
            break
        alpha = (g @ g) / denom
        dq = alpha * g
        n = np.max(np.abs(dq))
        if n > max_step:
            dq *= max_step / n
        q = q + dq
        if limits is not None:
            q = np.clip(q, limits[:, 0], limits[:, 1])
    return q, False, max_iters


def solve_ik(T_target, q_init, limits=None, attempts=5, seed=0):
    rng = np.random.default_rng(seed)
    q_seed = np.asarray(q_init, dtype=float)
    for _ in range(attempts):
        q, ok, _ = inverse_kinematics(T_target, q_seed, limits)
        if ok:
            return q, True
        q_seed = np.asarray(q_init, dtype=float) + rng.normal(0.0, 0.3, 7)
    return q, False