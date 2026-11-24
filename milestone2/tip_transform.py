import time

import numpy as np

from milestone1.kinematics import forward_kinematics


def saber_frame(p_tip, p_handle, y_ee):
    z = p_tip - p_handle
    z = z / np.linalg.norm(z)
    y = y_ee - (y_ee @ z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0] = x
    T[:3, 1] = y
    T[:3, 2] = z
    T[:3, 3] = p_tip
    return T


def pick_tip(p_a, p_b, p_ee):
    if np.linalg.norm(p_a - p_ee) > np.linalg.norm(p_b - p_ee):
        return p_a, p_b
    return p_b, p_a


def project_to_rotation(M):
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def tip_transform_once(q, p_a, p_b):
    T_base_ee = forward_kinematics(q)
    p_tip, p_handle = pick_tip(p_a, p_b, T_base_ee[:3, 3])
    T_base_tip = saber_frame(p_tip, p_handle, T_base_ee[:3, 1])
    return np.linalg.inv(T_base_ee) @ T_base_tip


def average_transforms(transforms):
    T = np.eye(4)
    T[:3, :3] = project_to_rotation(np.mean([X[:3, :3] for X in transforms], axis=0))
    T[:3, 3] = np.mean([X[:3, 3] for X in transforms], axis=0)
    return T


def calibrate_tip_transform(robot, detector, color, offsets, samples=20):
    q_home = np.asarray(robot.joint_values, dtype=float)
    results = []
    for dq in offsets:
        q = q_home + np.asarray(dq, dtype=float)
        if robot.check_safety(q) is not None:
            continue
        robot.set_joint_targets(q, speed=0.2, wait=True)
        time.sleep(0.5)
        for _ in range(samples):
            points = get_saber_points(detector, color)
            if points is not None:
                p_a, p_b = points
                results.append(tip_transform_once(robot.joint_values, p_a, p_b))
            time.sleep(0.03)
    robot.set_joint_targets(q_home, speed=0.2, wait=True)
    if not results:
        raise RuntimeError("No saber detections")
    return average_transforms(results)

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rotation_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = a @ b
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        K = skew(perp / np.linalg.norm(perp))
        return np.eye(3) + 2.0 * K @ K
    K = skew(axis / s)
    theta = np.arctan2(s, c)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def ee_target_for_blade_direction(T_base_ee, T_ee_tip, direction):
    blade = T_base_ee[:3, :3] @ T_ee_tip[:3, 2]
    R = rotation_between(blade, np.asarray(direction, dtype=float))
    T = T_base_ee.copy()
    T[:3, :3] = R @ T_base_ee[:3, :3]
    return T

def get_saber_points(detector, color):
    result = detector.query(color)
    if result is None:
        return None
    p_a = np.asarray(result[0], dtype=float)
    p_b = np.asarray(result[1], dtype=float)
    return p_a, p_b