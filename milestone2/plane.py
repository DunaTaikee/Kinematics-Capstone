import time

import numpy as np


def fit_plane(points):
    P = np.asarray(points, dtype=float)
    if P.shape[0] < 3:
        raise ValueError("Need at least 3 points")
    c = P.mean(axis=0)
    _, S, Vt = np.linalg.svd(P - c, full_matrices=False)
    if S[1] < 0.05 * S[0]:
        raise ValueError("Points are almost on a line")
    n = Vt[-1]
    d = n @ c
    if d < 0:
        n, d = -n, -d
    return n, d


def record_tip_points(detector, color, duration=3.0, rate=30):
    points = []
    t_end = time.time() + duration
    while time.time() < t_end:
        tip = get_tip_point(detector, color)
        if tip is not None:
            points.append(tip)
        time.sleep(1.0 / rate)
    return np.array(points)

def get_tip_point(detector, color):
    result = detector.query(color)
    if result is None:
        return None
    return np.asarray(result[0], dtype=float)