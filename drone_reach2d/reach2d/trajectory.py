"""Turn a geometric path into a dynamically consistent reference trajectory."""
from __future__ import annotations

import numpy as np


def make_reference(path, speed, h, A, B, smooth_time=1.0):
    """Polyline -> (x_ref [N+1, 4], u_ref [N, 2]); the reference obeys x+ = A x + B u exactly."""
    seg = np.diff(path, axis=0)
    L = np.linalg.norm(seg, axis=1)
    dirs = seg / L[:, None]
    s_end = np.concatenate([[0], np.cumsum(L)])
    n = int(np.ceil(s_end[-1] / speed / h))
    s = np.minimum(np.arange(n) * speed * h, s_end[-1] - 1e-9)
    k = np.searchsorted(s_end, s, side="right") - 1
    v = dirs[np.clip(k, 0, len(dirs) - 1)] * speed

    W = max(int(smooth_time / h), 1)  # moving-average smoothing of the velocity profile
    v = np.vstack([v, np.zeros((2 * W, 2))])
    kern = np.ones(W) / W
    v = np.stack([np.convolve(v[:, i], kern, mode="full") for i in range(2)], 1)
    v = np.vstack([np.zeros((1, 2)), v])  # start from rest
    v = v[: len(v) - W + 1]
    u = np.diff(v, axis=0) / h

    x = np.zeros((len(u) + 1, 4))
    x[0] = [*path[0], *v[0]]
    for i in range(len(u)):
        x[i + 1] = A @ x[i] + B @ u[i]
    return x, u
