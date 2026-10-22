"""Reach-tube computation: tracking-error tubes and position-space tubes with collision checks."""
from __future__ import annotations

import numpy as np

from .geometry import zono_intersects_polygon
from .zonotope import Zonotope, box_zonotope, hull_overapprox

PROJ = np.hstack([np.eye(2), np.zeros((2, 2))])  # state -> position


def error_tube(Acl, B, E0, dmax, N, method="zonotope", order=None):
    """Reach tube of e+ = Acl e + B d, d in box(dmax). Returns N+1 Zonotopes.

    method: 'zonotope' (exact, generators grow) | 'reduced' (Girard, needs `order`)
            | 'box' (interval hull every step -> wrapping effect).
    This is the swap point for other propagators (support functions, ellipsoids, ...).
    """
    W = Zonotope(np.zeros(Acl.shape[0]), B @ np.diag(dmax))
    E = E0.interval_hull() if method == "box" else E0
    out = [E]
    for _ in range(N):
        E = E.linear_map(Acl).minkowski_sum(W)
        if method == "box":
            E = E.interval_hull()
        elif method == "reduced":
            E = E.reduce(order)
        out.append(E)
    return out


def static_error_bound(tube):
    """Box bound (half-widths) on the position error over the whole tube -- for the planner."""
    boxes = [Z.linear_map(PROJ).interval_hull() for Z in tube]
    return np.array([max(b.G[i, i] for b in boxes) for i in range(2)])


def position_tube(x_ref, u_ref, E_sets, K, h, dmax):
    """Position-space tube.

    Returns (Omega, Seg): Omega[k] = P x_ref_k (+) P E_k at sampling instants, and
    Seg[k] covering ALL continuous-time positions in [t_k, t_{k+1}]:
        hull(Omega_k, Omega_{k+1}) (+) box(h^2/8 * a_max),
    with a_max bounding |acceleration| per axis (u_ref + K e + d): a C^1 trajectory with
    |p''| <= a deviates from its chord by at most a h^2 / 8.
    """
    N = len(E_sets) - 1
    Omega = [Zonotope(PROJ @ x_ref[k], np.zeros((2, 0))).minkowski_sum(E_sets[k].linear_map(PROJ))
             for k in range(N + 1)]
    Seg = []
    for k in range(N):
        a_max = np.abs(u_ref[k]) + np.abs(K @ E_sets[k].G).sum(1) + dmax
        Seg.append(hull_overapprox(Omega[k], Omega[k + 1])
                   .minkowski_sum(box_zonotope(np.zeros(2), a_max * h ** 2 / 8)))
    return Omega, Seg


def tube_collision_free(Seg, obstacles):
    """Returns (ok, first_bad_step)."""
    for k, Z in enumerate(Seg):
        for P in obstacles:
            if zono_intersects_polygon(Z, P):
                return False, k
    return True, None
