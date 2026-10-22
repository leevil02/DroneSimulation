"""Double-integrator dynamics, LQR feedback, and closed-loop simulation."""
from __future__ import annotations

import numpy as np


def double_integrator(h):
    """Exact zero-order-hold discretisation of a 2D double integrator, state [x, y, vx, vy]."""
    I, O = np.eye(2), np.zeros((2, 2))
    A = np.block([[I, h * I], [O, I]]) #system matrix
    B = np.vstack([h ** 2 / 2 * I, h * I]) #input matrix
    return A, B


def lqr_gain(A, B, Q, R, iters=5000):
    """Discrete LQR via Riccati iteration; returns K with u = K e."""
    P = Q.copy()
    for _ in range(iters):
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        done = np.max(np.abs(Pn - P)) < 1e-12
        P = Pn
        if done:
            break
    return -np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def simulate(A, B, K, x_ref, u_ref, E0, dmax, rng, mode):
    """One closed-loop run. mode: 'bangbang' | 'constant' | 'uniform' disturbance."""
    e = E0.c + E0.G @ rng.choice([-1, 1], E0.n_gen)  # vertex-like initial error
    x = x_ref[0] + e
    xs = [x.copy()]
    sign = rng.choice([-1, 1], 2)
    for k in range(len(u_ref)):
        if mode == "bangbang":
            d = rng.choice([-1, 1], 2) * dmax
        elif mode == "constant":
            d = sign * dmax
        else:
            d = rng.uniform(-dmax, dmax)
        x = A @ x + B @ (u_ref[k] + K @ (x - x_ref[k]) + d)
        xs.append(x.copy())
    return np.array(xs)
