"""Comparative studies of set representations."""
from __future__ import annotations

import time

import numpy as np

from .reachability import PROJ, error_tube

CONFIGS = [("zonotope (exact)", "zonotope", None), ("reduced, order 4", "reduced", 4),
           ("reduced, order 2", "reduced", 2), ("interval box", "box", None)]


def rotational_variant(Acl, theta=0.15, damping=0.98):
    """Closed loop with rotation coupled into the velocity block, to expose the wrapping effect."""
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = Acl.copy()
    A[2:, 2:] = damping * R @ Acl[2:, 2:]
    return A


def compare_representations(systems, B, E0, dmax, N, h, configs=CONFIGS):
    """systems: list of (name, A_cl). Returns {system: {config: dict(t, area, time)}} and prints a table."""
    results = {}
    print("representation study (position-projection area at final step, runtime):")
    for sysname, M in systems:
        results[sysname] = {}
        exact = None
        for name, method, order in configs:
            t0 = time.perf_counter()
            tube = error_tube(M, B, E0, dmax, N, method, order)
            dt = time.perf_counter() - t0
            areas = np.array([Z.linear_map(PROJ).area() for Z in tube])
            exact = areas if method == "zonotope" else exact
            results[sysname][name] = dict(t=np.arange(N + 1) * h, area=areas, time=dt)
            print(f"  [{sysname:24s}] {name:18s} area={areas[-1]:.4f}  "
                  f"ratio to exact={areas[-1] / exact[-1]:.2f}  time={dt * 1e3:.0f} ms")
    return results
