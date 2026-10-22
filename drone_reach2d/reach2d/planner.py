"""RRT planner whose edge checks use exact swept zonotopes (FaSTrack-style error bound)."""
from __future__ import annotations

import numpy as np

from .geometry import zono_intersects_polygon
from .zonotope import Zonotope


class RRTPlanner:
    def __init__(self, obstacles, bounds, err_half, margin=0.0, step=1.5, rng=None):
        self.obs, self.bounds = obstacles, bounds  # bounds = (xmin, ymin, xmax, ymax)
        self.err = np.asarray(err_half, float) + margin  # static tracking-error bound (+ margin)
        self.step = step
        self.rng = rng or np.random.default_rng(0)
        self.tree = None

    def _sweep(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        return Zonotope((a + b) / 2, np.hstack([((b - a) / 2)[:, None], np.diag(self.err)]))

    def edge_free(self, a, b):
        Z = self._sweep(a, b)
        x0, y0, x1, y1 = self.bounds
        if (Z.support([1, 0]) > x1 or Z.support([-1, 0]) > -x0 or
                Z.support([0, 1]) > y1 or Z.support([0, -1]) > -y0):
            return False
        return not any(zono_intersects_polygon(Z, P) for P in self.obs)

    def plan(self, start, goal, max_iter=4000, goal_bias=0.1):
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        nodes, parent = [start], [-1]
        x0, y0, x1, y1 = self.bounds
        for _ in range(max_iter):
            s = goal if self.rng.random() < goal_bias else self.rng.uniform([x0, y0], [x1, y1])
            i = int(np.argmin(np.linalg.norm(np.array(nodes) - s, axis=1)))
            d = s - nodes[i]
            new = nodes[i] + d / max(np.linalg.norm(d), 1e-9) * min(self.step, np.linalg.norm(d))
            if not self.edge_free(nodes[i], new):
                continue
            nodes.append(new)
            parent.append(i)
            if np.linalg.norm(new - goal) < self.step and self.edge_free(new, goal):
                nodes.append(goal)
                parent.append(len(nodes) - 2)
                self.tree = (np.array(nodes), parent)
                return self._shortcut(self._extract(nodes, parent))
        self.tree = (np.array(nodes), parent)
        return None

    @staticmethod
    def _extract(nodes, parent):
        path, j = [], len(nodes) - 1
        while j >= 0:
            path.append(nodes[j])
            j = parent[j]
        return path[::-1]

    def _shortcut(self, path, iters=300):
        path = list(path)
        for _ in range(iters):
            if len(path) < 3:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i > 1 and self.edge_free(path[i], path[j]):
                path = path[:i + 1] + path[j:]
        return np.array(path)
