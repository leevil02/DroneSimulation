"""Zonotope set representation: Z = { c + G a : a in [-1, 1]^m }."""
from __future__ import annotations

import numpy as np


class Zonotope:
    def __init__(self, c, G):
        self.c = np.asarray(c, float).reshape(-1)
        G = np.asarray(G, float).reshape(self.c.size, -1)
        self.G = G[:, np.linalg.norm(G, axis=0) > 1e-12]  # drop zero generators

    # -- properties
    @property
    def dim(self):
        return self.c.size

    @property
    def n_gen(self):
        return self.G.shape[1]

    @property
    def order(self):
        return self.n_gen / self.dim

    # -- set operations (any dimension)
    def linear_map(self, M):
        return Zonotope(M @ self.c, M @ self.G)

    def translate(self, v):
        return Zonotope(self.c + np.asarray(v, float), self.G)

    def minkowski_sum(self, other):
        return Zonotope(self.c + other.c, np.hstack([self.G, other.G]))

    def support(self, d):
        d = np.asarray(d, float)
        return float(d @ self.c + np.abs(d @ self.G).sum())

    def interval_hull(self):
        """Tightest axis-aligned box (returned as a zonotope)."""
        return Zonotope(self.c, np.diag(np.abs(self.G).sum(1)))

    def reduce(self, order):
        """Girard order reduction: keep the largest generators, box the rest."""
        n, m = self.dim, self.n_gen
        order = max(order, 1.0)
        if m <= order * n:
            return self
        n_keep = int(order * n) - n
        score = np.abs(self.G).sum(0) - np.abs(self.G).max(0)  # small = box-like => cheap to merge
        idx = np.argsort(score)
        n_red = m - n_keep
        small, large = self.G[:, idx[:n_red]], self.G[:, idx[n_red:]]
        return Zonotope(self.c, np.hstack([large, np.diag(np.abs(small).sum(1))]))

    # -- 2D helpers
    def area(self):
        assert self.dim == 2
        g = self.G
        D = np.abs(g[0][:, None] * g[1][None, :] - g[1][:, None] * g[0][None, :])
        return 2.0 * D.sum()

    def vertices(self):
        assert self.dim == 2
        G = self.G.copy()
        if G.shape[1] == 0:
            return self.c[None, :]
        flip = (G[1] < 0) | ((G[1] == 0) & (G[0] < 0))
        G[:, flip] *= -1
        G = G[:, np.argsort(np.arctan2(G[1], G[0]))]
        p = self.c - G.sum(1)
        pts = [p.copy()]
        for i in range(G.shape[1]):
            p = p + 2 * G[:, i]
            pts.append(p.copy())
        for i in range(G.shape[1]):
            p = p - 2 * G[:, i]
            pts.append(p.copy())
        return np.array(pts[:-1])

    def contains_point_2d(self, p, tol=1e-9):
        """Exact for 2D zonotopes (facet normals are perpendicular to the generators)."""
        N = np.stack([-self.G[1], self.G[0]])
        N = N / np.linalg.norm(N, axis=0)
        r = np.abs(N.T @ (np.asarray(p) - self.c))
        return bool(np.all(r <= np.abs(N.T @ self.G).sum(1) + tol))


def box_zonotope(center, half):
    """Axis-aligned box with the given center and half-widths."""
    return Zonotope(center, np.diag(np.asarray(half, float)))


def hull_overapprox(Z1: Zonotope, Z2: Zonotope) -> Zonotope:
    """Zonotope over-approximating conv(Z1 U Z2) (Girard-style, valid for independent parameters)."""
    m = max(Z1.n_gen, Z2.n_gen)

    def pad(G):
        return np.hstack([G, np.zeros((G.shape[0], m - G.shape[1]))])

    G1, G2 = pad(Z1.G), pad(Z2.G)
    return Zonotope((Z1.c + Z2.c) / 2,
                    np.hstack([(G1 + G2) / 2, (G1 - G2) / 2, ((Z1.c - Z2.c) / 2)[:, None]]))
