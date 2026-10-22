"""Obstacles and exact 2D intersection tests (separating axis theorem)."""
from __future__ import annotations

import numpy as np

from .zonotope import Zonotope


class ConvexPolygon:
    def __init__(self, vertices):
        V = np.asarray(vertices, float)
        c = V.mean(0)
        order = np.argsort(np.arctan2(V[:, 1] - c[1], V[:, 0] - c[0]))
        self.V = V[order]  # counter-clockwise
        E = np.roll(self.V, -1, axis=0) - self.V
        n = np.stack([E[:, 1], -E[:, 0]], axis=1)
        self.normals = n / np.linalg.norm(n, axis=1, keepdims=True)

    @staticmethod
    def circle(center, radius, n=16):
        """Circumscribed regular polygon (outer approximation => conservative)."""
        r = radius / np.cos(np.pi / n)
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return ConvexPolygon(np.asarray(center) + r * np.stack([np.cos(a), np.sin(a)], 1))

    @staticmethod
    def rect(x0, y0, x1, y1):
        return ConvexPolygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def zono_intersects_polygon(Z: Zonotope, P: ConvexPolygon) -> bool:
    """Exact test (2D): SAT over polygon edge normals and zonotope edge normals."""
    Gn = np.stack([-Z.G[1], Z.G[0]])
    Gn = (Gn / np.linalg.norm(Gn, axis=0)).T if Z.n_gen else np.zeros((0, 2))
    axes = np.vstack([P.normals, Gn])
    zc = axes @ Z.c
    zr = np.abs(axes @ Z.G).sum(1)
    proj = P.V @ axes.T
    separated = (zc - zr > proj.max(0)) | (zc + zr < proj.min(0))
    return not bool(separated.any())


def zono_intersects_zono(Z1: Zonotope, Z2: Zonotope) -> bool:
    """Z1 and Z2 intersect  <=>  0 in Z1 (+) (-Z2).  Exact in 2D."""
    return Z1.minkowski_sum(Zonotope(-Z2.c, Z2.G)).contains_point_2d(np.zeros(2))
