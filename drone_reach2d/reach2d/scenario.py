"""World definition and default experiment parameters."""
from __future__ import annotations

import numpy as np

from .geometry import ConvexPolygon

BOUNDS = (0, 0, 20, 20)
START = np.array([1.5, 1.5])
GOAL = np.array([7.0, 13.0])


def make_obstacles():
    return [
        ConvexPolygon.circle((6, 6), 2.0),
        ConvexPolygon.circle((10, 13), 2.5),
        ConvexPolygon.rect(12, 3, 14.5, 9),
        ConvexPolygon.rect(2, 11, 7, 12.5),
        ConvexPolygon([[15, 14], [18.5, 15], [17, 18.5]]),
        ConvexPolygon.circle((4, 17), 1.5),
    ]
