"""reach2d: set-based reachability for 2D drone path planning."""
from .dynamics import double_integrator, lqr_gain, simulate
from .geometry import ConvexPolygon, zono_intersects_polygon, zono_intersects_zono
from .planner import RRTPlanner
from .reachability import (PROJ, error_tube, position_tube, static_error_bound,
                           tube_collision_free)
from .trajectory import make_reference
from .zonotope import Zonotope, box_zonotope, hull_overapprox
