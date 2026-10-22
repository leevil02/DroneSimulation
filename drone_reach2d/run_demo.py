#!/usr/bin/env python3
"""End-to-end demo: plan with a static error bound, verify with the time-varying reach tube.

    python run_demo.py [--seed 3] [--h 0.1] [--speed 1.5] [--dmax 0.4] [--order 6] [--animate] [--outdir out]
"""
import argparse
import os
import time

import numpy as np

from reach2d import (PROJ, RRTPlanner, Zonotope, box_zonotope, double_integrator, error_tube,
                     lqr_gain, make_reference, position_tube, simulate, static_error_bound,
                     tube_collision_free)
from reach2d import scenario, studies
from reach2d.plotting import plot_representations, plot_scene, save_animation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--h", type=float, default=0.1, help="time step [s]")
    ap.add_argument("--speed", type=float, default=1.5, help="reference cruise speed [m/s]")
    ap.add_argument("--dmax", type=float, default=0.4, help="acceleration disturbance bound [m/s^2]")
    ap.add_argument("--order", type=float, default=6, help="zonotope order for the reduced tube")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    h = args.h
    A, B = double_integrator(h)
    K = lqr_gain(A, B, np.diag([10, 10, 2, 2.0]), np.eye(2))
    Acl = A + B @ K
    dmax = np.full(2, args.dmax)
    E0 = box_zonotope(np.zeros(4), [0.10, 0.10, 0.05, 0.05])  # initial tracking-error set
    print(f"closed-loop spectral radius: {max(abs(np.linalg.eigvals(Acl))):.3f}")

    obstacles, bounds = scenario.make_obstacles(), scenario.BOUNDS
    start, goal = scenario.START, scenario.GOAL

    # static tracking-error bound for the planner (FaSTrack-style)
    tube_inf = error_tube(Acl, B, E0, dmax, int(15 / h), "reduced", args.order)
    err_half = static_error_bound(tube_inf)
    print(f"static tracking-error bound (position half-widths): {err_half.round(3)} m")

    # plan -> smooth -> verify with the time-varying tube; inflate the margin until verified
    for margin in [0.0, 0.1, 0.2, 0.35, 0.5, 0.75]:
        planner = RRTPlanner(obstacles, bounds, err_half, margin, rng=np.random.default_rng(args.seed))
        t0 = time.perf_counter()
        path = planner.plan(start, goal)
        t_plan = time.perf_counter() - t0
        if path is None:
            print(f"margin {margin}: planner failed")
            continue
        x_ref, u_ref = make_reference(path, args.speed, h, A, B)
        N = len(u_ref)
        E_tube = error_tube(Acl, B, E0, dmax, N, "reduced", args.order)
        t0 = time.perf_counter()
        Omega, Seg = position_tube(x_ref, u_ref, E_tube, K, h, dmax)
        ok, k_bad = tube_collision_free(Seg, obstacles)
        t_ver = time.perf_counter() - t0
        print(f"margin {margin:4.2f}: {len(path)} waypoints, {N * h:5.1f}s trajectory, plan {t_plan:.2f}s, "
              f"verify {t_ver:.2f}s -> "
              f"{'VERIFIED SAFE' if ok else f'tube hits obstacle at t={k_bad * h:.1f}s'}")
        if ok:
            break
    else:
        raise SystemExit("could not find a verified path")

    # Monte-Carlo containment check
    n_runs, inside, sims = 100, 0, []
    for i in range(n_runs):
        xs = simulate(A, B, K, x_ref, u_ref, E0, dmax, rng, ["bangbang", "constant", "uniform"][i % 3])
        sims.append(xs)
        inside += all(Omega[k].contains_point_2d(xs[k, :2]) for k in range(len(xs)))
    print(f"simulated runs fully inside the reach tube: {inside}/{n_runs}")

    # representation study
    print()
    results = studies.compare_representations(
        [("closed-loop LQR", Acl), ("with rotational coupling", studies.rotational_variant(Acl))],
        B, E0, dmax, 150, h)

    static_area = 4 * err_half[0] * err_half[1]
    tv_area = np.mean([Z.linear_map(PROJ).area() for Z in E_tube])
    print(f"\nconservatism: static tracking bound area {static_area:.3f} m^2 "
          f"vs mean time-varying tube area {tv_area:.3f} m^2")

    outs = [os.path.join(args.outdir, "reach2d_scene.png"), os.path.join(args.outdir, "reach2d_representations.png")]
    plot_scene(outs[0], obstacles, bounds, planner, path, x_ref, Omega, sims, start, goal, margin)
    plot_representations(outs[1], results)
    if args.animate:
        outs.append(os.path.join(args.outdir, "reach2d.gif"))
        save_animation(outs[2], obstacles, bounds, x_ref, Omega, sims[0])
    print("\nwrote:", ", ".join(outs))


if __name__ == "__main__":
    main()
