"""Matplotlib figures (Agg backend; files are saved, not shown)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def draw_scene(ax, obstacles, bounds):
    for P in obstacles:
        ax.fill(*P.V.T, color="0.35", alpha=0.9, zorder=3)
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)


def plot_scene(path_out, obstacles, bounds, planner, path, x_ref, Omega, sims, start, goal, margin):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_scene(ax, obstacles, bounds)
    nodes, parent = planner.tree
    for j, p in enumerate(parent):
        if p >= 0:
            ax.plot(*nodes[[p, j]].T, color="0.8", lw=0.5, zorder=1)
    for k in range(0, len(Omega), 3):
        ax.fill(*Omega[k].vertices().T, color="tab:blue", alpha=0.10, lw=0, zorder=2)
    for xs in sims:
        ax.plot(xs[:, 0], xs[:, 1], color="tab:green", lw=0.5, alpha=0.5, zorder=4)
    ax.plot(path[:, 0], path[:, 1], "k--", lw=1, label="RRT path (shortcut)", zorder=5)
    ax.plot(x_ref[:, 0], x_ref[:, 1], color="tab:red", lw=1.5, label="smoothed reference", zorder=5)
    ax.plot(*start, "go", ms=9, zorder=6)
    ax.plot(*goal, "r*", ms=14, zorder=6)
    ax.plot([], [], color="tab:green", label="simulated runs (adversarial disturbance)")
    ax.fill([], [], color="tab:blue", alpha=0.3, label="reach tube Omega_k")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"2D drone: planning with tracking-error reach tube (margin {margin} m)")
    fig.savefig(path_out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_representations(path_out, results):
    fig, axs = plt.subplots(1, len(results), figsize=(6 * len(results), 4.5))
    for ax, (sysname, cfgs) in zip(axs, results.items()):
        for name, r in cfgs.items():
            ax.semilogy(r["t"], r["area"], label=f"{name} ({r['time'] * 1e3:.0f} ms)")
        ax.set(title=sysname, xlabel="time [s]", ylabel="area of position projection [m^2]")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.savefig(path_out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_animation(path_out, obstacles, bounds, x_ref, Omega, xs, step=2, fps=20):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_scene(ax, obstacles, bounds)
    ax.plot(x_ref[:, 0], x_ref[:, 1], color="tab:red", lw=1)
    poly, = ax.fill([], [], color="tab:blue", alpha=0.5)
    dot, = ax.plot([], [], "ko", ms=5, zorder=6)
    trail, = ax.plot([], [], "g-", lw=1, zorder=5)

    def update(k):
        poly.set_xy(Omega[k].vertices())
        dot.set_data([xs[k, 0]], [xs[k, 1]])
        trail.set_data(xs[:k + 1, 0], xs[:k + 1, 1])
        return poly, dot, trail

    FuncAnimation(fig, update, frames=range(0, len(Omega), step)).save(path_out, writer=PillowWriter(fps=fps))
    plt.close(fig)
