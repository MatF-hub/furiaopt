"""Visualizer for examples/sqp_delta_robot.cpp. Parses the solver's spdlog log file (no CSV):
one shared GEOMETRY line (robot constants), then per problem an "=== METHOD ... ===" marker
followed by a REFERENCE line (the tracked target sequence, home included at both ends), 0-2
OBSTACLE lines, a SOLVE line, and a RESULT line (the full decision vector).
"""
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

SEGMENT_LEN = 9  # 8 untracked interior points + 1 tracked endpoint per segment -- fixed convention


def parse_kv_line(line):
    kv = {}
    for key, val in re.findall(r"(\w+)=(-?[\w.+-]+)", line):
        try:
            kv[key] = float(val)
        except ValueError:
            kv[key] = val
    return kv


def parse_log(path):
    """Per-iteration SQP lines ("iter=...,cost=...,equality_constraint=...,inequality_constraint=
    ...,x=...") come from ConstrainedSolver's own outer-loop logging -- same line format the QP
    example's IPM loop already uses, so it's parsed the same way here."""
    geometry = None
    solves = []
    pending_label = None
    pending_obstacles = []
    pending_iters = []
    pending_reference = None
    with open(path) as f:
        for line in f:
            if "=== METHOD" in line:
                pending_label = line.split("=== METHOD", 1)[1].split("===")[0].strip()
                pending_obstacles = []
                pending_iters = []
                pending_reference = None
            elif "GEOMETRY" in line:
                geometry = parse_kv_line(line)
            elif "REFERENCE" in line:
                points_str = line.split("points=", 1)[1].strip()
                pending_reference = np.array([[float(v) for v in triple.split(",")]
                                               for triple in points_str.split(";")])
            elif "OBSTACLE" in line:
                pending_obstacles.append(parse_kv_line(line))
            elif "equality_constraint=" in line:
                pending_iters.append(parse_kv_line(line.split(",x=")[0]))
            elif "SOLVE" in line:
                if pending_reference is None:
                    raise ValueError(f"SOLVE line for {pending_label!r} with no preceding REFERENCE line in {path}")
                solve = parse_kv_line(line)
                solve["label"] = pending_label or f"Problem {len(solves) + 1}"
                solve["obstacles"] = pending_obstacles
                solve["reference"] = pending_reference
                solve["iters"] = pending_iters
                solves.append(solve)
            elif "RESULT" in line:
                x_str = line.split("x=", 1)[1].strip()
                x = np.array([float(v) for v in x_str.split()])  # furiaopt::utils::vec_to_string is whitespace-separated
                solves[-1]["x"] = x
                solves[-1]["n"] = len(x) // 3
    if geometry is None or not solves:
        raise ValueError(f"could not find GEOMETRY/SOLVE/RESULT lines in {path}")
    return geometry, solves


def display_title(label):
    """Strip the leading "Problem N" numbering, keeping a single descriptive phrase."""
    m = re.match(r"Problem\s+\d+\s*\((.*)\)\s*$", label)
    return m.group(1) if m else label


class DeltaGeometry:
    """Forward kinematics + rendering, driven only by the GEOMETRY log fields (robot constants
    shared across all problems) -- self-contained, no dependency on the C++ source."""

    def __init__(self, geom):
        self.r_b, self.r_e, self.l1, self.l2 = geom["R_B"], geom["R_E"], geom["L1"], geom["L2"]
        self.a = self.r_b - self.r_e
        self.phi = [np.radians(90 + i * 120) for i in range(3)]
        self.theta_min, self.theta_max = np.radians(geom["theta_min_deg"]), np.radians(geom["theta_max_deg"])

    def elbow_offset_point(self, theta, phi):
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        return (self.a + self.l1 * np.cos(theta)) * u + np.array([0.0, 0.0, -self.l1 * np.sin(theta)])

    def forward_kinematics(self, thetas):
        c0, c1, c2 = (self.elbow_offset_point(t, phi) for t, phi in zip(thetas, self.phi))
        lhs = np.array([2 * (c0 - c1), 2 * (c0 - c2)])
        rhs = np.array([c0 @ c0 - c1 @ c1, c0 @ c0 - c2 @ c2])
        p0, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
        d = np.cross(c0 - c1, c0 - c2)
        d = d / np.linalg.norm(d)
        diff = p0 - c0
        b = 2 * diff @ d
        c = diff @ diff - self.l2**2
        disc = b**2 - 4 * c
        t1, t2 = (-b + np.sqrt(disc)) / 2, (-b - np.sqrt(disc)) / 2
        return min(p0 + t1 * d, p0 + t2 * d, key=lambda pp: pp[2])

    def path(self, x, n):
        theta = x.reshape(3, n)
        return np.array([self.forward_kinematics(theta[:, k]) for k in range(n)])

    def arm_points(self, theta_j, phi, p):
        u = np.array([np.cos(phi), np.sin(phi), 0.0])
        base_pivot = self.r_b * u
        elbow = base_pivot + self.l1 * np.cos(theta_j) * u + np.array([0, 0, -self.l1 * np.sin(theta_j)])
        platform_joint = np.asarray(p) + self.r_e * u
        return base_pivot, elbow, platform_joint


ARM_COLORS = ["#d62728", "#2ca02c", "#1f77b4"]  # red, green, blue -- one per arm, consistent throughout


def tracked_k(n):
    return list(range(0, n, SEGMENT_LEN))


def draw_robot(ax, geo, theta_k, p):
    """Semi-realistic render: colored bicep/forearm per arm, filled base + platform plates,
    motor pivot markers -- not just wireframe lines."""
    base_pts, elbows, joints = [], [], []
    for j, phi in enumerate(geo.phi):
        base_pivot, elbow, joint = geo.arm_points(theta_k[j], phi, p)
        base_pts.append(base_pivot)
        elbows.append(elbow)
        joints.append(joint)
        color = ARM_COLORS[j]
        ax.plot(*zip(base_pivot, elbow), color=color, linewidth=4, solid_capstyle="round")
        ax.plot(*zip(elbow, joint), color=color, linewidth=2, solid_capstyle="round")
        ax.scatter(*base_pivot, color=color, marker="o", s=35, edgecolor="k", linewidth=0.5, zorder=5)
        ax.scatter(*elbow, color=color, marker="o", s=15, zorder=5)

    base_pts = np.array(base_pts + [base_pts[0]])
    ax.add_collection3d(Poly3DCollection([base_pts], color="0.6", alpha=0.25))
    joints_arr = np.array(joints)
    ax.add_collection3d(Poly3DCollection([joints_arr], color="0.2", alpha=0.6))
    ax.scatter(*p, color="k", marker="x", s=30, zorder=6)


def draw_obstacle(ax, center, radius, z, n_hatch=6):
    """Red circle with red hatch lines through it -- a keep-away zone, drawn on the path plane."""
    t = np.linspace(0, 2 * np.pi, 48)
    circle = np.stack([center[0] + radius * np.cos(t), center[1] + radius * np.sin(t), np.full_like(t, z)], axis=1)
    ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color="red", linewidth=1.5)
    direction, perp = np.array([1.0, 1.0]) / np.sqrt(2), np.array([-1.0, 1.0]) / np.sqrt(2)
    for d in np.linspace(-radius * 0.85, radius * 0.85, n_hatch):
        half_chord = np.sqrt(max(radius**2 - d**2, 0.0))
        c = np.asarray(center) + perp * d
        p1, p2 = c - direction * half_chord, c + direction * half_chord
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [z, z], color="red", linewidth=0.8, alpha=0.8)


def obstacle_clearance(path, obstacles):
    clearances = []
    for obs in obstacles:
        d = np.linalg.norm(path[:, :2] - [obs["center_x"], obs["center_y"]], axis=1)
        clearances.append((obs["name"], d.min() - obs["radius"]))
    return clearances


def plot_diagnostics(geo, solve, out_path):
    x, n = solve["x"], solve["n"]
    theta = x.reshape(3, n)
    theta_deg = np.degrees(theta)
    path = geo.path(x, n)
    tracked = tracked_k(n)
    track_err = np.linalg.norm(path[tracked] - solve["reference"], axis=1)

    limit_margin = min((theta - geo.theta_min).min(), (geo.theta_max - theta).min())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.axis("off")
    lines = [
        f"{solve['label']}",
        "",
        f"converged        : {solve.get('converged')}",
        f"termination_reason: {solve.get('termination_reason')}",
        f"iterations       : {int(solve['iterations'])}",
        f"final_cost       : {solve['final_cost']:.4f}",
        f"elapsed_ms       : {solve['elapsed_ms']:.2f}",
        "",
        f"max tracking error : {track_err.max():.4f} mm",
        f"mean tracking error: {track_err.mean():.4f} mm",
        f"joint-limit margin : {np.degrees(limit_margin):.2f} deg "
        f"({'OK' if limit_margin >= 0 else 'VIOLATED'})",
    ]
    for name, clearance in obstacle_clearance(path, solve["obstacles"]):
        lines.append(f"obstacle {name} clearance: {clearance:.3f} mm ({'OK' if clearance >= 0 else 'VIOLATED'})")
    ax.text(0, 1, "\n".join(lines), family="monospace", fontsize=10, va="top", transform=ax.transAxes)

    ax = axes[1]
    for j in range(3):
        ax.plot(theta_deg[j], color=ARM_COLORS[j], label=f"theta_{j}")
    ax.axhline(np.degrees(geo.theta_min), color="k", linestyle="--", linewidth=1)
    ax.axhline(np.degrees(geo.theta_max), color="k", linestyle="--", linewidth=1, label="limits")
    for k in tracked:
        ax.axvline(k, color="gray", linewidth=0.4)
    ax.set_xlabel("fine-grid index k")
    ax.set_ylabel("joint angle (deg)")
    ax.legend(fontsize=8)
    ax.set_title("joint trajectories")

    fig.suptitle(f"{display_title(solve['label'])} - diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(solve, out_path):
    """Genuine outer-SQP convergence: cost and constraint violation per iteration, from
    ConstrainedSolver's own per-iteration log line (see parse_log)."""
    iters = solve["iters"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    if iters:
        it = [d["iter"] for d in iters]
        cost = [d["cost"] for d in iters]
        eq_violation = [max(d["equality_constraint"], 1e-16) for d in iters]
        ineq_violation = [max(d["inequality_constraint"], 1e-16) for d in iters]

        axes[0].plot(it, cost, "o-", color="tab:blue")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("SQP iteration")
        axes[0].set_ylabel("cost")
        axes[0].set_title("cost convergence")

        axes[1].plot(it, eq_violation, "o-", label="equality violation")
        axes[1].plot(it, ineq_violation, "s-", label="inequality violation")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("SQP iteration")
        axes[1].set_ylabel("constraint violation")
        axes[1].legend(fontsize=8)
        axes[1].set_title("constraint violation")
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "no per-iteration data", ha="center", va="center", transform=ax.transAxes)

    ls_label = f"LS: {int(solve['iterations'])} iter - {solve['elapsed_ms']:.0f} ms"
    fig.suptitle(f"{display_title(solve['label'])} - {ls_label}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def set_equal_aspect(ax, points):
    points = np.array(points)
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    half_range = (points.max(axis=0) - points.min(axis=0)).max() / 2 * 1.15
    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)
    ax.set_box_aspect((1, 1, 1))


def animate(geo, solve, out_path, fps=15):
    x, n = solve["x"], solve["n"]
    theta = x.reshape(3, n)
    path = geo.path(x, n)
    reference = solve["reference"]

    base_circle = np.array([(geo.r_b * np.cos(t), geo.r_b * np.sin(t), 0.0)
                             for t in np.linspace(0, 2 * np.pi, 48)])
    all_points = np.vstack([path, base_circle])

    legend_handles = [Line2D([0], [0], color="k", linestyle=":", label="reference trajectory")]
    if solve["obstacles"]:
        legend_handles.append(Line2D([0], [0], color="red", label="obstacle"))

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection="3d")

    def update(k):
        ax.cla()
        ax.plot(base_circle[:, 0], base_circle[:, 1], base_circle[:, 2], "k--", linewidth=1, alpha=0.5)
        ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], "k:", linewidth=1, alpha=0.7)
        for obs in solve["obstacles"]:
            draw_obstacle(ax, (obs["center_x"], obs["center_y"]), obs["radius"], reference[:, 2].min())
        segs = np.stack([path[:-1], path[1:]], axis=1)[:k]
        if len(segs):
            progress = np.linspace(0, 1, len(path) - 1)[:k]
            lc = Line3DCollection(segs, colors=plt.cm.viridis(progress), linewidths=2)
            ax.add_collection3d(lc)
        draw_robot(ax, geo, theta[:, k], path[k])
        set_equal_aspect(ax, all_points)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
        ax.set_title(f"{display_title(solve['label'])} - k={k}/{n - 1}")
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=False)
        ax.view_init(elev=22, azim=-60)

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000 / fps)
    anim.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


def plot_topdown(geo, solve, out_path):
    """Flattened (x,y) overview -- the clearest way to see the obstacle detours on a dense
    self-intersecting shape like the star, where the 3D animation angle gets cluttered."""
    x, n = solve["x"], solve["n"]
    path = geo.path(x, n)
    reference = solve["reference"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot(reference[:, 0], reference[:, 1], "k:", linewidth=1, label="reference trajectory")
    segs = np.stack([path[:-1, :2], path[1:, :2]], axis=1)
    progress = np.linspace(0, 1, len(path) - 1)
    ax.add_collection(plt.matplotlib.collections.LineCollection(segs, colors=plt.cm.viridis(progress), linewidths=2))
    for obs in solve["obstacles"]:
        t = np.linspace(0, 2 * np.pi, 64)
        cx, cy, r = obs["center_x"], obs["center_y"], obs["radius"]
        ax.fill(cx + r * np.cos(t), cy + r * np.sin(t), color="red", alpha=0.25)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), color="red", linewidth=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"{display_title(solve['label'])} - top-down path")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", default=None, help="output directory (default: alongside the log file)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    geometry, solves = parse_log(args.log)
    geo = DeltaGeometry(geometry)
    out_dir = args.out or os.path.dirname(os.path.abspath(args.log))
    os.makedirs(out_dir, exist_ok=True)

    for i, solve in enumerate(solves):
        slug = re.sub(r"[^a-z0-9]+", "_", solve["label"].lower()).strip("_")
        diag_path = os.path.join(out_dir, f"sqp_delta_robot_{slug}_diagnostics.png")
        conv_path = os.path.join(out_dir, f"sqp_delta_robot_{slug}_convergence.png")
        topdown_path = os.path.join(out_dir, f"sqp_delta_robot_{slug}_topdown.png")
        gif_path = os.path.join(out_dir, f"sqp_delta_robot_{slug}_trajectory.gif")
        plot_diagnostics(geo, solve, diag_path)
        plot_convergence(solve, conv_path)
        plot_topdown(geo, solve, topdown_path)
        animate(geo, solve, gif_path)
        print(f"[{i}] {solve['label']}: wrote {diag_path}, {conv_path}, {topdown_path} and {gif_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
