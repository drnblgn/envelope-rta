from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation


# Rectangle vehicle (same as animate_step1)
def vehicle_rect(x, y, yaw, length=4.5, width=2.0):
    pts = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
        [ length/2,  width/2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts_w = pts @ R.T
    pts_w[:, 0] += x
    pts_w[:, 1] += y
    return pts_w


def main():
    log_path = Path(__file__).resolve().parent.parent / "sim_log_step3_rta.npz"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    data = np.load(log_path)

    dt = float(data["dt"])
    xs = data["xs"]
    ys = data["ys"]
    yaws = data["yaws"]
    vs = data["vs"]
    steers = data["steers"]
    alats = data["alats"]
    rta_active = data["rta_active"].astype(bool)

    curvature = float(data["curvature"])
    a_lat_max = float(data["a_lat_max"])

    # Normalize yaws
    yaws = (yaws + np.pi) % (2 * np.pi) - np.pi

    # ---- Road (match Step 1 style) ----
    lane_half_width = 2.0
    Nroad = 800

    if abs(curvature) < 1e-9:
        road_x = np.linspace(xs.min() - 30, xs.max() + 30, Nroad)
        road_y = np.zeros_like(road_x)

        dxr = np.gradient(road_x)
        dyr = np.gradient(road_y)
    else:
        R = 1.0 / curvature
        cx, cy = 0.0, R
        theta = np.linspace(-np.pi, np.pi, Nroad)  # full circle
        road_x = cx + R * np.cos(theta)
        road_y = cy + R * np.sin(theta)
        dxr = np.gradient(road_x)
        dyr = np.gradient(road_y)

    nrm = np.sqrt(dxr * dxr + dyr * dyr) + 1e-9
    nx = -dyr / nrm
    ny = dxr / nrm

    left_x = road_x + lane_half_width * nx
    left_y = road_y + lane_half_width * ny
    right_x = road_x - lane_half_width * nx
    right_y = road_y - lane_half_width * ny

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Envelope – Deterministic RTA (Safety Filter)")

    # Draw road like Step 1 (same alpha/line widths)
    ax.plot(road_x, road_y, "--", linewidth=2, alpha=0.7)
    ax.plot(left_x, left_y, "-", linewidth=4, alpha=0.35)
    ax.plot(right_x, right_y, "-", linewidth=4, alpha=0.35)

    # Camera limits: include road + trajectory
    pad = 8.0
    all_x = np.concatenate([xs, road_x, left_x, right_x])
    all_y = np.concatenate([ys, road_y, left_y, right_y])
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    # Trajectory + vehicle
    traj_line, = ax.plot([], [], linewidth=2)

    veh_patch = plt.Polygon(
        vehicle_rect(xs[0], ys[0], yaws[0]),
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.add_patch(veh_patch)

    # HUD box (match Step 1)
    hud = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round", alpha=0.25)
    )

    # Bottom warning/status text (match Step 1 placement)
    warn = ax.text(
        0.5, 0.08, "",
        transform=ax.transAxes,
        va="bottom", ha="center",
        fontsize=16,
        weight="bold"
    )

    def init():
        traj_line.set_data([], [])
        veh_patch.set_xy(vehicle_rect(xs[0], ys[0], yaws[0]))
        veh_patch.set_facecolor("tab:blue")
        hud.set_text("")
        warn.set_text("")
        return traj_line, veh_patch, hud, warn

    def update(i):
        traj_line.set_data(xs[:i+1], ys[:i+1])
        veh_patch.set_xy(vehicle_rect(xs[i], ys[i], yaws[i]))

        t = i * dt

        # HUD (same formatting vibe as Step 1)
        hud.set_text(
            f"t = {t:4.1f}s\n"
            f"v = {vs[i]:4.1f} m/s\n"
            f"steer = {steers[i]: .3f} rad\n"
            f"a_lat = {alats[i]:5.1f} m/s²\n"
            f"limit = {a_lat_max:.1f}"
        )

        # Match Step 1 visual language:
        # - blue when normal
        # - red when actively overriding (big, obvious)
        if rta_active[i]:
            veh_patch.set_facecolor("tab:red")
            warn.set_text("RTA ACTIVE (override engaged)")
            traj_line.set_linewidth(3)
        else:
            veh_patch.set_facecolor("tab:blue")
            traj_line.set_linewidth(2)
            # If near limit, show a subtle safety message (optional but nice)
            if abs(alats[i]) > a_lat_max:
                warn.set_text("SAFETY VIOLATION (caught by RTA)")
            else:
                warn.set_text("")

        return traj_line, veh_patch, hud, warn

    ani = FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=int(dt * 1000),
        blit=True
    )

    fps = int(round(1.0 / dt))
    output_dir = log_path.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    if animation.writers.is_available("ffmpeg"):
        out = output_dir / "step3_rta.mp4"
        ani.save(out, fps=fps, dpi=140, writer="ffmpeg")
        saved.append(str(out))
    if animation.writers.is_available("pillow"):
        out = output_dir / "step3_rta.gif"
        ani.save(out, fps=fps, dpi=140, writer="pillow")
        saved.append(str(out))
    if not saved:
        raise RuntimeError("No animation writer available. Install ffmpeg or pillow.")

    print(f"Saved: {', '.join(saved)}")

    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
