from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

from envelope.control.aggressive import AggressiveController
from envelope.sim.bicycle import (
    VehicleParams,
    VehicleState,
    lateral_acceleration,
    step,
)
from envelope.sim.world import CurvedRoad


# Rectangle vehicle (matches Step 3 style)
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


def generate_log(log_path: Path) -> None:
    from envelope.rta.failure import summarize_failure

    dt = 0.05
    T = 12.0
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=0.06)
    ctrl = AggressiveController(target_speed=22.0)

    state = VehicleState(0.0, 0.0, 0.0, 5.0)

    xs, ys, yaws, vs, steers, alats = [], [], [], [], [], []

    for _ in range(steps):
        u = ctrl.act(state, params, road.curvature)
        state = step(state, u, dt, params)

        xs.append(state.x)
        ys.append(state.y)
        yaws.append(state.yaw)
        vs.append(state.v)
        steers.append(u.steer)
        alats.append(lateral_acceleration(state.v, u.steer, params))

    xs = np.array(xs)
    ys = np.array(ys)
    yaws = np.array(yaws)
    vs = np.array(vs)
    steers = np.array(steers)
    alats = np.array(alats)

    a_lat_max = 5.0

    violation_mask = np.abs(alats) > a_lat_max
    failure = summarize_failure(dt, violation_mask)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        log_path,
        dt=dt,
        xs=xs,
        ys=ys,
        yaws=yaws,
        vs=vs,
        steers=steers,
        alats=alats,
        curvature=road.curvature,
        wheelbase=params.wheelbase,
        a_lat_max=a_lat_max,
        violation_mask=violation_mask,
        has_violation=failure["has_violation"],
        first_violation_t=failure["first_violation_t"],
        violation_rate=failure["violation_rate"],
    )


def resolve_log_path() -> Path:
    candidates = [
        Path("sim_log_step1.npz"),
        Path(__file__).resolve().parent / "sim_log_step1.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    log_path = candidates[-1]
    generate_log(log_path)
    return log_path


def main():
    log_path = resolve_log_path()
    data = np.load(log_path)

    dt = float(data["dt"])
    xs = data["xs"]
    ys = data["ys"]
    yaws = data["yaws"]
    vs = data["vs"]
    alats = data["alats"]
    curvature = float(data["curvature"])
    a_lat_max = float(data["a_lat_max"])

    # Normalize yaws
    yaws = (yaws + np.pi) % (2 * np.pi) - np.pi

    # Violation info
    violation_mask = np.abs(alats) > a_lat_max
    first_violation = np.argmax(violation_mask) if np.any(violation_mask) else None
    first_violation_t = (first_violation * dt) if first_violation is not None else None

    # Road center for curved road (your setup assumes this)
    R = 1.0 / curvature
    cx, cy = 0.0, R

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Envelope – Aggressive Controller (No Safety)")

    # ---- Road (Step 3 style): dashed center + two offset boundaries ----
    lane_half_width = 2.0
    Nroad = 800
    theta = np.linspace(-np.pi, np.pi, Nroad)  # full circle so road exists everywhere

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

    ax.plot(road_x, road_y, "--", linewidth=2, alpha=0.7)       # centerline
    ax.plot(left_x, left_y, "-", linewidth=4, alpha=0.35)       # boundaries
    ax.plot(right_x, right_y, "-", linewidth=4, alpha=0.35)

    # ---- Unsafe visualization: push vehicle outward when over a_lat limit ----
    dx = xs - cx
    dy = ys - cy
    r = np.hypot(dx, dy)
    r = np.maximum(r, 1e-6)
    unit_x = dx / r
    unit_y = dy / r

    overshoot = np.clip((np.abs(alats) / a_lat_max) - 1.0, 0.0, None)
    skid_offset = overshoot * lane_half_width * 1.5
    r_vis = r + skid_offset

    xs_vis = cx + unit_x * r_vis
    ys_vis = cy + unit_y * r_vis

    # Camera limits: include road + trajectory
    pad = 8.0
    all_x = np.concatenate([xs_vis, road_x, left_x, right_x])
    all_y = np.concatenate([ys_vis, road_y, left_y, right_y])
    ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
    ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

    # Trajectory + vehicle
    traj_line, = ax.plot([], [], linewidth=2)

    veh_patch = plt.Polygon(
        vehicle_rect(xs_vis[0], ys_vis[0], yaws[0]),
        closed=True,
        facecolor="tab:blue",
        edgecolor="black",
        linewidth=2.0,
    )
    ax.add_patch(veh_patch)

    # HUD + warning text (kept from your original)
    hud = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=12,
        bbox=dict(boxstyle="round", alpha=0.25)
    )

    warn = ax.text(
        0.5, 0.08, "",
        transform=ax.transAxes,
        va="bottom", ha="center",
        fontsize=16,
        weight="bold"
    )

    def init():
        traj_line.set_data([], [])
        veh_patch.set_xy(vehicle_rect(xs_vis[0], ys_vis[0], yaws[0]))
        veh_patch.set_facecolor("tab:blue")
        hud.set_text("")
        warn.set_text("")
        return traj_line, veh_patch, hud, warn

    def update(i):
        traj_line.set_data(xs_vis[:i+1], ys_vis[:i+1])
        veh_patch.set_xy(vehicle_rect(xs_vis[i], ys_vis[i], yaws[i]))

        t = i * dt

        # Determine off-road based on visual radial offset from centerline
        radial_offset = abs(np.hypot(xs_vis[i] - cx, ys_vis[i] - cy) - R)
        out_of_road = radial_offset > lane_half_width

        first_violation_text = (
            f"{first_violation_t:4.1f}s"
            if first_violation_t is not None
            else "none"
        )
        hud.set_text(
            f"t = {t:4.1f}s\n"
            f"v = {vs[i]:4.1f} m/s\n"
            f"a_lat = {alats[i]:5.1f} m/s²\n"
            f"limit = {a_lat_max:.1f}\n"
            f"first violation at = {first_violation_text}"
        )

        if out_of_road:
            veh_patch.set_facecolor("tab:red")
            warn.set_text("OFF ROAD (no RTA)")
            traj_line.set_linewidth(3)
        else:
            veh_patch.set_facecolor("tab:blue")
            traj_line.set_linewidth(2)
            if abs(alats[i]) > a_lat_max:
                warn.set_text("SAFETY VIOLATION (no RTA)")
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
        out = output_dir / "step1_unsafe_animation.mp4"
        ani.save(out, fps=fps, dpi=140, writer="ffmpeg")
        saved.append(str(out))
    if animation.writers.is_available("pillow"):
        out = output_dir / "step1_unsafe_animation.gif"
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
