from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation


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


def _get_optional(data, key, default):
    return data[key] if key in data.files else default


def main():
    # --------------------------
    # Load log
    # --------------------------
    log_path = Path(__file__).resolve().parents[1] / "sim_log_step4_risk_rta.npz"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing log: {log_path}")

    data = np.load(log_path, allow_pickle=True)
    print("Loaded log OK:", log_path, flush=True)

    dt = float(data["dt"])
    xs = data["xs"]; ys = data["ys"]; yaws = data["yaws"]
    vs = data["vs"]; alats = data["alats"]

    curvature = float(data["curvature"])
    base_a_lat_max = float(_get_optional(data, "base_a_lat_max", _get_optional(data, "a_lat_max", 5.0)))
    wetness_log = float(_get_optional(data, "wetness", 0.0))

    rta_active = data["rta_active"].astype(bool)
    p_violate = data["p_violate"].astype(float)
    p_threshold = data["p_threshold"].astype(float)
    conservatism = data["conservatism"].astype(float)
    a_lat_max_eff = data["a_lat_max_eff"].astype(float)
    ai_used = data["ai_used"].astype(bool)
    scenario_text = str(_get_optional(data, "scenario_text", "scenario: (unknown)"))

    # Normalize yaws
    yaws = (yaws + np.pi) % (2 * np.pi) - np.pi

    # Baseline params from log
    base_steer_noise_std = float(_get_optional(data, "steer_noise_std", 0.04))
    base_steer_bias = float(_get_optional(data, "steer_bias", 0.03))
    base_accel_noise_std = float(_get_optional(data, "accel_noise_std", 0.0))

    fps = int(round(1.0 / dt))
    output_dir = log_path.parent / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------
    # Road geometry (computed once)
    # --------------------------
    lane_half_width = 2.0
    Nroad = 800

    if abs(curvature) < 1e-9:
        road_x = np.linspace(xs.min() - 30, xs.max() + 30, Nroad)
        road_y = np.zeros_like(road_x)
        dxr = np.gradient(road_x); dyr = np.gradient(road_y)
    else:
        R = 1.0 / curvature
        cx, cy = 0.0, R
        theta = np.linspace(-np.pi, np.pi, Nroad)
        road_x = cx + R * np.cos(theta)
        road_y = cy + R * np.sin(theta)
        dxr = np.gradient(road_x); dyr = np.gradient(road_y)

    nrm = np.sqrt(dxr * dxr + dyr * dyr) + 1e-9
    nx = -dyr / nrm; ny = dxr / nrm
    left_x = road_x + lane_half_width * nx
    left_y = road_y + lane_half_width * ny
    right_x = road_x - lane_half_width * nx
    right_y = road_y - lane_half_width * ny

    # --------------------------
    # Variant configs
    # --------------------------
    variants = ["U0", "U1", "U2", "U3"]

    def variant_params(level: str):
        """
        Return (steer_noise_std, steer_bias, accel_noise_std, wetness, steer_effectiveness, label)
        steer_effectiveness < 1.0 mimics understeer on wet/slippery surface.
        """
        if level == "U0":
            return 0.0, 0.0, 0.0, 0.0, 1.0, "U0 — no uncertainty"
        if level == "U1":
            return base_steer_noise_std, 0.0, 0.0, 0.0, 1.0, "U1 — steering noise only"
        if level == "U2":
            return base_steer_noise_std, base_steer_bias, 0.0, 0.0, 1.0, "U2 — noise + steering bias"
        if level == "U3":
            w = float(np.clip(wetness_log, 0.0, 1.0))
            steer_eff = float(np.clip(1.0 - 0.45 * w, 0.55, 1.0))  # understeer-ish
            return base_steer_noise_std, base_steer_bias, base_accel_noise_std, w, steer_eff, "U3 — noise + bias + wet/slippery"
        raise ValueError(level)

    # --------------------------
    # Ghost sim dependencies
    # --------------------------
    from envelope.control.aggressive import AggressiveController
    from envelope.sim.bicycle import VehicleParams, VehicleState, step

    params = VehicleParams()
    ctrl = AggressiveController(target_speed=22.0)

    # Ghost initial position (ahead of main)
    ghost_lead_m = 0.0
    xg0 = float(xs[0])
    yg0 = float(ys[0])
    ygaw0 = float(yaws[0])
    vg0 = float(vs[0])

    # Noise shaping for visible yaw oscillation
    ghost_noise_alpha = 0.25     # faster/slower wiggle
    ghost_steer_std_scale = 2.5  # bigger/smaller yaw wiggle
    ghost_accel_std_scale = 1.0

    def render_one(level: str):
        steer_noise_std, steer_bias, accel_noise_std, wetness, steer_eff, label = variant_params(level)

        # --- Figure per variant (so each is independent) ---
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.90)
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.90)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Envelope – Step 4: Risk RTA + AI Conservatism")

        ax.plot(road_x, road_y, "--", linewidth=2, alpha=0.7)
        ax.plot(left_x, left_y, "-", linewidth=4, alpha=0.35)
        ax.plot(right_x, right_y, "-", linewidth=4, alpha=0.35)

        pad = 8.0
        all_x = np.concatenate([xs, road_x, left_x, right_x])
        all_y = np.concatenate([ys, road_y, left_y, right_y])
        ax.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax.set_ylim(all_y.min() - pad, all_y.max() + pad)

        # main trajectory (logged)
        traj_line, = ax.plot([], [], linewidth=2, zorder=6)

        # ghost trajectory (full history from start)
        ghost_traj_line, = ax.plot(
            [], [],
            linestyle="--",
            linewidth=2,
            color="gray",
            alpha=0.85,
            zorder=6,
        )

        # main vehicle
        veh_patch = plt.Polygon(
            vehicle_rect(xs[0], ys[0], yaws[0]),
            closed=True,
            facecolor="tab:blue",
            edgecolor="black",
            linewidth=2.0,
            zorder=7,
        )
        ax.add_patch(veh_patch)

        # ghost vehicle
        ghost_patch = plt.Polygon(
            vehicle_rect(xs[0], ys[0], yaws[0]),
            closed=True,
            facecolor="gray",
            edgecolor="none",
            alpha=0.35,
            zorder=5,
        )
        ax.add_patch(ghost_patch)

        hud = ax.text(
            0.02, 0.98, "",
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.18),
        )

        warn = ax.text(
            0.5, 0.5, "",
            transform=ax.transAxes,
            va="center", ha="center",
            fontsize=18,
            weight="bold"
        )

        scenario_box = ax.text(
            0.02, 0.02, f"scenario: {scenario_text}",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.20)
        )

        variant_box = ax.text(
            0.02, 0.10, f"Step 4 — {label}",
            transform=ax.transAxes,
            va="bottom", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", alpha=0.18),
        )

        # Ghost mutable state (reset per variant)
        ghost_state = VehicleState(xg0, yg0, ygaw0, vg0)
        ghost_rng = np.random.default_rng(1234)  # keep same seed so variants are comparable
        ghost_steer_n = 0.0
        ghost_accel_n = 0.0

        ghost_traj_x: list[float] = []
        ghost_traj_y: list[float] = []

        def _step_ghost_once():
            nonlocal ghost_state, ghost_steer_n, ghost_accel_n

            u_nom = ctrl.act(ghost_state, params, curvature)

            steer_std = float(steer_noise_std) * ghost_steer_std_scale
            ghost_steer_n = (1.0 - ghost_noise_alpha) * ghost_steer_n + ghost_noise_alpha * float(
                ghost_rng.normal(0.0, steer_std)
            )

            if accel_noise_std > 0:
                accel_std = float(accel_noise_std) * ghost_accel_std_scale
                ghost_accel_n = (1.0 - ghost_noise_alpha) * ghost_accel_n + ghost_noise_alpha * float(
                    ghost_rng.normal(0.0, accel_std)
                )
            else:
                ghost_accel_n = 0.0

            steer_exec = (float(u_nom.steer) + float(steer_bias) + ghost_steer_n) * float(steer_eff)
            accel_exec = float(u_nom.accel) + ghost_accel_n
            u_exec = type(u_nom)(steer=steer_exec, accel=accel_exec)

            ghost_state = step(ghost_state, u_exec, dt, params)

        def init():
            nonlocal ghost_state, ghost_steer_n, ghost_accel_n

            traj_line.set_data([], [])
            ghost_traj_line.set_data([], [])

            veh_patch.set_xy(vehicle_rect(xs[0], ys[0], yaws[0]))
            veh_patch.set_facecolor("tab:blue")

            # reset ghost
            ghost_state = VehicleState(xg0, yg0, ygaw0, vg0)
            ghost_steer_n = 0.0
            ghost_accel_n = 0.0
            ghost_patch.set_xy(vehicle_rect(float(ghost_state.x), float(ghost_state.y), float(ghost_state.yaw)))

            # seed ghost history
            ghost_traj_x.clear()
            ghost_traj_y.clear()
            ghost_traj_x.append(float(ghost_state.x))
            ghost_traj_y.append(float(ghost_state.y))
            ghost_traj_line.set_data(ghost_traj_x, ghost_traj_y)

            hud.set_text("")
            warn.set_text("")

            return [
                traj_line, ghost_traj_line, veh_patch, ghost_patch,
                hud, warn, scenario_box, variant_box
            ]

        def update(i: int):
            # main car (logged)
            traj_line.set_data(xs[:i+1], ys[:i+1])
            veh_patch.set_xy(vehicle_rect(xs[i], ys[i], yaws[i]))

            # ghost car (dynamic)
            _step_ghost_once()
            ghost_patch.set_xy(vehicle_rect(float(ghost_state.x), float(ghost_state.y), float(ghost_state.yaw)))

            # ghost full trajectory (never truncate)
            ghost_traj_x.append(float(ghost_state.x))
            ghost_traj_y.append(float(ghost_state.y))
            ghost_traj_line.set_data(ghost_traj_x, ghost_traj_y)

            t = i * dt
            hud.set_text(
                f"t={t:4.1f}s  v={vs[i]:4.1f} m/s\n"
                f"a_lat={alats[i]:4.1f}  lim={a_lat_max_eff[i]:.1f} (base {base_a_lat_max:.1f})\n"
                f"risk={p_violate[i]:.2f}  thr={p_threshold[i]:.2f}  C={conservatism[i]:.2f}  AI={int(ai_used[i])}\n"
                f"{level}  σ={steer_noise_std:.3f}  bias={steer_bias:.3f}  wet={wetness:.2f}  eff={steer_eff:.2f}"
            )

            if rta_active[i]:
                veh_patch.set_facecolor("tab:red")
                warn.set_text("RTA\nACTIVE\n(risk exceeded)")
                traj_line.set_linewidth(3)
            else:
                veh_patch.set_facecolor("tab:blue")
                warn.set_text("")
                traj_line.set_linewidth(2)

            return [
                traj_line, ghost_traj_line, veh_patch, ghost_patch,
                hud, warn, scenario_box, variant_box
            ]

        ani = FuncAnimation(
            fig,
            update,
            frames=len(xs),
            init_func=init,
            interval=int(dt * 1000),
            blit=False
        )

        out_tag = f"step4_{level.lower()}_risk_rta_dynamic_ghost"

        saved = []
        if animation.writers.is_available("ffmpeg"):
            out = output_dir / f"{out_tag}.mp4"
            print(f"[{level}] Saving MP4 -> {out}", flush=True)
            ani.save(out, fps=fps, dpi=140, writer="ffmpeg")
            saved.append(str(out))

        if animation.writers.is_available("pillow"):
            out = output_dir / f"{out_tag}.gif"
            print(f"[{level}] Saving GIF -> {out}", flush=True)
            ani.save(out, fps=fps, dpi=140, writer="pillow")
            saved.append(str(out))

        if not saved:
            raise RuntimeError("No animation writer available. Install ffmpeg or pillow.")

        plt.close(fig)
        print(f"[{level}] Saved: {', '.join(saved)}", flush=True)

    # --------------------------
    # Render all variants
    # --------------------------
    print("Rendering variants:", ", ".join(variants), flush=True)
    for level in variants:
        render_one(level)

    # Optional: display the last figure interactively if you want.
    # (We closed all figs after saving, so nothing pops up.)
    print("All variants rendered.", flush=True)


if __name__ == "__main__":
    main()
