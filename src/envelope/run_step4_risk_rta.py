from __future__ import annotations

from pathlib import Path
import numpy as np

from envelope.control.aggressive import AggressiveController
from envelope.control.safe import SafeController
from envelope.sim.bicycle import VehicleParams, VehicleState, lateral_acceleration, step
from envelope.sim.world import CurvedRoad

from envelope.rta.uncertainty import UncertaintyConfig, effective_a_lat_max
from envelope.rta.risk_mc import RiskRTAConfig, RiskRTAMonteCarlo
from envelope.ai.risk_interpreter import interpret_risk_with_openai


def main():
    print("=== Step 4: Risk (Monte Carlo) RTA + AI Conservatism sim starting ===", flush=True)

    dt = 0.05
    T = 12.0
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=0.08)

    aggressive = AggressiveController(target_speed=22.0)
    safe = SafeController(safe_speed=8.0)

    # --- Uncertainty: start simple, you can expand later ---
    unc = UncertaintyConfig(
        steer_noise_std=0.02,   # rad
        steer_bias=0.05,        # rad
        speed_noise_std=0.08,   # m/s-ish jitter
        wetness=0.6,            # 0..1 tightens envelope
        y_noise_std=0.0,
        yaw_noise_std=0.0,
        curvature_bias=0.0,
    )

    rta_cfg = RiskRTAConfig(
        horizon_s=1.0,
        dt_rollout=dt,
        num_rollouts=50,  # start smaller for speed; you can raise to 80-120 later
        base_a_lat_max=5.0,
        p_threshold_nominal=0.20,
        p_threshold_min=0.05,
        rng_seed=0,
    )
    risk_rta = RiskRTAMonteCarlo(rta_cfg, unc)

    # Scenario description feeds AI Risk Interpreter (or offline heuristic)
    scenario_text = "heavy rain, slippery asphalt, low visibility at night"

    state = VehicleState(0.0, 0.0, 0.0, 5.0)

    # Logs
    xs, ys, yaws, vs, steers, alats = [], [], [], [], [], []
    rta_active = []
    p_violate = []
    p_threshold = []
    conservatism_log = []
    a_lat_max_eff_log = []
    ai_used_log = []

    # Telemetry update cadence (AI call cadence)
    update_every = int(round(1.0 / dt))      # every 1.0s sim time
    recent_window = int(round(1.0 / dt))     # look back 1.0s
    heartbeat_every = int(round(0.5 / dt))   # print every 0.5s sim time

    conservatism = 0.5
    ai_used = False

    print(
        f"Config: steps={steps}, dt={dt}, rollouts={rta_cfg.num_rollouts}, horizon={rta_cfg.horizon_s}s, wetness={unc.wetness}",
        flush=True,
    )

    for i in range(steps):
        if i == 0:
            print("Loop started. Running simulation...", flush=True)

        if i % heartbeat_every == 0:
            print(f"t={i*dt:4.1f}s  step={i}/{steps}", flush=True)

        # --- Telemetry summary + AI conservatism update (not every step) ---
        if i % update_every == 0:
            if len(alats) > 0:
                recent = np.array(alats[-recent_window:], dtype=float)
                ratio = float(np.max(np.abs(recent)) / max(rta_cfg.base_a_lat_max, 1e-6))
            else:
                ratio = 0.0

            telemetry = {
                "speed_mps": float(state.v),
                "curvature": float(road.curvature),
                "max_recent_a_lat_ratio": ratio,
                "wetness": float(unc.wetness),
                "steer_noise_std": float(unc.steer_noise_std),
            }

            interp = interpret_risk_with_openai(scenario_text, telemetry)
            conservatism = float(np.clip(interp.conservatism, 0.0, 1.0))
            ai_used = bool(interp.ai_used)

            print(
                f"AI risk interp: conservatism={conservatism:.2f} ai_used={ai_used} telemetry_ratio={ratio:.2f}",
                flush=True,
            )

        # --- Nominal control ---
        u_nom = aggressive.act(state, params, road.curvature)

        # --- Risk RTA decision ---
        intervene, p_v, p_th = risk_rta.should_intervene(state, u_nom, params, conservatism)

        if intervene:
            u = safe.act(state, params, steer_hold=u_nom.steer)
        else:
            u = u_nom

        # --- Step dynamics ---
        t = i * dt

        # Mid-curve wet patch (visual + physical)
        if 3.0 <= t <= 7.0:
            wet = float(unc.wetness)
        else:
            wet = 0.0

        # Apply "slippery road" physics: understeer (reduces effective steer)
        steer_exec = float(u.steer) * (1.0 - 0.45 * wet)
        u_exec = type(u)(steer=steer_exec, accel=float(u.accel))

        state = step(state, u_exec, dt, params)

        # --- Log ---
        xs.append(state.x)
        ys.append(state.y)
        yaws.append(state.yaw)
        vs.append(state.v)
        steers.append(steer_exec)

        a_lat = float(lateral_acceleration(state.v, steer_exec, params))
        alats.append(a_lat)

        rta_active.append(bool(intervene))
        p_violate.append(float(p_v))
        p_threshold.append(float(p_th))
        conservatism_log.append(float(conservatism))
        a_lat_max_eff_log.append(
            effective_a_lat_max(rta_cfg.base_a_lat_max, wet, conservatism)
        )
        ai_used_log.append(bool(ai_used))

    # Convert arrays
    xs = np.array(xs); ys = np.array(ys); yaws = np.array(yaws)
    vs = np.array(vs); steers = np.array(steers); alats = np.array(alats)
    rta_active = np.array(rta_active, dtype=bool)
    p_violate = np.array(p_violate, dtype=float)
    p_threshold = np.array(p_threshold, dtype=float)
    conservatism_log = np.array(conservatism_log, dtype=float)
    a_lat_max_eff_log = np.array(a_lat_max_eff_log, dtype=float)
    ai_used_log = np.array(ai_used_log, dtype=bool)

    out = Path(__file__).resolve().parent / "sim_log_step4_risk_rta.npz"
    np.savez(
        out,
        dt=dt,
        xs=xs, ys=ys, yaws=yaws,
        vs=vs, steers=steers, alats=alats,
        curvature=float(road.curvature),
        wheelbase=float(params.wheelbase),
        base_a_lat_max=float(rta_cfg.base_a_lat_max),
        wetness=float(unc.wetness),
        rta_active=rta_active,
        p_violate=p_violate,
        p_threshold=p_threshold,
        conservatism=conservatism_log,
        a_lat_max_eff=a_lat_max_eff_log,
        ai_used=ai_used_log,
        scenario_text=str(scenario_text),
        num_rollouts=int(rta_cfg.num_rollouts),
        horizon_s=float(rta_cfg.horizon_s),
    )
    print("Saved:", out, flush=True)

    if not out.exists():
        raise RuntimeError(f"Expected log file not found: {out}")


if __name__ == "__main__":
    print("RUNNING run_step4_risk_rta.py...", flush=True)
    main()
    print("DONE run_step4_risk_rta.py", flush=True)
