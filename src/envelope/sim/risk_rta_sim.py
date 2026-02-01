from __future__ import annotations

from pathlib import Path
import numpy as np

from envelope.control.aggressive import AggressiveController
from envelope.control.safe import SafeController
from envelope.sim.bicycle import VehicleParams, VehicleState, lateral_acceleration, step
from envelope.sim.world import CurvedRoad

from envelope.rta.uncertainty import UncertaintyConfig, effective_a_lat_max
from envelope.rta.risk_mc import RiskRTAConfig, RiskRTAMonteCarlo
from envelope.ai.risk_interpreter import (
    interpret_risk_with_openai,
    interpret_risk_offline,
)


def main():
    print("=== Step 4: Risk (Monte Carlo) RTA + AI Conservatism sim starting ===", flush=True)

    dt = 0.05
    T = 12.0
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=0.05)

    aggressive = AggressiveController(target_speed=22.0)
    safe = SafeController(safe_speed=8.0)

    # --- Uncertainty ---
    unc = UncertaintyConfig(
        steer_noise_std=0.02,
        steer_bias=0.05,
        speed_noise_std=0.08,
        wetness=0.6,
        y_noise_std=0.0,
        yaw_noise_std=0.0,
        curvature_bias=0.0,
    )

    rta_cfg = RiskRTAConfig(
        horizon_s=1.0,
        dt_rollout=dt,
        num_rollouts=50,
        base_a_lat_max=5.0,
        p_threshold_nominal=0.70,
        p_threshold_min=0.05,
        rng_seed=0,
    )
    risk_rta = RiskRTAMonteCarlo(rta_cfg, unc)

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
    ai_rationale_log = []

    telemetry = {
        "speed_mps": float(state.v),
        "curvature": float(road.curvature),
        "max_recent_a_lat_ratio": 0.0,
        "wetness": float(unc.wetness),
        "steer_noise_std": float(unc.steer_noise_std),
        "steer_bias": float(unc.steer_bias),
    }

    interp = interpret_risk_with_openai(scenario_text, telemetry)
    if interp is None:
        interp = interpret_risk_offline(scenario_text, telemetry)

    conservatism = float(np.clip(interp.conservatism, 0.0, 1.0))
    ai_used = bool(interp.ai_used)
    ai_rationale = str(interp.rationale)

    print(
        f"Risk interpretation: conservatism={conservatism:.2f} ai_used={ai_used} rationale={ai_rationale}",
        flush=True,
    )

    for i in range(steps):
        # --- Nominal control ---
        u_nom = aggressive.act(state, params, road.curvature)

        # --- Risk RTA decision ---
        intervene, p_v, p_th = risk_rta.should_intervene(
            state, u_nom, params, conservatism
        )

        # --- RTA-safe control ---
        if intervene:
            u = safe.act(state, params, steer_hold=u_nom.steer)
        else:
            u = u_nom

        t = i * dt

        # Wet patch
        wet = float(unc.wetness) if (3.0 <= t <= 7.0) else 0.0

        # === FINAL STEERING LOGIC ===
        if intervene:
            steer_exec = 1.08 * float(u.steer)   # inward safety margin
        else:
            steer_exec = float(u.steer) * (1.0 - 0.45 * wet)

        steer_exec = np.clip(
            steer_exec, -params.max_steer, params.max_steer
        )

        u_exec = type(u)(
            steer=steer_exec,
            accel=float(u.accel),
        )

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
            float(effective_a_lat_max(rta_cfg.base_a_lat_max, wet, conservatism))
        )
        ai_used_log.append(bool(ai_used))
        ai_rationale_log.append(ai_rationale)

    # Save
    out = Path(__file__).resolve().parent / "sim_log_step4_risk_rta.npz"
    np.savez(
        out,
        dt=dt,
        xs=np.array(xs),
        ys=np.array(ys),
        yaws=np.array(yaws),
        vs=np.array(vs),
        steers=np.array(steers),
        alats=np.array(alats),
        curvature=float(road.curvature),
        wheelbase=float(params.wheelbase),
        base_a_lat_max=float(rta_cfg.base_a_lat_max),
        wetness=float(unc.wetness),
        rta_active=np.array(rta_active, dtype=bool),
        p_violate=np.array(p_violate),
        p_threshold=np.array(p_threshold),
        conservatism=np.array(conservatism_log),
        a_lat_max_eff=np.array(a_lat_max_eff_log),
        ai_used=np.array(ai_used_log),
        ai_rationale=np.array(ai_rationale_log, dtype=object),
        scenario_text=str(scenario_text),
        num_rollouts=int(rta_cfg.num_rollouts),
        horizon_s=float(rta_cfg.horizon_s),
    )

    print("Saved:", out, flush=True)


if __name__ == "__main__":
    print("RUNNING run_step4_risk_rta.py...", flush=True)
    main()
    print("DONE run_step4_risk_rta.py", flush=True)
