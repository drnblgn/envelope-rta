from __future__ import annotations

import json
import os
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

    # -----------------------
    # Defaults (used if no scenario JSON is provided)
    # -----------------------
    road = CurvedRoad(curvature=0.05)
    aggressive = AggressiveController(target_speed=22.0)
    safe = SafeController(safe_speed=8.0)

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

    scenario_text = os.environ.get(
        "ENVELOPE_SCENARIO_TEXT",
        "heavy rain, slippery asphalt, low visibility at night",
    )

    # -----------------------
    # Optional: load scenario JSON from env var (Step7 uses this)
    # -----------------------
    scenario_json = os.environ.get("ENVELOPE_SCENARIO_JSON", "").strip()
    if scenario_json:
        cfg = json.loads(Path(scenario_json).read_text())

        # world
        road = CurvedRoad(curvature=float(cfg.get("world", {}).get("curvature", 0.05)))

        # control
        target_speed = float(cfg.get("control", {}).get("target_speed", 22.0))
        safe_speed = float(cfg.get("control", {}).get("safe_speed", 8.0))
        aggressive = AggressiveController(target_speed=target_speed)
        safe = SafeController(safe_speed=safe_speed)

        # uncertainty
        unc_cfg = cfg.get("unc", {})
        unc = UncertaintyConfig(
            steer_noise_std=float(unc_cfg.get("steer_noise_std", 0.02)),
            steer_bias=float(unc_cfg.get("steer_bias", 0.05)),
            speed_noise_std=float(unc_cfg.get("speed_noise_std", 0.08)),
            wetness=float(unc_cfg.get("wetness", 0.6)),
            y_noise_std=float(unc_cfg.get("y_noise_std", 0.0)),
            yaw_noise_std=float(unc_cfg.get("yaw_noise_std", 0.0)),
            curvature_bias=float(unc_cfg.get("curvature_bias", 0.0)),
        )

        # scenario text
        scenario_text = str(cfg.get("scenario_text", scenario_text))

        # If scenario JSON optionally contains rta settings, keep defaults unless present
        rta_cfg = RiskRTAConfig(
            horizon_s=float(cfg.get("rta", {}).get("horizon_s", rta_cfg.horizon_s)),
            dt_rollout=float(cfg.get("rta", {}).get("dt_rollout", rta_cfg.dt_rollout)),
            num_rollouts=int(cfg.get("rta", {}).get("num_rollouts", rta_cfg.num_rollouts)),
            base_a_lat_max=float(cfg.get("rta", {}).get("base_a_lat_max", rta_cfg.base_a_lat_max)),
            p_threshold_nominal=float(cfg.get("rta", {}).get("p_threshold_nominal", rta_cfg.p_threshold_nominal)),
            p_threshold_min=float(cfg.get("rta", {}).get("p_threshold_min", rta_cfg.p_threshold_min)),
            rng_seed=int(cfg.get("rta", {}).get("rng_seed", rta_cfg.rng_seed)),
        )
        risk_rta = RiskRTAMonteCarlo(rta_cfg, unc)

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
        # Nominal
        u_nom = aggressive.act(state, params, road.curvature)

        # Risk RTA decision
        intervene, p_v, p_th = risk_rta.should_intervene(state, u_nom, params, conservatism)

        # Safe controller uses nominal steer as hold
        if intervene:
            u = safe.act(state, params, steer_hold=u_nom.steer)
        else:
            u = u_nom

        t = i * dt

        # Wet patch (keep your existing visual window)
        wet = float(unc.wetness) if (3.0 <= t <= 7.0) else 0.0

        # Steering execution (your fixed Step4 behavior)
        if intervene:
            steer_exec = 1.08 * float(u.steer)  # inward margin (keeps on-road)
        else:
            steer_exec = float(u.steer) * (1.0 - 0.45 * wet)

        steer_exec = np.clip(steer_exec, -params.max_steer, params.max_steer)

        u_exec = type(u)(steer=steer_exec, accel=float(u.accel))
        state = step(state, u_exec, dt, params)

        xs.append(state.x); ys.append(state.y); yaws.append(state.yaw)
        vs.append(state.v); steers.append(steer_exec)

        a_lat = float(lateral_acceleration(state.v, steer_exec, params))
        alats.append(a_lat)

        rta_active.append(bool(intervene))
        p_violate.append(float(p_v))
        p_threshold.append(float(p_th))

        conservatism_log.append(float(conservatism))
        a_lat_max_eff_log.append(float(effective_a_lat_max(rta_cfg.base_a_lat_max, wet, conservatism)))
        ai_used_log.append(bool(ai_used))
        ai_rationale_log.append(str(ai_rationale))

    # Save log to OUT_DIR (Step7 sets this)
    out_dir = Path(os.environ.get("ENVELOPE_OUT_DIR", str(Path(__file__).resolve().parent)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sim_log_step4_risk_rta.npz"

    np.savez(
        out,
        dt=dt,
        xs=np.array(xs), ys=np.array(ys), yaws=np.array(yaws),
        vs=np.array(vs), steers=np.array(steers), alats=np.array(alats),
        curvature=float(road.curvature),
        wheelbase=float(params.wheelbase),
        base_a_lat_max=float(rta_cfg.base_a_lat_max),
        wetness=float(unc.wetness),
        rta_active=np.array(rta_active, dtype=bool),
        p_violate=np.array(p_violate, dtype=float),
        p_threshold=np.array(p_threshold, dtype=float),
        conservatism=np.array(conservatism_log, dtype=float),
        a_lat_max_eff=np.array(a_lat_max_eff_log, dtype=float),
        ai_used=np.array(ai_used_log, dtype=bool),
        ai_rationale=np.array(ai_rationale_log, dtype=object),
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
