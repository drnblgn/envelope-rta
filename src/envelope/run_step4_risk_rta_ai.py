## WORK IN PROGRESS
from __future__ import annotations

import json
import os
import time
from pathlib import Path
import numpy as np

from envelope.control.aggressive import AggressiveController
from envelope.control.safe import SafeController
from envelope.sim.bicycle import VehicleParams, VehicleState, lateral_acceleration, step
from envelope.sim.world import CurvedRoad

from envelope.rta.uncertainty import UncertaintyConfig, effective_a_lat_max
from envelope.rta.risk_mc import RiskRTAConfig, RiskRTAMonteCarlo
from envelope.ai.risk_interpreter import interpret_risk_with_openai, interpret_risk_offline


def main():
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run_step4_ai_pre",
                "hypothesisId": "H1",
                "location": "run_step4_risk_rta_ai.py:main:entry",
                "message": "Start AI Step4 sim",
                "data": {
                    "cwd": os.getcwd(),
                    "scenario_env": os.environ.get("ENVELOPE_SCENARIO_JSON", ""),
                    "out_dir_env": os.environ.get("ENVELOPE_OUT_DIR", ""),
                    "scenario_text_env": os.environ.get("ENVELOPE_SCENARIO_TEXT", "")
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion
    print("=== Step 4 AI SIM: Risk RTA + Ghost ===", flush=True)

    dt = 0.05
    T = 12.0
    steps = int(T / dt)
    params = VehicleParams()

    # defaults
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
    scenario_text = os.environ.get("ENVELOPE_SCENARIO_TEXT", "heavy rain, sharp curve, fast highway at night")

    # load scenario JSON from env (Step7 sets this)
    scenario_json = os.environ.get("ENVELOPE_SCENARIO_JSON", "").strip()
    if scenario_json:
        # #region agent log
        try:
            _exists = Path(scenario_json).exists()
            with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run_step4_ai_pre",
                    "hypothesisId": "H2",
                    "location": "run_step4_risk_rta_ai.py:scenario_load",
                    "message": "Loading scenario JSON from env",
                    "data": {"path": scenario_json, "exists": _exists},
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        cfg = json.loads(Path(scenario_json).read_text())
        road = CurvedRoad(curvature=float(cfg.get("world", {}).get("curvature", 0.05)))

        target_speed = float(cfg.get("control", {}).get("target_speed", 22.0))
        safe_speed = float(cfg.get("control", {}).get("safe_speed", 8.0))
        aggressive = AggressiveController(target_speed=target_speed)
        safe = SafeController(safe_speed=safe_speed)

        unc_cfg = cfg.get("unc", {})
        # #region agent log
        try:
            with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
                _f.write(json.dumps({
                    "sessionId": "debug-session",
                    "runId": "run_step4_ai_pre",
                    "hypothesisId": "H2",
                    "location": "run_step4_risk_rta_ai.py:uncertainty_keys",
                    "message": "Uncertainty keys present",
                    "data": {
                        "has_unc": "unc" in cfg,
                        "has_uncertainty": "uncertainty" in cfg,
                        "unc_keys": sorted(list(unc_cfg.keys()))
                    },
                    "timestamp": int(time.time() * 1000)
                }) + "\n")
        except Exception:
            pass
        # #endregion
        unc = UncertaintyConfig(
            steer_noise_std=float(unc_cfg.get("steer_noise_std", 0.02)),
            steer_bias=float(unc_cfg.get("steer_bias", 0.05)),
            speed_noise_std=float(unc_cfg.get("speed_noise_std", 0.08)),
            wetness=float(unc_cfg.get("wetness", 0.6)),
            y_noise_std=float(unc_cfg.get("y_noise_std", 0.0)),
            yaw_noise_std=float(unc_cfg.get("yaw_noise_std", 0.0)),
            curvature_bias=float(unc_cfg.get("curvature_bias", 0.0)),
        )

        scenario_text = str(cfg.get("scenario_text", scenario_text))

    risk_rta = RiskRTAMonteCarlo(rta_cfg, unc)

    state_rta = VehicleState(0.0, 0.0, 0.0, 5.0)
    state_ghost = VehicleState(0.0, 0.0, 0.0, 5.0)

    telemetry = {
        "speed_mps": float(state_rta.v),
        "curvature": float(road.curvature),
        "max_recent_a_lat_ratio": 0.0,
        "wetness": float(unc.wetness),
        "steer_noise_std": float(unc.steer_noise_std),
        "steer_bias": float(unc.steer_bias),
    }
    interp = interpret_risk_with_openai(scenario_text, telemetry) or interpret_risk_offline(scenario_text, telemetry)
    conservatism = float(np.clip(interp.conservatism, 0.0, 1.0))
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run_step4_ai_pre",
                "hypothesisId": "H3",
                "location": "run_step4_risk_rta_ai.py:interp",
                "message": "Risk interpretation result",
                "data": {
                    "conservatism": conservatism,
                    "ai_used": bool(interp.ai_used),
                    "scenario_text": scenario_text
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion
    print(f"Risk interpretation: conservatism={conservatism:.2f} ai_used={bool(interp.ai_used)}", flush=True)

    xs, ys, vs, steers = [], [], [], []
    xs_g, ys_g, vs_g, steers_g = [], [], [], []
    rta_active, p_violate, p_threshold = [], [], []
    alats, alats_g = [], []

    for i in range(steps):
        t = i * dt
        wet = float(unc.wetness) if (3.0 <= t <= 7.0) else 0.0

        # ghost
        u_g = aggressive.act(state_ghost, params, road.curvature)
        steer_exec_g = float(u_g.steer) * (1.0 - 0.45 * wet)
        steer_exec_g = np.clip(steer_exec_g, -params.max_steer, params.max_steer)
        state_ghost = step(state_ghost, type(u_g)(steer=steer_exec_g, accel=float(u_g.accel)), dt, params)

        xs_g.append(state_ghost.x); ys_g.append(state_ghost.y); vs_g.append(state_ghost.v); steers_g.append(steer_exec_g)
        alats_g.append(float(lateral_acceleration(state_ghost.v, steer_exec_g, params)))

        # rta vehicle
        u_nom = aggressive.act(state_rta, params, road.curvature)
        intervene, p_v, p_th = risk_rta.should_intervene(state_rta, u_nom, params, conservatism)

        u = safe.act(state_rta, params, steer_hold=u_nom.steer) if intervene else u_nom

        steer_exec = 1.08 * float(u.steer) if intervene else float(u.steer) * (1.0 - 0.45 * wet)
        steer_exec = np.clip(steer_exec, -params.max_steer, params.max_steer)
        state_rta = step(state_rta, type(u)(steer=steer_exec, accel=float(u.accel)), dt, params)

        xs.append(state_rta.x); ys.append(state_rta.y); vs.append(state_rta.v); steers.append(steer_exec)
        alats.append(float(lateral_acceleration(state_rta.v, steer_exec, params)))

        rta_active.append(bool(intervene))
        p_violate.append(float(p_v))
        p_threshold.append(float(p_th))

    out_dir = Path(os.environ.get("ENVELOPE_OUT_DIR", "runs/ai/tmp")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "sim_log_step4_risk_rta.npz"

    np.savez(
        out,
        dt=dt,
        xs=np.array(xs), ys=np.array(ys), vs=np.array(vs), steers=np.array(steers), alats=np.array(alats),
        xs_ghost=np.array(xs_g), ys_ghost=np.array(ys_g), vs_ghost=np.array(vs_g), steers_ghost=np.array(steers_g), alats_ghost=np.array(alats_g),
        curvature=float(road.curvature),
        wheelbase=float(params.wheelbase),
        wetness=float(unc.wetness),
        rta_active=np.array(rta_active, dtype=bool),
        p_violate=np.array(p_violate, dtype=float),
        p_threshold=np.array(p_threshold, dtype=float),
        scenario_text=str(scenario_text),
        a_lat_max_eff=float(effective_a_lat_max(rta_cfg.base_a_lat_max, float(unc.wetness), conservatism)),
    )

    print("Saved:", out, flush=True)
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "run_step4_ai_pre",
                "hypothesisId": "H4",
                "location": "run_step4_risk_rta_ai.py:save",
                "message": "Saved AI sim log",
                "data": {
                    "out": str(out),
                    "exists": out.exists(),
                    "rta_active_frac": float(np.mean(rta_active)) if rta_active else 0.0
                },
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion


if __name__ == "__main__":
    main()
