from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np

from envelope.control.aggressive import AggressiveController
from envelope.control.safe import SafeController
from envelope.sim.bicycle import VehicleParams, VehicleState, lateral_acceleration, step
from envelope.sim.world import CurvedRoad

from envelope.rta.uncertainty import UncertaintyConfig, effective_a_lat_max
from envelope.rta.risk_mc import RiskRTAConfig, RiskRTAMonteCarlo
from envelope.ai.risk_interpreter import interpret_risk_with_openai, interpret_risk_offline

from envelope.scenarios.schema import ScenarioConfig


def run_risk_rta_sim(cfg: ScenarioConfig, out_dir: str | Path) -> Tuple[Path, Dict[str, Any]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dt = float(cfg.sim.dt)
    T = float(cfg.sim.T)
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=float(cfg.world.curvature))

    aggressive = AggressiveController(target_speed=float(cfg.control.target_speed))
    safe = SafeController(safe_speed=float(cfg.control.safe_speed))

    unc = UncertaintyConfig(
        steer_noise_std=float(cfg.unc.steer_noise_std),
        steer_bias=float(cfg.unc.steer_bias),
        speed_noise_std=float(cfg.unc.speed_noise_std),
        wetness=float(cfg.unc.wetness),
        y_noise_std=float(cfg.unc.y_noise_std),
        yaw_noise_std=float(cfg.unc.yaw_noise_std),
        curvature_bias=float(cfg.unc.curvature_bias),
    )

    rta_cfg = RiskRTAConfig(
        horizon_s=float(cfg.rta.horizon_s),
        dt_rollout=float(cfg.rta.dt_rollout),
        num_rollouts=int(cfg.rta.num_rollouts),
        base_a_lat_max=float(cfg.rta.base_a_lat_max),
        p_threshold_nominal=float(cfg.rta.p_threshold_nominal),
        p_threshold_min=float(cfg.rta.p_threshold_min),
        rng_seed=int(cfg.rta.rng_seed),
    )
    risk_rta = RiskRTAMonteCarlo(rta_cfg, unc)

    scenario_text = str(cfg.scenario_text)
    state = VehicleState(float(cfg.vehicle_init.x), float(cfg.vehicle_init.y), float(cfg.vehicle_init.yaw), float(cfg.vehicle_init.v))

    telemetry = {
        "speed_mps": float(state.v),
        "curvature": float(road.curvature),
        "max_recent_a_lat_ratio": 0.0,
        "wetness": float(unc.wetness),
        "steer_noise_std": float(unc.steer_noise_std),
        "steer_bias": float(unc.steer_bias),
    }

    if bool(cfg.use_ai):
        interp = interpret_risk_with_openai(scenario_text, telemetry)
    else:
        interp = interpret_risk_offline(scenario_text, telemetry)

    conservatism = float(np.clip(interp.conservatism, 0.0, 1.0))
    ai_used = bool(interp.ai_used)
    ai_rationale = str(interp.rationale)

    # Logs
    xs, ys, yaws, vs, steers, alats = [], [], [], [], [], []
    rta_active = []
    p_violate = []
    p_threshold = []
    conservatism_log = []
    a_lat_max_eff_log = []
    ai_used_log = []
    ai_rationale_log = []

    t_start = float(cfg.wet_patch.t_start)
    t_end = float(cfg.wet_patch.t_end)

    for i in range(steps):
        u_nom = aggressive.act(state, params, road.curvature)

        intervene, p_v, p_th = risk_rta.should_intervene(state, u_nom, params, conservatism)

        if intervene:
            u = safe.act(state, params, steer_hold=u_nom.steer)
        else:
            u = u_nom

        t = i * dt
        wet = float(unc.wetness) if (t_start <= t <= t_end) else 0.0

        steer_exec = float(u.steer) * (1.0 - 0.45 * wet)
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
        ai_rationale_log.append(ai_rationale)

    # Arrays
    xs = np.array(xs); ys = np.array(ys); yaws = np.array(yaws)
    vs = np.array(vs); steers = np.array(steers); alats = np.array(alats)
    rta_active = np.array(rta_active, dtype=bool)
    p_violate = np.array(p_violate, dtype=float)
    p_threshold = np.array(p_threshold, dtype=float)
    conservatism_log = np.array(conservatism_log, dtype=float)
    a_lat_max_eff_log = np.array(a_lat_max_eff_log, dtype=float)
    ai_used_log = np.array(ai_used_log, dtype=bool)
    ai_rationale_log = np.array(ai_rationale_log, dtype=object)

    out = out_dir / "sim_log.npz"
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
        ai_rationale=ai_rationale_log,
        scenario_text=str(scenario_text),
        scenario_id=str(cfg.id),
        num_rollouts=int(rta_cfg.num_rollouts),
        horizon_s=float(rta_cfg.horizon_s),
        scenario_config=np.array([asdict(cfg)], dtype=object),
    )

    # Simple metrics
    metrics: Dict[str, Any] = {
        "id": cfg.id,
        "conservatism": float(conservatism),
        "ai_used": int(ai_used),
        "p_threshold": float(p_threshold[0]) if len(p_threshold) else float("nan"),
        "rta_active_fraction": float(np.mean(rta_active)) if len(rta_active) else 0.0,
        "max_p_violate": float(np.max(p_violate)) if len(p_violate) else float("nan"),
        "min_p_violate": float(np.min(p_violate)) if len(p_violate) else float("nan"),
        "max_abs_a_lat": float(np.max(np.abs(alats))) if len(alats) else float("nan"),
        "out_log": str(out),
    }

    # First intervene time
    if len(rta_active) and np.any(rta_active):
        idx = int(np.argmax(rta_active))
        metrics["first_rta_time_s"] = float(idx * dt)
    else:
        metrics["first_rta_time_s"] = float("nan")

    return out, metrics
