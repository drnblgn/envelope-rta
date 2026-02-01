from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from envelope.scenarios.schema import (
    ScenarioConfig, SimSpec, WorldSpec, ControlSpec, VehicleInit,
    UncertaintySpec, RiskRTASpec, WetPatchSpec
)


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def load_scenario(path: str | Path) -> ScenarioConfig:
    path = Path(path)
    obj = json.loads(path.read_text())

    sim_d = obj.get("sim", {})
    world_d = obj.get("world", {})
    control_d = obj.get("control", {})
    init_d = obj.get("vehicle_init", {})
    unc_d = obj.get("uncertainty", obj.get("unc", {}))
    rta_d = obj.get("rta", {})
    wet_d = obj.get("wet_patch", {})

    cfg = ScenarioConfig(
        id=str(obj["id"]),
        scenario_text=str(obj.get("scenario_text", "")),
        use_ai=bool(obj.get("use_ai", False)),
        sim=SimSpec(
            dt=float(_get(sim_d, "dt", 0.05)),
            T=float(_get(sim_d, "T", 12.0)),
        ),
        world=WorldSpec(
            curvature=float(_get(world_d, "curvature", 0.08)),
        ),
        control=ControlSpec(
            target_speed=float(_get(control_d, "target_speed", 22.0)),
            safe_speed=float(_get(control_d, "safe_speed", 8.0)),
        ),
        vehicle_init=VehicleInit(
            x=float(_get(init_d, "x", 0.0)),
            y=float(_get(init_d, "y", 0.0)),
            yaw=float(_get(init_d, "yaw", 0.0)),
            v=float(_get(init_d, "v", 5.0)),
        ),
        unc=UncertaintySpec(
            steer_noise_std=float(_get(unc_d, "steer_noise_std", 0.02)),
            steer_bias=float(_get(unc_d, "steer_bias", 0.05)),
            speed_noise_std=float(_get(unc_d, "speed_noise_std", 0.08)),
            wetness=float(_get(unc_d, "wetness", 0.6)),
            y_noise_std=float(_get(unc_d, "y_noise_std", 0.0)),
            yaw_noise_std=float(_get(unc_d, "yaw_noise_std", 0.0)),
            curvature_bias=float(_get(unc_d, "curvature_bias", 0.0)),
        ),
        rta=RiskRTASpec(
            horizon_s=float(_get(rta_d, "horizon_s", 1.0)),
            dt_rollout=float(_get(rta_d, "dt_rollout", float(_get(sim_d, "dt", 0.05)))),
            num_rollouts=int(_get(rta_d, "num_rollouts", 50)),
            base_a_lat_max=float(_get(rta_d, "base_a_lat_max", 5.0)),
            p_threshold_nominal=float(_get(rta_d, "p_threshold_nominal", 0.85)),
            p_threshold_min=float(_get(rta_d, "p_threshold_min", 0.05)),
            rng_seed=int(_get(rta_d, "rng_seed", 0)),
        ),
        wet_patch=WetPatchSpec(
            t_start=float(_get(wet_d, "t_start", 3.0)),
            t_end=float(_get(wet_d, "t_end", 7.0)),
        ),
    )
    return cfg
