from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class UncertaintySpec:
    steer_noise_std: float = 0.02
    steer_bias: float = 0.05
    speed_noise_std: float = 0.08
    wetness: float = 0.6
    y_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    curvature_bias: float = 0.0


@dataclass(frozen=True)
class RiskRTASpec:
    horizon_s: float = 1.0
    dt_rollout: float = 0.05
    num_rollouts: int = 50
    base_a_lat_max: float = 5.0
    p_threshold_nominal: float = 0.85
    p_threshold_min: float = 0.05
    rng_seed: int = 0


@dataclass(frozen=True)
class WorldSpec:
    curvature: float = 0.08


@dataclass(frozen=True)
class ControlSpec:
    target_speed: float = 22.0
    safe_speed: float = 8.0


@dataclass(frozen=True)
class VehicleInit:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 5.0


@dataclass(frozen=True)
class WetPatchSpec:
    t_start: float = 3.0
    t_end: float = 7.0


@dataclass(frozen=True)
class SimSpec:
    dt: float = 0.05
    T: float = 12.0


@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    scenario_text: str
    sim: SimSpec = SimSpec()
    world: WorldSpec = WorldSpec()
    control: ControlSpec = ControlSpec()
    vehicle_init: VehicleInit = VehicleInit()
    unc: UncertaintySpec = UncertaintySpec()
    rta: RiskRTASpec = RiskRTASpec()
    wet_patch: WetPatchSpec = WetPatchSpec()
    use_ai: bool = False  # safe default: offline heuristic unless key+pkg available
