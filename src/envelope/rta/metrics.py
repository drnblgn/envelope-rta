import numpy as np
from envelope.sim.bicycle import VehicleParams


def lateral_accel(v: float, steer: float, params: VehicleParams) -> float:
    steer = float(np.clip(steer, -params.max_steer, params.max_steer))
    return (v ** 2) * np.tan(steer) / params.wheelbase


def violates_lateral_accel(a_lat: float, a_lat_max: float) -> bool:
    return abs(a_lat) > a_lat_max