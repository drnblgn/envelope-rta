from dataclasses import dataclass
import numpy as np

from envelope.sim.bicycle import VehicleParams, VehicleState, Control, step
from envelope.rta.metrics import lateral_accel


@dataclass
class RTADeterministic:
    horizon_s: float = 1.0
    dt_rollout: float = 0.05
    a_lat_max: float = 5.0
    trigger_prob: float = 1.0  # deterministic => effectively always 1

    def will_violate(self, state: VehicleState, u_nom: Control, params: VehicleParams) -> bool:
        """
        Predict forward assuming nominal control is held constant.
        If ANY future step violates a_lat, return True.
        """
        N = int(self.horizon_s / self.dt_rollout)
        s = state
        for _ in range(N):
            s = step(s, u_nom, self.dt_rollout, params)
            a_lat = lateral_accel(s.v, u_nom.steer, params)
            if abs(a_lat) > self.a_lat_max:
                return True
        return False