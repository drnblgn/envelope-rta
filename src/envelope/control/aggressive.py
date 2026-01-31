from dataclasses import dataclass
from envelope.sim.bicycle import Control, VehicleState, VehicleParams
import numpy as np


@dataclass
class AggressiveController:
    target_speed: float = 22.0  # m/s (intentionally high)
    kp_speed: float = 1.5

    def act(self, state: VehicleState, params: VehicleParams, curvature: float) -> Control:
        # Constant-curvature steering (no safety)
        steer = np.arctan(params.wheelbase * curvature)

        # Push speed aggressively
        accel = self.kp_speed * (self.target_speed - state.v)

        return Control(steer=steer, accel=accel)
