from dataclasses import dataclass
from envelope.sim.bicycle import Control, VehicleParams, VehicleState


@dataclass
class SafeController:
    """
    Simple safe fallback: brake to a target safe speed, no steering changes.
    """
    safe_speed: float = 8.0
    max_brake: float = -6.0  # must match VehicleParams.min_accel

    def act(self, state: VehicleState, params: VehicleParams, steer_hold: float) -> Control:
        # Keep steer from nominal controller, only override accel
        if state.v > self.safe_speed:
            accel = self.max_brake
        else:
            accel = 0.0
        return Control(steer=steer_hold, accel=accel)