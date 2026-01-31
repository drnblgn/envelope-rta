from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from envelope.rta.uncertainty import UncertaintyConfig, sample_uncertainty, effective_a_lat_max


@dataclass(frozen=True)
class RiskRTAConfig:
    horizon_s: float = 1.0
    dt_rollout: float = 0.05
    num_rollouts: int = 50

    base_a_lat_max: float = 5.0

    # risk threshold: intervene if estimated violation probability exceeds this
    p_threshold_nominal: float = 0.20  # baseline, without AI conservatism

    # How much conservatism tightens p_threshold
    # effective threshold = lerp(p_threshold_nominal, p_threshold_min, conservatism)
    p_threshold_min: float = 0.05

    rng_seed: int = 0


class RiskRTAMonteCarlo:
    def __init__(self, cfg: RiskRTAConfig, unc: UncertaintyConfig):
        self.cfg = cfg
        self.unc = unc
        self.rng = np.random.default_rng(cfg.rng_seed)

    def p_threshold(self, conservatism: float) -> float:
        c = float(np.clip(conservatism, 0.0, 1.0))
        return (1.0 - c) * self.cfg.p_threshold_nominal + c * self.cfg.p_threshold_min

    def estimate_violation_probability(self, state, u_nom, params, conservatism: float) -> float:
        """
        Roll out under sampled uncertainty and estimate P(violate lateral accel limit).
        Assumes your project has:
          - envelope.sim.bicycle.step(state, u, dt, params) -> next_state
          - envelope.sim.bicycle.lateral_acceleration(v, steer, params) -> a_lat
        """
        from envelope.sim.bicycle import step, lateral_acceleration, VehicleState  # local import to match your structure

        steps = int(np.ceil(self.cfg.horizon_s / self.cfg.dt_rollout))
        violations = 0

        # Effective safety limit depends on wetness + conservatism
        a_lat_max_eff = effective_a_lat_max(self.cfg.base_a_lat_max, self.unc.wetness, conservatism)

        for _ in range(self.cfg.num_rollouts):
            s = state
            violated = False

            sampled = sample_uncertainty(self.unc, self.rng)

            for _k in range(steps):
                # Apply sensor noise optionally (very light touch)
                if sampled.y_noise != 0.0 or sampled.yaw_noise != 0.0:
                    s = VehicleState(
                        s.x,
                        s.y + sampled.y_noise,
                        s.yaw + sampled.yaw_noise,
                        s.v,
                    )

                # Apply actuation uncertainty to the nominal command
                # u_nom must have .steer (like in your code)
                steer_exec = float(u_nom.steer) + sampled.steer_bias + sampled.steer_noise

                # Keep speed dynamics simple: add speed noise directly
                # If your model doesn't support speed command, this is a hacky but usable disturbance
                # We'll apply it by adjusting v after stepping.
                # Preserve nominal accel (your action requires it)
                u_exec = type(u_nom)(steer=steer_exec, accel=float(u_nom.accel))
                
                s_next = step(s, u_exec, self.cfg.dt_rollout, params)

                if sampled.speed_noise != 0.0:
                    s_next = VehicleState(s_next.x, s_next.y, s_next.yaw, max(0.0, s_next.v + sampled.speed_noise))

                a_lat = float(lateral_acceleration(s_next.v, steer_exec, params))
                if abs(a_lat) > a_lat_max_eff:
                    violated = True
                    break

                s = s_next

            if violated:
                violations += 1

        return violations / float(self.cfg.num_rollouts)

    def should_intervene(self, state, u_nom, params, conservatism: float) -> tuple[bool, float, float]:
        """
        Returns:
          (intervene, p_violate, p_threshold_eff)
        """
        p_v = self.estimate_violation_probability(state, u_nom, params, conservatism)
        p_th = self.p_threshold(conservatism)
        return (p_v > p_th), p_v, p_th
