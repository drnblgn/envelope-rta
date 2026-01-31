from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class UncertaintyConfig:
    # Steering actuation noise (added to steer command at execution)
    steer_noise_std: float = 0.0  # rad

    # Steering bias (systematic offset)
    steer_bias: float = 0.0  # rad

    # Speed actuation noise (added to speed update indirectly via accel proxy)
    # If you don't have accel control, keep this at 0.0 and ignore.
    speed_noise_std: float = 0.0  # m/s per step approx (simple)

    # Road curvature mismatch (controller thinks kappa_nom, reality kappa_true)
    curvature_bias: float = 0.0  # 1/m

    # "Wet road" effect via reduced allowable lateral accel (envelope tightens)
    # This does NOT change dynamics; it changes safety threshold.
    wetness: float = 0.0  # 0..1

    # Sensor noise (used in RTA rollouts if you want; optional)
    y_noise_std: float = 0.0  # meters
    yaw_noise_std: float = 0.0  # rad


@dataclass(frozen=True)
class SampledUncertainty:
    steer_noise: float
    steer_bias: float
    speed_noise: float
    curvature_bias: float
    y_noise: float
    yaw_noise: float


def sample_uncertainty(cfg: UncertaintyConfig, rng: np.random.Generator) -> SampledUncertainty:
    return SampledUncertainty(
        steer_noise=float(rng.normal(0.0, cfg.steer_noise_std)),
        steer_bias=float(cfg.steer_bias),
        speed_noise=float(rng.normal(0.0, cfg.speed_noise_std)) if cfg.speed_noise_std > 0 else 0.0,
        curvature_bias=float(cfg.curvature_bias),
        y_noise=float(rng.normal(0.0, cfg.y_noise_std)) if cfg.y_noise_std > 0 else 0.0,
        yaw_noise=float(rng.normal(0.0, cfg.yaw_noise_std)) if cfg.yaw_noise_std > 0 else 0.0,
    )


def effective_a_lat_max(base_a_lat_max: float, wetness: float, conservatism: float) -> float:
    """
    Tighten the envelope based on wetness and AI conservatism.
    - wetness: 0..1
    - conservatism: 0..1
    Returns a reduced a_lat_max.
    """
    wetness = float(np.clip(wetness, 0.0, 1.0))
    conservatism = float(np.clip(conservatism, 0.0, 1.0))

    # Up to 30% reduction from wetness, up to 30% from conservatism
    factor = 1.0 - 0.30 * wetness - 0.30 * conservatism
    factor = float(np.clip(factor, 0.4, 1.0))  # never below 40% for sanity
    return base_a_lat_max * factor
