# src/envelope/sim/bicycle.py

from dataclasses import dataclass
import numpy as np


@dataclass
class VehicleState:
    x: float # position (meters)
    y: float # position (meters)
    yaw: float # heading (rad)
    v: float # velocity (m/s)


@dataclass
class Control:
    steer: float   # front wheel angle (radians)
    accel: float   # longitudinal acceleration (m/s²)


@dataclass
class VehicleParams:
    wheelbase: float = 2.7 # distance between front and rear wheels (meters)
    max_steer: float = 0.5 # maximum front wheel angle (radians)
    max_accel: float = 3.0 # maximum longitudinal acceleration (m/s²)
    min_accel: float = -6.0 # minimum longitudinal acceleration (m/s²)
    max_speed: float = 30.0 # maximum speed (m/s)


def clamp(val, vmin, vmax):
    return max(vmin, min(vmax, val))


def step(state: VehicleState, control: Control, dt: float, params: VehicleParams):
    delta = clamp(control.steer, -params.max_steer, params.max_steer)
    accel = clamp(control.accel, params.min_accel, params.max_accel)

    v = clamp(state.v + accel * dt, 0.0, params.max_speed)

    yaw_rate = v / params.wheelbase * np.tan(delta)

    x = state.x + v * np.cos(state.yaw) * dt
    y = state.y + v * np.sin(state.yaw) * dt
    
    yaw = state.yaw + yaw_rate * dt
    # keep yaw in [-pi, pi]
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi

    return VehicleState(x=x, y=y, yaw=yaw, v=v)


def lateral_acceleration(v: float, steer: float, params: VehicleParams):
    steer = clamp(steer, -params.max_steer, params.max_steer)
    return v**2 * np.tan(steer) / params.wheelbase

