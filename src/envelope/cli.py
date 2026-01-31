import numpy as np
import matplotlib.pyplot as plt

from envelope.sim.bicycle import VehicleState, VehicleParams, step, lateral_acceleration
from envelope.sim.world import CurvedRoad
from envelope.control.aggressive import AggressiveController


def main():
    dt = 0.05
    T = 12.0
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=0.06)
    ctrl = AggressiveController(target_speed=22.0)

    state = VehicleState(0.0, 0.0, 0.0, 5.0)

    xs, ys, a_lat = [], [], []

    for _ in range(steps):
        u = ctrl.act(state, params, road.curvature)
        state = step(state, u, dt, params)

        xs.append(state.x)
        ys.append(state.y)
        a_lat.append(lateral_acceleration(state.v, u.steer, params))

    a_lat = np.array(a_lat)
    a_lat_max = 5.0  # safety limit for demo

    plt.figure()
    plt.plot(xs, ys)
    plt.axis("equal")
    plt.title("Trajectory (Aggressive Controller, No Safety)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")

    plt.figure()
    plt.plot(a_lat, label="lateral acceleration")
    plt.axhline(a_lat_max, linestyle="--", label="limit")
    plt.axhline(-a_lat_max, linestyle="--")
    plt.legend()
    plt.title("Clear Safety Violation")

    plt.show()


if __name__ == "__main__":
    main()
