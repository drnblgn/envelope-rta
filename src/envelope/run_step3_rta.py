from pathlib import Path
import numpy as np

from envelope.control.aggressive import AggressiveController
from envelope.control.safe import SafeController
from envelope.rta.deterministic import RTADeterministic
from envelope.sim.bicycle import VehicleParams, VehicleState, lateral_acceleration, step
from envelope.sim.world import CurvedRoad


def main():
    print("=== Step 3: Deterministic RTA sim starting ===", flush=True)
    dt = 0.05
    T = 12.0
    steps = int(T / dt)

    params = VehicleParams()
    road = CurvedRoad(curvature=0.06)

    aggressive = AggressiveController(target_speed=22.0)
    safe = SafeController(safe_speed=8.0)

    a_lat_max = 5.0
    rta = RTADeterministic(horizon_s=1.0, dt_rollout=dt, a_lat_max=a_lat_max)

    state = VehicleState(0.0, 0.0, 0.0, 5.0)

    xs, ys, yaws, vs, steers, alats = [], [], [], [], [], []
    rta_active = []

    for _ in range(steps):
        u_nom = aggressive.act(state, params, road.curvature)

        intervene = rta.will_violate(state, u_nom, params)
        if intervene:
            u = safe.act(state, params, steer_hold=u_nom.steer)
        else:
            u = u_nom

        state = step(state, u, dt, params)

        xs.append(state.x)
        ys.append(state.y)
        yaws.append(state.yaw)
        vs.append(state.v)
        steers.append(u.steer)
        alats.append(lateral_acceleration(state.v, u.steer, params))
        rta_active.append(intervene)

    xs = np.array(xs); ys = np.array(ys); yaws = np.array(yaws)
    vs = np.array(vs); steers = np.array(steers); alats = np.array(alats)
    rta_active = np.array(rta_active, dtype=bool)

    out = Path(__file__).resolve().parent / "sim_log_step3_rta.npz"
    np.savez(
        out,
        dt=dt,
        xs=xs, ys=ys, yaws=yaws, vs=vs, steers=steers, alats=alats,
        curvature=road.curvature,
        wheelbase=params.wheelbase,
        a_lat_max=a_lat_max,
        rta_active=rta_active,
    )
    print("Saved:", out, flush=True)

    if not out.exists():
        raise RuntimeError(f"Expected log file not found: {out}")



if __name__ == "__main__":
    print("RUNNING run_step3_rta.py...", flush=True)
    main()
    print("DONE run_step3_rta.py", flush=True)

