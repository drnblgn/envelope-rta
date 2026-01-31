from pathlib import Path
import numpy as np


def main():
    log_path = Path("src/envelope/sim_log_step1.npz")
    data = np.load(log_path)

    dt = float(data["dt"])
    vs = data["vs"]
    alats = data["alats"]
    a_lat_max = float(data["a_lat_max"])

    violation_mask = np.abs(alats) > a_lat_max
    has_violation = bool(np.any(violation_mask))

    if has_violation:
        first_idx = int(np.argmax(violation_mask))
        first_t = first_idx * dt
    else:
        first_idx = -1
        first_t = None

    print("=== Envelope Step 2: Safety Failure Summary (No RTA) ===")
    print("log:", log_path)
    print("a_lat_max:", a_lat_max)
    print("max(|a_lat|):", float(np.max(np.abs(alats))))
    print("max speed:", float(np.max(vs)))
    print("has_violation:", has_violation)
    print("first_violation_idx:", first_idx)
    print("first_violation_t:", first_t)
    print("violation_fraction:", float(np.mean(violation_mask)))


if __name__ == "__main__":
    main()