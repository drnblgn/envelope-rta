from __future__ import annotations

from pathlib import Path

from envelope.scenarios.loader import load_scenario
from envelope.sim.risk_rta_sim import run_risk_rta_sim


def main():
    here = Path(__file__).resolve().parent
    preset = here / "scenarios" / "presets" / "u3.json"  # change u0/u1/u2/u3 as needed
    cfg = load_scenario(preset)

    out_dir = Path("runs") / "single" / cfg.id
    log_path, metrics = run_risk_rta_sim(cfg, out_dir)

    print("Saved:", log_path)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
