from __future__ import annotations

import csv
from pathlib import Path

from envelope.scenarios.loader import load_scenario
from envelope.sim.risk_rta_sim import run_risk_rta_sim


def main():
    presets_dir = Path("src/envelope/scenarios/presets")
    out_root = Path("runs")
    out_root.mkdir(parents=True, exist_ok=True)

    scenario_paths = sorted(presets_dir.glob("*.json"))
    if not scenario_paths:
        raise RuntimeError(f"No scenario JSON files found in {presets_dir}")

    rows = []
    for sp in scenario_paths:
        cfg = load_scenario(sp)
        out_dir = out_root / cfg.id
        log_path, metrics = run_risk_rta_sim(cfg, out_dir)
        metrics["out_log"] = str(log_path)
        rows.append(metrics)

        print(
            f"[OK] {cfg.id:24s} "
            f"c={metrics['conservatism']:.3f} "
            f"p_th={metrics['p_threshold']:.3f} "
            f"rta_frac={metrics['rta_active_fraction']:.3f} "
            f"max_p={metrics['max_p_violate']:.3f}"
        )

    summary_path = out_root / "summary.csv"
    fieldnames = [
        "id",
        "conservatism",
        "ai_used",
        "p_threshold",
        "rta_active_fraction",
        "first_rta_time_s",
        "min_p_violate",
        "max_p_violate",
        "max_abs_a_lat",
        "out_log",
    ]

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print("Wrote:", summary_path)


if __name__ == "__main__":
    main()
