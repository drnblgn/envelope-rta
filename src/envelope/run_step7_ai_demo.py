## WORK IN PROGRESS
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from envelope.ai.scenario_generator import generate_scenario_json


def _envelope_root() -> Path:
    # src/envelope/run_step7_ai_demo.py -> src/envelope
    return Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", type=str, default=None, help="Scenario description text.")
    ap.add_argument("--text-file", type=str, default=None, help="Path to a file containing scenario text.")
    ap.add_argument("--base-preset", type=str, default=None, help="Optional base preset JSON path.")
    args = ap.parse_args()

    if args.text_file:
        text = Path(args.text_file).read_text().strip()
    elif args.text:
        text = args.text.strip()
    else:
        text = sys.stdin.read().strip()

    if not text:
        print("ERROR: Provide --text or --text-file (or pipe stdin).", file=sys.stderr)
        return 2

    envelope_root = _envelope_root()

    # 1) Generate scenario JSON (safe location, unique names)
    gen = generate_scenario_json(
        text,
        out_dir=envelope_root / "scenarios" / "ai_generated",
        base_preset_path=Path(args.base_preset) if args.base_preset else None,
    )
    print(f"[AI] scenario_json={gen.json_path}")
    print(f"[AI] {gen.summary}")

    # 2) Run Step4 sim using your existing runner, but force unique output dir + scenario via env vars
    runs_dir = (envelope_root.parent / "runs" / "ai" / gen.scenario_id)
    runs_dir.mkdir(parents=True, exist_ok=True)

    step4_runner = envelope_root / "run_step4_risk_rta_ai.py"

    if not step4_runner.exists():
        print(f"ERROR: step4 runner not found: {step4_runner}", file=sys.stderr)
        return 3

    env = dict(os.environ)
    env["ENVELOPE_OUT_DIR"] = str(runs_dir)
    env["ENVELOPE_SCENARIO_JSON"] = str(gen.json_path)

    cmd = [sys.executable, str(step4_runner)]
    print(f"[SIM] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)

    # Step4 writes log here (guaranteed by patched Step4 file below)
    log_path = runs_dir / "sim_log_step4_risk_rta.npz"
    if not log_path.exists():
        candidates = sorted(runs_dir.glob("*.npz"))
        if not candidates:
            raise RuntimeError(f"No .npz log found in {runs_dir}")
        log_path = candidates[0]

    print(f"[SIM] log={log_path}")

    # 3) Render video (single animation w/ ghost)
    videos_out = envelope_root / "videos" / "ai"
    videos_out.mkdir(parents=True, exist_ok=True)

    animate_script = envelope_root / "rendering" / "animate_step4_risk_rta_ai.py"

    if not animate_script.exists():
        print(f"ERROR: renderer not found: {animate_script}", file=sys.stderr)
        return 4

    render_cmd = [
        sys.executable,
        str(animate_script),
        "--log",
        str(log_path),
        "--out-dir",
        str(videos_out),
    ]
    print(f"[RENDER] {' '.join(render_cmd)}")
    subprocess.run(render_cmd, check=True)

    print(f"[DONE] scenario={gen.json_path}")
    print(f"[DONE] log={log_path}")
    print(f"[DONE] videos_dir={videos_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
