Envelope RTA – Overview and How-To
=================================

This project simulates a vehicle on a curved road and demonstrates how
Runtime Assurance (RTA) keeps the vehicle within safety limits under
uncertainty. It includes:
- Baseline unsafe behavior (Step 1)
- Deterministic RTA (Step 3)
- Risk-based RTA with uncertainty ladder (Step 4)
- Optional AI-driven scenario generation and interpretation (Steps 6-7)


Concept summary
---------------
- A vehicle follows a curved road using an aggressive controller.
- RTA modules estimate safety risk (deterministic or Monte Carlo).
- When risk exceeds a threshold, a safe controller overrides the nominal one.
- Videos visualize unsafe behavior, RTA intervention, and uncertainty effects.


Project layout (high-level)
---------------------------
- src/envelope/sim/              Core vehicle dynamics
- src/envelope/control/          Controllers (aggressive, safe)
- src/envelope/rta/              RTA logic (deterministic + Monte Carlo)
- src/envelope/scenarios/        Scenario presets and AI-generated scenarios
- src/envelope/rendering/        Animation scripts
- src/envelope/videos/           Rendered outputs (mp4/gif)


Setup
-----
1) Create/activate a Python environment
2) Install dependencies (numpy, matplotlib, etc.)

If you already have a venv:
  source .venv/bin/activate


Step 1 (unsafe baseline)
------------------------
Run the baseline simulation:

  PYTHONPATH=src python -m envelope.cli

Render the animation:

  PYTHONPATH=src python -m envelope.rendering.animate_step1

Outputs:
  src/envelope/videos/step1_unsafe_animation.mp4 (and .gif if pillow)


Step 3 (deterministic RTA)
--------------------------
Run the RTA-enabled simulation:

  PYTHONPATH=src python -m envelope.run_step3_rta

Render:

  PYTHONPATH=src python -m envelope.rendering.animate_step3_rta

Outputs:
  src/envelope/videos/step3_rta.mp4 (and .gif if pillow)


Step 4 (risk RTA, non-AI) — Generate logs
-----------------------------------------
Use the preset ladder (U0–U3) and run the batch sim:

  PYTHONPATH=src python -m envelope.run_batch

This will create per-scenario logs in:
  runs/<scenario_id>/sim_log.npz

Example scenario IDs:
  runs/u0_no_uncertainty/
  runs/u1_steer_noise/
  runs/u2_noise_plus_bias/
  runs/u3_noise_bias_wet_patch/

Note: Each preset sets "use_ai": false, so this is non-AI.


Step 4 (risk RTA, non-AI) — Render videos
-----------------------------------------
Render a specific scenario log:

  PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/u3_noise_bias_wet_patch

Or pass the log file directly:

  PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/u3_noise_bias_wet_patch/sim_log.npz

Outputs are saved under the same parent folder:
  runs/u3_noise_bias_wet_patch/videos/

If you run with the default log in src/envelope, outputs go to:
  src/envelope/videos/


Step 4 (risk RTA, single run, non-AI)
-------------------------------------
If you prefer the single-run script:

  PYTHONPATH=src python -m envelope.run_step4_risk_rta

This writes:
  src/envelope/sim_log_step4_risk_rta.npz

Then render:

  PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta src/envelope/sim_log_step4_risk_rta.npz


Step 4 (risk RTA, AI scenario run)
----------------------------------
This path uses AI-generated scenarios and AI risk interpretation.
If you do not want AI, skip this section.

Run the AI scenario simulation:

  PYTHONPATH=src python -m envelope.run_step4_risk_rta_ai

Render the AI simulation log:

  PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta_ai --log runs/ai/tmp/sim_log_step4_risk_rta.npz

Outputs (default):
  src/envelope/videos/ai/<scenario_id>_step4_risk_rta_dynamic_ghost.mp4


Step 6 (scenario presets)
-------------------------
Run a single preset scenario by editing the preset in:
  src/envelope/run_step6_scenario.py

Then run:

  PYTHONPATH=src python -m envelope.run_step6_scenario


Step 7 (AI demo)
----------------
This script generates an AI scenario JSON and runs the AI pipeline.

  PYTHONPATH=src python -m envelope.run_step7_ai_demo


Troubleshooting
---------------
- If MP4 export fails, install ffmpeg:
    brew install ffmpeg
- If you only get GIFs, Pillow is being used as a fallback.
- If the output folder seems wrong, check the log path you passed.


Quick reference
---------------
Batch sim (U0–U3):
  PYTHONPATH=src python -m envelope.run_batch

Render a scenario:
  PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/<id>
