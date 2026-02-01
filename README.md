# Envelope RTA

Envelope RTA is a compact simulation and visualization project that demonstrates **Runtime Assurance (RTA)** for a vehicle driving on a curved road under increasing uncertainty.

The project shows how safety supervisors can intervene when a nominal controller becomes unsafe, progressing from simple deterministic guards to probabilistic, risk-aware runtime assurance with Monte Carlo estimation and optional AI-driven scenario generation.

---

## What this project demonstrates

You will see the following progression:

- **Step 1** – Unsafe baseline (no safety supervision)
- **Step 3** – Deterministic Runtime Assurance
- **Step 4** – Risk-based Runtime Assurance with uncertainty ladder (U0–U3)
- **Steps 6–7 (optional)** – AI-generated scenarios and AI-assisted risk interpretation

Each step produces logs and rendered videos that visualize controller behavior, safety intervention, and uncertainty effects.

---

## Core concept

- A vehicle follows a curved road using an **aggressive nominal controller**
- A **Runtime Assurance (RTA)** module monitors safety constraints
- When safety risk exceeds a threshold, control switches to a **safe controller**
- Risk-based RTA estimates probability of failure using **Monte Carlo rollouts**
- An **uncertainty ladder** progressively adds noise, bias, and environmental effects

Videos visualize unsafe behavior, RTA intervention, and how uncertainty changes system behavior.

---

## Repository structure

All source code lives under `src/envelope/`.

### Simulation
```
src/envelope/sim/
├── bicycle.py        # Vehicle dynamics
├── world.py          # Road + environment
└── risk_rta_sim.py   # Integrated simulation loop
```

### Controllers
```
src/envelope/control/
├── aggressive.py     # Nominal controller
└── safe.py           # Safety controller
```

### Runtime Assurance (RTA)
```
src/envelope/rta/
├── deterministic.py
├── risk_mc.py
├── uncertainty.py
├── metrics.py
├── failure.py
└── print_failure_summary.py
```

### Scenarios
```
src/envelope/scenarios/
├── schema.py
├── loader.py
├── presets/
└── ai_generated/
```

### AI (optional)
```
src/envelope/ai/
├── scenario_generator.py
└── risk_interpreter.py
```

### Rendering / Visualization
```
src/envelope/rendering/
├── animate_step1.py
├── animate_step3_rta.py
├── animate_step4_risk_rta.py
└── animate_step4_risk_rta_ai.py
```

### Videos (outputs)
```
src/envelope/videos/
```

### Top-level runnable modules
```
src/envelope/
├── cli.py
├── run_batch.py
├── run_step3_rta.py
├── run_step4_risk_rta.py
├── run_step4_risk_rta_ai.py
├── run_step6_scenario.py
└── run_step7_ai_demo.py
```

---

## Setup

### Create and activate a virtual environment
```
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies
```
pip install numpy matplotlib pillow
```

> Video export to MP4 requires ffmpeg.
> - macOS: brew install ffmpeg
> - Ubuntu: sudo apt-get install ffmpeg

---

## Running the project

All commands assume you are in the repo root and use:

```
PYTHONPATH=src
```

---

## Step 1 — Unsafe baseline

Run the baseline simulation:
```
PYTHONPATH=src python -m envelope.cli
```

Render the animation:
```
PYTHONPATH=src python -m envelope.rendering.animate_step1
```

Outputs (default):
```
src/envelope/videos/step1_unsafe_animation.mp4
src/envelope/videos/step1_unsafe_animation.gif
```

---

## Step 3 — Deterministic RTA

Run the RTA-enabled simulation:
```
PYTHONPATH=src python -m envelope.run_step3_rta
```

Render:
```
PYTHONPATH=src python -m envelope.rendering.animate_step3_rta
```

Outputs:
```
src/envelope/videos/step3_rta.mp4
```

---

## Step 4 — Risk-based RTA (non-AI)

### Batch run (uncertainty ladder U0–U3)

Run all preset scenarios:
```
PYTHONPATH=src python -m envelope.run_batch
```

This generates logs under:
```
runs/<scenario_id>/sim_log.npz
```

Example scenario IDs:
```
runs/u0_no_uncertainty/
runs/u1_steer_noise/
runs/u2_noise_plus_bias/
runs/u3_noise_bias_wet_patch/
```

---

### Render a batch scenario

Render by scenario folder:
```
PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/u3_noise_bias_wet_patch
```

Or render by explicit log file:
```
PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/u3_noise_bias_wet_patch/sim_log.npz
```

Outputs are saved to:
```
runs/u3_noise_bias_wet_patch/videos/
```

---

### Single-run risk RTA (non-AI)

Run:
```
PYTHONPATH=src python -m envelope.run_step4_risk_rta
```

This writes:
```
src/envelope/sim_log_step4_risk_rta.npz
```

Render:
```
PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta src/envelope/sim_log_step4_risk_rta.npz
```

---

## Step 4 — Risk-based RTA (AI scenario + AI interpretation)

Run the AI-based simulation:
```
PYTHONPATH=src python -m envelope.run_step4_risk_rta_ai
```

Render the AI simulation log:
```
PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta_ai --log runs/ai/tmp/sim_log_step4_risk_rta.npz
```

Default outputs:
```
src/envelope/videos/ai/<scenario_id>_step4_risk_rta_dynamic_ghost.mp4
```

---

## Step 6 — Scenario presets

Edit the selected preset in:
```
src/envelope/run_step6_scenario.py
```

Run:
```
PYTHONPATH=src python -m envelope.run_step6_scenario
```

---

## Step 7 — AI demo

Generates an AI scenario JSON and runs the full AI pipeline:
```
PYTHONPATH=src python -m envelope.run_step7_ai_demo
```

---

## Troubleshooting

- Only GIFs generated → install ffmpeg
- Output folder unexpected → outputs are written next to the log path
- Module not found → ensure PYTHONPATH=src and run from repo root

---

## Quick reference

Batch simulation:
```
PYTHONPATH=src python -m envelope.run_batch
```

Render a scenario:
```
PYTHONPATH=src python -m envelope.rendering.animate_step4_risk_rta runs/<scenario_id>
```

---

## Repo artifacts

Pitch decks included in this repository:
- Envelope. - short pitch.pdf
- Envelope. - long pitch.pdf

---

## Tools used

This project was built with support from:
- Cursor Pro
- ChatGPT Plus
- Gemini (Banana Nano & Pro)
- Perplexity Pro
