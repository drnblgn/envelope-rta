from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class GeneratedScenario:
    scenario_id: str
    json_path: Path
    summary: str


def _repo_envelope_root() -> Path:
    # src/envelope/ai/scenario_generator.py -> src/envelope
    return Path(__file__).resolve().parents[1]


def _load_base_preset(envelope_root: Path) -> Dict[str, Any]:
    presets_dir = envelope_root / "scenarios" / "presets"
    candidates = ["u2.json", "u3.json", "u1.json", "u0.json"]
    for name in candidates:
        p = presets_dir / name
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"No preset found in: {presets_dir}")


def _safe_set(d: Dict[str, Any], path: str, value: Any) -> None:
    """Set nested dict value by dot-path if keys exist; create intermediate dicts if needed."""
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    keys = path.split(".")
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _keyword_score(text: str, words: Tuple[str, ...]) -> bool:
    t = text.lower()
    return any(w in t for w in words)


def generate_scenario_json(
    scenario_text: str,
    *,
    out_dir: Optional[Path] = None,
    base_preset_path: Optional[Path] = None,
) -> GeneratedScenario:
    """
    Generate a NEW scenario JSON (never overwrites presets).
    Uses a preset as template, then tweaks parameters based on keywords.
    """
    envelope_root = _repo_envelope_root()

    if out_dir is None:
        out_dir = envelope_root / "scenarios" / "ai_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if base_preset_path is not None:
        base = json.loads(Path(base_preset_path).read_text())
    else:
        base = _load_base_preset(envelope_root)

    # Unique, non-colliding ID + filenames
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_id = f"ai_{ts}"

    # --- Heuristic mapping from text -> parameters (fast, deterministic) ---
    text = scenario_text.strip()
    t = text.lower()

    rainy = _keyword_score(t, ("rain", "storm", "wet", "slippery", "hydroplan"))
    icy = _keyword_score(t, ("ice", "icy", "snow", "black ice"))
    fog = _keyword_score(t, ("fog", "low visibility", "night", "dark"))
    sharp = _keyword_score(t, ("sharp", "hairpin", "tight curve", "hard turn"))
    fast = _keyword_score(t, ("highway", "fast", "speeding", "racing"))

    # Defaults from base (if missing, fall back)
    wetness = float(_safe_get(base, "unc.wetness", 0.0))
    steer_noise = float(_safe_get(base, "unc.steer_noise_std", 0.02))
    steer_bias = float(_safe_get(base, "unc.steer_bias", 0.03))
    speed_noise = float(_safe_get(base, "unc.speed_noise_std", 0.08))
    curvature = float(_safe_get(base, "world.curvature", 0.05))
    target_speed = float(_safe_get(base, "control.target_speed", 18.0))
    safe_speed = float(_safe_get(base, "control.safe_speed", 8.0))

    if rainy:
        wetness = max(wetness, 0.6)
        steer_noise = max(steer_noise, 0.03)
        speed_noise = max(speed_noise, 0.10)
    if icy:
        wetness = max(wetness, 0.85)
        steer_noise = max(steer_noise, 0.04)
        speed_noise = max(speed_noise, 0.12)
        safe_speed = min(safe_speed, 6.0)
    if fog:
        steer_noise = max(steer_noise, 0.03)
        steer_bias = max(steer_bias, 0.03)
    if sharp:
        curvature = max(curvature, 0.06)
    if fast:
        target_speed = max(target_speed, 22.0)

    # Clamp to reasonable demo ranges
    wetness = _clamp(wetness, 0.0, 1.0)
    steer_noise = _clamp(steer_noise, 0.0, 0.08)
    steer_bias = _clamp(steer_bias, -0.12, 0.12)
    speed_noise = _clamp(speed_noise, 0.0, 0.30)
    curvature = _clamp(curvature, 0.01, 0.10)
    target_speed = _clamp(target_speed, 5.0, 30.0)
    safe_speed = _clamp(safe_speed, 3.0, target_speed)

    # --- Write into scenario dict ---
    _safe_set(base, "id", scenario_id)
    _safe_set(base, "scenario_text", text)
    _safe_set(base, "use_ai", True)

    _safe_set(base, "world.curvature", float(curvature))

    _safe_set(base, "control.target_speed", float(target_speed))
    _safe_set(base, "control.safe_speed", float(safe_speed))

    _safe_set(base, "unc.wetness", float(wetness))
    _safe_set(base, "unc.steer_noise_std", float(steer_noise))
    _safe_set(base, "unc.steer_bias", float(steer_bias))
    _safe_set(base, "unc.speed_noise_std", float(speed_noise))

    # Ensure we never accidentally label as u0/u1/u2/u3
    if re.fullmatch(r"u[0-3]", str(_safe_get(base, "id", ""))):
        _safe_set(base, "id", scenario_id)

    out_path = out_dir / f"{scenario_id}.json"
    out_path.write_text(json.dumps(base, indent=2, sort_keys=False))

    summary_bits = []
    if rainy or icy:
        summary_bits.append(f"wetness={wetness:.2f}")
    if sharp:
        summary_bits.append(f"curvature={curvature:.3f}")
    if fast:
        summary_bits.append(f"target_speed={target_speed:.1f}")
    summary = f"{scenario_id} -> " + (", ".join(summary_bits) if summary_bits else "generated")

    return GeneratedScenario(scenario_id=scenario_id, json_path=out_path, summary=summary)
