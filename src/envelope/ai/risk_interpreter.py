from __future__ import annotations

from dataclasses import dataclass
import json
import os
import numpy as np


@dataclass(frozen=True)
class RiskInterpretation:
    conservatism: float  # 0..1
    rationale: str
    ai_used: bool


def _clamp01(x: float) -> float:
    return float(np.clip(float(x), 0.0, 1.0))


def interpret_risk_offline(scenario_text: str, telemetry: dict) -> RiskInterpretation:
    """
    Deterministic fallback. Always available.
    """
    text = (scenario_text or "").lower()
    c = 0.3

    if "wet" in text or "rain" in text or "slippery" in text:
        c += 0.4
    if "night" in text or "fog" in text or "low visibility" in text:
        c += 0.2
    if "crosswind" in text or "gust" in text:
        c += 0.2

    ratio = float(telemetry.get("max_recent_a_lat_ratio", 0.0))
    if ratio > 0.9:
        c += 0.3
    elif ratio > 0.75:
        c += 0.15

    c = _clamp01(c)
    return RiskInterpretation(
        conservatism=c,
        rationale=f"offline heuristic (ratio={ratio:.2f})",
        ai_used=False,
    )


def interpret_risk_with_openai(scenario_text: str, telemetry: dict) -> RiskInterpretation:
    """
    Uses OpenAI if OPENAI_API_KEY is set and 'openai' package is installed.
    Falls back to offline heuristic otherwise.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return interpret_risk_offline(scenario_text, telemetry)

    try:
        from openai import OpenAI
    except Exception:
        return interpret_risk_offline(scenario_text, telemetry)

    client = OpenAI(api_key=api_key)

    system = (
        "You are a risk interpreter for a driving safety demo. "
        "Return ONLY valid JSON with keys: conservatism (number 0..1), rationale (string). "
        "Conservatism means how cautious the safety filter should be."
    )
    user = {
        "scenario": scenario_text,
        "telemetry_summary": telemetry,
        "instruction": "Output only JSON, no markdown, no extra text."
    }

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        obj = json.loads(content)
        c = _clamp01(obj.get("conservatism", 0.6))
        rationale = str(obj.get("rationale", ""))
        return RiskInterpretation(conservatism=c, rationale=rationale, ai_used=True)
    except Exception:
        return interpret_risk_offline(scenario_text, telemetry)
