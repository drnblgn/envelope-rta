## WORK IN PROGRESS
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import time
from pathlib import Path


def _pick_ghost_renderer(mod):
    # pick a callable whose name suggests it renders the dynamic ghost animation
    candidates = []
    for name, obj in vars(mod).items():
        if callable(obj):
            lname = name.lower()
            if "ghost" in lname and ("render" in lname or "animate" in lname):
                candidates.append((name, obj))

    if not candidates:
        # show helpful options
        available = sorted(
            n for n, o in vars(mod).items()
            if callable(o) and ("render" in n.lower() or "animate" in n.lower())
        )
        raise ImportError(
            "Could not find a ghost renderer function in envelope.rendering.animate_step4_risk_rta.\n"
            f"Available render/animate callables: {available}"
        )

    # Prefer ones that explicitly say dynamic_ghost
    candidates.sort(key=lambda x: ("dynamic" not in x[0].lower(), x[0]))
    return candidates[0]  # (name, fn)


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = set(sig.parameters.keys())
    call_kwargs = {k: v for k, v in kwargs.items() if k in supported}
    return fn(**call_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to AI sim .npz log")
    ap.add_argument("--out-dir", default=None, help="Output dir (default: src/envelope/videos/ai)")
    ap.add_argument("--no-gif", action="store_true", help="Disable GIF output")
    args = ap.parse_args()

    log_path = Path(args.log).resolve()
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "render_step4_ai_pre",
                "hypothesisId": "H4",
                "location": "animate_step4_risk_rta_ai.py:log_path",
                "message": "Resolved log path",
                "data": {"log_path": str(log_path), "exists": log_path.exists()},
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else (Path(__file__).resolve().parents[1] / "videos" / "ai")
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_id = log_path.parent.name  # runs/ai/<scenario_id>/
    mp4_path = out_dir / f"{scenario_id}_step4_risk_rta_dynamic_ghost.mp4"
    gif_path = out_dir / f"{scenario_id}_step4_risk_rta_dynamic_ghost.gif"

    mod = importlib.import_module("envelope.rendering.animate_step4_risk_rta")
    fn_name, fn = _pick_ghost_renderer(mod)
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "render_step4_ai_pre",
                "hypothesisId": "H5",
                "location": "animate_step4_risk_rta_ai.py:renderer",
                "message": "Selected renderer",
                "data": {"fn_name": fn_name, "module": str(mod.__name__)},
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion

    print(f"[AI RENDER] Using renderer: {fn_name}", flush=True)
    print(f"[AI RENDER] log={log_path}", flush=True)
    print(f"[AI RENDER] mp4={mp4_path}", flush=True)
    if not args.no_gif:
        print(f"[AI RENDER] gif={gif_path}", flush=True)

    # Try common parameter names across repos
    _call_with_supported_kwargs(
        fn,
        log=str(log_path),
        log_path=str(log_path),
        npz=str(log_path),
        out_dir=str(out_dir),
        mp4_path=str(mp4_path),
        out_mp4=str(mp4_path),
        mp4=str(mp4_path),
        gif_path=None if args.no_gif else str(gif_path),
        out_gif=None if args.no_gif else str(gif_path),
        gif=None if args.no_gif else str(gif_path),
        no_gif=bool(args.no_gif),
    )

    # Ensure at least MP4 exists
    if not mp4_path.exists():
        raise RuntimeError(f"Renderer ran but MP4 not found: {mp4_path}")
    # #region agent log
    try:
        with open("/Users/derinbilgin/envelope-rta/.cursor/debug.log", "a", encoding="utf-8") as _f:
            _f.write(json.dumps({
                "sessionId": "debug-session",
                "runId": "render_step4_ai_pre",
                "hypothesisId": "H4",
                "location": "animate_step4_risk_rta_ai.py:mp4",
                "message": "MP4 created",
                "data": {"mp4_path": str(mp4_path), "exists": mp4_path.exists()},
                "timestamp": int(time.time() * 1000)
            }) + "\n")
    except Exception:
        pass
    # #endregion

    print("[AI RENDER] Done.", flush=True)


if __name__ == "__main__":
    main()
