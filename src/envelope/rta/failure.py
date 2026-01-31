import numpy as np

def first_true_index(mask: np.ndarray):
    if mask is None or len(mask) == 0:
        return None
    if not np.any(mask):
        return None
    return int(np.argmax(mask))


def summarize_failure(dt: float, violation_mask: np.ndarray) -> dict:
    idx = first_true_index(violation_mask)
    return {
        "has_violation": bool(np.any(violation_mask)),
        "first_violation_idx": idx,
        "first_violation_t": (idx * dt) if idx is not None else None,
        "violation_rate": float(np.mean(violation_mask)) if len(violation_mask) else 0.0,
    }