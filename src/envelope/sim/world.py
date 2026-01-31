from dataclasses import dataclass


@dataclass
class CurvedRoad:
    curvature: float = 0.06  # 1/m (tight curve)