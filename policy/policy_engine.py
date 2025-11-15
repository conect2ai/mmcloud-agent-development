# policy/policy_engine.py
from typing import Optional
from agents.schemas import PolicyState

def _score_behavior(behavior: str | None) -> int:
    b = (behavior or "Normal").lower()
    if b == "aggressive": return 2
    if b == "normal":     return 1
    return 0  # cautious/unknown

def assess_policy_combined(
    *,
    driver_behavior: str | None,
    road_type: str | None,
    speed: float,
    radar_area: float | None,
    ml_score: Optional[float] = None,  # <- default None
) -> PolicyState:
    score = 0
    reasons = []

    if radar_area is not None:
        if radar_area >= 6000: score += 2; reasons.append("radar_area alta")
        elif radar_area >= 2000: score += 1; reasons.append("radar_area moderada")

    if ml_score is not None:
        if ml_score >= 0.7: score += 2; reasons.append("ml_score alto")
        elif ml_score >= 0.5: score += 1; reasons.append("ml_score moderado")

    rt = (road_type or "").lower()
    if rt == "city" and speed > 60:
        score += 2; reasons.append("velocidade alta na cidade")
    elif rt == "highway" and speed > 120:
        score += 1; reasons.append("velocidade alta na rodovia")

    score += _score_behavior(driver_behavior)
    if driver_behavior:
        reasons.append(f"comportamento={driver_behavior}")

    if score >= 5:
        behavior = "Aggressive"; severity = "high"; advice = "reduce_speed"
    elif score >= 3:
        behavior = "Normal";     severity = "medium"; advice = "reduce_throttle"
    else:
        behavior = "Cautious";   severity = "low";    advice = "maintain"

    return PolicyState(behavior=behavior, severity=severity, advice_code=advice, reasons=reasons)