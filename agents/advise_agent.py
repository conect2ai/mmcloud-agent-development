# agents/advise_agent.py
from typing import List, Tuple, Dict, Any
from agents.schemas import PolicyState, Alert

DEFAULT_FALLBACK = (
    "Behavior: Normal. PRF zone: none. Driving is within expected range. Stay attentive."
)

SYSTEM_PROMPT = (
    "You are in-car assistant."
    "Start with: 'Behavior: {behavior}. PRF zone: {risk}.' "
    "Be concise. No extra disclaimers. One short tip."
    # "Rewrite the draft with different words."
)


def _risk_label(alerts: List[Alert]) -> str:
    """
    Returns one of: 'none', 'accidents', 'fines', or 'accidents and fines'.
    Separation is based on the Alert.type field ('accident' | 'fine').
    If both types are present at once, we report 'accidents and fines'.
    """
    if not alerts: return "none"
    has_acc = any(a.type == "accident" for a in alerts)
    has_fin = any(a.type == "fine" for a in alerts)
    if has_acc and has_fin: return "accidents and fines"
    if has_acc: return "accidents"
    if has_fin: return "fines"
    return "none"

def _rule_draft(policy: PolicyState, alerts: List[Alert]) -> str:
    """
    Build a short English draft combining behavior + PRF zone.
    Keep this minimal; the LLM will refine.
    """
    beh = (policy.behavior or "Normal").capitalize()
    risk = _risk_label(alerts)
    if beh.lower()=="aggressive" and risk!="none":
        return "Aggressive driving in a risk zone. Slow down and increase space."
    if beh.lower()=="aggressive":
        return "Aggressive driving. Ease off throttle and avoid harsh braking."
    if beh.lower()=="cautious" and risk!="none":
        return "Cautious driving in a risk zone. Keep attention; good for safety and economy."
    if beh.lower()=="cautious":
        return "Cautious driving—good for safety and fuel economy."
    # Normal
    if risk!="none": return "Risk zone ahead. Stay alert and adjust speed."
    return "Driving within expected range. Maintain defensive driving."

def _ensure_labels(text: str, behavior: str, risk: str) -> str:
    """
    Guarantee the output begins with:
      'Behavior: <Cautious|Normal|Aggressive>. PRF zone: <none|accidents|fines|accidents and fines>.'
    If missing, prepend those labels.
    """
    text = (text or "").strip()
    prefix = f"Behavior: {behavior}. PRF zone: {risk}."
    low = text.lower()
    if "behavior:" in low and "prf zone:" in low:
        return text
    return f"{prefix} {text}".strip()

def _sanitize_ascii(s: str) -> str:
    try:
        return s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        return s  # fallback silencioso


async def advise_agent(
    policy: PolicyState,
    alerts: List[Alert],
    llm
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (message, source, meta):
      - message: final English text, max ~2 sentences, with explicit labels
      - source: "model" | "fallback"
      - meta: may include usage/timings/proc + agent_inserted_behavior_prf
    """
    behavior = (policy.behavior or "Normal").capitalize()
    risk = _risk_label(alerts)
    draft = _rule_draft(policy, alerts)

    # -----------------------------
    # CASE 1: Fallback (sem LLM)
    # -----------------------------
    if llm is None:
        final_text = _ensure_labels(draft, behavior, risk)
        meta = {
            "agent_inserted_behavior_prf": True  # sempre true no fallback
        }
        return final_text, "fallback", meta

    # -----------------------------
    # CASE 2: Com LLM
    # -----------------------------
    system = SYSTEM_PROMPT.format(behavior=behavior, risk=risk)
    user = (
        f"Fraft: {draft}\n"
        f"Severity: {policy.severity or 'low'}."
        + (f" Alert: {alerts[0].type} ~{alerts[0].distance_m}m {alerts[0].direction}." if alerts else "")
    )

    system = _sanitize_ascii(system)
    user = _sanitize_ascii(user)

    try:
        out = await llm.chat(system, user)  # {"message": str, "meta": {...}}

        text = (out.get("message") or "").strip()
        meta = out.get("meta") or {}

        if not text:
            text = draft

        # -----------------------------------------------
        # NEW: Detecta se o LLM incluiu Behavior/PRF zone
        # -----------------------------------------------
        low = text.lower()
        has_behavior = "behavior:" in low
        has_prf = "prf zone" in low

        if has_behavior and has_prf:
            # LLM gerou espontaneamente as labels
            inserted = False
            final_text = text
        else:
            # Agente precisou forçar via _ensure_labels()
            inserted = True
            final_text = _ensure_labels(text, behavior, risk)

        # Marca explicitamente no meta
        meta["agent_inserted_behavior_prf"] = inserted

        return final_text, "model", meta

    except Exception:
        # Fallback em caso de erro do LLM
        final_text = _ensure_labels(draft, behavior, risk)
        meta = {
            "agent_inserted_behavior_prf": True
        }
        return final_text, "fallback", meta