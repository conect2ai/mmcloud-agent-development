# agents/advise_agent.py
import httpx
from typing import List, Tuple, Dict, Any
from agents.schemas import PolicyState, Alert
from utils.proc_utils import sample_process_metrics, find_llama_server_pid

DEFAULT_FALLBACK = "OK. Keeping an eye on the route. I will notify you if I detect nearby incidents or fines."

async def advise_agent(policy: PolicyState, alerts: List[Alert], llm) -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (message, source, meta)
      - source in {"model","fallback","timeout","error"}
      - meta: {"usage": {...}, "timings": {...}, "proc": {...}, "metrics_source": "..."} (may be empty)
    """
    system = (
        "You are an in-car assistant. Be brief (<= 2 sentences). "
        "Use neutral, helpful tone. If there is any nearby incident/fine, mention distance and advice."
    )

    bh = policy.behavior
    sev = policy.severity
    reason = ", ".join(getattr(policy, "reasons", []) or []) or "n/a"
    if alerts:
        a = alerts[0]
        al = f"{a.type} ~{a.distance_m} m {a.direction}"
    else:
        al = "none"

    user = (
        f"Driver behavior: {bh} (severity: {sev}). Reasons: {reason}.\n"
        f"Nearby alerts: {al}.\n"
        "Give one short advice line to the driver."
    )

    if llm is None:
        return DEFAULT_FALLBACK, "fallback", {}

    try:
        out = await llm.chat(system, user)  # expected: {"message": str, "meta": {...}}
        text = (out.get("message") or "").strip()
        meta: Dict[str, Any] = dict(out.get("meta") or {})

        # ---- attach process metrics (server) ----
        proc = {}
        pid = getattr(llm, "monitor_pid", None)
        if not pid:
            pid = find_llama_server_pid(getattr(llm, "base_url", None), default_port=8080)
            try:
                setattr(llm, "monitor_pid", pid)
            except Exception:
                pass
        if pid:
            try:
                proc = sample_process_metrics(pid)
            except Exception:
                proc = {}

        meta["proc"] = proc
        meta.setdefault("metrics_source", "server+client")

        if not text:
            return DEFAULT_FALLBACK, "error", meta
        return text, "model", meta

    except httpx.TimeoutException:
        return DEFAULT_FALLBACK, "timeout", {}
    except Exception:
        return DEFAULT_FALLBACK, "error", {}
