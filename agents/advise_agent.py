# agents/advise_agent.py
import json
from agents.schemas import PolicyState, Alert
from nlg.llm_runtime_http import LLMRuntimeHTTP
from nlg.templates import render_prompt

def _fallback_message(policy: PolicyState, alerts: list[Alert]) -> str:
    base = {
        "Cautious": "Condução estável, continue assim.",
        "Normal": "Condução ok, mantenha distância segura.",
        "Aggressive": "Reduza a velocidade e aceleração para segurança.",
    }[policy.behavior]
    nearby = "Sem ocorrências próximas." if not alerts else \
             f"Atenção: {alerts[0].type} a {alerts[0].distance_m} m à frente."
    return f"{base} {nearby}"

async def advise_agent(policy: PolicyState, alerts: list[Alert], llm: LLMRuntimeHTTP | None) -> str:
    # Em severidade alta, seja determinístico (sem LLM) para latência e clareza
    if llm is None or policy.severity in {"high"}:
        return _fallback_message(policy, alerts)

    prompt = render_prompt(
        policy_json=json.dumps(policy.__dict__, ensure_ascii=False),
        alerts_json=json.dumps([a.__dict__ for a in alerts], ensure_ascii=False),
    )
    text = await llm.generate(prompt, timeout_ms=350)
    return text or _fallback_message(policy, alerts)