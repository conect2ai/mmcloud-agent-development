# nlg/templates.py
def render_prompt(policy_json: str, alerts_json: str) -> str:
    return f"""Você é um coach de direção. Responda em no máximo 2 frases, português simples.
Dados de condução: {policy_json}
Ocorrências próximas: {alerts_json}
Formato:
- Conselho: <1 frase direta>
- Ocorrências: <se houver, 1 frase curta; senão "Sem ocorrências próximas.">
"""