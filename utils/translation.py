# utils/translation.py
import time
from typing import Any, Dict

# --- mensagens que se alternam com o tempo ---
ROTATING_MESSAGES = [
    "Accidents and Fines nearby",
    "Drive carefully — risky area",
    "Stay alert and maintain distance"
]
ROTATION_INTERVAL_S = 7  # tempo em segundos para trocar o label

def _get_rotating_label() -> str:
    """Returns one of the rotating messages based on current time."""
    idx = int(time.time() / ROTATION_INTERVAL_S) % len(ROTATING_MESSAGES)
    return ROTATING_MESSAGES[idx]

# --- dicionários de tradução fixos ---
COMPASS_PT_TO_EN_BASE = {
    "S": "South", "L": "East", "O": "West",
    "Sul": "South", "Leste": "East", "Oeste": "West",
}
SENTIDO_PT_TO_EN = {"Frente": "Forward", "Ré": "Reverse", "Parado": "Stopped"}

def _get_compass_map() -> Dict[str, str]:
    """
    Retorna o dicionário de bússola dinâmico, onde as direções norte
    são substituídas pela mensagem rotativa.
    """
    rotating_label = _get_rotating_label()
    compass = dict(COMPASS_PT_TO_EN_BASE)
    compass.update({
        "N": rotating_label,
        "Norte": rotating_label,
    })
    return compass

# --- funções principais (mantidas compatíveis) ---
def translate_value(key: str, value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        base = value.strip()
        compass_map = _get_compass_map()

        if key in {"bussola", "heading", "direcao_cardinal"}:
            return compass_map.get(base, base)
        if key == "sentido":
            return SENTIDO_PT_TO_EN.get(base, base)
        if base in compass_map:
            return compass_map[base]
        return value
    if isinstance(value, (list, tuple)):
        return [translate_value(key, v) for v in value]
    return value

def translate_payload_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: translate_value(k, v) for k, v in payload.items()}
