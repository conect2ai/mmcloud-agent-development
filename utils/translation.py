# utils/translation.py
from typing import Any, Dict, List

COMPASS_PT_TO_EN = {
    "N": "North", "S": "South", "L": "East", "O": "West",
    "Norte": "North", "Sul": "South", "Leste": "East", "Oeste": "West",
}
SENTIDO_PT_TO_EN = {"Frente": "Forward", "Ré": "Reverse", "Parado": "Stopped"}

def translate_value(key: str, value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        base = value.strip()
        if key in {"bussola", "heading", "direcao_cardinal"}:
            return COMPASS_PT_TO_EN.get(base, base)
        if key == "sentido":
            return SENTIDO_PT_TO_EN.get(base, base)
        if base in COMPASS_PT_TO_EN:
            return COMPASS_PT_TO_EN[base]
        return value
    if isinstance(value, (list, tuple)):
        return [translate_value(key, v) for v in value]
    return value

def translate_payload_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: translate_value(k, v) for k, v in payload.items()}