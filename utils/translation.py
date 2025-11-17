# utils/translation.py
import time
from typing import Any, Dict
import numpy as np

# --- dicionários de tradução fixos ---
COMPASS_PT_TO_EN_BASE = {
    "S": "South", "L": "East", "O": "West",
    "Sul": "South", "Leste": "East", "Oeste": "West",
}
SENTIDO_PT_TO_EN = {"Frente": "Forward", "Ré": "Reverse", "Parado": "Stopped"}

def build_heading_message_from_alerts(alerts) -> str | None:
    """
    Gera uma mensagem curta em PT para a UI, baseada nos alerts do Orchestrator.
    Ex: 'Acidentes próximos (~300m à frente)'.
    Retorna None se não houver nada relevante.
    """
    if not alerts:
        return None

    # Os alerts vêm de agents.schemas.Alert (type: 'accident' | 'fine', etc.)
    has_acc = any(getattr(a, "type", None) == "accident" for a in alerts)
    has_fin = any(getattr(a, "type", None) == "fine" for a in alerts)

    if has_acc and has_fin:
        base = "Accidents and fines nearby"
    elif has_acc:
        base = "Accidents nearby"
    elif has_fin:
        base = "Fines nearby"
    else:
        return None

    # Usa o alerta mais próximo (se tiver distance/direction)
    nearest = min(
        alerts,
        key=lambda a: float(getattr(a, "distance_m", float("inf")) or float("inf")),
    )
    dist = getattr(nearest, "distance_m", None)
    direction = getattr(nearest, "direction", "") or ""

    # Monta detalhe bonitinho
    detail_parts = []
    if isinstance(dist, (int, float)) and np.isfinite(dist):
        detail_parts.append(f"~{int(dist)}m")
    if direction:
        detail_parts.append(direction)

    if detail_parts:
        return f"{base} ({' '.join(detail_parts)})"
    return base

def _get_compass_map() -> Dict[str, str]:
    """
    Retorna o dicionário de bússola estático (PT -> EN).

    Não faz mais substituição por mensagens rotativas; isso permite que:
      - "bussola" continue sendo traduzida para North/South/etc.
      - "heading" possa carregar mensagens vindas do Orchestrator
        (por ex.: 'Acidentes próximos (~300m à frente)') sem ser alterada.
    """
    # copia simples do mapa base
    return dict(COMPASS_PT_TO_EN_BASE)

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
