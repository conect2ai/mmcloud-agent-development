# agents/safety_agent.py
from typing import List, Optional
from agents.schemas import Alert
from services.alerts_service import get_nearby_alerts_by_gps
import asyncio

async def safety_agent_with_gps(
    speed_kmh: float,
    lat: Optional[float],
    lon: Optional[float],
    radius_m: int = 500,
    timeout_ms: int = 250
) -> List[Alert]:
    if lat is None or lon is None:
        return []
    try:
        return await asyncio.wait_for(get_nearby_alerts_by_gps(lat, lon, radius_m), timeout=timeout_ms/1000)
    except asyncio.TimeoutError:
        return []