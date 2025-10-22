# agents/schemas.py
from dataclasses import dataclass, field
from typing import List, Optional, Literal

@dataclass
class Processed:
    ts: str
    speed: float
    rpm: float
    radar_area: Optional[float] = None
    engine_load: Optional[float] = None
    driver_behavior: Optional[str] = None
    road_type: Optional[str] = None
    city_highway: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    # IMPORTANTE: NADA de Pydantic aqui

@dataclass
class PolicyState:
    behavior: Literal["Cautious","Normal","Aggressive"]
    severity: Literal["low","medium","high"]
    advice_code: Literal["ok","reduce_throttle","reduce_speed","maintain"]
    reasons: List[str] = field(default_factory=list)

@dataclass
class Alert:
    type: Literal["accident","fine"]
    distance_m: int
    direction: Literal["ahead","left","right","behind"]
    confidence: float

@dataclass
class OrchestratorOutput:
    policy: PolicyState
    alerts: List[Alert]
    message: str