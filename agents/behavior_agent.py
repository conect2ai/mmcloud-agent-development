# agents/behavior_agent.py
from agents.schemas import Processed, PolicyState
from policy.policy_engine import assess_policy_combined

async def behavior_agent(p: Processed) -> PolicyState:
    road = p.road_type or p.city_highway
    return assess_policy_combined(
        driver_behavior=p.driver_behavior,
        road_type=road,
        speed=p.speed,
        radar_area=p.radar_area,
        ml_score=getattr(p, "ml_score", None),
    )