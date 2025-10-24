# agents/orchestrator.py
import traceback
from agents.schemas import Processed, OrchestratorOutput
from agents.behavior_agent import behavior_agent
from agents.safety_agent import safety_agent_with_gps
from agents.advise_agent import advise_agent
from nlg.llm_runtime_http import LLMRuntimeHTTP
from agents.safety_agent import safety_agent_with_gps
from utils.metrics import RowMetrics

class Orchestrator:
    def __init__(self, llm: LLMRuntimeHTTP | None):
        self.llm = llm

    async def run_once(self, processed: Processed) -> OrchestratorOutput:
        try:
            rec = RowMetrics()

            with rec.block("agent.behavior"):
                policy = await behavior_agent(processed)

            radius = 500 if processed.speed < 60 else 1000
            with rec.block("agent.safety_gps"):
                alerts = await safety_agent_with_gps(processed.speed, processed.latitude, processed.longitude, radius_m=radius)

            with rec.block("agent.advise_llm"):
                message = await advise_agent(policy, alerts, self.llm)

            return OrchestratorOutput(
                policy=policy,
                alerts=alerts,
                message=message,
                metrics=rec.as_flat()  # m.agent.behavior.*, m.agent.safety_gps.*, m.agent.advise_llm.*
            )
        
        except Exception as e:
            print("\n[ERRO em run_once]", type(e).__name__)
            print(traceback.format_exc())
            raise