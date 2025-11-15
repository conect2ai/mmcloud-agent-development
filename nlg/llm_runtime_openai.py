# nlg/runtime_openai.py
from __future__ import annotations
import httpx, time
from typing import Optional, Dict, Any
from utils.inference_profiler import InferenceProfiler

QWEN_CHAT_TEMPLATE = """<|im_start|>system
{system}
<|im_end|>
<|im_start|>user
{user}
<|im_end|>
<|im_start|>assistant
"""

class LLMRuntimeOpenAI:
    def __init__(self, base_url: str, model: str, max_tokens: int = 64, temperature: float = 0.1,
                 timeout_s: float = 6.0, monitor_pid: Optional[int] = None, sample_interval_s: float = 0.1):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.monitor_pid = monitor_pid
        self.sample_interval_s = sample_interval_s

    async def chat(self, system: str, user: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "add_generation_prompt": True,
            "stop": ["<|im_end|>"],
        }

        prof = InferenceProfiler(self.monitor_pid, interval_s=self.sample_interval_s)
        t0 = time.monotonic()
        prof.start()
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.base_url}/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
        t1 = time.monotonic()
        proc_metrics = prof.stop()

        msg = (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        usage   = data.get("usage")   or {}
        timings = data.get("timings") or {}

        total_ms_client = int((t1 - t0) * 1000)
        meta = {
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "timings": {
                "prompt_ms": timings.get("prompt_ms"),
                "completion_ms": timings.get("completion_ms"),
                "total_ms": timings.get("total_ms"),
                "total_ms_client": total_ms_client,
            },
            "proc": {
                "cpu_avg_pct": proc_metrics.get("cpu_avg_pct"),
                "cpu_max_pct": proc_metrics.get("cpu_max_pct"),
                "rss_peak_mb": proc_metrics.get("rss_peak_mb"),
                "samples": proc_metrics.get("samples"),
                "pid": self.monitor_pid,
            },
            "metrics_source": ("server+client" if (usage or timings) else "client_only"),
        }
        return {"message": msg, "raw": data, "meta": meta}