# nlg/llm_runtime_http.py
import aiohttp, asyncio
from typing import Optional

class LLMRuntimeHTTP:
    def __init__(self, url="http://127.0.0.1:8080/completion", max_tokens=48, temperature=0.0):
        self.url = url
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, prompt: str, timeout_ms: int = 350) -> Optional[str]:
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "cache_prompt": True,
            "stop": ["</s>"]
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(self.url, json=payload, timeout=timeout_ms/1000) as r:
                    j = await r.json()
                    # server retorna {"content": "..."} ou stream; aqui supomos modo simples
                    return (j.get("content") or "").strip() or None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return None