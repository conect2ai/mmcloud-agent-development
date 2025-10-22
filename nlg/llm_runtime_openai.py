# nlg/llm_runtime_openai.py
import aiohttp, asyncio
from typing import Optional, List, Dict, Any

class LLMRuntimeOpenAI:
    def __init__(self, base_url="http://127.0.0.1:8080/v1", model="local-model",
                 max_tokens=48, temperature=0.0):
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def generate(self, prompt: str, timeout_ms: int = 350) -> Optional[str]:
        """
        Usa /v1/chat/completions. Generate 1 sentence
        """
        payload = {
            "model": self.model,  # valor ignorado pelo server; precisa existir
            "messages": [
                {"role": "system", "content": "You are a driver assistant. Answer in one short sentence."},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.post(f"{self.base_url}/chat/completions",
                                     json=payload, timeout=timeout_ms/1000) as r:
                    j = await r.json()
                    choices = j.get("choices") or []
                    if not choices: return None
                    return (choices[0].get("message", {}).get("content") or "").strip() or None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            return None