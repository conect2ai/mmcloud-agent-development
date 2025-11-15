# nlg/healthcheck.py
import aiohttp, asyncio

async def wait_llm_ready(
    base_url: str = "http://127.0.0.1:8080/v1",
    total_timeout_s: float = 10.0,
    interval_s: float = 0.5
) -> bool:
    deadline = asyncio.get_event_loop().time() + total_timeout_s
    async with aiohttp.ClientSession() as sess:
        while asyncio.get_event_loop().time() < deadline:
            # 1) /v1/models
            try:
                async with sess.get(f"{base_url}/models", timeout=interval_s) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass

            # 2) fallback: /v1/chat/completions com 1 token
            try:
                payload = {
                    "model": "local-model",
                    "messages": [{"role":"user","content":"ping"}],
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": False
                }
                async with sess.post(f"{base_url}/chat/completions",
                                     json=payload, timeout=interval_s) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass

            await asyncio.sleep(interval_s)
    return False