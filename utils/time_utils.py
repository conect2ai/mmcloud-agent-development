# utils/time_utils.py
import time
from typing import Dict

class LoopTimer:
    def __init__(self, period_s: float):
        self.period = float(period_s)
        self.last = time.monotonic()

    def step(self) -> Dict[str, float]:
        now = time.monotonic()
        dt = now - self.last
        self.last = now
        return {"dt_s": dt, "drift_ms": 1000.0 * (dt - self.period)}