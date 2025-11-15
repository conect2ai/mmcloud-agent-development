# utils/inference_profiler.py
from __future__ import annotations
import time, threading
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import psutil
except Exception:
    psutil = None  # fallback sem psutil

@dataclass
class ProcSample:
    t: float
    cpu_pct: float
    rss_bytes: int

class InferenceProfiler:
    """
    Samples CPU% and RSS of a target process while inference is running.
    - Uses a background thread to sample every `interval_s`.
    - Cross-platform (macOS, Linux, RPi) via psutil.
    """
    def __init__(self, pid: Optional[int], interval_s: float = 0.1):
        self.pid = pid
        self.interval_s = float(interval_s)
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._samples: List[ProcSample] = []
        self._proc = psutil.Process(pid) if (psutil and pid) else None
        # Prime cpu_percent baseline (non-blocking)
        if self._proc:
            try:
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._proc = None

    def start(self):
        if not self._proc:
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            t = time.monotonic()
            try:
                cpu = self._proc.cpu_percent(interval=None)  # instantaneous since primed
                mem = self._proc.memory_info().rss
                self._samples.append(ProcSample(t, cpu, mem))
            except Exception:
                break
            time.sleep(self.interval_s)

    def stop(self) -> Dict[str, Any]:
        if not self._proc:
            return {
                "cpu_avg_pct": None,
                "cpu_max_pct": None,
                "rss_peak_mb": None,
                "samples": 0,
            }
        if self._thr:
            self._stop.set()
            self._thr.join(timeout=1.0)
        if not self._samples:
            return {
                "cpu_avg_pct": 0.0,
                "cpu_max_pct": 0.0,
                "rss_peak_mb": self._proc.memory_info().rss / (1024 * 1024),
                "samples": 0,
            }
        cpu_vals = [s.cpu_pct for s in self._samples]
        rss_vals = [s.rss_bytes for s in self._samples]
        return {
            "cpu_avg_pct": sum(cpu_vals) / len(cpu_vals),
            "cpu_max_pct": max(cpu_vals),
            "rss_peak_mb": max(rss_vals) / (1024 * 1024),
            "samples": len(self._samples),
        }