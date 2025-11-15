# utils/metrics.py
from __future__ import annotations
import os, time, json, contextlib
from typing import Any, Dict, Optional

try:
    import psutil
    _PSUTIL = True
    _PROC = psutil.Process(os.getpid())
except Exception:
    psutil = None
    _PSUTIL = False

try:
    import resource  # Unix (macOS/Linux)
    _RESOURCE = True
except Exception:
    _RESOURCE = False

def _proc_memory():
    if not _PSUTIL: return None, None
    try:
        mi = _PROC.memory_info()
        return mi.rss / (1024*1024), _PROC.memory_percent()
    except Exception:
        return None, None

def _cpu_usage_times():
    if _RESOURCE:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return float(ru.ru_utime), float(ru.ru_stime)
    return None, None  # fallback sem resource

class RowMetrics:
    """
    Coleciona métricas (tempo/CPU/RAM) por bloco e deixa pronto para 'merge' no processed.
    Uso:
        rec = RowMetrics()
        with rec.block("teda"):
            ...
        processed.update(rec.as_flat(prefix="m."))
    """
    def __init__(self):
        self.data: Dict[str, Any] = {}  # m.<name>.*  => valor

    class _Block:
        def __init__(self, parent: "RowMetrics", name: str, extra: Optional[Dict[str, Any]] = None):
            self.p = parent
            self.name = name
            self.extra = extra or {}
            self.t0 = None
            self.u0 = None
            self.s0 = None
            self.th0 = None

        def __enter__(self):
            self.t0 = time.perf_counter()
            self.th0 = time.thread_time() if hasattr(time, "thread_time") else None
            self.u0, self.s0 = _cpu_usage_times()
            return self

        def __exit__(self, exc_type, exc, tb):
            t1 = time.perf_counter()
            th1 = time.thread_time() if hasattr(time, "thread_time") else None
            u1, s1 = _cpu_usage_times()
            rss_mb, mem_pct = _proc_memory()

            base = f"m.{self.name}"
            self.p.data[f"{base}.ok"] = (exc_type is None)
            self.p.data[f"{base}.wall_ms"] = round(1000.0 * (t1 - (self.t0 or t1)), 3)
            self.p.data[f"{base}.cpu_user_s"] = None if (self.u0 is None or u1 is None) else round(u1 - self.u0, 6)
            self.p.data[f"{base}.cpu_sys_s"]  = None if (self.s0 is None or s1 is None) else round(s1 - self.s0, 6)
            self.p.data[f"{base}.thread_cpu_s"] = None if (self.th0 is None or th1 is None) else round(th1 - self.th0, 6)
            if rss_mb is not None:  self.p.data[f"{base}.rss_mb"]  = round(rss_mb, 2)
            if mem_pct is not None: self.p.data[f"{base}.mem_pct"] = round(mem_pct, 2)
            if self.extra:
                self.p.data[f"{base}.extra"] = json.dumps(self.extra, ensure_ascii=False)
            # não suprime exceção
            return False

    def block(self, name: str, extra: Optional[Dict[str, Any]] = None) -> "_Block":
        return RowMetrics._Block(self, name, extra)

    def as_flat(self, prefix: str = "m.") -> Dict[str, Any]:
        # já estamos armazenando com prefixo "m.<name>.*"
        if prefix == "m.": 
            return dict(self.data)
        return { (k if k.startswith(prefix) else prefix + k[2:]) : v for k, v in self.data.items() }