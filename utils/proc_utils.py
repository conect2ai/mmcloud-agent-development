# utils/proc_utils.py
from __future__ import annotations
import subprocess
import psutil
from urllib.parse import urlparse
from typing import Optional, Dict, Any, List

def port_from_base_url(base_url: str | None) -> Optional[int]:
    if not base_url:
        return None
    try:
        u = urlparse(base_url)
        return u.port
    except Exception:
        return None

def find_pid_by_port_psutil(port: int) -> Optional[int]:
    try:
        for c in psutil.net_connections(kind="inet"):
            if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
                return c.pid
    except Exception:
        pass
    return None

def find_pid_by_port_lsof(port: int) -> Optional[int]:
    """
    Fallback usando `lsof` (macOS/Unix). Pode exigir permissão para listar processos.
    """
    try:
        out = subprocess.check_output(["lsof", "-i", f"TCP:{port}", "-sTCP:LISTEN", "-n", "-P"], text=True)
        # Exemplo de linha: "python3  12345 user  ... TCP *:8080 (LISTEN)"
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    except Exception:
        pass
    return None

def find_pid_by_cmdline(port: Optional[int] = None) -> Optional[int]:
    """
    Procura por processos com 'llama_cpp.server' (ou 'python -m llama_cpp.server').
    Se `port` for dado, ainda confere se a linha de comando contém '--port <port>'.
    """
    try:
        for p in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            info = p.info
            cmd: List[str] = info.get("cmdline") or []
            joined = " ".join(cmd).lower()
            if "llama_cpp.server" in joined or "python -m llama_cpp.server" in joined:
                if port is None or f"--port {port}" in joined or f":{port}" in joined:
                    return info["pid"]
    except Exception:
        pass
    return None

def find_llama_server_pid(base_url: Optional[str], default_port: int = 8080) -> Optional[int]:
    """
    Estratégia em cascata para descobrir o PID do servidor llama.cpp:
      1) psutil pela porta
      2) lsof pela porta
      3) varrer processos por cmdline
    """
    port = port_from_base_url(base_url) or default_port
    pid = find_pid_by_port_psutil(port)
    if pid:
        return pid
    pid = find_pid_by_port_lsof(port)
    if pid:
        return pid
    pid = find_pid_by_cmdline(port)
    return pid

def sample_process_metrics(pid: int, duration_s: float = 0.30, samples: int = 3) -> Dict[str, Any]:
    """
    Amostra CPU% (média/máximo) e RSS pico (MB) de um processo por uma janela curta.
    """
    try:
        import time
        p = psutil.Process(pid)
        cpu_vals, rss_vals = [], []
        _ = p.cpu_percent(interval=None)  # warm-up

        step = max(duration_s / max(samples, 1), 0.05)
        for _ in range(max(samples, 1)):
            time.sleep(step)
            cpu_vals.append(p.cpu_percent(interval=None))
            rss_vals.append(p.memory_info().rss / (1024 * 1024.0))
        return {
            "pid": pid,
            "samples": len(cpu_vals),
            "cpu_avg_pct": round(sum(cpu_vals) / len(cpu_vals), 2),
            "cpu_max_pct": round(max(cpu_vals), 2),
            "rss_peak_mb": round(max(rss_vals), 2),
        }
    except Exception:
        return {}
