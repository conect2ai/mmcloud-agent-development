# utils/replay.py
from __future__ import annotations
import os, time, csv
from typing import Dict, Any, Optional, List, Iterable, Callable

def _to_float(x, default=None):
    try:
        if x is None or x == "": return default
        return float(x)
    except Exception:
        return default

def _to_str(x, default=None):
    return str(x) if x is not None and x != "" else default

class CsvReplayer:
    """
    Reproduz amostras de um CSV como dicionários 'raw' compatíveis com seu pipeline.
    - Suporta tempo pelo timestamp do arquivo (CLOCK=file) ou por intervalo fixo (CLOCK=realtime).
    - Pode acelerar/desacelerar com REPLAY_SPEED.
    - Faz mapeamento de colunas via 'colmap'.
    """
    def __init__(
        self,
        path: str,
        colmap: Dict[str, str] | None = None,
        ts_col: str = "ts",
        clock: str = "file",         # "file" | "realtime"
        default_dt_s: float = 1.0,
        speed: float = 1.0,
        loop: bool = True,
    ):
        self.path = path
        self.colmap = colmap or {}
        self.ts_col = ts_col
        self.clock = clock
        self.default_dt_s = float(default_dt_s)
        self.speed = float(speed) if float(speed) > 0 else 1.0
        self.loop = loop

        self._rows: List[Dict[str, Any]] = []
        self._i = 0
        self._last_file_ts: Optional[float] = None   # segundos (epoch ou relativo)
        self._last_wall = None

        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(self.path)
        with open(self.path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            self._rows = [dict(row) for row in r]
        if not self._rows:
            raise ValueError("CSV vazio")

        # normaliza timestamps (aceita ISO ou epoch em segundos)
        def parse_ts(v: str) -> Optional[float]:
            if v is None or v == "": return None
            v = v.strip()
            # epoch?
            try:
                return float(v)
            except Exception:
                pass
            # ISO?
            try:
                from datetime import datetime
                # tenta vários formatos rápidos
                for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ","%Y-%m-%dT%H:%M:%S.%f",
                            "%Y-%m-%dT%H:%M:%SZ","%Y-%m-%dT%H:%M:%S"):
                    try:
                        return datetime.fromisoformat(v.replace("Z","")).timestamp()
                    except Exception:
                        continue
            except Exception:
                return None
            return None

        # guarda ts (float) por linha (ou None)
        for row in self._rows:
            if self.ts_col in row:
                row["_ts_float"] = parse_ts(row[self.ts_col])
            else:
                row["_ts_float"] = None

    def _sleep_until_next(self, cur_ts: Optional[float]):
        if self.clock == "realtime":
            # ritmo fixo
            time.sleep(self.default_dt_s)
            return
        # clock = file; respeita delta de timestamps do arquivo
        if self._last_file_ts is None or cur_ts is None:
            self._last_file_ts = cur_ts
            self._last_wall = time.perf_counter()
            return
        dt_file = max(0.0, (cur_ts - self._last_file_ts))
        dt_wall_target = dt_file / self.speed
        if self._last_wall is None:
            self._last_wall = time.perf_counter()
        elapsed = time.perf_counter() - self._last_wall
        delay = max(0.0, dt_wall_target - elapsed)
        if delay > 0:
            time.sleep(delay)
        # atualiza marcadores
        self._last_file_ts = cur_ts
        self._last_wall = time.perf_counter()

    def _map_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aplica colmap e faz casts essenciais p/ seu pipeline.
        """
        out: Dict[str, Any] = {}
        # 1) aplica renomeações
        for src, val in row.items():
            dst = self.colmap.get(src, src)  # mantém nome se não mapear
            out[dst] = val

        # 2) casts comuns do seu pipeline
        # OBD básicos
        out["speed"] = _to_float(out.get("speed") or out.get("velocidade"), 0.0)
        out["rpm"] = _to_float(out.get("rpm"), 0.0)
        out["throttle"] = _to_float(out.get("throttle"), None)
        out["engine_load"] = _to_float(out.get("engine_load"), None)
        out["maf"] = _to_float(out.get("maf"), None)
        out["fuel_type"] = _to_str(out.get("fuel_type"), None)
        out["road_type"] = _to_str(out.get("road_type") or out.get("city_highway"), None)

        # GPS (se houver no CSV)
        out["latitude"]  = _to_float(out.get("latitude"), None)
        out["longitude"] = _to_float(out.get("longitude"), None)

        # IMU (se houver)
        out["gyro_z_dps"] = _to_float(out.get("gyro_z_dps") or out.get("gyro_z"), 0.0)

        return out

    def next_raw(self) -> Optional[Dict[str, Any]]:
        if not self._rows:
            return None
        if self._i >= len(self._rows):
            if not self.loop:
                return None
            # loop
            self._i = 0
            self._last_file_ts = None
            self._last_wall = None

        row = self._rows[self._i]
        self._i += 1
        # timing
        self._sleep_until_next(row.get("_ts_float"))
        # mapeia para 'raw'
        return self._map_row(row)