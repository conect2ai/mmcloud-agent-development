# services/alerts_service.py
from __future__ import annotations
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.schemas import Alert
from utils.haversine import haversine_vectorized, degree_bbox

# ======= Paths (ajuste via env se preferir) =======
ACIDENTES_CSV = os.getenv("ACIDENTES_CSV", "data/acidentes_processado.csv")
MULTAS_CSV    = os.getenv("MULTAS_CSV",    "data/multas_processado.csv")

# ======= Colunas esperadas =======
ACC_COLS = ["data", "hora", "rodovia", "km", "municipio", "tipo", "gravidade", "latitude", "longitude"]
MUL_COLS = ["data", "hora", "rodovia", "km", "municipio", "descricao", "enquadramento", "latitude", "longitude"]

# ======= Index em memória =======
class AlertsIndex:
    """
    Índice leve para consultas por proximidade (Haversine).
    - Carrega CSVs uma única vez.
    - Mantém arrays NumPy de lat/lon para rápido pré-filtro por bounding box.
    - Se sklearn estiver disponível, usa BallTree(haversine) automaticamente.
    """
    def __init__(self, acidentes_path: str, multas_path: str):
        self.acidentes_df = self._load_acidentes(acidentes_path)
        self.multas_df    = self._load_multas(multas_path)

        # Arrays para cálculos rápidos
        self.acc_lat = self.acidentes_df["latitude"].to_numpy(dtype=np.float64)
        self.acc_lon = self.acidentes_df["longitude"].to_numpy(dtype=np.float64)
        self.mul_lat = self.multas_df["latitude"].to_numpy(dtype=np.float64)
        self.mul_lon = self.multas_df["longitude"].to_numpy(dtype=np.float64)

        # BallTree opcional
        self._use_balltree = False
        try:
            from sklearn.neighbors import BallTree
            self._use_balltree = True
            self._acc_tree = BallTree(np.radians(np.c_[self.acc_lat, self.acc_lon]), metric="haversine")
            self._mul_tree = BallTree(np.radians(np.c_[self.mul_lat, self.mul_lon]), metric="haversine")
        except Exception:
            self._acc_tree = None
            self._mul_tree = None

    @staticmethod
    def _load_acidentes(path: str) -> pd.DataFrame:
        # carrega apenas as colunas declaradas; dtypes enxutos
        df = pd.read_csv(path, usecols=ACC_COLS, dtype={
            "data": "string", "hora": "string",
            "rodovia": "string", "km": "float64", "municipio": "string",
            "tipo": "string", "gravidade": "string",
            "latitude": "float64", "longitude": "float64"
        })
        # drop de linhas com lat/lon inválidas
        df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        return df

    @staticmethod
    def _load_multas(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, usecols=MUL_COLS, dtype={
            "data": "string", "hora": "string",
            "rodovia": "string", "km": "float64", "municipio": "string",
            "descricao": "string", "enquadramento": "string",
            "latitude": "float64", "longitude": "float64"
        })
        df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
        return df

    # ---------- Consulta pública ----------
    def query(self, lat: float, lon: float, radius_m: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self._use_balltree:
            return self._query_balltree(lat, lon, radius_m)
        return self._query_numpy(lat, lon, radius_m)

    # ---------- Implementações ----------
    def _query_numpy(self, lat: float, lon: float, radius_m: int):
        # Pré-filtro por bounding box (reduz drástico o N de pontos a calcular)
        lat_min, lat_max, lon_min, lon_max = degree_bbox(lat, lon, radius_m)

        # acidentes
        idx_a = np.where(
            (self.acc_lat >= lat_min) & (self.acc_lat <= lat_max) &
            (self.acc_lon >= lon_min) & (self.acc_lon <= lon_max)
        )[0]
        acc_res = self.acidentes_df.iloc[idx_a]
        if not acc_res.empty:
            dists = haversine_vectorized(lat, lon,
                                         acc_res["latitude"].to_numpy(),
                                         acc_res["longitude"].to_numpy())
            acc_res = acc_res.assign(dist_m=dists)
            acc_res = acc_res[acc_res["dist_m"] <= radius_m].sort_values("dist_m")

        # multas
        idx_m = np.where(
            (self.mul_lat >= lat_min) & (self.mul_lat <= lat_max) &
            (self.mul_lon >= lon_min) & (self.mul_lon <= lon_max)
        )[0]
        mul_res = self.multas_df.iloc[idx_m]
        if not mul_res.empty:
            dists = haversine_vectorized(lat, lon,
                                         mul_res["latitude"].to_numpy(),
                                         mul_res["longitude"].to_numpy())
            mul_res = mul_res.assign(dist_m=dists)
            mul_res = mul_res[mul_res["dist_m"] <= radius_m].sort_values("dist_m")

        return acc_res, mul_res

    def _query_balltree(self, lat: float, lon: float, radius_m: int):
        # BallTree opera em radianos e retorna distâncias em radianos
        R = 6371008.8
        radius_rad = radius_m / R
        q = np.radians([[lat, lon]])

        a_idx_arr, a_dist_arr = self._acc_tree.query_radius(q, r=radius_rad, return_distance=True)
        m_idx_arr, m_dist_arr = self._mul_tree.query_radius(q, r=radius_rad, return_distance=True)

        # acidentes
        a_idx = a_idx_arr[0]
        a_dist_m = (a_dist_arr[0] * R) if a_dist_arr[0].size else np.array([], dtype=float)
        acc_res = self.acidentes_df.iloc[a_idx].copy()
        if not acc_res.empty:
            acc_res.loc[:, "dist_m"] = a_dist_m
            acc_res = acc_res.sort_values("dist_m")

        # multas
        m_idx = m_idx_arr[0]
        m_dist_m = (m_dist_arr[0] * R) if m_dist_arr[0].size else np.array([], dtype=float)
        mul_res = self.multas_df.iloc[m_idx].copy()
        if not mul_res.empty:
            mul_res.loc[:, "dist_m"] = m_dist_m
            mul_res = mul_res.sort_values("dist_m")

        return acc_res, mul_res

# ---------- Singleton ----------
ALERTS_INDEX: Optional[AlertsIndex] = None

async def init_alerts_index(acidentes_path: str = ACIDENTES_CSV, multas_path: str = MULTAS_CSV):
    """
    Carrega em memória uma única vez (chamar no startup).
    """
    global ALERTS_INDEX
    ALERTS_INDEX = AlertsIndex(acidentes_path, multas_path)

# ---------- API para agentes ----------
async def get_nearby_alerts_by_gps(lat: float, lon: float, radius_m: int = 500) -> List[Alert]:
    """
    Converte o match próximo em objetos Alert usados pelo SafetyAgent.
    Neste exemplo retornamos apenas o ponto mais próximo de cada tipo.
    """
    if ALERTS_INDEX is None:
        return []
    acc_df, mul_df = ALERTS_INDEX.query(lat, lon, radius_m)
    out: List[Alert] = []

    if not acc_df.empty:
        r = acc_df.iloc[0]
        out.append(Alert(
            type="accident",
            distance_m=int(r["dist_m"]),
            direction="ahead",  # TODO: pode derivar usando heading + bearing
            confidence=0.85
        ))

    if not mul_df.empty:
        r = mul_df.iloc[0]
        out.append(Alert(
            type="fine",
            distance_m=int(r["dist_m"]),
            direction="ahead",
            confidence=0.75
        ))
    return out