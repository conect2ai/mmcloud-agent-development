# utils/haversine.py
import numpy as np

EARTH_RADIUS_M = 6371008.8  # raio médio da Terra em metros

def haversine_vectorized(lat1_deg, lon1_deg, lat2_deg_arr, lon2_deg_arr):
    """
    Distância haversine entre 1 ponto (lat1, lon1) e vetores (lat2_arr, lon2_arr), em METROS.
    Entradas em graus. Saída: ndarray float64 em metros.
    """
    lat1 = np.radians(lat1_deg)
    lon1 = np.radians(lon1_deg)
    lat2 = np.radians(lat2_deg_arr)
    lon2 = np.radians(lon2_deg_arr)

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_M * c

def degree_bbox(lat_deg, lon_deg, radius_m):
    """
    Bounding box aproximado em graus para pré-filtro rápido.
    """
    dlat = (radius_m / EARTH_RADIUS_M) * (180.0 / np.pi)
    dlon = dlat / max(np.cos(np.radians(lat_deg)), 1e-6)
    return (lat_deg - dlat, lat_deg + dlat, lon_deg - dlon, lon_deg + dlon)