# utils/common.py
from typing import Any, Dict
import csv

def get_first(d: Dict[str, Any], *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def safe_float(x, default=0.0):
    try:
        return float(x if x is not None else default)
    except Exception:
        return float(default)

def safe_int(x, default=0):
    try:
        return int(float(x if x is not None else default))
    except Exception:
        return int(default)

def safe_round(x, default, ndigits=2):
    return round(safe_float(x, default), ndigits)