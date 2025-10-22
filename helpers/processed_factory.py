# helpers/processed_factory.py
from dataclasses import fields
from agents.schemas import Processed

def to_processed(d: dict) -> Processed:
    allowed = {f.name for f in fields(Processed)}
    filt = {k: d[k] for k in d.keys() & allowed}
    # campos obrigatórios com defaults se faltarem (ajuste se quiser)
    if "ts" not in filt:    filt["ts"] = d.get("ts") or ""
    if "speed" not in filt: filt["speed"] = float(d.get("speed", 0.0))
    if "rpm" not in filt:   filt["rpm"] = float(d.get("rpm", 0.0))
    return Processed(**filt)