
import os
import time
import json
import random
import asyncio
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from utils.predictions import calculate_radar_area
from models.outlier_detection import TEDA
from models.mmcloud import MMCloud
from utils.predictions import predict_fuel_type, predict_city_highway
from utils.emissions import calculate_emissions_maf_afr
from utils.accelerometer import calculate_heading, mock_acelerometer
from utils.consumption import instant_fuel_consumption
from utils.commom import get_first, safe_int, safe_float, safe_round
from utils.heading import update_heading_deg, heading_deg_to_cardinal_pt
from utils.translation import translate_payload_values
from utils.time_utils import LoopTimer

# =========================
# Configuration
# =========================
TEST_MODE = os.getenv("TEST_MODE", "1") in {"1", "true", "True", "YES", "yes"}
SEND_INTERVAL_S = float(os.getenv("SEND_INTERVAL_S", "1.0"))
MOCK_ACC = True

# =========================
# Models
# =========================
teda = TEDA()
mmcloud = MMCloud(dimension=2, max_clusters=3)

# =========================
# Data collectors
# =========================

def read_obd_snapshot() -> Dict[str, Any]:
    import obd
    connection = obd.OBD()
    sensors = {
        "speed": obd.commands.SPEED,
        "rpm": obd.commands.RPM,
        "engine_load": obd.commands.ENGINE_LOAD,
        "coolant_temp": obd.commands.COOLANT_TEMP,
        "timing_advance": obd.commands.TIMING_ADVANCE,
        "intake_temp": obd.commands.INTAKE_TEMP,
        "maf": obd.commands.MAF,
        "throttle": obd.commands.THROTTLE_POS,
        "fuel_level": obd.commands.FUEL_LEVEL,
        "ethanol_percentage": obd.commands.ETHANOL_PERCENT,
        "tempAmbiente": obd.commands.AMBIANT_AIR_TEMP,
        "bateria": obd.commands.CONTROL_MODULE_VOLTAGE,
        "temperaturaMotor": obd.commands.COOLANT_TEMP,
    }
    
def read_test_snapshot() -> Dict[str, Any]:
    return {
        "battery": round(random.uniform(11.8, 14.4), 2),         # V
        "engine_temp": random.randint(70, 105),                  # °C
        "fuel_type": random.choice(["Gasoline", "Ethanol"]),
        "road_type": random.choice(["City", "Highway"]),
        "compass": random.choice(["North", "South", "East", "West"]),
        "co2": round(random.uniform(90, 280), 2),                # g/km
        "driver_profile": random.choice(["Cautious", "Normal", "Aggressive"]),
        "speed": random.randint(0, 120),                         # km/h
        "fuel_level": random.randint(0, 100),                    # %
        "eco_mode": random.choice([True, False]),
        "rating_imetro": random.choice(list("ABCDE")),
        "ambient_temp": random.randint(18, 38),                  # °C
        "rpm": random.randint(700, 4000),                        # rpm
        "consumption": round(random.uniform(5.0, 14.0), 2),      # L/100 km
        "distance": round(random.uniform(0, 500), 2),            # km
        "gyro_z_dps": random.uniform(-2, 2),                     # simulated gyro
        "throttle" : random.uniform(0, 1), 
        "engine_load" : random.uniform(0, 1), 
        "maf" : random.uniform(0, 100),
        "ethanol_percentage" : random.uniform(0, 1), 
        "timing_advance" : random.uniform(0, 100), 
    }

# =========================
# FastAPI app
# =========================

app = FastAPI(title="OBD-II WS (PT keys, EN values)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_connections: List[WebSocket] = []

async def broadcast(payload: Dict[str, Any]):
    to_remove = []
    for ws in list(_connections):
        try:
            await ws.send_json(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        if ws in _connections:
            _connections.remove(ws)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    _connections.append(ws)
    try:
        # Immediately send a status + one first sample so the client sees data at once
        await ws.send_json({"status": "connected", "test_mode": TEST_MODE})
        # sample = build_payload()
        # await ws.send_json(translate_payload_values(sample))

        # Keep the connection alive; we don't require client messages
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send a keepalive ping
                await ws.send_json({"ping": datetime.now(timezone.utc).isoformat()})
    except WebSocketDisconnect:
        pass
    finally:
        if ws in _connections:
            _connections.remove(ws)

# =========================
# Core payload builder
# =========================

_heading_deg = 0.0
_timer = LoopTimer(SEND_INTERVAL_S)

def build_payload_interface(raw) -> Dict[str, Any]:
    global _heading_deg
    dt_info = _timer.step()
    dt_s = dt_info["dt_s"]
    elapsed_s = 0.0 if _start_monotonic is None else (time.monotonic() - _start_monotonic)

    gyro = safe_float(get_first(raw, "gyro_z_dps", "gyro_z", "gyroZ", "gyro", default=0.0), 0.0)
    _heading_deg = update_heading_deg(_heading_deg, gyro, dt_s)
    heading_pt = heading_deg_to_cardinal_pt(_heading_deg)

    bateria            = safe_round(get_first(raw, "battery", "battery_voltage", "bateria"), 13, 2)
    temperatura_motor  = safe_int(get_first(raw, "engine_temp", "coolant_temp", "temperaturaMotor"), 90)
    tipo_combustivel   = get_first(raw, "fuel_type", "fuel", "tipoCombustivel", default="Gasolina")
    tipo_via           = get_first(raw, "road_type", "tipoVia", "city_highway", default="Desconhecida")
    co2_val            = safe_round(get_first(raw, "co2", "co2_emission_per_km"), 200, 2)
    perfil_motorista   = get_first(raw, "driver_profile", "driver_behavior", "perfilMotorista", default="Calmo")
    velocidade         = safe_int(get_first(raw, "speed", "velocidade"), 60)
    fuel_level         = safe_int(get_first(raw, "fuel_level", "fuelLevel"), 50)
    eco_mode           = bool(get_first(raw, "eco_mode", "eco", default=False))
    nota_imetro        = get_first(raw, "notaImetro", default="A")
    temp_ambiente      = safe_int(get_first(raw, "ambient_temp", "tempAmbiente"), 25)
    rpm_val            = safe_int(get_first(raw, "rpm"), 2000)
    consumo_val        = safe_round(get_first(raw, "consumption", "consumo", "consumo_medio"), 10, 2)
    distancia_val      = safe_round(get_first(raw, "distance", "distancia_total", "distancia"), 100, 2)

    payload_pt: Dict[str, Any] = {
        "bateria": bateria,
        "temperaturaMotor": temperatura_motor,
        "tipoCombustivel": tipo_combustivel,
        "tipoVia": tipo_via,
        "bussola": heading_pt,  # N/L/S/O
        "co2": co2_val,
        "perfilMotorista": perfil_motorista,
        "velocidade": velocidade,
        "fuelLevel": fuel_level,
        "tempTotal": round(elapsed_s, 0),
        "eco": eco_mode,
        "notaImetro": nota_imetro,
        "tempAmbiente": temp_ambiente,
        "rpm": rpm_val,
        "consumo": consumo_val,
        "distancia": distancia_val,
        "heading": heading_pt,
    }

    return payload_pt

def compute_features_and_predictions(raw):
    
    # 1. Calculate radar area
    raw['radar_area'] = calculate_radar_area({
        "rpm": raw["rpm"],
        "speed": raw["speed"],
        "throttle": raw["throttle"],
        "engine_load": raw["engine_load"]
    })

    # 2. Run TEDA model on radar area soft-sensor
    raw["teda_flag"] = teda.run([raw["radar_area"]])

    # 3. Run MMCloud to identify the driver profile
    raw["driver_behavior"] = mmcloud.process_point([raw["radar_area"], raw["engine_load"]], 1)

    # 4. Identify fuel type (Gasoline or Ethanol)
    raw["fuel_type"], raw["fuel_type_prob"] = predict_fuel_type(raw)

    # 5.1 Get the accelerometer data and calculate the magnitude
    if MOCK_ACC:
        raw = mock_acelerometer(raw)
    else:
        from utils.accelerometer import read_acelerometer
        raw = read_acelerometer(raw)

    raw['accel_magnitude'] = raw["accel_x"]**2 + raw["accel_y"]**2 + raw["accel_z"]**2

    # 5.2 Identify city or highway
    raw["city_highway"], raw["city_highway_prob"] = predict_city_highway(raw)

    # 6. Emissions estimation
    raw = calculate_emissions_maf_afr(raw)

    # 7. Calculate instant fuel consuptiom
    raw['instant_fuel_consumption'] = instant_fuel_consumption(raw["speed"], maf=raw["maf"], combustivel=raw["fuel_type"])

    # 8. Estimated distance
    if "total_distance" not in raw:
        raw["total_distance"] = 0.0
    raw["total_distance"] += raw["speed"] / 3600

    # 9. Average consuption
    if "total_consumption" not in raw:
        raw["total_consumption"] = 0.0
        raw["consuptiom_count"] = 0

    raw["total_consumption"] += raw["instant_fuel_consumption"]
    raw["consuptiom_count"] += 1
    raw["consumo_medio"] = raw["total_consumption"] / raw["consuptiom_count"]

    # 10. Calculate eco flag
    if raw["driver_behavior"] == "cautious":
        raw["eco"] = True
    else:
        raw["eco"] = False

    # 11. Calculate heading
    raw['heading'] = calculate_heading(raw)

    return raw

# =========================
# Background loop (startup hook)
# =========================

@app.on_event("startup")
async def _startup():
    global _start_monotonic
    _start_monotonic = time.monotonic()  # marco zero da coleta
    asyncio.create_task(_main_loop_task())

async def _main_loop_task():
    """
    Loop Principal
    """
    while True:
        try:

            raw = read_test_snapshot() if TEST_MODE else read_obd_snapshot()

            processed = compute_features_and_predictions(raw)

            # print(processed)

            payload_interface = build_payload_interface(raw)

            await broadcast(translate_payload_values(payload_interface))
        except Exception as e:
            # soft log to all clients
            await broadcast({"erro": f"{type(e).__name__}: {e}"})
        await asyncio.sleep(max(0.0, SEND_INTERVAL_S))

# =========================
# Entrypoint
# =========================

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    run()
