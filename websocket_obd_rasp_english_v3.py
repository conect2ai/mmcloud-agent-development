import os
import time
import json
import random
import asyncio
import traceback
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timezone
from typing import Dict, Any, List

import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Utils
from utils.predictions import calculate_radar_area
from models.outlier_detection import TEDA
from models.mmcloud import MMCloud
from utils.predictions import predict_fuel_type, predict_city_highway
from utils.emissions import calculate_emissions_maf_afr, estimate_maf, _get_first
from utils.accelerometer import calculate_heading, mock_acelerometer
from utils.consumption import instant_fuel_consumption
from utils.commom import get_first, safe_int, safe_float, safe_round
from utils.heading import update_heading_deg, heading_deg_to_cardinal_pt
from utils.translation import translate_payload_values, build_heading_message_from_alerts
from utils.time_utils import LoopTimer
from utils.gps import get_gps_coordinates_async
from utils.trip_log import init_trip_log, save_row_dynamic, update_row_by_key
from services.alerts_service import init_alerts_index
from helpers.processed_factory import to_processed
from utils.metrics import RowMetrics
from utils.replay import CsvReplayer

# Agents
from agents.orchestrator import Orchestrator
from agents.behavior_agent import behavior_agent
from agents.safety_agent import safety_agent_with_gps
from nlg.llm_runtime_openai import LLMRuntimeOpenAI
from nlg.healthcheck import wait_llm_ready

# =========================
# Configuration
# =========================
TEST_MODE = os.getenv("TEST_MODE", "1") in {"1", "true", "True", "YES", "yes"}
SEND_INTERVAL_S = float(os.getenv("SEND_INTERVAL_S", "1.0"))
MOCK_ACC = os.getenv("MOCK_ACC", "1") in {"1", "true", "True", "yes", "YES"}
MOCK_GPS = os.getenv("MOCK_GPS", "1") in {"1", "true", "True", "yes", "YES"}
TRIP_DIR = Path("data/trips")
TRIP_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_ROWS = 0

TRIP_START_ISO = None      # ex.: '2025-10-21T11-03-27Z'
TRIP_LOG_FILE = None       # Path to CSV of the current trip
REPLAYER: CsvReplayer | None = None

# None to turn off and just use the fallback
LLM = None
ORCH = None

GPS_PORT = os.getenv("GPS_PORT", "/dev/ttyAMA0")

# liga o replay
REPLAY_MODE=1
# arquivo com os dados
REPLAY_CSV="./replays/exp1_driver1_car2.csv"
# aceleração do tempo (2.0 = 2x, 0 = ignore timestamps e solta a cada SEND_INTERVAL_S)
REPLAY_SPEED=1.0
# reinicia quando terminar
REPLAY_LOOP=0
# “tempo” do replay: file=reproduz ritmo pelo timestamp; realtime=ignora ts e usa intervalo fixo
REPLAY_CLOCK="file"   # ou realtime

ENGINE_VE = 1.1 # Turbo cars
ENGINE_DISPLACEMENT_L = 1.0

# --- Config ---
SAFETY_CHECK_INTERVAL_S = float(os.getenv("SAFETY_CHECK_INTERVAL_S", "8.0"))
SAFETY_ALERT_BACKOFF_S  = float(os.getenv("SAFETY_ALERT_BACKOFF_S", "20.0"))
LLM_MIN_INTERVAL_S      = float(os.getenv("LLM_MIN_INTERVAL_S", "12.0"))

# --- Estado compartilhado ---
LATEST_STATE = {}         # último processed/tick para safety usar
# LLM_QUEUE: asyncio.Queue = asyncio.Queue()
LAST_UI_PAYLOAD: Dict[str, Any] = {}
_last_llm_enqueued_ts: float | None = None
_last_safety_alert_time: float = 0.0

LLM_ENQUEUED: set[str] = set()  # evita jobs duplicados por ts

ROW_SEQ = 0

def next_row_id() -> int:
    global ROW_SEQ
    ROW_SEQ += 1
    return ROW_SEQ

# =========================
# Models
# =========================
teda = TEDA()
mmcloud = MMCloud(dimension=2, max_clusters=3)

# =========================
# Data collectors
# =========================

import os, time
from typing import Dict, Any

# --- Configuráveis por env (opcionais) ---
OBD_PORT = os.getenv("OBD_PORT")           # ex.: "/dev/ttyUSB0" ou "/dev/rfcomm0"
OBD_BAUD = int(os.getenv("OBD_BAUD", "0")) # 0 = auto
OBD_FAST = os.getenv("OBD_FAST", "1") not in ("0", "false", "False")

# Frequências (em segundos)
FAST_INTERVAL = float(os.getenv("OBD_FAST_INTERVAL_S", "0.3"))   # rpm/speed/throttle/load
SLOW_INTERVAL = float(os.getenv("OBD_SLOW_INTERVAL_S", "2.0"))   # temps, trims, etc.

# Chaves críticas para a sua pipeline (sempre presentes com default 0.0)
CRITICAL_KEYS = ("rpm", "speed", "throttle", "engine_load", "timing_advance", "ethanol_percentage", "intake_temp", "maf", "map")

# Estado cacheado (evita re-descobrir tudo a cada chamada)
__obd_conn = None
__supported: Dict[str, Any] = {}
__last_values: Dict[str, float] = {}
__last_ts: Dict[str, float] = {}

def __connect_obd():
    """Cria (ou reutiliza) a conexão OBD."""
    global __obd_conn
    import obd
    if __obd_conn and __obd_conn.is_connected():
        return __obd_conn
    try:
        kw = {}
        if OBD_PORT:
            kw["portstr"] = OBD_PORT
        if OBD_BAUD > 0:
            kw["baudrate"] = OBD_BAUD
        kw["fast"] = OBD_FAST
        __obd_conn = obd.OBD(**kw)
    except Exception:
        # Tenta um fallback mais tolerante
        try:
            __obd_conn = obd.OBD(fast=False)
        except Exception:
            __obd_conn = None
    return __obd_conn

def __discover_supported(conn):
    """Monta o dict de sensores suportados e cacheia."""
    global __supported
    if __supported:
        return __supported
    import obd
    # Grupo rápido
    sensors_fast = {
        "speed":          obd.commands.SPEED,
        "rpm":            obd.commands.RPM,
        "throttle":       obd.commands.THROTTLE_POS,
        "engine_load":    obd.commands.ENGINE_LOAD,
        "timing_advance": obd.commands.TIMING_ADVANCE,
        "intake_temp":    obd.commands.INTAKE_TEMP,
    }
    # Grupo lento
    sensors_slow = {
        "ethanol_percentage":getattr(obd.commands, "ETHANOL_PERCENT", 0.0),
        "coolant_temp":        obd.commands.COOLANT_TEMP,
        "maf":                 obd.commands.MAF,
        "map":                 obd.commands.INTAKE_PRESSURE,
        "baro":                obd.commands.BAROMETRIC_PRESSURE,
        "fuel_level":          obd.commands.FUEL_LEVEL,
        "air_fuel_ratio":      getattr(obd.commands, "AIR_FUEL_RATIO", None),
        "o2_b1s1":             getattr(obd.commands, "O2_B1S1_VOLTAGE", None),
        "battery_voltage":     obd.commands.CONTROL_MODULE_VOLTAGE,
        "ambient_temp":        obd.commands.AMBIANT_AIR_TEMP,
        "cat_temp1":           getattr(obd.commands, "CATALYST_TEMP_B1S1", None),
        "oil_temp":            getattr(obd.commands, "OIL_TEMP", None),
        "fuel_pressure":       getattr(obd.commands, "FUEL_PRESSURE", None),
        "distance_since_clear":getattr(obd.commands, "DISTANCE_SINCE_DTC_CLEAR", None),
    }
    # Remove None (pode não existir em versões antigas da lib)
    sensors_slow = {k:v for k,v in sensors_slow.items() if v is not None}
    # Interseção com suportados pela ECU
    try:
        supported_cmds = conn.supported_commands
        fast_ok = {k:v for k,v in sensors_fast.items() if v in supported_cmds}
        slow_ok = {k:v for k,v in sensors_slow.items() if v in supported_cmds}
    except Exception:
        # Se não conseguir listar, tenta tudo mesmo assim
        fast_ok, slow_ok = sensors_fast, sensors_slow
    __supported = {"fast": fast_ok, "slow": slow_ok}
    return __supported

def __to_float(val) -> float:
    """Converte objetos Unit/Ratio/String da python-OBD para float."""
    try:
        # objects com magnitude (pint)
        mag = getattr(val, "magnitude", None)
        if mag is not None:
            return float(mag)
        # às vezes é algo que já faz float(...)
        return float(val)
    except Exception:
        # último recurso: parse do início da string
        s = str(val)
        try:
            return float(s.split()[0].replace(",", "."))
        except Exception:
            return 0.0

def __query(conn, cmd):
    """Consulta um comando com tolerância a falhas."""
    try:
        resp = conn.query(cmd)
        if resp is None or resp.value is None:
            return None
        return __to_float(resp.value)
    except Exception:
        return None

def read_obd_snapshot() -> Dict[str, Any]:
    """
    Lê um snapshot OBD com:
      - apenas PIDs suportados,
      - cache por frequência (FAST_INTERVAL / SLOW_INTERVAL),
      - sem quebrar se falhar algo (defaults em CRITICAL_KEYS).
    """
    global __last_values, __last_ts
    now = time.monotonic()

    conn = __connect_obd()
    data: Dict[str, Any] = {}

    if not conn or not conn.is_connected():
        # Se não conectar, devolve valores anteriores (se houver) + defaults
        for k in set(__last_values.keys()) | set(CRITICAL_KEYS):
            data[k] = float(__last_values.get(k, 0.0))
        return data

    groups = __discover_supported(conn)

    # FAST group (sempre tenta dentro do intervalo)
    for key, cmd in groups["fast"].items():
        last_t = __last_ts.get(key, 0.0)
        if now - last_t >= FAST_INTERVAL or key not in __last_values:
            val = __query(conn, cmd)
            if val is not None:
                __last_values[key] = val
                __last_ts[key] = now
        data[key] = float(__last_values.get(key, 0.0))

    # SLOW group (só atualiza quando interval expirar)
    for key, cmd in groups["slow"].items():
        last_t = __last_ts.get(key, 0.0)
        if now - last_t >= SLOW_INTERVAL or key not in __last_values:
            val = __query(conn, cmd)
            if val is not None:
                __last_values[key] = val
                __last_ts[key] = now
        if key in __last_values:
            data[key] = float(__last_values[key])

    # Garantias mínimas p/ sua pipeline
    for k in CRITICAL_KEYS:
        data.setdefault(k, 0.0)

    # (Opcional) aliases amigáveis, se quiser manter compatibilidade com sua UI
    # data["bateria"] = data.get("battery_voltage", data.get("bateria", 0.0))
    # data["temperaturaMotor"] = data.get("coolant_temp", data.get("temperaturaMotor", 0.0))

    # print(data)  # debug
    return data

    
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
        "map": random.uniform(30, 100),          # kPa
        "intake_air_temp": random.uniform(20,40) # °C
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

async def broadcast(message: Dict[str, Any]):
    """
    Envia 'message' para todos os WebSockets conectados nesta app.
    Sobrescreve o broadcast importado de utils.websocket, garantindo
    que usamos a lista _connections local.
    """
    dead: List[WebSocket] = []
    for ws in list(_connections):
        try:
            await ws.send_json(message)
        except WebSocketDisconnect:
            dead.append(ws)
        except Exception:
            # Não derruba tudo se um cliente der erro
            try:
                await ws.close()
            except Exception:
                pass
            dead.append(ws)

    # Remove conexões mortas
    for ws in dead:
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
                # Keepalive: reutiliza o último payload completo + flag de ping
                try:
                    if LAST_UI_PAYLOAD:
                        payload_pt = dict(LAST_UI_PAYLOAD)
                    else:
                        payload_pt = {"status": "connected", "test_mode": TEST_MODE}
                    payload_pt["ping"] = datetime.now(timezone.utc).isoformat()
                    await ws.send_json(payload_pt)
                except Exception:
                    # Se der algum erro aqui, manda ao menos um ping simples
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
    global _heading_deg, LATEST_STATE
    dt_info = _timer.step()
    dt_s = dt_info["dt_s"]
    elapsed_s = 0.0 if _start_monotonic is None else (time.monotonic() - _start_monotonic)

    gyro = safe_float(get_first(raw, "gyro_z_dps", "gyro_z", "gyroZ", "gyro", default=0.0), 0.0)
    _heading_deg = update_heading_deg(_heading_deg, gyro, dt_s)
    heading_pt = heading_deg_to_cardinal_pt(_heading_deg)

    # Se o Orchestrator produziu uma mensagem de risco, usamos aqui
    heading_msg = None
    try:
        heading_msg = LATEST_STATE.get("heading_message")
    except Exception:
        heading_msg = None

    # Se existir mensagem, ela vai em 'heading'; senão usamos a direção mesmo
    heading_ui = heading_msg or heading_pt

    bateria            = safe_round(get_first(raw, "battery", "battery_voltage", "bateria"), 13, 2)
    temperatura_motor  = safe_int(get_first(raw, "engine_temp", "coolant_temp", "temperaturaMotor"), 90)
    tipo_combustivel   = get_first(raw, "fuel_type", "fuel", default="Gasoline")
    tipo_via           = get_first(raw, "road_type", "city_highway", "tipoVia", default="Desconhecida")
    co2_val            = safe_round(get_first(raw, "co2", "co2_emission_per_km"), 200, 2)
    perfil_motorista   = get_first(raw, "driver_profile", "driver_behavior", "perfilMotorista", default="Normal")
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
        "heading": heading_ui,
    }

    return payload_pt

async def safety_scheduler():
    """
    Periodically checks for PRF accidents/fines near the current GPS position and,
    when relevant, asks the Orchestrator to enqueue a background LLM job.

    IMPORTANT:
    - LLM_QUEUE / _last_llm_enqueued_ts / LLM_MIN_INTERVAL_S are no longer used.
    - The Orchestrator now handles dedupe, rate-limit and LLM calls.
    - safety_scheduler only:
        * queries GPS-based alerts
        * applies SAFETY_ALERT_BACKOFF_S
        * calls ORCH.enqueue_llm_job()
    """
    import time, asyncio, random

    global _last_safety_alert_time
    global TRIP_LOG_FILE, LATEST_STATE, ORCH

    # Ensure trip log exists before using row_id in callbacks
    while TRIP_LOG_FILE is None:
        await asyncio.sleep(0.1)

    def _sleep_with_jitter(base_s: float) -> float:
        return max(0.05, base_s + random.uniform(-0.25, 0.25))

    while True:
        try:
            # Snapshot do estado atual
            snap = dict(LATEST_STATE)
            lat = snap.get("latitude")
            lon = snap.get("longitude")
            spd = float(snap.get("speed") or 0.0)

            if lat is None or lon is None:
                await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))
                continue

            # Raio dinâmico baseado na velocidade
            radius = 500 if spd < 60 else 1000

            # Checa PRFs / multas / acidentes
            alerts = await safety_agent_with_gps(spd, lat, lon, radius_m=radius)

            now = time.monotonic()

            if alerts:
                # Backoff entre mensagens de safety
                if now - _last_safety_alert_time >= SAFETY_ALERT_BACKOFF_S:
                    _last_safety_alert_time = now

                    # Política rápida para contextualizar o evento
                    policy = await behavior_agent(to_processed(snap))

                    # Se o snapshot atual tem um row_id válido, usamos ele
                    row_id = snap.get("row_id")

                    if row_id is not None and ORCH is not None:
                        # Enfileira um job de LLM de forma segura
                        # O Orchestrator aplica rate-limit e dedupe interno
                        await ORCH.enqueue_llm_job(
                            row_id,
                            policy,
                            alerts,
                            snap,
                        )

                # Quando houver alertas, checar com intervalo menor
                await asyncio.sleep(
                    _sleep_with_jitter(min(SAFETY_CHECK_INTERVAL_S, 3.0))
                )
                continue

            # Sem alertas → intervalo normal
            await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))

        except Exception as e:
            import traceback
            print(f"[safety_scheduler] ERROR {type(e).__name__}\n{traceback.format_exc()}\n")
            await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))

def compute_features_and_predictions(raw, rec: RowMetrics | None = None):

    rec = rec or RowMetrics()
    
    # 1. Calculate radar area
    raw['radar_area'] = calculate_radar_area({
        "rpm": float(raw.get("rpm", 0.0)),
        "speed": float(raw.get("speed", 0.0)),
        "throttle": float(raw.get("throttle", 0.0)),
        "engine_load": float(raw.get("engine_load", 0.0)),
    })

    # print(raw['radar_area'])

    # 2. Run TEDA model on radar area soft-sensor
    with rec.block("teda.run"):
        raw["teda_flag"] = teda.run([raw["radar_area"]])

    # 3. Run MMCloud to identify the driver profile
    with rec.block("mmcloud.process_point"):
        raw["driver_behavior"] = mmcloud.process_point(COLLECTED_ROWS, [raw["radar_area"], raw["engine_load"]])

    # 4. Identify fuel type (Gasoline or Ethanol)
    with rec.block("rf.fuel_type"):
        raw["fuel_type"], raw["fuel_type_prob"] = predict_fuel_type(raw)

    # print(raw["fuel_type"])

    # 5.1 Get the accelerometer data and calculate the magnitude
    if MOCK_ACC:
        raw = mock_acelerometer(raw)
    else:
        from utils.accelerometer import read_acelerometer
        raw = read_acelerometer(raw)

    raw['accel_magnitude'] = raw["accel_x"]**2 + raw["accel_y"]**2 + raw["accel_z"]**2

    # 5.2 Identify city or highway
    with rec.block("rf.city_highway"):
        raw["city_highway_int"], raw["city_highway_prob"] = predict_city_highway(raw)

    if raw["city_highway_int"] == 0:
        raw["city_highway"] = "City"
    else:
        raw["city_highway"] = "Highway"

    # 6. Emissions estimation
    ## Estimate MAF if the doens't have
    maf_val = raw.get("maf")
    if maf_val is None:
        maf_val = estimate_maf(
            rpm=raw.get("rpm"),
            intake_temp_c=_get_first(raw, "iat", "intake_air_temp", "intake_temp", "ambient_temp"),
            intake_pressure_kpa=_get_first(raw, "map", "intake_pressure", "map_kpa", "MAP"),
            displacement_l=ENGINE_VE,  # defina por env/config
            ve=ENGINE_DISPLACEMENT_L
        )
        if maf_val is not None:
            raw["maf"] = maf_val
            raw["maf_estimated"] = True
        else:
            raw["maf_estimated"] = False
    raw = calculate_emissions_maf_afr(raw)

    # 7. Calculate instant fuel consuptiom
    try:
        raw['instant_fuel_consumption'] = instant_fuel_consumption(
            raw.get("speed", 0.0),
            maf=maf_val,
            rpm=raw.get("rpm"),
            map_value=_get_first(raw, "map", "intake_pressure", "map_kpa", "MAP"),
            iat=_get_first(raw, "iat", "intake_air_temp", "intake_temp", "ambient_temp"),
            vdm=ENGINE_DISPLACEMENT_L
        )
    except ValueError:
        raw['instant_fuel_consumption'] = 0.0

    # 8. Estimated distance
    if "total_distance" not in raw:
        raw["total_distance"] = 0.0
    raw["total_distance"] += raw["speed"] / 3600.0

    # 9. Average consumption (typos!)
    if "total_consumption" not in raw:
        raw["total_consumption"] = 0.0
        raw["consumption_count"] = 0

    raw["total_consumption"] += raw["instant_fuel_consumption"]
    raw["consumption_count"] += 1
    raw["average_consumption"] = raw["total_consumption"] / max(1, raw["consumption_count"])

    # 10. Calculate eco flag
    if raw["driver_behavior"] == "cautious":
        raw["eco"] = True
    else:
        raw["eco"] = False

    # 11. Calculate heading
    raw['heading'] = calculate_heading(raw)

    # Adding the metrics to the dict
    raw.update(rec.as_flat())

    return raw

async def on_llm_result(
    row_id: int,
    msg: str,
    src: str,
    meta: dict[str, Any],
    snapshot: dict[str, Any],
):
    """
    Callback chamado pelo Orchestrator sempre que o worker interno terminar
    uma chamada de LLM.
    Agora: além de atualizar o CSV, envia para a UI um payload COMPLETO
    (números + llm_*), reaproveitando o último payload da UI
    para não mexer no timer nem nos números.
    """
    global LATEST_STATE, LAST_UI_PAYLOAD

    try:
        # Sanitize
        txt = (msg or "").strip().replace("\n", " ").replace("\r", " ")
        src = str(src or "")
        meta = meta or {}

        usage   = (meta or {}).get("usage")   or {}
        timings = (meta or {}).get("timings") or {}
        proc    = (meta or {}).get("proc")    or {}

        # Atualiza CSV
        updates = {
            "llm_message": txt,
            "llm_source": src,

            # Latência (em segundos) e em ms, vindas do runtime_openai
            "llm_latency": meta.get("latency", None),
            # "llm_latency_ms": meta.get("latency_ms", None),

            # Tokens (mapeando corretamente pros campos do usage)
            "llm_total_tokens":  usage.get("total_tokens", None),
            "llm_input_tokens":  usage.get("prompt_tokens", None),
            "llm_output_tokens": usage.get("completion_tokens", None),

            # Timings do servidor/cliente
            # "llm_prompt_ms":        timings.get("prompt_ms", None),
            # "llm_completion_ms":    timings.get("completion_ms", None),
            # "llm_total_ms_server":  timings.get("total_ms", None),
            "llm_total_ms_client":  timings.get("total_ms_client", None),

            # Métricas de CPU/RAM do InferenceProfiler
            "llm_cpu_avg_pct":  proc.get("cpu_avg_pct", None),
            "llm_cpu_max_pct":  proc.get("cpu_max_pct", None),
            "llm_rss_peak_mb":  proc.get("rss_peak_mb", None),
            "llm_proc_samples": proc.get("samples", None),
            "llm_proc_pid":     proc.get("pid", None),

            # De onde vieram as métricas (server+client ou só client)
            "llm_metrics_source": meta.get("metrics_source", None),

            # Já existia
            "llm_agent_inserted_behavior_prf": meta.get("agent_inserted_behavior_prf"),
        }

        if updates["llm_output_tokens"] is not None and updates["llm_latency"] not in (None, 0):
            updates["llm_tokens_per_s"] = updates["llm_output_tokens"] / updates["llm_latency"]

        update_row_by_key(TRIP_LOG_FILE, "row_id", row_id, updates)

        # ---- Monta payload completo para UI reaproveitando o último estado ----
        if LAST_UI_PAYLOAD:
            payload_pt = dict(LAST_UI_PAYLOAD)
        else:
            # fallback raro: se ainda não houve loop, montamos algo básico
            base_state: Dict[str, Any] = dict(LATEST_STATE or snapshot)
            payload_interface = build_payload_interface(base_state)
            payload_pt = translate_payload_values(payload_interface)

        # Inclui os campos de LLM junto com os números
        payload_pt.update({
            "row_id": row_id,
            **updates,
        })

        await broadcast(payload_pt)

    except Exception:
        print("[on_llm_result] erro ao processar resultado do LLM")
        print(traceback.format_exc())
        print("[on_llm_result] erro ao processar resultado do LLM")
        print(traceback.format_exc())

# =========================
# Background loop (startup hook)
# =========================
@app.on_event("startup")
async def _startup():
    """
    - Inicializa relógio
    - Prepara TRIP log e (opcional) replayer
    - Sobe LLM (se disponível) e tenta descobrir o PID do server
    - Cria tasks de loop principal, worker do LLM e scheduler de safety
    """
    import os, time, asyncio
    from datetime import datetime, timezone

    global _start_monotonic, LLM, ORCH, REPLAYER, TRIP_LOG_FILE, _last_llm_enqueued_ts

    _start_monotonic = time.monotonic()
    _last_llm_enqueued_ts = None

    # ---- índices/estruturas auxiliares (ex.: spatial index PRF) ----
    await init_alerts_index()

    # ---- Trip log: sempre inicializa ----
    TRIP_LOG_FILE = init_trip_log(base_dir=os.getenv("TRIP_LOG_DIR", "./trips"))
    print(f"[trip] logging em: {TRIP_LOG_FILE}")

    # ---- Replay opcional ----
    if REPLAY_MODE:
        if not REPLAY_CSV:
            print("[replay] REPLAY_MODE=1 mas REPLAY_CSV não foi definido.")
            REPLAYER = None
        else:
            REPLAYER = CsvReplayer(
                path=REPLAY_CSV,
                colmap={},            # ajuste se headers diferirem
                ts_col="ts",
                clock=REPLAY_CLOCK,
                default_dt_s=SEND_INTERVAL_S,
                speed=REPLAY_SPEED,
                loop=REPLAY_LOOP,
            )
            print(f"[replay] ON: {REPLAY_CSV} speed={REPLAY_SPEED} clock={REPLAY_CLOCK}")
    else:
        REPLAYER = None

    # ---- LLM server (OpenAI-style via llama.cpp) ----
    base_url = "http://127.0.0.1:8080/v1"

        # ---- LLM server (OpenAI-style via llama.cpp) ----
    base_url = "http://127.0.0.1:8080/v1"

    try:
        is_ready = await wait_llm_ready(
            base_url=base_url,
            total_timeout_s=10.0,
            interval_s=0.5
        )
    except Exception as e:
        print(f"[startup] wait_llm_ready falhou: {e}")
        is_ready = False

    if is_ready:
        # 1) instancia o runtime (ainda sem monitor_pid definido)
        LLM = LLMRuntimeOpenAI(
            base_url=base_url,
            model="local-model",
            max_tokens=48,
            temperature=0.1,
            timeout_s=8.0,
            monitor_pid=None,  # explícito, só pra ficar claro
        )

        # 2) tenta descobrir/cachear o PID do servidor LLM
        try:
            from utils.proc_utils import find_llama_server_pid

            pid = None
            for _ in range(6):
                pid = find_llama_server_pid(getattr(LLM, "base_url", None), default_port=8080)
                if pid:
                    break
                time.sleep(0.3)

            if pid:
                LLM.monitor_pid = pid
                print(f"[startup] monitorando PID do LLM: {pid}")
            else:
                # fallback: monitora o próprio processo (menos ideal, mas melhor que nada)
                import os
                LLM.monitor_pid = os.getpid()
                print(f"[startup] não achei PID do server; usando o próprio processo {LLM.monitor_pid}")

        except Exception as e:
            print(f"[startup] ensure_llm_pid erro: {e}")
            # se quiser ser explícito:
            # LLM.monitor_pid = None
    else:
        LLM = None
        print("[startup] LLM indisponível (server não respondeu). Seguindo sem métricas do LLM.")

    # LLM=None
    # ---- Orquestrador ----
    ORCH = Orchestrator(
        llm=LLM,
        llm_min_interval_s=LLM_MIN_INTERVAL_S,  # mesmo valor que você usa hoje (12s)
        on_llm_result=on_llm_result,
    )

    await ORCH.start_background_tasks()

    # ---- Tasks (apenas uma do main loop!) ----
    asyncio.create_task(_main_loop_task())
    # asyncio.create_task(llm_worker())
    asyncio.create_task(safety_scheduler())


async def _main_loop_task():
    """
    Main Loop:
      1) Lê fonte (replay ou real) + GPS
      2) Processa features/predições (com RowMetrics por tick)
      3) Orquestrador (FSM) -> policy/alerts
      4) Enriquecimento de métricas
      5) Persiste a linha no CSV (save_row_dynamic)
      6) Só DEPOIS enfileira job do LLM usando row_id (chave estável)
      7) Atualiza estado e envia payload para UI
    """
    while True:
        try:
            global LATEST_STATE, ORCH, TRIP_LOG_FILE

            # ---------- Fonte de dados: replay ou real ----------
            if REPLAY_MODE and REPLAYER is not None:
                raw = REPLAYER.next_raw()
                if raw is None:
                    await asyncio.sleep(SEND_INTERVAL_S)
                    continue
            else:
                raw = read_test_snapshot() if TEST_MODE else read_obd_snapshot()

                # GPS real/mock (se houver). Não quebre se a porta não existir no Mac.
                try:
                    lat, lon = await get_gps_coordinates_async(port=GPS_PORT, baudrate=9600, timeout=0.5)
                    if lat is not None and lon is not None:
                        raw["latitude"], raw["longitude"] = lat, lon
                except Exception:
                    # GPS opcional
                    pass

            # ---------- Métricas por tick ----------
            rec = RowMetrics()

            # ---------- Processamento ----------
            processed = compute_features_and_predictions(raw, rec=rec)

            # Chave estável para backfill
            rid = next_row_id()
            processed["row_id"] = rid

            # Timestamp/coords (mantemos o ts para análises, mas a chave é o row_id)
            ts_iso = datetime.now(timezone.utc).isoformat()
            processed["ts"] = ts_iso
            processed["latitude"]  = raw.get("latitude")
            processed["longitude"] = raw.get("longitude")

            # ---------- Orquestrador ----------
            if ORCH is not None:
                orch_out = await ORCH.run_once(to_processed(processed))
                processed["policy_behavior"] = orch_out.policy.behavior
                processed["policy_severity"] = orch_out.policy.severity
                # Métricas dos agentes (se houver)
                if hasattr(orch_out, "metrics") and isinstance(orch_out.metrics, dict):
                    processed.update(orch_out.metrics)
                # Enfileirar LLM depois de salvar (mais abaixo)
                enqueue_policy = orch_out.policy
                enqueue_alerts = orch_out.alerts

                # Mensagem para a UI baseada em acidentes/multas próximos
                heading_msg = build_heading_message_from_alerts(enqueue_alerts)
                processed["heading_message"] = heading_msg
            else:
                # Fallback se orquestrador não estiver pronto
                processed["policy_behavior"] = processed.get("driver_behavior", "Normal")
                processed["policy_severity"] = "low"
                enqueue_policy = None
                enqueue_alerts = []
                processed["heading_message"] = None

            # ---------- Métricas do processamento ----------
            processed.update(rec.as_flat())  # m.* do compute_features...

            # ---------- Persistência (primeiro salva a linha) ----------
            save_row_dynamic(processed, TRIP_LOG_FILE)

            print("[saved]", processed.get("ts"))  # ou row_id, se você usar row_id

            # ---------- Enfileirar LLM (depois de salvar a linha), usando row_id ----------
            if enqueue_policy is not None:
                await ORCH.enqueue_llm_job(
                    rid,
                    enqueue_policy,
                    enqueue_alerts,
                    dict(processed),
                )

            # ---------- Estado corrente ----------
            LATEST_STATE.clear()
            LATEST_STATE.update(processed)

            # ---------- UI ----------
            payload_interface = build_payload_interface(raw)
            payload_pt = translate_payload_values(payload_interface)

            # cache do último payload completo enviado para a UI
            LAST_UI_PAYLOAD.clear()
            LAST_UI_PAYLOAD.update(payload_pt)

            await broadcast(payload_pt)

        except Exception as e:
            err_type = type(e).__name__
            tb_str = traceback.format_exc()
            print(f"\n[ERRO {err_type}]\n{tb_str}\n")

            try:
                # Usa o último payload conhecido e só acrescenta o erro
                if LAST_UI_PAYLOAD:
                    payload_pt = dict(LAST_UI_PAYLOAD)
                else:
                    payload_pt = {}
                payload_pt["erro"] = f"{err_type}: {e}"
                await broadcast(payload_pt)
            except Exception:
                # Se até aqui der erro, cai no modo simples mesmo
                await broadcast({"erro": f"{err_type}: {e}"})
        finally:
            await asyncio.sleep(max(0.0, SEND_INTERVAL_S))

# =========================
# Entrypoint
# =========================

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    run()
