
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
from utils.translation import translate_payload_values
from utils.time_utils import LoopTimer
from utils.gps import get_gps_coordinates_async
from utils.trip_log import init_trip_log, save_row_dynamic, update_row_by_key
from services.alerts_service import init_alerts_index
from helpers.processed_factory import to_processed
from utils.metrics import RowMetrics
from utils.replay import CsvReplayer
from utils.csv_sanitize import sanitize_cell

# Agents
from agents.orchestrator import Orchestrator
from agents.advise_agent import advise_agent
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
REPLAY_CSV="./replays/trip_log_eclipse_0.csv"
# aceleração do tempo (2.0 = 2x, 0 = ignore timestamps e solta a cada SEND_INTERVAL_S)
REPLAY_SPEED=1.0
# reinicia quando terminar
REPLAY_LOOP=1
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
LLM_QUEUE: asyncio.Queue = asyncio.Queue()
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

async def safety_scheduler():
    """
    Periodically checks for PRF accidents/fines near the current GPS position and,
    when relevant, enqueues a background LLM job to generate a user message.
    - Non-blocking: never stalls the main data collection loop.
    - Backoff: avoids spamming when alerts persist.
    - Dedupe: prevents multiple LLM jobs for the same 'ts'.
    - Respects a minimum interval between LLM calls.

    Globals expected:
        - TRIP_LOG_FILE: str | PathLike (the trip CSV path; must be initialized in startup)
        - LATEST_STATE: dict (latest processed snapshot with keys: ts, speed, latitude, longitude, etc.)
        - LLM_QUEUE: asyncio.Queue (jobs: (ts, policy, alerts, snapshot))
        - LLM_ENQUEUED: set[str] (dedupe set for ts already enqueued)
        - _last_safety_alert_time: float (monotonic seconds for backoff control)
        - _last_llm_enqueued_ts: float | None (monotonic seconds for min-interval LLM)
        - SAFETY_CHECK_INTERVAL_S: float
        - SAFETY_ALERT_BACKOFF_S: float
        - LLM_MIN_INTERVAL_S: float
        - LLM: runtime or None (used indirectly inside the LLM worker)
    """
    import time, asyncio, random

    global _last_safety_alert_time, _last_llm_enqueued_ts
    global TRIP_LOG_FILE, LATEST_STATE, LLM_QUEUE, LLM_ENQUEUED

    # Ensure trip log is ready before attempting backfills/enqueues tied to 'ts'
    while TRIP_LOG_FILE is None:
        await asyncio.sleep(0.1)

    def _sleep_with_jitter(base_s: float) -> float:
        # small jitter to avoid temporal resonance when multiple tasks run
        return max(0.05, base_s + random.uniform(-0.25, 0.25))

    while True:
        try:
            # Shallow snapshot of the latest state (speed, lat, lon, ts, etc.)
            snap = dict(LATEST_STATE)
            lat = snap.get("latitude")
            lon = snap.get("longitude")
            spd = float(snap.get("speed") or 0.0)

            # If GPS not ready, wait and retry
            if lat is None or lon is None:
                await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))
                continue

            # Dynamic search radius based on speed
            radius = 500 if spd < 60 else 1000

            # Query safety agent (PRF accidents/fines around current location)
            alerts = await safety_agent_with_gps(spd, lat, lon, radius_m=radius)

            now = time.monotonic()

            if alerts:
                # Respect alert backoff to avoid spamming while user stays near the same hotspot
                if now - _last_safety_alert_time >= SAFETY_ALERT_BACKOFF_S:
                    _last_safety_alert_time = now

                    # Build a quick policy snapshot (reuse your behavior agent)
                    policy = await behavior_agent(to_processed(snap))

                    # Enqueue LLM job if we can associate to a row key (ts), not duplicated, and respecting min interval
                    ts = snap.get("ts")
                    can_call_llm = (_last_llm_enqueued_ts is None) or ((now - _last_llm_enqueued_ts) >= LLM_MIN_INTERVAL_S)
                    if ts and (ts not in LLM_ENQUEUED) and can_call_llm:
                        await LLM_QUEUE.put((ts, policy, alerts, snap))
                        LLM_ENQUEUED.add(ts)
                        _last_llm_enqueued_ts = now

                # When alerts exist, poll a bit faster but still with jitter (and backoff above)
                await asyncio.sleep(_sleep_with_jitter(min(SAFETY_CHECK_INTERVAL_S, 3.0)))
                continue

            # No alerts → check again on the normal interval
            await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))

        except Exception as e:
            import traceback
            print(f"[safety_scheduler] ERROR {type(e).__name__}\n{traceback.format_exc()}\n")
            # On error, wait the normal interval to avoid tight error loops
            await asyncio.sleep(_sleep_with_jitter(SAFETY_CHECK_INTERVAL_S))

async def llm_worker():
    """
    Consumes (row_id, policy, alerts, snapshot) jobs; calls the LLM (with retries);
    then backfills the CSV row keyed by 'row_id'. If the row is not yet present,
    it retries the backfill briefly so we don't lose the message.

    Expected globals:
      - TRIP_LOG_FILE: Path or str, trip CSV path (already initialized)
      - LLM: runtime object with .chat(...) used by advise_agent (may be None)
      - LLM_QUEUE: asyncio.Queue carrying tuples (row_id, policy, alerts, snap)
      - advise_agent(policy, alerts, llm) -> (message, source, meta) or (message, source)
      - update_row_by_key(path, key_col, key_val, updates) -> bool
      - broadcast(payload) -> websocket fanout (non-critical)
    """
    import asyncio, random, traceback

    global TRIP_LOG_FILE, LLM, LLM_QUEUE

    # Wait until trip log path is ready
    while TRIP_LOG_FILE is None:
        await asyncio.sleep(0.1)

    while True:
        row_id, policy, alerts, snap = await LLM_QUEUE.get()
        try:
            # Optional: lazy PID discovery for server metrics (non-blocking if it fails)
            try:
                if hasattr(LLM, "monitor_pid") and not getattr(LLM, "monitor_pid"):
                    from utils.proc_utils import find_pid_by_port
                    LLM.monitor_pid = find_pid_by_port(8080)  # adjust if server runs on another port
            except Exception:
                pass

            # 1) Generate message (up to 3 attempts with short backoff)
            attempts = 0
            final_msg, final_src, final_meta = "", "error", {}
            while attempts < 3:
                attempts += 1
                try:
                    # Accept (msg, src, meta) or (msg, src)
                    ret = await advise_agent(policy, alerts, LLM)
                    if isinstance(ret, tuple) and len(ret) == 3:
                        msg, src, meta = ret
                    elif isinstance(ret, tuple) and len(ret) == 2:
                        msg, src = ret
                        meta = {}
                    else:
                        msg, src, meta = "", "error", {}
                except Exception:
                    msg, src, meta = "", "error", {}

                if src == "model" and (msg or "").strip():
                    final_msg, final_src, final_meta = msg, src, (meta or {})
                    break
                else:
                    final_msg, final_src, final_meta = (msg or ""), (src or "error"), (meta or {})
                    await asyncio.sleep(0.5 * attempts + random.uniform(0.0, 0.25))

            # 2) Assemble updates (safe defaults)
            usage    = (final_meta or {}).get("usage")   or {}
            timings  = (final_meta or {}).get("timings") or {}
            proc     = (final_meta or {}).get("proc")    or {}
            msrc     = (final_meta or {}).get("metrics_source") or "unknown"

            updates = {
                "llm_message": sanitize_cell(final_msg),
                "llm_source": final_src,
                "llm_attempts": attempts,
                "llm_metrics_source": msrc,

                # usage (if the server provided it)
                "llm_prompt_tokens": usage.get("prompt_tokens"),
                "llm_completion_tokens": usage.get("completion_tokens"),
                "llm_total_tokens": usage.get("total_tokens"),

                # timings (server and/or client; your runtime may inject client-side)
                "llm_prompt_ms": timings.get("prompt_ms"),
                "llm_completion_ms": timings.get("completion_ms"),
                "llm_total_ms": timings.get("total_ms"),
                "llm_total_ms_client": timings.get("total_ms_client"),
                "llm_completion_tps_client": timings.get("completion_tps_client"),

                # server process CPU / memory sampled client-side (may be None if no PID)
                "llm_cpu_avg_pct": proc.get("cpu_avg_pct"),
                "llm_cpu_max_pct": proc.get("cpu_max_pct"),
                "llm_rss_peak_mb": proc.get("rss_peak_mb"),
                "llm_proc_samples": proc.get("samples"),
                "llm_proc_pid": proc.get("pid"),
            }

            print("[llm_worker] backfill row_id", row_id, "src=", final_src)  # ou row_id
            print("[llm_worker] monitor_pid =", getattr(LLM, "monitor_pid", None))


            # 3) Robust backfill by row_id (retry up to ~3s)
            ok = False
            for _ in range(30):  # 30 x 100ms = 3s
                ok = update_row_by_key(TRIP_LOG_FILE, "row_id", row_id, updates)
                if ok:
                    break
                await asyncio.sleep(0.1)

            if not ok:
                print(f"[llm_worker] WARN: row_id '{row_id}' not found for backfill after retries")

            # 4) Notify UI (best-effort)
            try:
                await broadcast({"row_id": row_id, **updates})
            except Exception:
                pass

        except Exception:
            print(f"[llm_worker] ERROR\n{traceback.format_exc()}\n")
        finally:
            LLM_QUEUE.task_done()

def compute_features_and_predictions(raw, rec: RowMetrics | None = None):

    rec = rec or RowMetrics()
    
    # 1. Calculate radar area
    raw['radar_area'] = calculate_radar_area({
        "rpm": raw["rpm"],
        "speed": raw["speed"],
        "throttle": raw["throttle"],
        "engine_load": raw["engine_load"]
    })

    # 2. Run TEDA model on radar area soft-sensor
    with rec.block("teda.run"):
        raw["teda_flag"] = teda.run([raw["radar_area"]])

    # 3. Run MMCloud to identify the driver profile
    with rec.block("mmcloud.process_point"):
        raw["driver_behavior"] = mmcloud.process_point([raw["radar_area"], raw["engine_load"]], 1)

    # 4. Identify fuel type (Gasoline or Ethanol)
    with rec.block("rf.fuel_type"):
        raw["fuel_type"], raw["fuel_type_prob"] = predict_fuel_type(raw)

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
        # 1) instancia o runtime
        LLM = LLMRuntimeOpenAI(
            base_url=base_url,
            model="local-model",
            max_tokens=48,
            temperature=0.1,
            timeout_s=8.0,
        )
        # 2) tenta descobrir/cachear o PID (sem quebrar se não achar)
        try:
            from utils.proc_utils import ensure_llm_pid
            pid = ensure_llm_pid(LLM, default_port=8080, attempts=6, sleep_s=0.3)
            print(f"[startup] LLM.monitor_pid = {pid}")
        except Exception as e:
            print(f"[startup] ensure_llm_pid erro: {e}")
    else:
        LLM = None
        print("[startup] LLM indisponível (server não respondeu). Seguindo sem métricas do LLM.")

    # ---- Orquestrador ----
    ORCH = Orchestrator(llm=LLM)

    # ---- Tasks (apenas uma do main loop!) ----
    asyncio.create_task(_main_loop_task())
    asyncio.create_task(llm_worker())
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
            global _last_llm_enqueued_ts, LATEST_STATE, LLM_QUEUE, ORCH, TRIP_LOG_FILE

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
            else:
                # Fallback se orquestrador não estiver pronto
                processed["policy_behavior"] = processed.get("driver_behavior", "Normal")
                processed["policy_severity"] = "low"
                enqueue_policy = None
                enqueue_alerts = []

            # ---------- Métricas do processamento ----------
            processed.update(rec.as_flat())  # m.* do compute_features...

            # ---------- Persistência (primeiro salva a linha) ----------
            save_row_dynamic(processed, TRIP_LOG_FILE)

            print("[saved]", processed.get("ts"))  # ou row_id, se você usar row_id

            # ---------- Enfileirar LLM (depois de salvar a linha), usando row_id ----------
            now = time.monotonic()
            if (
                LLM is not None
                and enqueue_policy is not None
                and (_last_llm_enqueued_ts is None or (now - _last_llm_enqueued_ts) >= LLM_MIN_INTERVAL_S)
            ):
                await LLM_QUEUE.put((rid, enqueue_policy, enqueue_alerts, dict(processed)))
                _last_llm_enqueued_ts = now

            # ---------- Estado corrente ----------
            LATEST_STATE.clear()
            LATEST_STATE.update(processed)

            # ---------- UI ----------
            payload_interface = build_payload_interface(raw)
            await broadcast(translate_payload_values(payload_interface))

        except Exception as e:
            err_type = type(e).__name__
            tb_str = traceback.format_exc()
            print(f"\n[ERRO {err_type}]\n{tb_str}\n")
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
