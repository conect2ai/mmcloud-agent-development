# utils/gps.py
import os
import time
# import serial
import random
from typing import Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

MOCK_GPS = os.getenv("MOCK_GPS", "1") in {"1", "true", "True", "yes", "YES"}

def parse_GPGGA(sentence: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        parts = sentence.split(",")
        if parts[2] == '' or parts[4] == '':
            return None, None

        lat_deg = float(parts[2][:2]); lat_min = float(parts[2][2:]); lat_dir = parts[3]
        latitude = lat_deg + (lat_min / 60.0)
        if lat_dir == 'S': latitude = -latitude

        lon_deg = float(parts[4][:3]); lon_min = float(parts[4][3:]); lon_dir = parts[5]
        longitude = lon_deg + (lon_min / 60.0)
        if lon_dir == 'W': longitude = -longitude

        return latitude, longitude
    except Exception:
        return None, None

def get_gps_coordinates_sync(port="/dev/ttyAMA0", baudrate=9600, timeout=1.0) -> Tuple[Optional[float], Optional[float]]:
    if not MOCK_GPS:
        import serial
        with serial.Serial(port, baudrate, timeout=0.2) as ser:
            start_time = time.time()
            while time.time() - start_time < timeout:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA'):
                    lat, lon = parse_GPGGA(line)
                    if lat is not None and lon is not None:
                        return lat, lon
            return None, None

_executor: ThreadPoolExecutor | None = None

async def get_gps_coordinates_async(port="/dev/ttyAMA0", baudrate=9600, timeout=1.0):
    """
    Executa a leitura em pool de threads para não bloquear o loop.
    Em macOS, a porta costuma ser algo como '/dev/tty.usbmodemXXXX' ou '/dev/tty.usbserial-XXXX'.
    """
    global _executor
    if MOCK_GPS:
        # Simula uma pequena variação em torno de Natal-RN
        base_lat, base_lon = -5.7945, -35.211
        jitter = lambda x: x + random.uniform(-0.0005, 0.0005)
        return jitter(base_lat), jitter(base_lon)

    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_executor, get_gps_coordinates_sync, port, baudrate, timeout)