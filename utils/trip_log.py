# utils/triplog.py
import os, csv, json, tempfile, shutil
from typing import Dict, Any, Iterable, List, Tuple

TRIP_LOG_FILE: str | None = None
TRIP_HEADER: List[str] = []  # ordem das colunas atual
TRIP_DIR: str = "./trips"

# ---------- helpers de serialização/flatten ----------

def _serialize_value(v: Any) -> str | int | float | None:
    """
    CSV é texto; convertemos valores “não-primários” p/ JSON.
    """
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)

def _flatten(d: Dict[str, Any], parent: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Aplana dicts aninhados (ex.: emissions.something).
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            out.update(_flatten(v, key, sep))
        else:
            out[key] = v
    return out

# ---------- header / arquivo ----------

def _load_existing_header(path: str) -> List[str]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            return list(header)
        except StopIteration:
            return []

def _write_all_rows_with_header(path: str, header: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # escreve num arquivo temporário e troca atomically
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".trip_", suffix=".csv", dir=dir_name)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                row = {k: _serialize_value(r.get(k)) for k in header}
                w.writerow(row)
        os.replace(tmp_path, path)
    except Exception:
        # limpa temp se algo falhar
        with contextlib.suppress(Exception):
            os.remove(tmp_path)
        raise

def init_trip_log(base_dir: str = "./trips", filename: str | None = None) -> str:
    """
    Inicializa o log da viagem. Se filename não vier, cria com timestamp sanitizado.
    Retorna o caminho do arquivo.
    """
    global TRIP_LOG_FILE, TRIP_HEADER, TRIP_DIR
    TRIP_DIR = base_dir
    os.makedirs(TRIP_DIR, exist_ok=True)
    if filename is None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat().replace(":", "-").replace("T", "_")
        base = ts.split(".")[0].replace("Z","")
        filename = f"{base}_trip.csv"
    TRIP_LOG_FILE = os.path.join(TRIP_DIR, filename)
    TRIP_HEADER = _load_existing_header(TRIP_LOG_FILE)
    return TRIP_LOG_FILE

def save_row_dynamic(row: Dict[str, Any], path: str | None = None) -> None:
    """
    Salva uma linha no CSV, expandindo header se surgirem colunas novas.
    """
    global TRIP_LOG_FILE, TRIP_HEADER
    if path is None:
        path = TRIP_LOG_FILE
    if not path:
        print("[triplog] caminho não definido; chame init_trip_log() primeiro.")
        return

    # aplanar e coletar chaves
    flat = _flatten(row)
    keys_now = list(flat.keys())

    # header existente do arquivo (se mudou por outro processo)
    if not TRIP_HEADER:
        TRIP_HEADER = _load_existing_header(path)

    # se não houver header ainda, usa o conjunto corrente
    if not TRIP_HEADER:
        TRIP_HEADER = list(dict.fromkeys(sorted(keys_now)))  # ordenado e único
        _write_all_rows_with_header(path, TRIP_HEADER, [])
    
    # detectar colunas novas
    new_cols = [k for k in keys_now if k not in TRIP_HEADER]
    if new_cols:
        # reescreve o arquivo com header expandido
        old_rows: List[Dict[str, Any]] = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                for rrow in r:
                    old_rows.append(rrow)
        # expande header mantendo ordem (header antigo + novas ordenadas)
        TRIP_HEADER = TRIP_HEADER + sorted(new_cols)
        _write_all_rows_with_header(path, TRIP_HEADER, old_rows)

    # apende a nova linha
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRIP_HEADER, extrasaction="ignore")
        # se o arquivo for novo (sem header), escreve header
        if os.path.getsize(path) == 0:
            w.writeheader()
        row_out = {k: _serialize_value(flat.get(k)) for k in TRIP_HEADER}
        w.writerow(row_out)