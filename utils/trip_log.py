# utils/triplog.py
import os, csv, json, tempfile, shutil, contextlib
from datetime import datetime
from typing import Dict, Any, Iterable, List, Tuple
from utils.csv_sanitize import sanitize_cell
from pathlib import Path

TRIP_LOG_FILE: str | None = None
TRIP_HEADER: List[str] = []  # ordem das colunas atual
TRIP_DIR: str = "./trips"

_TRIP_PATH: Path | None = None
_FIELDS: List[str] = []  # header corrente (evolutivo)

# ---------- helpers de serialização/flatten ----------

def _serialize_value(v):
    if v is None or isinstance(v, (int, float, bool)):
        return v
    # strings: sanitize to single line
    if isinstance(v, str):
        return sanitize_cell(v)
    # dict/list/other: JSON numa linha
    try:
        import json
        return sanitize_cell(json.dumps(v, ensure_ascii=False))
    except Exception:
        return sanitize_cell(str(v))

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

def init_trip_log(base_dir="./trips") -> str:
    """
    Cria um novo arquivo de log de viagem no diretório especificado.
    O nome inclui data e hora (ex: trip_2025-10-24_14-40-15.csv).
    """
    os.makedirs(base_dir, exist_ok=True)
    ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"trip_{ts_str}.csv"
    path = os.path.join(base_dir, filename)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("")  # header dinâmico será criado na 1ª linha
    # (opcional) link simbólico para “trip_current.csv”
    try:
        link = os.path.join(base_dir, "trip_current.csv")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(filename, link)
    except OSError:
        pass
    return path

def _read_all_rows(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Reads entire CSV; returns (fieldnames, rows_as_dicts). If file absent/empty, returns ([], []).
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return [], []
    with p.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            return [], []
        rows = list(r)
        return list(r.fieldnames), rows
    
def _write_all_rows(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    """
    Overwrites CSV with the given header and rows. Ensures utf-8 and fsync.
    """
    # remove duplicatas mantendo ordem
    seen = set()
    ordered = []
    for k in fieldnames:
        k2 = k.strip()
        if k2 and k2 not in seen:
            seen.add(k2)
            ordered.append(k2)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            # normaliza: garante todas as chaves
            out = {k: row.get(k, "") for k in ordered}
            w.writerow(out)
        f.flush()
        os.fsync(f.fileno())

def _evolve_fields(current: List[str], new_keys: Iterable[str]) -> List[str]:
    """
    Returns the union of current header + new_keys, preserving order.
    Ensures 'row_id' (int) and 'ts' are early in the header if present.
    """
    base = list(current or [])
    # já deixe row_id e ts no começo se ainda não tiver header
    if not base:
        base = ["row_id", "ts"]  # força header inicial útil para backfill
    seen = set(k for k in base)
    for k in new_keys:
        k2 = str(k).strip()
        if k2 and k2 not in seen:
            base.append(k2)
            seen.add(k2)
    return base

def save_row_dynamic(row: Dict, path: str | Path | None = None) -> None:
    """
    Appends a row to the CSV, evolving header if new columns appear.
    Ensures flush+fsync before returning (para o worker achar a linha).
    """
    global _TRIP_PATH, _FIELDS
    if path is None:
        if _TRIP_PATH is None:
            raise RuntimeError("Trip log path is not initialized.")
        path = _TRIP_PATH
    path = str(path)
    fields, rows = _read_all_rows(path)

    # evolui header combinando já existente com novas chaves
    if not fields:
        fields = _evolve_fields([], row.keys())
    else:
        fields = _evolve_fields(fields, row.keys())

    # converte tudo pra string (CSV), mantendo chave ausente como ""
    row_str = {k: ("" if row.get(k) is None else str(row.get(k))) for k in fields}
    rows.append(row_str)

    _write_all_rows(path, fields, rows)
    _FIELDS = fields  # guarda header corrente

def update_row_by_key(path: str | Path, key_col: str, key_val, updates: Dict) -> bool:
    """
    Updates the FIRST row where str(row[key_col]).strip() == str(key_val).strip()
    Returns True if updated, False if not found.
    Evolves the header if updates bring new keys.
    """
    p = str(path)
    fields, rows = _read_all_rows(p)
    if not rows:
        return False

    # se a coluna-chave não existe ainda, não temos como localizar: retorna False
    if not fields or key_col not in fields:
        return False

    key_val_s = str(key_val).strip()

    # encontra a linha
    idx = -1
    for i, r in enumerate(rows):
        rv = str(r.get(key_col, "")).strip()
        if rv == key_val_s:
            idx = i
            break
    if idx < 0:
        return False

    # evolui header com novas chaves de updates (se for o caso)
    new_fields = _evolve_fields(fields, updates.keys())
    # aplica updates na linha-alvo
    for k, v in updates.items():
        rows[idx][k] = "" if v is None else str(v)

    _write_all_rows(p, new_fields, rows)
    return True