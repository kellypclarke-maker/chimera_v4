from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, Iterable

from .schema import SHADOW_LEDGER_HEADERS, init_shadow_ledger


def _normalize_row(row: dict) -> dict:
    out = {k: row.get(k, "") for k in SHADOW_LEDGER_HEADERS}
    for k in SHADOW_LEDGER_HEADERS:
        if out[k] is None:
            out[k] = ""
    return out


def load_ledger(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("shadow_order_id") or "").strip()
            if not sid:
                continue
            out[sid] = _normalize_row(row)
    return out


def _atomic_write_rows(path: Path, rows: Iterable[dict]) -> None:
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHADOW_LEDGER_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def upsert_rows(path: Path, rows: Iterable[dict]) -> None:
    init_shadow_ledger(path)
    existing = load_ledger(path)

    for row in rows:
        sid = str(row.get("shadow_order_id") or "").strip()
        if not sid:
            continue
        existing[sid] = _normalize_row(row)

    ordered = sorted(
        existing.values(),
        key=lambda r: (str(r.get("created_ts") or ""), str(r.get("shadow_order_id") or "")),
    )
    _atomic_write_rows(path, ordered)
