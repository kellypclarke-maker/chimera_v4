from __future__ import annotations

import datetime as dt
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_ledger_path(*, cmd: str, league: str, date_iso: str) -> Path:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_cmd = str(cmd or "run").strip().lower() or "run"
    safe_lg = str(league or "na").strip().lower() or "na"
    safe_date = str(date_iso or "").strip() or "unknown-date"
    return Path("reports/ledger") / safe_date / f"{safe_cmd}_{safe_lg}_{ts}.jsonl"


@dataclass
class LedgerWriter:
    path: Path
    run_id: str

    @staticmethod
    def create(*, cmd: str, league: str, date_iso: str, path: Optional[Path] = None) -> "LedgerWriter":
        p = Path(path) if path is not None else _default_ledger_path(cmd=cmd, league=league, date_iso=date_iso)
        p.parent.mkdir(parents=True, exist_ok=True)
        run_id = os.urandom(8).hex()
        return LedgerWriter(path=p, run_id=run_id)

    def write(self, event: Dict[str, Any]) -> None:
        rec = dict(event or {})
        rec.setdefault("ts_utc", _utc_now_iso())
        rec.setdefault("run_id", self.run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")


