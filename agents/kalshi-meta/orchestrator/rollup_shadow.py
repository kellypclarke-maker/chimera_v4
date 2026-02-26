#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
DEFAULT_LEDGER = REPO_ROOT / "shadow_ledger.csv"


@dataclass(frozen=True)
class ShadowLedgerRow:
    timestamp: str
    oracle_type: str
    ticker: str
    action: str
    price_cents: int
    quantity: int
    spot_price: Optional[float]
    expected_value: Optional[float]

    @property
    def expected_pnl(self) -> float:
        if self.expected_value is None:
            return 0.0
        return float(self.expected_value) * float(self.quantity)


def _safe_int(raw: object) -> Optional[int]:
    try:
        if raw is None:
            return None
        return int(float(str(raw).strip()))
    except Exception:
        return None


def _safe_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        v = float(str(raw).strip())
    except Exception:
        return None
    if not (v == v):  # NaN guard
        return None
    return float(v)


def _normalize_oracle_type(raw: object) -> str:
    s = str(raw or "").strip().lower()
    if s == "crypto":
        return "crypto"
    if s == "weather":
        return "weather"
    return "other"


def _parse_row(row: Dict[str, str]) -> Optional[ShadowLedgerRow]:
    price = _safe_int(row.get("price_cents"))
    qty = _safe_int(row.get("quantity"))
    if price is None or qty is None or price <= 0 or qty <= 0:
        return None
    return ShadowLedgerRow(
        timestamp=str(row.get("timestamp") or "").strip(),
        oracle_type=_normalize_oracle_type(row.get("oracle_type")),
        ticker=str(row.get("ticker") or "").strip().upper(),
        action=str(row.get("action") or "").strip(),
        price_cents=int(price),
        quantity=int(qty),
        spot_price=_safe_float(row.get("spot_price")),
        expected_value=_safe_float(row.get("expected_value")),
    )


def _load_rows(path: Path) -> tuple[List[ShadowLedgerRow], int]:
    rows: List[ShadowLedgerRow] = []
    skipped = 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            parsed = _parse_row(raw)
            if parsed is None:
                skipped += 1
                continue
            rows.append(parsed)
    return rows, skipped


def _group(rows: Iterable[ShadowLedgerRow]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        bucket = out.setdefault(
            r.oracle_type,
            {
                "trades": 0.0,
                "quantity": 0.0,
                "sum_expected_value": 0.0,
                "expected_value_count": 0.0,
                "net_expected_pnl": 0.0,
            },
        )
        bucket["trades"] += 1.0
        bucket["quantity"] += float(r.quantity)
        if r.expected_value is not None:
            bucket["sum_expected_value"] += float(r.expected_value)
            bucket["expected_value_count"] += 1.0
        bucket["net_expected_pnl"] += float(r.expected_pnl)
    return out


def _print_summary(*, path: Path, rows: List[ShadowLedgerRow], skipped: int) -> None:
    grouped = _group(rows)
    total_qty = sum(r.quantity for r in rows)
    total_net = sum(r.expected_pnl for r in rows)
    with_ev = sum(1 for r in rows if r.expected_value is not None)

    print("Shadow Ledger Rollup")
    print(f"ledger={path}")
    print(f"rows={len(rows)} skipped={skipped} rows_with_expected_value={with_ev}")
    print(f"net_simulated_pnl={total_net:.6f}")
    print(f"total_quantity={int(total_qty)}")
    print("")
    print("By Oracle Type")
    print("oracle_type | trades | quantity | avg_expected_value | net_simulated_pnl")
    print("----------- | ------ | -------- | ------------------ | -----------------")
    for oracle in ("crypto", "weather", "other"):
        if oracle not in grouped:
            continue
        g = grouped[oracle]
        ev_count = max(1.0, float(g["expected_value_count"]))
        avg_ev = float(g["sum_expected_value"]) / ev_count
        print(
            f"{oracle:11s} | "
            f"{int(g['trades']):6d} | "
            f"{int(g['quantity']):8d} | "
            f"{avg_ev:18.6f} | "
            f"{float(g['net_expected_pnl']):17.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Roll up root shadow_ledger.csv into net simulated PNL grouped by oracle type."
    )
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_LEDGER),
        help="Path to shadow_ledger.csv (default: repo-root shadow_ledger.csv).",
    )
    args = parser.parse_args()
    path = Path(str(args.ledger)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        print(f"ledger_missing={path}")
        return 1

    try:
        rows, skipped = _load_rows(path)
    except Exception as exc:
        print(f"rollup_failed={type(exc).__name__}:{exc}")
        return 1

    _print_summary(path=path, rows=rows, skipped=skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
