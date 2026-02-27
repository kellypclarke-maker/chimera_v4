from __future__ import annotations

import csv
from pathlib import Path

SHADOW_LEDGER_HEADERS = [
    "shadow_order_id",
    "created_ts",
    "strategy_id",
    "ticker",
    "event_ticker",
    "category",
    "side",
    "action",
    "maker_or_taker",
    "intended_price_cents",
    "assumed_fill_price_cents",
    "size_contracts",
    "fees_assumed_dollars",
    "slippage_assumed_dollars",
    "Gross_EV",
    "Net_EV",
    "Fee_Paid",
    "Execution_Mode",
    "rules_text_hash",
    "rules_pointer",
    "resolution_pointer",
    "status",
    "resolved_ts",
    "resolved_payout_dollars",
    "realized_pnl_dollars",
    "notes",
]


def init_shadow_ledger(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SHADOW_LEDGER_HEADERS)
        writer.writeheader()
