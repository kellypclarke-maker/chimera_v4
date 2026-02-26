from __future__ import annotations

import os


def live_trading_enabled(*, confirm: bool) -> bool:
    if not bool(confirm):
        return False
    return str(os.environ.get("KALSHI_TRADING_ENABLED") or "").strip() == "1"

