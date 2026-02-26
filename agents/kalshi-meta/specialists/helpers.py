from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional


def safe_int(x: object) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def maker_fill_prob(bid: Optional[int], ask: Optional[int], intended: Optional[int]) -> float:
    if intended is None or ask is None:
        return 0.0
    ask_i = int(ask)
    if ask_i <= 0:
        return 0.0

    intended_i = int(intended)
    if intended_i < 1 or intended_i >= ask_i:
        return 0.0

    bid_i = int(bid or 0)

    if bid_i <= 0:
        # Empty bid-side: allow posting at ask-1/1c but assume very low fill probability.
        dist_to_ask = max(0, ask_i - intended_i)
        if dist_to_ask <= 1:
            base = 0.20
        elif dist_to_ask <= 2:
            base = 0.10
        else:
            base = 0.05
        spread = ask_i
        spread_penalty = min(0.40, float(spread) / 100.0)
        return max(0.01, min(0.50, base - spread_penalty))

    if ask_i <= bid_i:
        return 0.0

    spread = max(0, ask_i - bid_i)
    dist_to_ask = max(0, ask_i - intended_i)
    if dist_to_ask <= 1:
        base = 0.65
    elif dist_to_ask <= 2:
        base = 0.45
    else:
        base = 0.25
    spread_penalty = min(0.35, float(spread) / 100.0)
    return max(0.05, min(0.95, base - spread_penalty))

def intended_maker_yes_price(*, yes_bid: Optional[int], yes_ask: Optional[int], join_ticks: int) -> Optional[int]:
    if yes_ask is None:
        return None
    ask_i = int(yes_ask)
    if ask_i <= 1:
        return None

    bid_i = int(yes_bid or 0)

    if bid_i <= 0:
        px = ask_i - 1
        return px if px >= 1 else None

    px = min(ask_i - 1, bid_i + int(join_ticks))
    if px < 1 or px >= ask_i:
        return None
    return px

def passes_maker_liquidity(
    *,
    yes_bid: Optional[int],
    yes_ask: Optional[int],
    min_yes_bid_cents: int,
    max_yes_spread_cents: int,
) -> bool:
    if yes_ask is None:
        return False
    ask_i = int(yes_ask)
    if ask_i <= 0:
        return False

    bid_i = int(yes_bid or 0)
    if bid_i < 0:
        return False

    if bid_i > 0 and bid_i < int(min_yes_bid_cents):
        return False

    spread = ask_i - bid_i
    if spread < 0:
        return False
    if spread > int(max_yes_spread_cents):
        return False

    # If there is no bid, only consider it "tradable" if the ask itself is not too wide.
    if bid_i == 0 and ask_i > int(max_yes_spread_cents):
        return False

    return True

def rules_hash_from_row(row: Dict[str, str]) -> str:
    h = str(row.get("rules_text_hash") or "").strip()
    if h:
        return h
    rp = str(row.get("rules_primary") or "")
    rs = str(row.get("rules_secondary") or "")
    return hashlib.sha256((rp + "\n" + rs).encode("utf-8")).hexdigest()


def resolution_pointer_from_row(row: Dict[str, str]) -> str:
    terms = str(row.get("series_contract_terms_url") or "").strip()
    if terms:
        return terms
    src = str(row.get("series_settlement_sources_json") or "").strip()
    if src:
        try:
            arr = json.loads(src)
            if isinstance(arr, list) and arr:
                first = arr[0]
                if isinstance(first, dict):
                    url = str(first.get("url") or "").strip()
                    if url:
                        return url
        except Exception:
            pass
    return ""


def rules_pointer_from_row(row: Dict[str, str]) -> tuple[str, bool]:
    rp = str(row.get("rules_primary") or "").strip()
    rs = str(row.get("rules_secondary") or "").strip()
    rules_missing = not (rp or rs)
    pointer = "rules_missing_in_market_fields" if rules_missing else "rules_primary/rules_secondary from snapshot"
    return (pointer, rules_missing)


def category_from_row(row: Dict[str, str]) -> str:
    return str(
        row.get("_series_category_auth")
        or row.get("series_category")
        or row.get("_event_category_auth")
        or row.get("event_category")
        or ""
    ).strip()


def read_cached_json(path: Path, *, ttl_minutes: float) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        fetched_at = str(payload.get("fetched_at") or "").strip()
        if not fetched_at:
            return None
        ts = dt.datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=dt.timezone.utc)
        age_min = (dt.datetime.now(dt.timezone.utc) - ts).total_seconds() / 60.0
        if age_min > float(ttl_minutes):
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def write_cached_json(path: Path, data: Dict[str, object]) -> None:
    payload = {
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data": data,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

