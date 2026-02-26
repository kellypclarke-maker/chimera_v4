from __future__ import annotations

import re
from typing import Dict, List, Optional

import requests

from kalshi_core.clients.fees import maker_fee_dollars
from kalshi_core.clients.kalshi_public import (
    discover_market_tickers_for_series_date,
    kalshi_date_token_from_iso_date,
)
from specialists.base import CandidateProposal, RunContext
from specialists.helpers import (
    category_from_row,
    intended_maker_yes_price,
    maker_fill_prob,
    passes_maker_liquidity,
    resolution_pointer_from_row,
    rules_hash_from_row,
    rules_pointer_from_row,
    safe_int,
)

_P_TRUE_RE = re.compile(r"\bp_true(?:_cal)?=([0-9]*\.?[0-9]+)")


def _safe_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        v = float(str(raw).strip())
    except Exception:
        return None
    if not (0.0 <= v <= 1.0):
        return None
    return float(v)


def _is_nhl_row(row: Dict[str, str]) -> bool:
    ticker = str(row.get("ticker") or "").strip().upper()
    event_ticker = str(row.get("event_ticker") or "").strip().upper()
    title = str(row.get("title") or "").strip().upper()
    category = category_from_row(row).strip().lower()
    return (
        ticker.startswith("KXNHL")
        or event_ticker.startswith("KXNHL")
        or " NHL " in f" {title} "
        or category == "sports"
    )


def _extract_p_true(row: Dict[str, str]) -> Optional[float]:
    direct = _safe_float(row.get("p_true"))
    if direct is not None:
        return direct
    notes = str(row.get("liquidity_notes") or "")
    for m in _P_TRUE_RE.finditer(notes):
        v = _safe_float(m.group(1))
        if v is not None:
            return v
    return None


def discover_nhl_startup_tickers(*, day_iso: str, config: Dict[str, object]) -> List[str]:
    token = kalshi_date_token_from_iso_date(day_iso)
    if not token:
        return []
    base_url = str(config.get("base_url") or "").strip()
    series = str(config.get("shadow_nhl_series_ticker", "KXNHLGAME")).strip().upper() or "KXNHLGAME"
    max_events = max(1, int(config.get("shadow_bootstrap_max_nhl_events", 60)))
    max_markets = max(1, int(config.get("shadow_bootstrap_max_nhl_markets", 120)))
    try:
        with requests.Session() as s:
            return discover_market_tickers_for_series_date(
                session=s,
                date_token=token,
                series_ticker=series,
                base_url=base_url,
                max_pages=10,
                max_events=max_events,
                max_markets_per_event=max_markets,
            )
    except Exception:
        return []


class NHLSpecialist:
    name = "nhl"
    categories = ["Sports"]

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        cfg = context.config
        join_ticks = int(cfg.get("nhl_join_ticks", cfg.get("sports_join_ticks", 1)))
        slippage_dollars = float(cfg.get("nhl_slippage_cents_per_contract", cfg.get("sports_slippage_cents_per_contract", 0.25))) / 100.0
        min_ev = float(cfg.get("nhl_min_ev_dollars", cfg.get("sports_min_ev_dollars", 0.0)))
        maker_fee_rate = float(cfg.get("maker_fee_rate", 0.0))
        min_yes_bid_cents = int(cfg.get("min_yes_bid_cents", 1))
        max_yes_spread_cents = int(cfg.get("max_yes_spread_cents", 10))

        out: List[CandidateProposal] = []
        for row in context.markets:
            if not _is_nhl_row(row):
                continue
            p_true = _extract_p_true(row)
            if p_true is None:
                continue

            yes_bid = safe_int(row.get("yes_bid"))
            yes_ask = safe_int(row.get("yes_ask"))
            if not passes_maker_liquidity(
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                min_yes_bid_cents=min_yes_bid_cents,
                max_yes_spread_cents=max_yes_spread_cents,
            ):
                continue

            intended = intended_maker_yes_price(yes_bid=yes_bid, yes_ask=yes_ask, join_ticks=join_ticks)
            if intended is None:
                continue
            p_fill = maker_fill_prob(yes_bid, yes_ask, intended)
            if p_fill <= 0.0:
                continue

            price = float(intended) / 100.0
            fees = maker_fee_dollars(contracts=1, price=price, rate=maker_fee_rate)
            ev = float(p_fill) * (float(p_true) - price - float(fees) - float(slippage_dollars))
            if ev <= min_ev:
                continue

            rules_pointer, rules_missing = rules_pointer_from_row(row)
            out.append(
                CandidateProposal(
                    strategy_id="NHL_live_maker",
                    ticker=str(row.get("ticker") or "").strip().upper(),
                    title=str(row.get("title") or ""),
                    category=category_from_row(row) or "Sports",
                    close_time=str(row.get("close_time") or ""),
                    event_ticker=str(row.get("event_ticker") or "").strip().upper(),
                    side="yes",
                    action="post_yes",
                    maker_or_taker="maker",
                    yes_price_cents=intended,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=safe_int(row.get("no_bid")),
                    no_ask=safe_int(row.get("no_ask")),
                    p_fill_assumed=float(p_fill),
                    fees_assumed_dollars=float(fees),
                    slippage_assumed_dollars=float(slippage_dollars),
                    ev_dollars=float(ev),
                    ev_pct=(float(ev) / price if price > 0 else 0.0),
                    per_contract_notional=1.0,
                    size_contracts=1,
                    liquidity_notes=f"p_true={p_true:.6f};sport=nhl",
                    risk_flags="sport=nhl",
                    verification_checklist="verify sportsbook/live model p_true and goalie/status updates before live usage",
                    rules_text_hash=rules_hash_from_row(row),
                    rules_missing=rules_missing,
                    rules_pointer=rules_pointer,
                    resolution_pointer=resolution_pointer_from_row(row),
                    market_url=str(row.get("market_url") or ""),
                )
            )
        return out
