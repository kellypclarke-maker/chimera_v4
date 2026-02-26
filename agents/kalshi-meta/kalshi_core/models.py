from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RiskConfig:
    max_notional_per_market: float
    max_notional_per_category: float
    max_open_shadow_orders: int
    kill_switch_on_freshness_failure: bool = True


@dataclass(frozen=True)
class EventInfo:
    event_ticker: str
    series_ticker: str
    category: str
    title: str
    mutually_exclusive: bool
    collateral_return_type: str
    market_tickers: List[str]


@dataclass
class Candidate:
    strategy_id: str
    ticker: str
    title: str
    category: str
    close_time: str
    event_ticker: str
    side: str
    action: str
    maker_or_taker: str
    yes_price_cents: Optional[int]
    no_price_cents: Optional[int]
    yes_bid: Optional[int]
    yes_ask: Optional[int]
    no_bid: Optional[int]
    no_ask: Optional[int]
    sum_prices_cents: Optional[int]
    p_fill_assumed: Optional[float]
    fees_assumed_dollars: Optional[float]
    slippage_assumed_dollars: Optional[float]
    ev_dollars: Optional[float]
    ev_pct: Optional[float]
    per_contract_notional: float
    size_contracts: int
    liquidity_notes: str
    risk_flags: str
    verification_checklist: str
    rules_text_hash: str
    rules_missing: bool
    rules_pointer: str
    resolution_pointer: str
    market_url: str
    data_as_of_ts: str
    input_snapshot: str
    bundle_tickers_json: str = ""

    def to_row(self, rank: int) -> Dict[str, object]:
        return {
            "rank": rank,
            "strategy_id": self.strategy_id,
            "ticker": self.ticker,
            "title": self.title,
            "category": self.category,
            "close_time": self.close_time,
            "event_ticker": self.event_ticker,
            "side": self.side,
            "action": self.action,
            "maker_or_taker": self.maker_or_taker,
            "yes_price_cents": "" if self.yes_price_cents is None else self.yes_price_cents,
            "no_price_cents": "" if self.no_price_cents is None else self.no_price_cents,
            "yes_bid": "" if self.yes_bid is None else self.yes_bid,
            "yes_ask": "" if self.yes_ask is None else self.yes_ask,
            "no_bid": "" if self.no_bid is None else self.no_bid,
            "no_ask": "" if self.no_ask is None else self.no_ask,
            "sum_prices_cents": "" if self.sum_prices_cents is None else self.sum_prices_cents,
            "p_fill_assumed": "" if self.p_fill_assumed is None else round(float(self.p_fill_assumed), 6),
            "fees_assumed_dollars": "" if self.fees_assumed_dollars is None else round(float(self.fees_assumed_dollars), 6),
            "slippage_assumed_dollars": "" if self.slippage_assumed_dollars is None else round(float(self.slippage_assumed_dollars), 6),
            "ev_dollars": "" if self.ev_dollars is None else round(float(self.ev_dollars), 6),
            "ev_pct": "" if self.ev_pct is None else round(float(self.ev_pct), 6),
            "per_contract_notional": round(float(self.per_contract_notional), 6),
            "size_contracts": int(self.size_contracts),
            "liquidity_notes": self.liquidity_notes,
            "risk_flags": self.risk_flags,
            "verification_checklist": self.verification_checklist,
            "rules_text_hash": self.rules_text_hash,
            "rules_missing": str(bool(self.rules_missing)).lower(),
            "rules_pointer": self.rules_pointer,
            "resolution_pointer": self.resolution_pointer,
            "market_url": self.market_url,
            "data_as_of_ts": self.data_as_of_ts,
            "input_snapshot": self.input_snapshot,
            "bundle_tickers_json": self.bundle_tickers_json,
        }

