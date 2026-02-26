from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from kalshi_core.models import Candidate


@dataclass(frozen=True)
class RunContext:
    as_of_ts: str
    report_date: str
    config: Dict[str, Any]
    data_root: Path
    markets: List[Dict[str, str]]
    events: Dict[str, Any]
    input_snapshot: str


@dataclass
class CandidateProposal:
    strategy_id: str
    ticker: str
    title: str
    category: str
    close_time: str
    event_ticker: str
    side: str
    action: str
    maker_or_taker: str
    yes_price_cents: Optional[int] = None
    no_price_cents: Optional[int] = None
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None
    sum_prices_cents: Optional[int] = None
    p_fill_assumed: Optional[float] = None
    fees_assumed_dollars: Optional[float] = None
    slippage_assumed_dollars: Optional[float] = None
    ev_dollars: Optional[float] = None
    ev_pct: Optional[float] = None
    per_contract_notional: float = 0.0
    size_contracts: int = 0
    liquidity_notes: str = ""
    risk_flags: str = ""
    verification_checklist: str = ""
    rules_text_hash: str = ""
    rules_missing: bool = True
    rules_pointer: str = ""
    resolution_pointer: str = ""
    market_url: str = ""
    bundle_tickers_json: str = ""

    def to_candidate(self, *, as_of_ts: str, input_snapshot: str) -> Candidate:
        return Candidate(
            strategy_id=self.strategy_id,
            ticker=self.ticker,
            title=self.title,
            category=self.category,
            close_time=self.close_time,
            event_ticker=self.event_ticker,
            side=self.side,
            action=self.action,
            maker_or_taker=self.maker_or_taker,
            yes_price_cents=self.yes_price_cents,
            no_price_cents=self.no_price_cents,
            yes_bid=self.yes_bid,
            yes_ask=self.yes_ask,
            no_bid=self.no_bid,
            no_ask=self.no_ask,
            sum_prices_cents=self.sum_prices_cents,
            p_fill_assumed=self.p_fill_assumed,
            fees_assumed_dollars=self.fees_assumed_dollars,
            slippage_assumed_dollars=self.slippage_assumed_dollars,
            ev_dollars=self.ev_dollars,
            ev_pct=self.ev_pct,
            per_contract_notional=self.per_contract_notional,
            size_contracts=self.size_contracts,
            liquidity_notes=self.liquidity_notes,
            risk_flags=self.risk_flags,
            verification_checklist=self.verification_checklist,
            rules_text_hash=self.rules_text_hash,
            rules_missing=self.rules_missing,
            rules_pointer=self.rules_pointer,
            resolution_pointer=self.resolution_pointer,
            market_url=self.market_url,
            data_as_of_ts=as_of_ts,
            input_snapshot=input_snapshot,
            bundle_tickers_json=self.bundle_tickers_json,
        )


class Specialist(Protocol):
    name: str
    categories: List[str]

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        ...

