from __future__ import annotations

import csv
import datetime as dt
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from kalshi_core.clients.fees import maker_fee_dollars
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

FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPILFESL"
EXACT_RE = re.compile(r"exactly\s*(-?\d+(?:\.\d+)?)\s*%", re.IGNORECASE)


def _load_cached_fred_csv(path: Path, *, ttl_minutes: float) -> Optional[str]:
    if not path.exists():
        return None
    age_min = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)).total_seconds() / 60.0
    if age_min > float(ttl_minutes):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _fetch_fred_csv(*, path: Path, ttl_minutes: float) -> Optional[str]:
    cached = _load_cached_fred_csv(path, ttl_minutes=ttl_minutes)
    if cached is not None:
        return cached
    try:
        with requests.Session() as s:
            resp = s.get(FRED_CPI_URL, timeout=25.0)
            resp.raise_for_status()
            text = resp.text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text
    except Exception:
        return None


def _parse_fred_series(csv_text: str) -> List[Tuple[dt.date, float]]:
    out: List[Tuple[dt.date, float]] = []
    reader = csv.DictReader(csv_text.splitlines())
    value_col = "CPILFESL"
    for row in reader:
        d = str(row.get("DATE") or row.get("observation_date") or "").strip()
        v = str(row.get(value_col) or "").strip()
        if not d or not v or v == ".":
            continue
        try:
            dd = dt.date.fromisoformat(d)
            vv = float(v)
            out.append((dd, vv))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def _round_one_decimal(x: float) -> float:
    # Keep deterministic one-decimal bucketing.
    return float(round(float(x), 1))


def _mom_pmf(*, series: List[Tuple[dt.date, float]], lookback_years: int) -> Tuple[Dict[float, float], str, int]:
    if len(series) < 2:
        return ({}, "", 0)
    moms: List[Tuple[dt.date, float]] = []
    for i in range(1, len(series)):
        d, v = series[i]
        _, prev = series[i - 1]
        if prev == 0:
            continue
        moms.append((d, ((v / prev) - 1.0) * 100.0))
    if not moms:
        return ({}, "", 0)

    n = max(1, int(lookback_years) * 12)
    tail = moms[-n:]
    counts: Dict[float, int] = {}
    for _, x in tail:
        k = _round_one_decimal(x)
        counts[k] = int(counts.get(k, 0) + 1)
    total = sum(counts.values())
    if total <= 0:
        return ({}, "", 0)
    pmf = {k: float(v) / float(total) for k, v in counts.items()}
    asof = tail[-1][0].isoformat()
    return (pmf, asof, total)


def _is_cpi_core_market(row: Dict[str, str]) -> bool:
    blob = " ".join(
        [
            str(row.get("ticker") or ""),
            str(row.get("event_ticker") or ""),
            str(row.get("title") or ""),
            str(row.get("yes_sub_title") or ""),
        ]
    ).upper()
    return ("CPI" in blob) and ("CORE" in blob)


def _parse_exact_percent(yes_sub_title: str) -> Optional[float]:
    m = EXACT_RE.search(str(yes_sub_title or ""))
    if not m:
        return None
    try:
        return _round_one_decimal(float(m.group(1)))
    except Exception:
        return None


class EconSpecialist:
    name = "econ"
    categories = ["Economics", "Financials", "Companies", "Crypto"]

    def _cache_dir(self, context: RunContext) -> Path:
        return context.data_root / "external" / "econ"

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        out: List[CandidateProposal] = []
        cfg = context.config

        join_ticks = int(cfg.get("econ_join_ticks", 1))
        slippage_cents = float(cfg.get("econ_slippage_cents_per_contract", 0.25))
        min_ev = float(cfg.get("econ_min_ev_dollars", 0.0))
        lookback_years = int(cfg.get("econ_lookback_years", 10))
        fred_ttl = float(cfg.get("econ_cache_ttl_minutes", 1440.0))
        maker_fee_rate = float(cfg.get("maker_fee_rate", 0.0))
        min_yes_bid_cents = int(cfg.get("min_yes_bid_cents", 1))
        max_yes_spread_cents = int(cfg.get("max_yes_spread_cents", 10))

        cache_dir = self._cache_dir(context)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fred_csv_path = cache_dir / "fred_CPILFESL.csv"
        fred_text = _fetch_fred_csv(path=fred_csv_path, ttl_minutes=fred_ttl)
        if fred_text is None:
            return []

        series = _parse_fred_series(fred_text)
        pmf, asof, lookback_months = _mom_pmf(series=series, lookback_years=lookback_years)
        if not pmf:
            return []

        for row in context.markets:
            if category_from_row(row) not in self.categories:
                continue
            if not _is_cpi_core_market(row):
                continue
            exact = _parse_exact_percent(str(row.get("yes_sub_title") or ""))
            if exact is None:
                continue

            p_true = float(pmf.get(float(exact), 0.0))
            if not (0.0 <= p_true <= 1.0):
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
            slippage = float(slippage_cents) / 100.0
            ev = float(p_fill) * (float(p_true) - price - fees - slippage)
            if ev <= float(min_ev):
                continue

            rules_pointer, rules_missing = rules_pointer_from_row(row)
            out.append(
                CandidateProposal(
                    strategy_id="ECON_cpi_core_exact_maker",
                    ticker=str(row.get("ticker") or ""),
                    title=str(row.get("title") or ""),
                    category=category_from_row(row),
                    close_time=str(row.get("close_time") or ""),
                    event_ticker=str(row.get("event_ticker") or ""),
                    side="yes",
                    action="post_yes",
                    maker_or_taker="maker",
                    yes_price_cents=intended,
                    no_price_cents=None,
                    yes_bid=yes_bid,
                    yes_ask=yes_ask,
                    no_bid=safe_int(row.get("no_bid")),
                    no_ask=safe_int(row.get("no_ask")),
                    sum_prices_cents=None,
                    p_fill_assumed=float(p_fill),
                    fees_assumed_dollars=float(fees),
                    slippage_assumed_dollars=float(slippage),
                    ev_dollars=float(ev),
                    ev_pct=(float(ev) / price if price > 0 else float("nan")),
                    per_contract_notional=1.0,
                    size_contracts=1,
                    liquidity_notes=(
                        f"p_true={p_true:.4f};exact={exact:.1f};lookback_months={lookback_months};"
                        f"source=fred:CPILFESL;asof={asof}"
                    ),
                    risk_flags="model=empirical_pmf_cpi_core",
                    verification_checklist=(
                        "verify CPI core market parsing; verify FRED series freshness; "
                        "verify lookback window; verify maker fill/fee/slippage assumptions"
                    ),
                    rules_text_hash=rules_hash_from_row(row),
                    rules_missing=rules_missing,
                    rules_pointer=rules_pointer,
                    resolution_pointer=resolution_pointer_from_row(row),
                    market_url=str(row.get("market_url") or ""),
                )
            )
        return out

