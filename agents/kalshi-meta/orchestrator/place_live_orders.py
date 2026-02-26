#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kalshi_core.clients.fees import maker_fee_dollars
from kalshi_core.clients.http import get_json
from kalshi_core.clients.kalshi_private import KalshiPrivateClient
from kalshi_core.config import load_config
from kalshi_core.envfile import load_env_file
from kalshi_core.safety import live_trading_enabled
from specialists.helpers import intended_maker_yes_price, maker_fill_prob, passes_maker_liquidity, safe_int

from specialists.weather.plugin import (
    WEATHER_MARKET_RE,
    _bucket_kind,
    _bucket_matches_comparator,
    _infer_comparator_hints,
    _parse_bucket_bounds,
)


DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
PTRUE_RE = re.compile(r"p_true=([0-9]*\.?[0-9]+)")
PTRUE_CAL_RE = re.compile(r"p_true_cal=([0-9]*\.?[0-9]+)")


@dataclass
class LiveOrderTicket:
    ticker: str
    event_ticker: str
    category: str
    strategy_id: str
    market_url: str
    close_time: str
    yes_bid: int
    yes_ask: int
    limit_yes_price_cents: int
    size_contracts: int
    p_true: float
    p_fill: float
    ev_dollars: float
    spread_cents: int
    model_source: str
    model_mu_f: Optional[float]
    model_sigma_f: Optional[float]
    model_forecast_ts: str
    model_target_date: str
    model_station_id: str
    model_lead_hours: Optional[float]
    model_disagreement_f: Optional[float]
    model_calibration_segment: str
    model_calibration_method: str
    model_p_true_raw: Optional[float]
    model_p_true_cal: Optional[float]
    model_notes_json: str
    rules_text_hash: str
    rules_pointer: str
    resolution_pointer: str


BALANCE_DOLLAR_KEYS = (
    "available_cash_dollars",
    "available_balance_dollars",
    "buying_power_dollars",
    "withdrawable_balance_dollars",
    "cash_balance_dollars",
    "cash_dollars",
    "balance_dollars",
)

BALANCE_GENERIC_KEYS = (
    "available_cash",
    "available_balance",
    "buying_power",
    "withdrawable_balance",
    "cash_balance",
    "cash",
    "balance",
)


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _resolve_day(date_arg: str) -> str:
    s = str(date_arg or "").strip()
    if s:
        return dt.date.fromisoformat(s).isoformat()
    return _utc_now().date().isoformat()


def _parse_ts(x: object) -> Optional[dt.datetime]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return None


def _as_float(x: object) -> Optional[float]:
    if isinstance(x, bool) or x is None:
        return None
    try:
        v = float(x)
    except Exception:
        try:
            v = float(str(x).strip())
        except Exception:
            return None
    if not math.isfinite(v):
        return None
    return float(v)


def _collect_numeric_fields(obj: object, *, out: Dict[str, List[float]]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).strip().lower()
            fv = _as_float(v)
            if fv is not None:
                out.setdefault(key, []).append(float(fv))
            _collect_numeric_fields(v, out=out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_numeric_fields(v, out=out)


def _to_dollars(*, key: str, value: float, units: str) -> float:
    mode = str(units or "auto").strip().lower()
    if mode == "dollars":
        return float(value)
    if mode == "cents":
        return float(value) / 100.0
    k = str(key or "").strip().lower()
    if "dollars" in k:
        return float(value)
    if "cents" in k:
        return float(value) / 100.0
    # Heuristic for integer-like cent balances.
    if abs(float(value)) >= 100000.0:
        return float(value) / 100.0
    return float(value)


def _extract_available_cash_dollars(payload: Dict[str, Any], *, units: str) -> Optional[float]:
    collected: Dict[str, List[float]] = {}
    _collect_numeric_fields(payload, out=collected)

    for key in BALANCE_DOLLAR_KEYS:
        vals = collected.get(key)
        if vals:
            return max(0.0, _to_dollars(key=key, value=float(vals[0]), units=units))
    for key in BALANCE_GENERIC_KEYS:
        vals = collected.get(key)
        if vals:
            return max(0.0, _to_dollars(key=key, value=float(vals[0]), units=units))
    return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_settlement_cli(*, day_key: str, ticker: str) -> Optional[Dict[str, Any]]:
    p = ROOT / "reports" / "research" / str(day_key) / str(ticker).strip().upper() / "settlement_cli.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _weather_p_true_from_cli(*, ticker: str, market_row: Dict[str, Any], cli_payload: Dict[str, Any]) -> Optional[float]:
    m = WEATHER_MARKET_RE.match(str(ticker or "").strip().upper())
    if not m:
        return None
    kind = str(m.group("kind") or "").strip().upper()

    try:
        observed_max = cli_payload.get("max_temp_f")
        observed_min = cli_payload.get("min_temp_f")
        observed = int(observed_max) if kind == "HIGH" else int(observed_min)
    except Exception:
        return None

    bounds = _parse_bucket_bounds(str(market_row.get("yes_sub_title") or "").strip())
    if bounds is None:
        return None
    lower, upper = bounds

    bucket_kind = _bucket_kind(lower=lower, upper=upper)
    hints = _infer_comparator_hints(
        " ".join([str(market_row.get("rules_primary") or ""), str(market_row.get("title") or "")])
    )
    if not _bucket_matches_comparator(bucket_kind=bucket_kind, hints=hints):
        return None

    if lower is not None and observed < float(lower):
        return 0.0
    if upper is not None and observed > float(upper):
        return 0.0
    return 1.0


def _parse_p_true(row: Dict[str, str]) -> Optional[float]:
    for key in ("liquidity_notes", "notes"):
        m = PTRUE_CAL_RE.search(str(row.get(key) or ""))
        if m:
            try:
                p = float(m.group(1))
                if 0.0 <= p <= 1.0:
                    return p
            except Exception:
                pass

    for key in ("liquidity_notes", "notes"):
        m = PTRUE_RE.search(str(row.get(key) or ""))
        if m:
            try:
                p = float(m.group(1))
                if 0.0 <= p <= 1.0:
                    return p
            except Exception:
                pass

    raw = str(row.get("p_true") or "").strip()
    if raw:
        try:
            p = float(raw)
            if 0.0 <= p <= 1.0:
                return p
        except Exception:
            return None
    return None


def _parse_notes_map(notes: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in str(notes or "").split(";"):
        p = str(part).strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = str(v).strip()
    return out


def _safe_float_str(raw: str) -> Optional[float]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _extract_model_snapshot(row: Dict[str, str]) -> Dict[str, str]:
    notes = _parse_notes_map(str(row.get("liquidity_notes") or ""))
    return {
        "source": str(notes.get("source") or ""),
        "mu": str(notes.get("mu") or ""),
        "sigma": str(notes.get("sigma") or ""),
        "forecast_ts": str(notes.get("forecast_ts") or ""),
        "target_date": str(notes.get("target_date") or ""),
        "station_id": str(notes.get("station_id") or ""),
        "lead_h": str(notes.get("sigma_lead_h") or ""),
        "disagreement_f": str(notes.get("ensemble_disagreement_f") or ""),
        "cal_segment": str(notes.get("cal_segment") or ""),
        "cal_method": str(notes.get("cal_method") or ""),
        "p_true_raw": str(notes.get("p_true_raw") or ""),
        "p_true_cal": str(notes.get("p_true_cal") or ""),
        "providers": str(notes.get("providers") or ""),
        "weights": str(notes.get("weights") or ""),
        "det_lock": str(notes.get("det_lock") or ""),
    }


def _conservative_fill_prob(
    *,
    base_p_fill: float,
    spread_cents: int,
    cfg: Dict[str, Any],
    strategy_id: str,
) -> float:
    p = max(0.0, min(1.0, float(base_p_fill)))
    if strategy_id.startswith("WTHR_"):
        mult = float(cfg.get("weather_conservative_fill_mult", 0.80))
        soft_cap = max(0, int(cfg.get("weather_conservative_fill_spread_soft_cap_cents", 3)))
        per_cent = max(0.0, float(cfg.get("weather_conservative_fill_spread_penalty_per_cent", 0.05)))
    elif strategy_id.startswith("ECON_"):
        mult = float(cfg.get("econ_conservative_fill_mult", 0.90))
        soft_cap = max(0, int(cfg.get("econ_conservative_fill_spread_soft_cap_cents", 4)))
        per_cent = max(0.0, float(cfg.get("econ_conservative_fill_spread_penalty_per_cent", 0.03)))
    else:
        mult = float(cfg.get("orders_conservative_fill_mult", 0.90))
        soft_cap = max(0, int(cfg.get("orders_conservative_fill_spread_soft_cap_cents", 4)))
        per_cent = max(0.0, float(cfg.get("orders_conservative_fill_spread_penalty_per_cent", 0.03)))

    p *= max(0.05, min(1.0, mult))
    spread = max(0, int(spread_cents))
    if spread > soft_cap:
        penalty = 1.0 - (float(spread - soft_cap) * per_cent)
        p *= max(0.10, penalty)
    return max(0.01, min(0.95, p))


def _candidate_action_ok(row: Dict[str, str]) -> bool:
    return str(row.get("action") or "").strip().lower() == "post_yes" and str(row.get("maker_or_taker") or "").strip().lower() == "maker"


def _infer_as_of_ts(rows: Sequence[Dict[str, str]]) -> Optional[dt.datetime]:
    best: Optional[dt.datetime] = None
    for row in rows:
        t = _parse_ts(row.get("data_as_of_ts"))
        if t is None:
            continue
        if best is None or t > best:
            best = t
    return best


def _select_candidates(
    *,
    day: str,
    top_n: int,
    min_ev_dollars: float,
    max_close_hours: float,
) -> Tuple[List[Dict[str, str]], dt.datetime]:
    daily_csv = ROOT / "reports" / "daily" / f"{day}_candidates.csv"
    rows = _read_csv_rows(daily_csv)
    if not rows:
        raise FileNotFoundError(f"missing or empty candidates csv: {daily_csv}")

    as_of = _infer_as_of_ts(rows) or _utc_now()
    latest_close = as_of + dt.timedelta(hours=float(max_close_hours))

    filtered: List[Dict[str, str]] = []
    for row in rows:
        if not _candidate_action_ok(row):
            continue
        ev_raw = str(row.get("ev_dollars") or "").strip()
        if not ev_raw:
            continue
        try:
            ev = float(ev_raw)
        except Exception:
            continue
        if ev < float(min_ev_dollars):
            continue
        close_t = _parse_ts(row.get("close_time"))
        if close_t is None:
            continue
        if close_t < as_of or close_t > latest_close:
            continue
        filtered.append(row)

    filtered.sort(key=lambda r: float(r.get("ev_dollars") or "-1e9"), reverse=True)
    buffer_n = max(1, int(top_n) * 10)
    return (filtered[:buffer_n], as_of)


def _candidate_row_map_for_day(day_key: str) -> Dict[str, Dict[str, str]]:
    daily_csv = ROOT / "reports" / "daily" / f"{day_key}_candidates.csv"
    rows = _read_csv_rows(daily_csv)
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if not _candidate_action_ok(row):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        prev = out.get(ticker)
        if prev is None:
            out[ticker] = row
            continue
        try:
            ev_new = float(str(row.get("ev_dollars") or "-1e9"))
        except Exception:
            ev_new = -1e9
        try:
            ev_prev = float(str(prev.get("ev_dollars") or "-1e9"))
        except Exception:
            ev_prev = -1e9
        if ev_new > ev_prev:
            out[ticker] = row
    return out


def _public_base(cfg: Dict[str, Any]) -> str:
    return (
        str(os.environ.get("KALSHI_PUBLIC_BASE") or "")
        or str(os.environ.get("KALSHI_BASE") or "")
        or str(os.environ.get("KALSHI_API_BASE") or "")
        or str(cfg.get("base_url") or "")
        or DEFAULT_BASE
    ).rstrip("/")


def _fetch_market(session: requests.Session, base: str, ticker: str) -> Dict[str, Any]:
    url = f"{base}/markets/{str(ticker).strip().upper()}"
    payload = get_json(session, url)
    market = payload.get("market")
    return market if isinstance(market, dict) else payload


def _slippage_dollars_for(row: Dict[str, str], cfg: Dict[str, Any], size_contracts: int) -> float:
    strategy_id = str(row.get("strategy_id") or "").strip()
    if strategy_id.startswith("WTHR_"):
        cents = float(cfg.get("weather_slippage_cents_per_contract", 0.25))
    elif strategy_id.startswith("ECON_"):
        cents = float(cfg.get("econ_slippage_cents_per_contract", 0.25))
    else:
        cents = float(cfg.get("s1b_slippage_cents_per_leg", 0.25))
    return max(0.0, float(size_contracts) * (cents / 100.0))


def _rules_fields(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(row.get("rules_text_hash") or "").strip(),
        str(row.get("rules_pointer") or "").strip(),
        str(row.get("resolution_pointer") or "").strip(),
    )


def _ticket_from_candidate(
    *,
    candidate_row: Dict[str, str],
    market_row: Dict[str, Any],
    cfg: Dict[str, Any],
    size_contracts: int,
    day_key: str,
) -> Optional[LiveOrderTicket]:
    ticker = str(candidate_row.get("ticker") or "").strip().upper()
    if not ticker:
        return None

    strategy_id = str(candidate_row.get("strategy_id") or "").strip()
    category = str(candidate_row.get("category") or "").strip()

    yes_bid = safe_int(market_row.get("yes_bid"))
    yes_ask = safe_int(market_row.get("yes_ask"))
    if yes_bid is None or yes_ask is None:
        return None

    spread_cents = max(0, int(yes_ask) - max(0, int(yes_bid)))

    if not passes_maker_liquidity(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        min_yes_bid_cents=int(cfg.get("min_yes_bid_cents", 1)),
        max_yes_spread_cents=int(cfg.get("max_yes_spread_cents", 10)),
    ):
        return None

    if category == "Climate and Weather":
        if ticker.startswith("KXLOWT") and not bool(cfg.get("weather_trade_low_enabled", True)):
            return None
        if ticker.startswith("KXHIGHT") and not bool(cfg.get("weather_trade_high_enabled", True)):
            return None
        weather_max_spread = int(cfg.get("weather_max_spread_cents", cfg.get("max_yes_spread_cents", 10)))
        if int(spread_cents) > max(0, int(weather_max_spread)):
            return None

    candidate_intended = safe_int(candidate_row.get("yes_price_cents"))
    intended: Optional[int] = None
    if candidate_intended is not None:
        ci = int(candidate_intended)
        if 1 <= ci < int(yes_ask):
            intended = ci
    if intended is None:
        if strategy_id.startswith("WTHR_"):
            join_ticks = int(cfg.get("weather_join_ticks", 1))
        elif strategy_id.startswith("ECON_"):
            join_ticks = int(cfg.get("econ_join_ticks", 1))
        else:
            join_ticks = int(cfg.get("s1b_maker_join_ticks", 1))
        intended = intended_maker_yes_price(yes_bid=yes_bid, yes_ask=yes_ask, join_ticks=join_ticks)
    if intended is None:
        return None

    p_true = _parse_p_true(candidate_row)
    if p_true is None:
        return None

    model_snapshot = _extract_model_snapshot(candidate_row)
    model_source = str(model_snapshot.get("source") or "")
    model_mu_f = _safe_float_str(str(model_snapshot.get("mu") or ""))
    model_sigma_f = _safe_float_str(str(model_snapshot.get("sigma") or ""))
    model_forecast_ts = str(model_snapshot.get("forecast_ts") or "")
    model_target_date = str(model_snapshot.get("target_date") or "")
    model_station_id = str(model_snapshot.get("station_id") or "")
    model_lead_hours = _safe_float_str(str(model_snapshot.get("lead_h") or ""))
    model_disagreement_f = _safe_float_str(str(model_snapshot.get("disagreement_f") or ""))
    model_calibration_segment = str(model_snapshot.get("cal_segment") or "")
    model_calibration_method = str(model_snapshot.get("cal_method") or "")
    model_p_true_raw = _safe_float_str(str(model_snapshot.get("p_true_raw") or ""))
    model_p_true_cal = _safe_float_str(str(model_snapshot.get("p_true_cal") or ""))

    # Weather: if we have observed temps from the settlement CLI report, override p_true to 0/1.
    if category == "Climate and Weather":
        cli = _load_settlement_cli(day_key=str(day_key), ticker=ticker)
        if isinstance(cli, dict):
            p_true_cli = _weather_p_true_from_cli(ticker=ticker, market_row=market_row, cli_payload=cli)
            if p_true_cli is not None:
                p_true = float(p_true_cli)
                model_source = (model_source + "+settlement_cli").strip("+")
                model_p_true_cal = float(p_true_cli)

    p_fill_base = maker_fill_prob(yes_bid, yes_ask, intended)
    p_fill = _conservative_fill_prob(
        base_p_fill=float(p_fill_base),
        spread_cents=int(spread_cents),
        cfg=cfg,
        strategy_id=strategy_id,
    )
    if p_fill <= 0.0 or p_fill_base <= 0.0:
        return None

    price = float(intended) / 100.0
    edge_per_contract = float(p_true) - float(price)
    if category == "Climate and Weather":
        min_model_edge = float(cfg.get("weather_min_model_edge_per_contract", 0.0))
        if edge_per_contract < float(min_model_edge):
            return None

    fee = maker_fee_dollars(contracts=int(size_contracts), price=price, rate=float(cfg.get("maker_fee_rate", 0.0)))
    slippage = _slippage_dollars_for(candidate_row, cfg, int(size_contracts))

    # Expected profit per contract is (p_true - price). Multiply by size and conservative maker fill.
    ev = float(p_fill) * float(size_contracts) * float(edge_per_contract) - float(fee) - float(slippage)

    rules_hash, rules_pointer, resolution_pointer = _rules_fields(candidate_row)
    model_notes_json = json.dumps(model_snapshot, sort_keys=True)

    return LiveOrderTicket(
        ticker=ticker,
        event_ticker=str(candidate_row.get("event_ticker") or "").strip().upper(),
        category=category,
        strategy_id=strategy_id,
        market_url=str(candidate_row.get("market_url") or "").strip(),
        close_time=str(candidate_row.get("close_time") or "").strip(),
        yes_bid=int(yes_bid),
        yes_ask=int(yes_ask),
        limit_yes_price_cents=int(intended),
        size_contracts=int(size_contracts),
        p_true=float(p_true),
        p_fill=float(p_fill),
        ev_dollars=float(ev),
        spread_cents=int(spread_cents),
        model_source=model_source,
        model_mu_f=model_mu_f,
        model_sigma_f=model_sigma_f,
        model_forecast_ts=model_forecast_ts,
        model_target_date=model_target_date,
        model_station_id=model_station_id,
        model_lead_hours=model_lead_hours,
        model_disagreement_f=model_disagreement_f,
        model_calibration_segment=model_calibration_segment,
        model_calibration_method=model_calibration_method,
        model_p_true_raw=model_p_true_raw,
        model_p_true_cal=model_p_true_cal,
        model_notes_json=model_notes_json,
        rules_text_hash=rules_hash,
        rules_pointer=rules_pointer,
        resolution_pointer=resolution_pointer,
    )


def _tickets_to_csv(path: Path, tickets: Sequence[LiveOrderTicket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "ticker",
                "event_ticker",
                "category",
                "strategy_id",
                "market_url",
                "close_time",
                "yes_bid",
                "yes_ask",
                "limit_yes_price_cents",
                "size_contracts",
                "p_true",
                "p_fill",
                "ev_dollars",
                "spread_cents",
                "model_source",
                "model_mu_f",
                "model_sigma_f",
                "model_forecast_ts",
                "model_target_date",
                "model_station_id",
                "model_lead_hours",
                "model_disagreement_f",
                "model_calibration_segment",
                "model_calibration_method",
                "model_p_true_raw",
                "model_p_true_cal",
                "model_notes_json",
                "rules_text_hash",
                "rules_pointer",
                "resolution_pointer",
            ],
        )
        w.writeheader()
        for t in tickets:
            w.writerow(
                {
                    "ticker": t.ticker,
                    "event_ticker": t.event_ticker,
                    "category": t.category,
                    "strategy_id": t.strategy_id,
                    "market_url": t.market_url,
                    "close_time": t.close_time,
                    "yes_bid": t.yes_bid,
                    "yes_ask": t.yes_ask,
                    "limit_yes_price_cents": t.limit_yes_price_cents,
                    "size_contracts": t.size_contracts,
                    "p_true": f"{t.p_true:.6f}",
                    "p_fill": f"{t.p_fill:.4f}",
                    "ev_dollars": f"{t.ev_dollars:.6f}",
                    "spread_cents": int(t.spread_cents),
                    "model_source": t.model_source,
                    "model_mu_f": "" if t.model_mu_f is None else f"{t.model_mu_f:.4f}",
                    "model_sigma_f": "" if t.model_sigma_f is None else f"{t.model_sigma_f:.4f}",
                    "model_forecast_ts": t.model_forecast_ts,
                    "model_target_date": t.model_target_date,
                    "model_station_id": t.model_station_id,
                    "model_lead_hours": "" if t.model_lead_hours is None else f"{t.model_lead_hours:.2f}",
                    "model_disagreement_f": "" if t.model_disagreement_f is None else f"{t.model_disagreement_f:.4f}",
                    "model_calibration_segment": t.model_calibration_segment,
                    "model_calibration_method": t.model_calibration_method,
                    "model_p_true_raw": "" if t.model_p_true_raw is None else f"{t.model_p_true_raw:.6f}",
                    "model_p_true_cal": "" if t.model_p_true_cal is None else f"{t.model_p_true_cal:.6f}",
                    "model_notes_json": t.model_notes_json,
                    "rules_text_hash": t.rules_text_hash,
                    "rules_pointer": t.rules_pointer,
                    "resolution_pointer": t.resolution_pointer,
                }
            )


def _write_live_results(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_client_order_id(*, day_key: str, ticker: str, price_cents: int, count: int) -> str:
    # Keep IDs compact and deterministic to avoid exchange-side invalid-parameter rejects.
    day_tok = re.sub(r"[^A-Za-z0-9]", "", str(day_key or ""))[:12] or "d"
    ticker_tok = re.sub(r"[^A-Za-z0-9]", "", str(ticker or "").upper())[:16] or "t"
    raw = f"{day_key}|{ticker}|{int(price_cents)}|{int(count)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"chimera-{day_tok}-{ticker_tok}-{digest}"


def _count_bought_yes_fills(client: KalshiPrivateClient, ticker: str, *, limit: int = 200) -> int:
    total = 0
    cursor = None
    for _ in range(10):
        resp = client.get_fills(ticker=str(ticker).strip().upper(), limit=int(limit), cursor=cursor)
        fills = resp.get("fills") if isinstance(resp, dict) else None
        if not isinstance(fills, list):
            break
        for f in fills:
            if not isinstance(f, dict):
                continue
            if str(f.get("action") or "").strip().lower() != "buy":
                continue
            if str(f.get("side") or "").strip().lower() != "yes":
                continue
            try:
                total += int(f.get("count") or 0)
            except Exception:
                continue
        cursor = resp.get("cursor") if isinstance(resp, dict) else None
        if not cursor:
            break
    return int(total)


def _portfolio_positions_by_ticker(client: KalshiPrivateClient) -> Dict[str, int]:
    try:
        payload = client.get_portfolio_positions()
    except Exception:
        return {}
    rows = payload.get("market_positions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        rows = payload.get("positions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}

    out: Dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or row.get("market_ticker") or "").strip().upper()
        if not ticker:
            continue
        pos = safe_int(row.get("position"))
        if pos is None:
            pos = safe_int(row.get("position_fp"))
        if pos is None:
            continue
        out[ticker] = int(pos)
    return out


def _net_yes_position_from_fills(client: KalshiPrivateClient, ticker: str, *, limit: int = 200) -> int:
    # Fallback-only estimator. Prefer _portfolio_positions_by_ticker where possible.
    # Kalshi fill payloads represent sell-side fills with inverted "side" semantics
    # relative to submitted order side, so action cannot be trusted for netting.
    # Empirically, side=yes increases YES exposure and side=no decreases it.
    net = 0
    cursor = None
    for _ in range(20):
        resp = client.get_fills(ticker=str(ticker).strip().upper(), limit=int(limit), cursor=cursor)
        fills = resp.get("fills") if isinstance(resp, dict) else None
        if not isinstance(fills, list):
            break
        for f in fills:
            if not isinstance(f, dict):
                continue
            side = str(f.get("side") or "").strip().lower()
            try:
                count = int(f.get("count") or 0)
            except Exception:
                continue
            if count <= 0:
                continue
            if side == "yes":
                net += count
            elif side == "no":
                net -= count
        cursor = resp.get("cursor") if isinstance(resp, dict) else None
        if not cursor:
            break
    return int(net)


def _open_sell_yes_qty_for_ticker(*, open_orders: Sequence[Dict[str, Any]], ticker: str, prefixes: Sequence[str]) -> int:
    target = str(ticker).strip().upper()
    total = 0
    for o in open_orders:
        if not isinstance(o, dict):
            continue
        cid = str(o.get("client_order_id") or "")
        if not any(cid.startswith(p) for p in prefixes):
            continue
        ot = str(o.get("ticker") or "").strip().upper()
        if ot != target:
            continue
        if str(o.get("action") or "").strip().lower() != "sell":
            continue
        side = str(o.get("side") or "").strip().lower()
        # Explicit unwind path only emits SELL YES.
        if side and side != "yes":
            continue
        qty = safe_int(o.get("remaining_count"))
        if qty is None:
            qty = safe_int(o.get("resting_count"))
        if qty is None:
            qty = safe_int(o.get("count"))
        if qty is None or qty <= 0:
            continue
        total += int(qty)
    return int(total)


def _remaining_yes_contracts_to_target(*, target_size: int, current_yes_position: int) -> Tuple[int, int]:
    target = max(0, int(target_size))
    pos = int(current_yes_position)
    long_yes = max(0, pos)
    remaining = max(0, target - long_yes)
    return (long_yes, remaining)


def _iter_open_orders(client: KalshiPrivateClient, *, limit: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = None
    for _ in range(20):
        resp = client.get_orders(status="open", limit=int(limit), cursor=cursor)
        orders = resp.get("orders") if isinstance(resp, dict) else None
        if not isinstance(orders, list):
            break
        for o in orders:
            if isinstance(o, dict):
                out.append(o)
        cursor = resp.get("cursor") if isinstance(resp, dict) else None
        if not cursor:
            break
    return out


def _client_order_id_prefixes(day_key: str) -> Tuple[str, ...]:
    # Match both current and legacy client_order_id formats.
    day_tok = re.sub(r"[^A-Za-z0-9]", "", str(day_key or ""))[:12] or "d"
    return (
        f"chimera-{day_tok}-",
        f"chimera-live-{str(day_key).strip()}-",
    )


def _prior_managed_tickers_from_live_csv(*, day_key: str, prefixes: Sequence[str]) -> List[str]:
    path = ROOT / "reports" / "live" / f"{day_key}_live_orders.csv"
    if not path.exists():
        return []
    out: List[str] = []
    seen: set[str] = set()
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ticker = str(row.get("ticker") or "").strip().upper()
                if not ticker:
                    continue
                cid = str(row.get("client_order_id") or "")
                if cid and not any(cid.startswith(p) for p in prefixes):
                    continue
                if ticker in seen:
                    continue
                seen.add(ticker)
                out.append(ticker)
    except Exception:
        return []
    return out


def _historical_buy_activity_from_live_csv(
    *,
    day_key: str,
    prefixes: Sequence[str],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, float], Dict[str, float]]:
    path = ROOT / "reports" / "live" / f"{day_key}_live_orders.csv"
    by_ticker_count: Dict[str, int] = {}
    by_category_count: Dict[str, int] = {}
    by_ticker_notional: Dict[str, float] = {}
    by_category_notional: Dict[str, float] = {}
    if not path.exists():
        return (by_ticker_count, by_category_count, by_ticker_notional, by_category_notional)
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("status") or "").strip().lower() != "placed":
                    continue
                cid = str(row.get("client_order_id") or "")
                if cid and not any(cid.startswith(p) for p in prefixes):
                    continue
                ticker = str(row.get("ticker") or "").strip().upper()
                category = str(row.get("category") or "").strip()
                px = safe_int(row.get("limit_yes_price_cents"))
                qty = safe_int(row.get("size_contracts"))
                if ticker:
                    by_ticker_count[ticker] = int(by_ticker_count.get(ticker, 0)) + 1
                if category:
                    by_category_count[category] = int(by_category_count.get(category, 0)) + 1
                if px is not None and qty is not None and px > 0 and qty > 0:
                    notional = (float(px) / 100.0) * float(qty)
                    if ticker:
                        by_ticker_notional[ticker] = float(by_ticker_notional.get(ticker, 0.0)) + float(notional)
                    if category:
                        by_category_notional[category] = float(by_category_notional.get(category, 0.0)) + float(notional)
    except Exception:
        return ({}, {}, {}, {})
    return (by_ticker_count, by_category_count, by_ticker_notional, by_category_notional)


def _trim_by_notional_cap(
    *,
    desired_count: int,
    limit_yes_price_cents: int,
    current_market_notional: float,
    current_category_notional: float,
    market_cap: float,
    category_cap: float,
) -> int:
    qty = max(0, int(desired_count))
    if qty <= 0:
        return 0
    per_contract = max(0.01, float(limit_yes_price_cents) / 100.0)
    if market_cap > 0:
        room_market = max(0.0, float(market_cap) - float(current_market_notional))
        qty = min(qty, int(room_market // per_contract))
    if category_cap > 0:
        room_cat = max(0.0, float(category_cap) - float(current_category_notional))
        qty = min(qty, int(room_cat // per_contract))
    return max(0, int(qty))


def _ticket_cost_per_contract_dollars(
    *,
    ticket: LiveOrderTicket,
    cfg: Dict[str, Any],
    price_buffer_dollars: float,
) -> float:
    price = max(0.01, float(ticket.limit_yes_price_cents) / 100.0)
    fee_one = maker_fee_dollars(contracts=1, price=price, rate=float(cfg.get("maker_fee_rate", 0.0)))
    return max(0.01, price + float(fee_one) + max(0.0, float(price_buffer_dollars)))


def _size_tickets_by_bankroll(
    *,
    tickets: Sequence[LiveOrderTicket],
    cfg: Dict[str, Any],
    available_cash_dollars: float,
    reserve_ratio: float,
    max_total_pct: float,
    max_order_pct: float,
    min_order_dollars: float,
    price_buffer_dollars: float,
) -> Tuple[List[LiveOrderTicket], Dict[str, float]]:
    available = max(0.0, float(available_cash_dollars))
    reserve = min(0.95, max(0.0, float(reserve_ratio)))
    total_pct = min(1.0, max(0.0, float(max_total_pct)))
    order_pct = min(1.0, max(0.0, float(max_order_pct)))
    min_order = max(0.0, float(min_order_dollars))

    deployable = max(0.0, available * (1.0 - reserve) * total_pct)
    per_order_budget = deployable if order_pct <= 0 else (deployable * order_pct)
    remaining_budget = float(deployable)

    resized: List[LiveOrderTicket] = []
    for t in tickets:
        desired = max(0, int(t.size_contracts))
        if desired <= 0:
            continue
        per_contract = _ticket_cost_per_contract_dollars(ticket=t, cfg=cfg, price_buffer_dollars=price_buffer_dollars)
        by_total = int(remaining_budget // per_contract) if per_contract > 0 else 0
        by_order = int(per_order_budget // per_contract) if per_contract > 0 else 0
        cap = min(desired, by_total, by_order)
        if cap <= 0:
            continue
        est_cost = float(cap) * per_contract
        if est_cost < min_order:
            continue
        remaining_budget = max(0.0, remaining_budget - est_cost)
        scale = (float(cap) / float(desired)) if desired > 0 else 0.0
        resized.append(
            LiveOrderTicket(
                ticker=t.ticker,
                event_ticker=t.event_ticker,
                category=t.category,
                strategy_id=t.strategy_id,
                market_url=t.market_url,
                close_time=t.close_time,
                yes_bid=t.yes_bid,
                yes_ask=t.yes_ask,
                limit_yes_price_cents=t.limit_yes_price_cents,
                size_contracts=int(cap),
                p_true=t.p_true,
                p_fill=t.p_fill,
                ev_dollars=float(t.ev_dollars) * scale,
                spread_cents=t.spread_cents,
                model_source=t.model_source,
                model_mu_f=t.model_mu_f,
                model_sigma_f=t.model_sigma_f,
                model_forecast_ts=t.model_forecast_ts,
                model_target_date=t.model_target_date,
                model_station_id=t.model_station_id,
                model_lead_hours=t.model_lead_hours,
                model_disagreement_f=t.model_disagreement_f,
                model_calibration_segment=t.model_calibration_segment,
                model_calibration_method=t.model_calibration_method,
                model_p_true_raw=t.model_p_true_raw,
                model_p_true_cal=t.model_p_true_cal,
                model_notes_json=t.model_notes_json,
                rules_text_hash=t.rules_text_hash,
                rules_pointer=t.rules_pointer,
                resolution_pointer=t.resolution_pointer,
            )
        )

    meta = {
        "available_cash_dollars": float(available),
        "deployable_dollars": float(deployable),
        "remaining_budget_dollars": float(remaining_budget),
        "reserve_ratio": float(reserve),
        "max_total_pct": float(total_pct),
        "max_order_pct": float(order_pct),
        "min_order_dollars": float(min_order),
    }
    return (resized, meta)




def main() -> int:
    parser = argparse.ArgumentParser(description="Place live Kalshi orders from top mispricing candidates (maker-only, safety-gated).")
    parser.add_argument("--config", default=str(ROOT / "config" / "defaults.json"), help="Path to config JSON.")
    parser.add_argument("--date", default="", help="Report date YYYY-MM-DD (default: UTC today).")
    parser.add_argument("--tag", default="", help="Optional output tag to select a specific daily run.")
    parser.add_argument("--top-n", type=int, default=5, help="How many candidates to consider.")
    parser.add_argument("--min-ev-dollars", type=float, default=0.01, help="Minimum recomputed EV in dollars to place.")
    parser.add_argument("--max-close-hours", type=float, default=72.0, help="Only place for markets closing within this horizon.")
    parser.add_argument("--size-contracts", type=int, default=10, help="Contracts per order.")
    parser.add_argument("--env-file", default="", help="Optional env file to load (default: repo env.list if present).")
    parser.add_argument("--confirm", action="store_true", help="Required to enable live placement (also needs KALSHI_TRADING_ENABLED=1).")
    parser.add_argument("--fill-aware", action="store_true", help="In live mode, only buy remaining contracts up to size_contracts per ticker.")
    parser.add_argument("--cancel-stale", action="store_true", help="In live mode, cancel open chimera orders for this day/tag that are no longer selected.")
    parser.add_argument("--cancel-replace", action="store_true", help="In live mode, cancel existing open chimera orders for selected tickers before placing updated limits.")
    parser.add_argument("--unwind-unselected", action="store_true", help="In live mode, place SELL YES limits for previously managed tickers that are no longer selected.")
    parser.add_argument("--unwind-max-contracts", type=int, default=0, help="Cap contracts per unwind order (0 means full detected net YES position).")
    parser.add_argument("--bankroll-aware", action="store_true", help="Scale per-ticket contract counts from available portfolio cash.")
    parser.add_argument("--bankroll-available-dollars", type=float, default=0.0, help="Optional manual available-cash override in dollars (<=0 means fetch from portfolio API).")
    parser.add_argument("--bankroll-balance-units", choices=("auto", "dollars", "cents"), default="auto", help="Units for numeric balance fields when parsing portfolio API response.")
    parser.add_argument("--bankroll-reserve-ratio", type=float, default=0.10, help="Keep this fraction of available cash unallocated (0..0.95).")
    parser.add_argument("--bankroll-max-total-pct", type=float, default=0.50, help="Max fraction of post-reserve cash to deploy this cycle (0..1).")
    parser.add_argument("--bankroll-max-order-pct", type=float, default=0.20, help="Max fraction of deployable cycle budget per order (0..1).")
    parser.add_argument("--bankroll-min-order-dollars", type=float, default=1.0, help="Minimum estimated dollars per order after sizing.")
    parser.add_argument("--bankroll-price-buffer-dollars", type=float, default=0.01, help="Safety buffer added to per-contract cost for bankroll sizing.")
    parser.add_argument("--max-net-yes-contracts-per-ticker", type=int, default=10, help="Do not buy beyond this net YES position per ticker (0 disables).")
    parser.add_argument("--max-buy-orders-per-ticker-per-day", type=int, default=1, help="Max BUY placements per ticker per day_key (0 disables).")
    parser.add_argument("--max-buy-orders-per-category-per-day", type=int, default=5, help="Max BUY placements per category per day_key (0 disables).")
    parser.add_argument("--max-new-notional-per-market", type=float, default=0.0, help="Max new BUY notional dollars per ticker per day_key (<=0 uses config max_notional_per_market).")
    parser.add_argument("--max-new-notional-per-category", type=float, default=0.0, help="Max new BUY notional dollars per category per day_key (<=0 uses config max_notional_per_category).")
    parser.add_argument("--no-derisk-negative-edge", action="store_true", help="Disable SELL YES de-risking when current model edge turns negative.")
    parser.add_argument("--derisk-edge-threshold-dollars", type=float, default=0.0, help="Per-contract edge threshold for de-risk SELL YES (default 0.0 => sell when edge < 0).")
    parser.add_argument("--derisk-max-contracts-per-ticker", type=int, default=0, help="Cap contracts per de-risk SELL YES order (0 means full eligible size).")
    args = parser.parse_args()

    # Load env without printing any values.
    env_path = Path(str(args.env_file).strip()) if str(args.env_file).strip() else (REPO_ROOT / "env.list")
    load_env_file(
        env_path,
        overwrite=False,
        allowlist={
            "KALSHI_API_KEY_ID",
            "KALSHI_PRIVATE_KEY_PATH",
            "KALSHI_API_PRIVATE_KEY",
            "KALSHI_API_BASE",
            "KALSHI_BASE",
            "KALSHI_PUBLIC_BASE",
        },
    )

    cfg = load_config(Path(args.config))
    day = _resolve_day(args.date)
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", str(args.tag or "").strip()).strip("-")
    suffix = f"_{tag}" if tag else ""
    day_key = f"{day}{suffix}"
    candidate_map = _candidate_row_map_for_day(day_key)

    candidates, as_of = _select_candidates(
        day=day_key,
        top_n=int(args.top_n),
        min_ev_dollars=float(cfg.get("shadow_min_ev_dollars", 0.0)) if float(args.min_ev_dollars) <= 0 else float(args.min_ev_dollars),
        max_close_hours=float(args.max_close_hours),
    )

    base = _public_base(cfg)
    session = requests.Session()

    tickets: List[LiveOrderTicket] = []
    for row in candidates:
        ticker = str(row.get("ticker") or "").strip().upper()
        try:
            market = _fetch_market(session, base, ticker)
        except Exception:
            continue
        t = _ticket_from_candidate(candidate_row=row, market_row=market, cfg=cfg, size_contracts=int(args.size_contracts), day_key=day_key)
        if t is None:
            continue
        if t.ev_dollars < float(args.min_ev_dollars):
            continue
        tickets.append(t)

    tickets.sort(key=lambda t: t.ev_dollars, reverse=True)
    tickets = tickets[: max(1, int(args.top_n))]

    if float(args.bankroll_available_dollars) < 0:
        raise ValueError("--bankroll-available-dollars must be >= 0")
    if not (0.0 <= float(args.bankroll_reserve_ratio) <= 0.95):
        raise ValueError("--bankroll-reserve-ratio must be in [0, 0.95]")
    if not (0.0 <= float(args.bankroll_max_total_pct) <= 1.0):
        raise ValueError("--bankroll-max-total-pct must be in [0, 1]")
    if not (0.0 <= float(args.bankroll_max_order_pct) <= 1.0):
        raise ValueError("--bankroll-max-order-pct must be in [0, 1]")
    if float(args.bankroll_min_order_dollars) < 0:
        raise ValueError("--bankroll-min-order-dollars must be >= 0")
    if float(args.bankroll_price_buffer_dollars) < 0:
        raise ValueError("--bankroll-price-buffer-dollars must be >= 0")
    if int(args.max_net_yes_contracts_per_ticker) < 0:
        raise ValueError("--max-net-yes-contracts-per-ticker must be >= 0")
    if int(args.max_buy_orders_per_ticker_per_day) < 0:
        raise ValueError("--max-buy-orders-per-ticker-per-day must be >= 0")
    if int(args.max_buy_orders_per_category_per_day) < 0:
        raise ValueError("--max-buy-orders-per-category-per-day must be >= 0")
    if float(args.max_new_notional_per_market) < 0:
        raise ValueError("--max-new-notional-per-market must be >= 0")
    if float(args.max_new_notional_per_category) < 0:
        raise ValueError("--max-new-notional-per-category must be >= 0")
    if int(args.derisk_max_contracts_per_ticker) < 0:
        raise ValueError("--derisk-max-contracts-per-ticker must be >= 0")

    client: Optional[KalshiPrivateClient] = None
    bankroll_meta: Dict[str, Any] = {}
    bankroll_error = ""
    if bool(args.bankroll_aware) and tickets:
        available_cash_dollars: Optional[float] = None
        if float(args.bankroll_available_dollars) > 0:
            available_cash_dollars = float(args.bankroll_available_dollars)
        else:
            try:
                client = KalshiPrivateClient(base_url=str(cfg.get("base_url") or ""))
                bal_payload = client.get_portfolio_balance()
                if isinstance(bal_payload, dict):
                    available_cash_dollars = _extract_available_cash_dollars(
                        bal_payload, units=str(args.bankroll_balance_units)
                    )
            except Exception as exc:
                bankroll_error = f"bankroll_fetch_error:{type(exc).__name__}"

        if available_cash_dollars is None:
            bankroll_error = bankroll_error or "bankroll_unavailable"
            tickets = []
        else:
            tickets, bankroll_meta = _size_tickets_by_bankroll(
                tickets=tickets,
                cfg=cfg,
                available_cash_dollars=float(available_cash_dollars),
                reserve_ratio=float(args.bankroll_reserve_ratio),
                max_total_pct=float(args.bankroll_max_total_pct),
                max_order_pct=float(args.bankroll_max_order_pct),
                min_order_dollars=float(args.bankroll_min_order_dollars),
                price_buffer_dollars=float(args.bankroll_price_buffer_dollars),
            )
            if not tickets and not bankroll_error:
                bankroll_error = "bankroll_sized_to_zero"

    out_tickets = ROOT / "reports" / "orders" / f"{day_key}_live_order_tickets.csv"
    _tickets_to_csv(out_tickets, tickets)

    enabled = live_trading_enabled(confirm=bool(args.confirm))

    results: List[Dict[str, Any]] = []
    if bool(args.bankroll_aware):
        if bankroll_meta:
            results.append(
                {
                    "ticker": "",
                    "client_order_id": "",
                    "order_id": "",
                    "status": "bankroll_info",
                    "available_cash_dollars": f"{float(bankroll_meta.get('available_cash_dollars', 0.0)):.2f}",
                    "deployable_dollars": f"{float(bankroll_meta.get('deployable_dollars', 0.0)):.2f}",
                    "remaining_budget_dollars": f"{float(bankroll_meta.get('remaining_budget_dollars', 0.0)):.2f}",
                }
            )
        if bankroll_error:
            results.append(
                {
                    "ticker": "",
                    "client_order_id": "",
                    "order_id": "",
                    "status": "bankroll_blocked",
                    "error": bankroll_error,
                }
            )

    if enabled:
        if client is None:
            client = KalshiPrivateClient(base_url=str(cfg.get("base_url") or ""))
        fill_aware = bool(args.fill_aware)
        max_net_yes = max(0, int(args.max_net_yes_contracts_per_ticker))
        derisk_enabled = not bool(args.no_derisk_negative_edge)
        derisk_threshold = float(args.derisk_edge_threshold_dollars)
        derisk_max_contracts = max(0, int(args.derisk_max_contracts_per_ticker))
        need_positions = bool(fill_aware) or (max_net_yes > 0) or bool(derisk_enabled) or bool(args.unwind_unselected)
        fill_positions: Dict[str, int] = {}
        if need_positions:
            try:
                fill_positions = _portfolio_positions_by_ticker(client)
            except Exception:
                fill_positions = {}
        cancel_stale = bool(args.cancel_stale)
        cancel_replace = bool(args.cancel_replace)
        unwind_unselected = bool(args.unwind_unselected)
        unwind_max_contracts = max(0, int(args.unwind_max_contracts))

        prefixes = _client_order_id_prefixes(day_key)
        market_cap = float(args.max_new_notional_per_market) if float(args.max_new_notional_per_market) > 0 else float(cfg.get("max_notional_per_market", 0.0))
        category_cap = float(args.max_new_notional_per_category) if float(args.max_new_notional_per_category) > 0 else float(cfg.get("max_notional_per_category", 0.0))
        max_buy_ticker_day = max(0, int(args.max_buy_orders_per_ticker_per_day))
        max_buy_category_day = max(0, int(args.max_buy_orders_per_category_per_day))
        hist_buy_by_ticker, hist_buy_by_category, hist_notional_by_ticker, hist_notional_by_category = _historical_buy_activity_from_live_csv(
            day_key=day_key,
            prefixes=prefixes,
        )
        run_buy_by_ticker: Dict[str, int] = {}
        run_buy_by_category: Dict[str, int] = {}
        run_notional_by_ticker: Dict[str, float] = {}
        run_notional_by_category: Dict[str, float] = {}
        open_orders = _iter_open_orders(client)
        selected_tickers = {t.ticker for t in tickets}
        managed_tickers: set[str] = set(selected_tickers)
        for o in open_orders:
            cid = str(o.get("client_order_id") or "")
            if not any(cid.startswith(p) for p in prefixes):
                continue
            ticker = str(o.get("ticker") or "").strip().upper()
            if ticker:
                managed_tickers.add(ticker)
        for ticker in _prior_managed_tickers_from_live_csv(day_key=day_key, prefixes=prefixes):
            managed_tickers.add(str(ticker).strip().upper())

        if derisk_enabled:
            open_orders = _iter_open_orders(client)
            positions_by_ticker = fill_positions if isinstance(fill_positions, dict) else _portfolio_positions_by_ticker(client)
            derisk_universe = sorted({str(t).strip().upper() for t in managed_tickers if str(t).strip()} | {str(t).strip().upper() for t in positions_by_ticker.keys()})
            for ticker in derisk_universe:
                net_yes = int(positions_by_ticker.get(ticker, 0))
                if net_yes <= 0:
                    continue
                row = candidate_map.get(ticker)
                if not isinstance(row, dict):
                    continue
                p_true = _parse_p_true(row)
                if p_true is None:
                    continue
                try:
                    market = _fetch_market(session, base, ticker)
                except Exception:
                    continue
                yes_bid = safe_int(market.get("yes_bid"))
                if yes_bid is None or int(yes_bid) < 1:
                    continue

                if str(row.get("category") or "").strip() == "Climate and Weather":
                    cli = _load_settlement_cli(day_key=str(day_key), ticker=ticker)
                    if isinstance(cli, dict):
                        p_true_cli = _weather_p_true_from_cli(ticker=ticker, market_row=market, cli_payload=cli)
                        if p_true_cli is not None:
                            p_true = float(p_true_cli)

                edge_per_contract = float(p_true) - (float(yes_bid) / 100.0)
                if edge_per_contract >= float(derisk_threshold):
                    continue

                pending_sell_yes = _open_sell_yes_qty_for_ticker(open_orders=open_orders, ticker=ticker, prefixes=prefixes)
                available_to_sell = max(0, int(net_yes) - int(pending_sell_yes))
                if available_to_sell <= 0:
                    results.append(
                        {
                            "ticker": ticker,
                            "category": str(row.get("category") or ""),
                            "order_id": "",
                            "status": "derisk_skipped_pending",
                            "client_order_id": "",
                            "net_yes_position": int(net_yes),
                            "pending_derisk_contracts": int(pending_sell_yes),
                            "edge_per_contract": f"{edge_per_contract:.6f}",
                            "threshold": f"{float(derisk_threshold):.6f}",
                        }
                    )
                    continue
                qty = int(available_to_sell) if derisk_max_contracts <= 0 else min(int(available_to_sell), int(derisk_max_contracts))
                if qty <= 0:
                    continue
                client_order_id = _build_client_order_id(
                    day_key=f"{day_key}-drk",
                    ticker=ticker,
                    price_cents=int(yes_bid),
                    count=int(qty),
                )
                try:
                    resp = client.place_order(
                        ticker=ticker,
                        side="yes",
                        action="sell",
                        count=int(qty),
                        price_cents=int(yes_bid),
                        client_order_id=client_order_id,
                    )
                    order = resp.get("order") if isinstance(resp, dict) else None
                    order_id = str(order.get("order_id") or "") if isinstance(order, dict) else ""
                    results.append(
                        {
                            "ticker": ticker,
                            "category": str(row.get("category") or ""),
                            "client_order_id": client_order_id,
                            "order_id": order_id,
                            "status": "derisk_placed",
                            "limit_yes_price_cents": int(yes_bid),
                            "size_contracts": int(qty),
                            "net_yes_position": int(net_yes),
                            "pending_derisk_contracts": int(pending_sell_yes),
                            "edge_per_contract": f"{edge_per_contract:.6f}",
                            "threshold": f"{float(derisk_threshold):.6f}",
                        }
                    )
                    open_orders.append(
                        {
                            "ticker": ticker,
                            "action": "sell",
                            "side": "yes",
                            "remaining_count": int(qty),
                            "client_order_id": client_order_id,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "ticker": ticker,
                            "category": str(row.get("category") or ""),
                            "client_order_id": client_order_id,
                            "order_id": "",
                            "status": "derisk_error",
                            "error": type(exc).__name__,
                            "limit_yes_price_cents": int(yes_bid),
                            "size_contracts": int(qty),
                            "net_yes_position": int(net_yes),
                            "pending_derisk_contracts": int(pending_sell_yes),
                            "edge_per_contract": f"{edge_per_contract:.6f}",
                            "threshold": f"{float(derisk_threshold):.6f}",
                        }
                    )

        if cancel_stale:
            for o in open_orders:
                cid = str(o.get("client_order_id") or "")
                ticker = str(o.get("ticker") or "").strip().upper()
                oid = str(o.get("order_id") or "")
                if not oid or not any(cid.startswith(p) for p in prefixes):
                    continue
                if ticker not in selected_tickers:
                    try:
                        client.cancel_order(order_id=oid)
                        results.append({"ticker": ticker, "order_id": oid, "status": "canceled_stale", "client_order_id": cid})
                    except Exception as exc:
                        results.append({"ticker": ticker, "order_id": oid, "status": "cancel_error", "client_order_id": cid, "error": type(exc).__name__})

        if cancel_replace:
            for o in open_orders:
                cid = str(o.get("client_order_id") or "")
                ticker = str(o.get("ticker") or "").strip().upper()
                oid = str(o.get("order_id") or "")
                if not oid or not any(cid.startswith(p) for p in prefixes):
                    continue
                if ticker in selected_tickers:
                    try:
                        client.cancel_order(order_id=oid)
                        results.append({"ticker": ticker, "order_id": oid, "status": "canceled_replace", "client_order_id": cid})
                    except Exception as exc:
                        results.append({"ticker": ticker, "order_id": oid, "status": "cancel_error", "client_order_id": cid, "error": type(exc).__name__})

        if unwind_unselected:
            # Refresh open orders after optional cancels so we only count still-resting orders.
            open_orders = _iter_open_orders(client)
            positions_by_ticker = _portfolio_positions_by_ticker(client)
            dropped = sorted(t for t in managed_tickers if t and t not in selected_tickers)
            for ticker in dropped:
                try:
                    net_yes = int(positions_by_ticker.get(str(ticker).strip().upper(), 0))
                    if net_yes == 0:
                        net_yes = _net_yes_position_from_fills(client, ticker)
                except Exception as exc:
                    results.append({"ticker": ticker, "order_id": "", "status": "unwind_error", "client_order_id": "", "error": type(exc).__name__})
                    continue
                if net_yes <= 0:
                    continue
                pending_sell_yes = _open_sell_yes_qty_for_ticker(open_orders=open_orders, ticker=ticker, prefixes=prefixes)
                available_to_unwind = max(0, int(net_yes) - int(pending_sell_yes))
                if available_to_unwind <= 0:
                    results.append(
                        {
                            "ticker": ticker,
                            "order_id": "",
                            "status": "unwind_skipped_pending",
                            "client_order_id": "",
                            "net_yes_position": int(net_yes),
                            "pending_unwind_contracts": int(pending_sell_yes),
                        }
                    )
                    continue

                qty = int(available_to_unwind) if unwind_max_contracts <= 0 else min(int(available_to_unwind), int(unwind_max_contracts))
                if qty <= 0:
                    continue
                try:
                    mkt = _fetch_market(session, base, ticker)
                    yes_bid = safe_int(mkt.get("yes_bid"))
                except Exception:
                    yes_bid = None
                if yes_bid is None or int(yes_bid) < 1:
                    results.append(
                        {
                            "ticker": ticker,
                            "order_id": "",
                            "status": "unwind_skipped_no_bid",
                            "client_order_id": "",
                            "size_contracts": qty,
                        }
                    )
                    continue
                client_order_id = _build_client_order_id(
                    day_key=f"{day_key}-uw",
                    ticker=ticker,
                    price_cents=int(yes_bid),
                    count=int(qty),
                )
                try:
                    resp = client.place_order(
                        ticker=ticker,
                        side="yes",
                        action="sell",
                        count=int(qty),
                        price_cents=int(yes_bid),
                        client_order_id=client_order_id,
                    )
                    order = resp.get("order") if isinstance(resp, dict) else None
                    order_id = str(order.get("order_id") or "") if isinstance(order, dict) else ""
                    results.append(
                        {
                            "ticker": ticker,
                            "client_order_id": client_order_id,
                            "order_id": order_id,
                            "status": "unwind_placed",
                            "limit_yes_price_cents": int(yes_bid),
                            "size_contracts": qty,
                            "net_yes_position": int(net_yes),
                            "pending_unwind_contracts": int(pending_sell_yes),
                        }
                    )
                    # Avoid duplicate unwind sizing inside this cycle.
                    open_orders.append(
                        {
                            "ticker": ticker,
                            "action": "sell",
                            "side": "yes",
                            "remaining_count": int(qty),
                            "client_order_id": client_order_id,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "ticker": ticker,
                            "client_order_id": client_order_id,
                            "order_id": "",
                            "status": "unwind_error",
                            "error": type(exc).__name__,
                            "limit_yes_price_cents": int(yes_bid),
                            "size_contracts": qty,
                            "net_yes_position": int(net_yes),
                            "pending_unwind_contracts": int(pending_sell_yes),
                        }
                    )

        for t in tickets:
            ticker_key = str(t.ticker).strip().upper()
            category_key = str(t.category).strip()
            model_fields = {
                "p_true": f"{t.p_true:.6f}",
                "p_fill": f"{t.p_fill:.6f}",
                "spread_cents": int(t.spread_cents),
                "model_source": t.model_source,
                "model_mu_f": "" if t.model_mu_f is None else f"{t.model_mu_f:.4f}",
                "model_sigma_f": "" if t.model_sigma_f is None else f"{t.model_sigma_f:.4f}",
                "model_forecast_ts": t.model_forecast_ts,
                "model_target_date": t.model_target_date,
                "model_station_id": t.model_station_id,
                "model_lead_hours": "" if t.model_lead_hours is None else f"{t.model_lead_hours:.2f}",
                "model_disagreement_f": "" if t.model_disagreement_f is None else f"{t.model_disagreement_f:.4f}",
                "model_calibration_segment": t.model_calibration_segment,
                "model_calibration_method": t.model_calibration_method,
                "model_p_true_raw": "" if t.model_p_true_raw is None else f"{t.model_p_true_raw:.6f}",
                "model_p_true_cal": "" if t.model_p_true_cal is None else f"{t.model_p_true_cal:.6f}",
                "model_notes_json": t.model_notes_json,
            }
            if max_buy_ticker_day > 0:
                prior_ticker_buys = int(hist_buy_by_ticker.get(ticker_key, 0)) + int(run_buy_by_ticker.get(ticker_key, 0))
                if prior_ticker_buys >= max_buy_ticker_day:
                    results.append(
                        {
                            "ticker": t.ticker,
                            "category": t.category,
                            "client_order_id": "",
                            "order_id": "",
                            "status": "cap_ticker_daily",
                            "max_buy_orders_per_ticker_per_day": max_buy_ticker_day,
                            "prior_buy_orders": prior_ticker_buys,
                            **model_fields,
                        }
                    )
                    continue
            if category_key and max_buy_category_day > 0:
                prior_category_buys = int(hist_buy_by_category.get(category_key, 0)) + int(run_buy_by_category.get(category_key, 0))
                if prior_category_buys >= max_buy_category_day:
                    results.append(
                        {
                            "ticker": t.ticker,
                            "category": t.category,
                            "client_order_id": "",
                            "order_id": "",
                            "status": "cap_category_daily",
                            "max_buy_orders_per_category_per_day": max_buy_category_day,
                            "prior_buy_orders": prior_category_buys,
                            **model_fields,
                        }
                    )
                    continue

            already = 0
            current_yes_position = 0
            remaining = int(t.size_contracts)
            if need_positions:
                try:
                    tk = str(t.ticker).strip().upper()
                    if tk in fill_positions:
                        current_yes_position = int(fill_positions.get(tk) or 0)
                    else:
                        current_yes_position = _net_yes_position_from_fills(client, t.ticker)
                except Exception:
                    current_yes_position = 0
                already, remaining = _remaining_yes_contracts_to_target(
                    target_size=int(t.size_contracts),
                    current_yes_position=int(current_yes_position),
                )
            if max_net_yes > 0:
                target_size = min(max_net_yes, max(0, int(t.size_contracts)))
                already, remaining = _remaining_yes_contracts_to_target(
                    target_size=int(target_size),
                    current_yes_position=int(current_yes_position),
                )
                if remaining <= 0:
                    results.append(
                        {
                            "ticker": t.ticker,
                            "category": t.category,
                            "client_order_id": "",
                            "order_id": "",
                            "status": "cap_net_position",
                            "max_net_yes_contracts_per_ticker": max_net_yes,
                            "current_yes_position": current_yes_position,
                            "remaining": remaining,
                            **model_fields,
                        }
                    )
                    continue

            remaining = _trim_by_notional_cap(
                desired_count=int(remaining),
                limit_yes_price_cents=int(t.limit_yes_price_cents),
                current_market_notional=float(hist_notional_by_ticker.get(ticker_key, 0.0)) + float(run_notional_by_ticker.get(ticker_key, 0.0)),
                current_category_notional=float(hist_notional_by_category.get(category_key, 0.0)) + float(run_notional_by_category.get(category_key, 0.0)),
                market_cap=float(market_cap),
                category_cap=float(category_cap),
            )

            if remaining <= 0:
                results.append(
                    {
                        "ticker": t.ticker,
                        "category": t.category,
                        "client_order_id": "",
                        "order_id": "",
                        "status": "cap_notional_or_filled",
                        "limit_yes_price_cents": t.limit_yes_price_cents,
                        "size_contracts": t.size_contracts,
                        "already_bought": already,
                        "current_yes_position": current_yes_position,
                        "remaining": remaining,
                        "ev_dollars": f"{t.ev_dollars:.6f}",
                        **model_fields,
                    }
                    )
                continue

            client_order_id = _build_client_order_id(
                day_key=day_key,
                ticker=t.ticker,
                price_cents=int(t.limit_yes_price_cents),
                count=int(remaining),
            )
            try:
                resp = client.place_order(
                    ticker=t.ticker,
                    side="yes",
                    action="buy",
                    count=int(remaining),
                    price_cents=int(t.limit_yes_price_cents),
                    client_order_id=client_order_id,
                )
                order = resp.get("order") if isinstance(resp, dict) else None
                order_id = str(order.get("order_id") or "") if isinstance(order, dict) else ""
                results.append(
                    {
                        "ticker": t.ticker,
                        "category": t.category,
                        "client_order_id": client_order_id,
                        "order_id": order_id,
                        "status": "placed",
                        "limit_yes_price_cents": t.limit_yes_price_cents,
                        "size_contracts": remaining,
                        "already_bought": already,
                        "current_yes_position": current_yes_position,
                        "remaining": remaining,
                        "ev_dollars": f"{t.ev_dollars:.6f}",
                        **model_fields,
                    }
                )
                placed_notional = (float(t.limit_yes_price_cents) / 100.0) * float(remaining)
                run_buy_by_ticker[ticker_key] = int(run_buy_by_ticker.get(ticker_key, 0)) + 1
                if category_key:
                    run_buy_by_category[category_key] = int(run_buy_by_category.get(category_key, 0)) + 1
                run_notional_by_ticker[ticker_key] = float(run_notional_by_ticker.get(ticker_key, 0.0)) + float(placed_notional)
                if category_key:
                    run_notional_by_category[category_key] = float(run_notional_by_category.get(category_key, 0.0)) + float(placed_notional)
            except Exception as exc:
                results.append(
                    {
                        "ticker": t.ticker,
                        "category": t.category,
                        "client_order_id": client_order_id,
                        "order_id": "",
                        "status": "error",
                        "error": type(exc).__name__,
                        "limit_yes_price_cents": t.limit_yes_price_cents,
                        "size_contracts": remaining,
                        "already_bought": already,
                        "current_yes_position": current_yes_position,
                        "remaining": remaining,
                        "ev_dollars": f"{t.ev_dollars:.6f}",
                        **model_fields,
                    }
                )

    out_results = ROOT / "reports" / "live" / f"{day_key}_live_orders.csv"
    _write_live_results(out_results, results)

    print(f"as_of_ts={as_of.isoformat()}")
    print(f"candidates_considered={len(candidates)}")
    print(f"tickets_written={out_tickets}")
    print(f"tickets_selected={len(tickets)}")
    print(f"live_enabled={enabled}")
    if enabled:
        print(f"live_results_written={out_results}")
    else:
        print("dry_run=1 (set KALSHI_TRADING_ENABLED=1 and pass --confirm to place real orders)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

