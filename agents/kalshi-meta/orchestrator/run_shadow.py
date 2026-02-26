#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import argparse
import csv
import re
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print(f"[DEBUG] Python Path: {sys.path[0]}")

from kalshi_core.clients.fees import maker_fee_dollars, taker_fee_dollars
from kalshi_core.clients.kalshi_public import (
    discover_viable_crypto_tickers_for_date,
    discover_sports_tickers_for_date,
    discover_weather_tickers_for_date,
    fetch_event as fetch_public_event,
    get_markets as fetch_public_markets,
    fetch_open_event_tickers_for_date,
    extract_kalshi_date_token_from_ticker,
    fetch_btc_spot_rest_fallback,
)
from kalshi_core.clients.kalshi_ws import ws_collect_ticker_snapshot
from kalshi_core.ledger import SHADOW_LEDGER_HEADERS, load_ledger, upsert_rows
from specialists.helpers import (
    intended_maker_yes_price,
    maker_fill_prob,
    passes_maker_liquidity,
    safe_int,
)
from specialists.econ.plugin import (
    _fetch_fred_csv,
    _is_cpi_core_market,
    _mom_pmf,
    _parse_exact_percent,
    _parse_fred_series,
)
from specialists.weather.plugin import (
    WEATHER_MARKET_RE,
    _bucket_kind,
    _bucket_matches_comparator,
    _bucket_probability,
    _effective_weather_sigma,
    _extract_station_coords,
    _extract_station_name,
    _fetch_nws_forecast,
    _fetch_nws_hourly_forecast,
    _fetch_station_observations,
    _fetch_station_geojson,
    _forecast_mu_for_date,
    _infer_comparator_hints,
    _latest_observation_timestamp,
    _parse_forecast_timestamp,
    _parse_bucket_bounds,
    _parse_date,
    _resolve_local_tz,
    _resolve_station_context,
    _station_id_from_rules,
)
from specialists.nba.plugin import discover_nba_startup_tickers, lookup_nba_live_p_true
from specialists.nhl.plugin import discover_nhl_startup_tickers, lookup_nhl_live_p_true

from specialists.crypto.plugin import (
    _estimate_sigma_and_spot_from_candles as _crypto_estimate_sigma_and_spot_from_candles,
    _fetch_candles as _crypto_fetch_candles,
    _parse_bounds as _crypto_parse_bounds,
    _parse_iso_dt as _crypto_parse_iso_dt,
    _parse_rule_target_dt as _crypto_parse_rule_target_dt,
    _p_hit_lower as _crypto_p_hit_lower,
    _p_hit_upper as _crypto_p_hit_upper,
    _p_terminal_between as _crypto_p_terminal_between,
    _product_for_event as _crypto_product_for_event,
    _sigma_floor_per_sqrt_s as _crypto_sigma_floor_per_sqrt_s,
    _spot_proxy as _crypto_spot_proxy,
)

DEFAULT_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_KALSHI_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_DATE_TOKEN_FULL_RE = re.compile(r"^\d{2}[A-Z]{3}\d{2}$")
_SPORTS_EVENT_PREFIXES: Tuple[str, ...] = ("KXNBAGAME-", "KXNHLGAME-")
_SPORTS_DIRECT_MARKET_SUFFIXES: Tuple[str, ...] = ("MIA", "PHI", "CHA", "IND", "NYR")


def log_shadow_trade(
    oracle_type: str,
    ticker: str,
    action: str,
    price_cents: int,
    quantity: int,
    spot_price: Optional[float],
    expected_value: Optional[float],
) -> None:
    ledger_path = REPO_ROOT / "shadow_ledger.csv"
    headers = [
        "timestamp",
        "oracle_type",
        "ticker",
        "action",
        "price_cents",
        "quantity",
        "spot_price",
        "expected_value",
    ]
    write_header = not ledger_path.exists()
    with ledger_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow(
            [
                _utc_now_iso(),
                str(oracle_type),
                str(ticker).strip().upper(),
                str(action),
                int(price_cents),
                int(quantity),
                "" if spot_price is None else float(spot_price),
                "" if expected_value is None else float(expected_value),
            ]
        )


def _parse_float(raw: object) -> Optional[float]:
    try:
        if raw is None:
            return None
        v = float(str(raw).strip())
    except Exception:
        return None
    if not (v == v):  # NaN guard
        return None
    return float(v)


def _parse_kv_notes(blob: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in str(blob or "").split(";"):
        p = str(part).strip()
        if not p or "=" not in p:
            continue
        k, v = p.split("=", 1)
        key = str(k).strip().lower()
        if not key:
            continue
        out[key] = str(v).strip()
    return out


def _infer_oracle_type(*, ticker: str, category: str) -> str:
    cat = str(category or "").strip().lower()
    t = str(ticker or "").strip().upper()
    if "nba" in cat or t.startswith("KXNBA"):
        return "nba"
    if "nhl" in cat or t.startswith("KXNHL"):
        return "nhl"
    if "crypto" in cat or t.startswith("KXBTC-") or t.startswith("KXETH-") or t.startswith("KXSOL"):
        return "crypto"
    if "weather" in cat or t.startswith("KXHIGH") or t.startswith("KXLOW"):
        return "weather"
    return "other"


def _spot_from_candidate_row(row: Dict[str, str]) -> Optional[float]:
    notes = _parse_kv_notes(str(row.get("liquidity_notes") or ""))
    for key in ("spot", "mu"):
        v = _parse_float(notes.get(key))
        if v is not None:
            return float(v)
    return None


def _is_sports_ticker(ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    return t.startswith("KXNBA") or t.startswith("KXNHL")


def _is_parent_sports_event_ticker(ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    if not any(t.startswith(prefix) for prefix in _SPORTS_EVENT_PREFIXES):
        return False
    return t.count("-") == 1


def _is_specific_sports_market_ticker(ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    if not any(t.startswith(prefix) for prefix in _SPORTS_EVENT_PREFIXES):
        return False
    return t.count("-") >= 2


def _expand_parent_sports_event_to_market_tickers(
    *,
    session: requests.Session,
    base_url: str,
    parent_event_ticker: str,
) -> List[str]:
    parent = str(parent_event_ticker or "").strip().upper()
    if not _is_parent_sports_event_ticker(parent):
        return []
    try:
        markets = fetch_public_markets(
            session=session,
            event_tickers=[parent],
            base_url=base_url,
            max_events=1,
        )
    except Exception as exc:
        print(f"[SHADOW][DISCOVERY] event->market expansion failed event_ticker={parent} error={exc}")
        return []

    out: set[str] = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        mt = str(market.get("ticker") or "").strip().upper()
        if not mt:
            continue
        if _is_specific_sports_market_ticker(mt) and mt.startswith(parent + "-"):
            out.add(mt)

    if not out:
        for market in markets:
            if not isinstance(market, dict):
                continue
            mt = str(market.get("ticker") or "").strip().upper()
            if mt and _is_specific_sports_market_ticker(mt):
                out.add(mt)

    resolved = sorted(out)
    if resolved:
        print(
            f"[SHADOW][DISCOVERY] flattened event_ticker={parent} "
            f"market_count={len(resolved)} first3={resolved[:3]}"
        )
    else:
        print(f"[SHADOW][DISCOVERY] flattened event_ticker={parent} market_count=0")
    return resolved


def _is_sports_order(order: Dict[str, Any]) -> bool:
    category = str(order.get("category") or "").strip().lower()
    strategy = str(order.get("strategy_id") or "").strip().upper()
    ticker = str(order.get("ticker") or "").strip().upper()
    return (
        category == "sports"
        or strategy.startswith("NBA_")
        or strategy.startswith("NHL_")
        or strategy.startswith("SPORT_")
        or _is_sports_ticker(ticker)
    )


def _p_true_from_candidate_row(row: Dict[str, str]) -> Optional[float]:
    direct = _parse_float(row.get("p_true"))
    if direct is not None and 0.0 <= float(direct) <= 1.0:
        return float(direct)
    notes = _parse_kv_notes(str(row.get("liquidity_notes") or ""))
    for key in ("p_true_cal", "p_true"):
        v = _parse_float(notes.get(key))
        if v is not None and 0.0 <= float(v) <= 1.0:
            return float(v)
    return None


def _p_true_from_order(order: Dict[str, Any]) -> Optional[float]:
    direct = _parse_float(order.get("_runtime_last_p_true"))
    if direct is not None and 0.0 <= float(direct) <= 1.0:
        return float(direct)
    notes = _parse_kv_notes(str(order.get("notes") or ""))
    for key in ("p_true_cal", "p_true"):
        v = _parse_float(notes.get(key))
        if v is not None and 0.0 <= float(v) <= 1.0:
            return float(v)
    return None


def _ws_ticker_snapshot(
    *,
    market_tickers: Sequence[str],
    use_private_auth: bool,
    timeout_s: float,
) -> Dict[str, Dict[str, Any]]:
    tickers = sorted({str(t).strip().upper() for t in market_tickers if str(t).strip()})
    if not tickers:
        return {}
    print(f"[SHADOW][WS] connecting for ticker snapshot count={len(tickers)} timeout_s={float(timeout_s):.1f}")
    try:
        out = asyncio.run(
            ws_collect_ticker_snapshot(
                market_tickers=tickers,
                use_private_auth=bool(use_private_auth),
                timeout_s=max(0.5, float(timeout_s)),
            )
        )
        print(f"[SHADOW][WS] snapshot received={len(out)}")
        return out
    except Exception:
        print("[SHADOW][WS] snapshot failed; falling back to REST")
        return {}


def _kalshi_date_token_from_iso(date_iso: str) -> Optional[str]:
    s = str(date_iso or "").strip()
    if not s:
        return None
    upper = s.upper()
    if _DATE_TOKEN_FULL_RE.fullmatch(upper):
        return upper
    if len(s) < 10:
        return None
    try:
        d = dt.date.fromisoformat(s[:10])
    except Exception:
        return None
    return f"{d.strftime('%y')}{_KALSHI_MONTHS[d.month - 1]}{d.strftime('%d')}"


def _seed_yes_price_cents(*, yes_bid: Optional[int], yes_ask: Optional[int]) -> int:
    if yes_ask is not None and int(yes_ask) > 1:
        return int(yes_ask) - 1
    if yes_bid is not None and int(yes_bid) > 0:
        return int(yes_bid)
    return 50


def _build_candidate_row_from_market(
    *,
    market: Dict[str, Any],
    category: str,
    p_true: Optional[float],
    extra_notes: str = "",
) -> Optional[Dict[str, str]]:
    ticker = str(market.get("ticker") or "").strip().upper()
    if not ticker:
        return None
    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))
    no_bid = safe_int(market.get("no_bid"))
    no_ask = safe_int(market.get("no_ask"))
    px = _seed_yes_price_cents(yes_bid=yes_bid, yes_ask=yes_ask)
    p = p_true
    if p is None and yes_bid is not None and yes_ask is not None and int(yes_ask) >= int(yes_bid):
        p = ((float(yes_bid) + float(yes_ask)) / 2.0) / 100.0
    if p is None:
        p = 0.5
    notes = f"p_true={float(p):.6f}"
    if extra_notes:
        notes += f";{extra_notes}"
    return {
        "strategy_id": "",
        "ticker": ticker,
        "event_ticker": str(market.get("event_ticker") or "").strip().upper(),
        "title": str(market.get("title") or "").strip(),
        "category": str(category),
        "close_time": str(market.get("close_time") or market.get("expiration_time") or "").strip(),
        "action": "post_yes",
        "maker_or_taker": "maker",
        "yes_price_cents": str(int(px)),
        "yes_bid": "" if yes_bid is None else str(int(yes_bid)),
        "yes_ask": "" if yes_ask is None else str(int(yes_ask)),
        "no_bid": "" if no_bid is None else str(int(no_bid)),
        "no_ask": "" if no_ask is None else str(int(no_ask)),
        "ev_dollars": "",
        "fees_assumed_dollars": "0.0",
        "slippage_assumed_dollars": "0.0",
        "rules_text_hash": "",
        "rules_pointer": "",
        "resolution_pointer": "",
        "liquidity_notes": notes,
        "p_true": f"{float(p):.6f}",
        "data_as_of_ts": _utc_now_iso(),
        "market_url": "",
    }


def _discover_candidates_for_series(
    *,
    session: requests.Session,
    base_url: str,
    date_token: str,
    series_ticker: str,
    category: str,
    max_events: int,
    max_markets: int,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        events = fetch_open_event_tickers_for_date(
            session=session,
            date_token=str(date_token).strip().upper(),
            series_ticker=str(series_ticker).strip().upper(),
            base_url=base_url,
            max_pages=10,
            status="open",
        )
    except Exception:
        return []
    if not events:
        return []
    for event_ticker in events[: max(1, int(max_events))]:
        try:
            ev = fetch_public_event(session=session, event_ticker=event_ticker, base_url=base_url)
        except Exception:
            continue
        markets = ev.get("markets") if isinstance(ev, dict) else None
        if not isinstance(markets, list):
            continue
        for m in markets[: max(1, int(max_markets))]:
            if not isinstance(m, dict):
                continue
            status = str(m.get("status") or "").strip().lower()
            if status and status not in {"open", "active"}:
                continue
            cand = _build_candidate_row_from_market(
                market=m,
                category=category,
                p_true=None,
                extra_notes=f"series={str(series_ticker).strip().upper()}",
            )
            if cand is not None:
                out.append(cand)
    uniq: Dict[str, Dict[str, str]] = {}
    for row in out:
        t = str(row.get("ticker") or "").strip().upper()
        if t and t not in uniq:
            uniq[t] = row
    return list(uniq.values())


def _bootstrap_candidates_from_live_discovery(*, day: str, config: Dict[str, Any]) -> List[Dict[str, str]]:
    enabled = str(config.get("shadow_bootstrap_discovery_enabled", "1")).strip().lower() not in {"0", "false", "no"}
    if not enabled:
        return []
    date_token = _kalshi_date_token_from_iso(day)
    if not date_token:
        return []

    base_url = str(config.get("base_url") or DEFAULT_BASE).strip().rstrip("/") or DEFAULT_BASE
    out: List[Dict[str, str]] = []
    print(f"[SHADOW][DISCOVERY] bootstrap start day={day} date_token={date_token}")
    with requests.Session() as s:
        # Crypto dynamic discovery
        print("[DEBUG] Attempting Crypto Discovery...")
        crypto_variance = max(0.0, float(config.get("shadow_crypto_variance_pct", 0.02)))
        crypto_series = str(config.get("shadow_crypto_series_ticker", "KXBTC")).strip().upper() or "KXBTC"
        max_crypto = max(1, int(config.get("shadow_bootstrap_max_crypto_tickers", 40)))
        spot = fetch_btc_spot_rest_fallback(s)
        crypto_count = 0
        if spot is not None:
            try:
                tickers = discover_viable_crypto_tickers_for_date(
                    session=s,
                    current_spot=float(spot),
                    date_token=str(date_token).strip().upper(),
                    series_ticker=crypto_series,
                    variance_pct=float(crypto_variance),
                    base_url=base_url,
                )
            except Exception:
                tickers = []
            for t in tickers[:max_crypto]:
                if str(date_token).strip().upper() not in str(t).strip().upper():
                    print(
                        f"[DEBUG] Discarded {str(t).strip().upper()} due to missing date token "
                        f"{str(date_token).strip().upper()} in ticker"
                    )
                    continue
                try:
                    mkt = _fetch_market(session=s, base_url=base_url, ticker=t)
                except Exception:
                    continue
                cand = _build_candidate_row_from_market(
                    market=mkt,
                    category="Crypto",
                    p_true=0.5,
                    extra_notes=f"spot={float(spot):.4f};series={crypto_series}",
                )
                if cand is not None:
                    out.append(cand)
                    crypto_count += 1
        print(f"[SHADOW][DISCOVERY] crypto spot={spot} candidates={crypto_count}")

        # Weather discovery (direct API, no local queue dependency).
        print("[DEBUG] Attempting Weather Discovery...")
        weather_series = config.get("shadow_weather_series_tickers", ["KXHIGHNY", "KXLOWNY"])
        if not isinstance(weather_series, list):
            weather_series = ["KXHIGHNY", "KXLOWNY"]
        max_weather = max(1, int(config.get("shadow_bootstrap_max_weather_tickers", 20)))
        weather_tickers: List[str]
        try:
            weather_tickers = discover_weather_tickers_for_date(
                session=s,
                date_token=str(date_token).strip().upper(),
                base_url=base_url,
                series_tickers=[str(x) for x in weather_series],
                max_pages=10,
                max_events=20,
                max_markets_per_event=max_weather,
            )
        except Exception:
            weather_tickers = []
        weather_count = 0
        for t in weather_tickers[:max_weather]:
            if str(date_token).strip().upper() not in str(t).strip().upper():
                print(
                    f"[DEBUG] Discarded {str(t).strip().upper()} due to missing date token "
                    f"{str(date_token).strip().upper()} in ticker"
                )
                continue
            try:
                mkt = _fetch_market(session=s, base_url=base_url, ticker=t)
            except Exception:
                continue
            cand = _build_candidate_row_from_market(
                market=mkt,
                category="Climate and Weather",
                p_true=None,
                extra_notes="source=weather_discovery",
            )
            if cand is not None:
                out.append(cand)
                weather_count += 1
        print(f"[SHADOW][DISCOVERY] weather candidates={weather_count}")

        # Sports discovery (plugin startup + direct API fallback).
        print("[DEBUG] Attempting Sports Discovery...")
        sports_series = config.get("shadow_sports_series_tickers", ["KXNBAGAME", "KXNHLGAME"])
        if not isinstance(sports_series, list):
            sports_series = ["KXNBAGAME", "KXNHLGAME"]
        max_sports = max(1, int(config.get("shadow_bootstrap_max_sports_tickers", 80)))
        try:
            core_sports_tickers = discover_sports_tickers_for_date(
                session=s,
                date_token=str(date_token).strip().upper(),
                base_url=base_url,
                series_tickers=[str(x) for x in sports_series],
                max_pages=10,
                max_events=100,
                max_markets_per_event=max_sports,
            )
        except Exception:
            core_sports_tickers = []
        plugin_nba_tickers = discover_nba_startup_tickers(day_iso=day, config=config)
        plugin_nhl_tickers = discover_nhl_startup_tickers(day_iso=day, config=config)
        print(
            f"[SHADOW][DISCOVERY] sports startup nba={len(plugin_nba_tickers)} "
            f"nhl={len(plugin_nhl_tickers)} core={len(core_sports_tickers)}"
        )
        sports_tickers: List[str] = []
        seen_sports: set[str] = set()
        for mt in list(plugin_nba_tickers) + list(plugin_nhl_tickers) + list(core_sports_tickers):
            t = str(mt).strip().upper()
            if not t or t in seen_sports:
                continue
            seen_sports.add(t)
            sports_tickers.append(t)
        sports_count = 0
        for t in sports_tickers[:max_sports]:
            if str(date_token).strip().upper() not in str(t).strip().upper():
                print(
                    f"[DEBUG] Discarded {str(t).strip().upper()} due to missing date token "
                    f"{str(date_token).strip().upper()} in ticker"
                )
                continue
            try:
                mkt = _fetch_market(session=s, base_url=base_url, ticker=t)
            except Exception:
                continue
            cand = _build_candidate_row_from_market(
                market=mkt,
                category="Sports",
                p_true=None,
                extra_notes="source=sports_discovery",
            )
            if cand is not None:
                out.append(cand)
                sports_count += 1
        print(f"[SHADOW][DISCOVERY] sports candidates={sports_count}")

    uniq: Dict[str, Dict[str, str]] = {}
    for row in out:
        t = str(row.get("ticker") or "").strip().upper()
        if t and t not in uniq:
            uniq[t] = row
    merged = list(uniq.values())
    print(f"[SHADOW][DISCOVERY] bootstrap total_candidates={len(merged)}")
    return merged


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


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


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = _read_json(path)
    return cfg if isinstance(cfg, dict) else {}


def _load_env_list_defaults(path: Path) -> None:
    candidates: List[Path] = []
    raw = Path(str(path)).expanduser()
    candidates.append(raw)
    try:
        candidates.append(raw.resolve())
    except Exception:
        pass
    candidates.append(REPO_ROOT / "env.list")
    candidates.append(ROOT / "env.list")
    candidates.append(Path.cwd() / "env.list")

    seen: set[str] = set()
    env_path: Optional[Path] = None
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            env_path = cand
            break

    if env_path is None:
        print(f"[DEBUG] Env defaults file not found. checked={list(seen)}")
        return

    set_count = 0
    parsed = 0
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = str(raw_line or "").strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = str(k or "").strip()
        if not key:
            continue
        value = str(v or "").strip().strip('"').strip("'")
        parsed += 1
        os.environ[key] = value
        set_count += 1
        print(f"[DEBUG] Force-set key: {key}.")
    print(f"[DEBUG] Env defaults loaded from {env_path} parsed={parsed} set={set_count}")


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    s = str(value).strip().lower()
    if not s:
        return bool(default)
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _normalize_ev_thresholds_for_shadow_test(config: Dict[str, Any]) -> None:
    keys = (
        "shadow_min_ev_dollars",
        "crypto_min_ev_dollars",
        "weather_min_ev_dollars",
        "sports_min_ev_dollars",
        "nba_min_ev_dollars",
        "nhl_min_ev_dollars",
        "econ_min_ev_dollars",
    )
    if "shadow_min_ev_dollars" not in config:
        config["shadow_min_ev_dollars"] = 0.0

    lowered: List[Tuple[str, float]] = []
    for key in keys:
        if key not in config:
            continue
        try:
            val = float(config.get(key))
        except Exception:
            continue
        if val > 0.05:
            lowered.append((key, val))
            config[key] = 0.0

    if lowered:
        detail = ", ".join([f"{k}:{v:.4f}->0.0000" for k, v in lowered])
        print(f"[SHADOW][CONFIG] lowered EV thresholds for test sensitivity ({detail})")

    print(
        "[SHADOW][CONFIG] EV thresholds "
        f"shadow={float(config.get('shadow_min_ev_dollars', 0.0)):.4f} "
        f"crypto={float(config.get('crypto_min_ev_dollars', 0.0)):.4f} "
        f"weather={float(config.get('weather_min_ev_dollars', 0.0)):.4f} "
        f"sports={float(config.get('sports_min_ev_dollars', 0.0)):.4f} "
        f"nba={float(config.get('nba_min_ev_dollars', 0.0)):.4f} "
        f"nhl={float(config.get('nhl_min_ev_dollars', 0.0)):.4f} "
        f"econ={float(config.get('econ_min_ev_dollars', 0.0)):.4f}"
    )


def _shadow_execution_mode(config: Dict[str, Any]) -> str:
    mode = str(config.get("shadow_execution_mode", "maker")).strip().lower()
    if mode not in {"maker", "taker"}:
        return "maker"
    return mode


def _shadow_fee_rate(config: Dict[str, Any], *, execution_mode: str) -> float:
    if str(execution_mode).strip().lower() == "taker":
        if "shadow_taker_fee_rate" in config:
            return float(config.get("shadow_taker_fee_rate", 0.07))
        if "taker_fee_rate" in config:
            return float(config.get("taker_fee_rate", 0.07))
        return 0.07
    if "shadow_maker_fee_rate" in config:
        return float(config.get("shadow_maker_fee_rate", 0.0))
    return float(config.get("maker_fee_rate", 0.0))


def _parse_ticker_override(raw: str) -> List[str]:
    return [t.strip().upper() for t in str(raw or "").split(",") if t.strip()]


def _sorted_with_priority_prefix(
    tickers: Iterable[str],
    *,
    priority_prefix: str,
    priority_active: bool,
) -> List[str]:
    pref = str(priority_prefix or "").strip().upper()
    items = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not pref or not priority_active:
        return items
    return sorted(items, key=lambda t: (0 if t.startswith(pref) else 1, t))


def _is_direct_market_override_ticker(ticker: str) -> bool:
    t = str(ticker or "").strip().upper()
    if not t:
        return False
    if t.count("-") < 2:
        return False
    if not _is_sports_ticker(t):
        return False
    tail = t.rsplit("-", 1)[-1]
    return tail in _SPORTS_DIRECT_MARKET_SUFFIXES


def _load_queue_tickers_from_md(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s.startswith("### "):
            continue
        parts = s.split(". ", 1)
        if len(parts) == 2:
            t = parts[1].strip().upper()
            if t:
                out.append(t)
    return out


def _load_daily_candidates(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = str(row.get("ticker") or "").strip().upper()
            if t.startswith("KXNBAGAME") or t.startswith("KXNHLGAME"):
                row["category"] = "Sports"
            if t and t not in out:
                out[t] = row
    return out


def _ticker_matches_date_token(*, ticker: str, date_token: str) -> bool:
    t = str(ticker or "").strip().upper()
    tok = str(date_token or "").strip().upper()
    if not tok:
        return True
    return tok in t


def _prune_parent_sports_event_candidate_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    children_by_parent: set[str] = set()
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if _is_specific_sports_market_ticker(ticker):
            children_by_parent.add(ticker.rsplit("-", 1)[0])
    if not children_by_parent:
        return list(rows)

    out: List[Dict[str, str]] = []
    dropped = 0
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if _is_parent_sports_event_ticker(ticker) and ticker in children_by_parent:
            dropped += 1
            continue
        out.append(row)
    if dropped > 0:
        print(f"[SHADOW][DISCOVERY] pruned parent sports event rows={dropped}")
    return out


def _debug_preview_event_tickers_from_api(*, day: str, config: Dict[str, Any]) -> None:
    base_url = str(config.get("base_url") or DEFAULT_BASE).strip().rstrip("/") or DEFAULT_BASE
    series: List[str] = []
    series.append(str(config.get("shadow_crypto_series_ticker", "KXBTC")).strip().upper() or "KXBTC")
    weather_series = config.get("shadow_weather_series_tickers", ["KXHIGHNY", "KXLOWNY"])
    sports_series = config.get("shadow_sports_series_tickers", ["KXNBAGAME", "KXNHLGAME"])
    if isinstance(weather_series, list):
        series.extend([str(x).strip().upper() for x in weather_series if str(x).strip()])
    if isinstance(sports_series, list):
        series.extend([str(x).strip().upper() for x in sports_series if str(x).strip()])

    seen_series: set[str] = set()
    token = _kalshi_date_token_from_iso(day) or str(day).strip().upper()
    with requests.Session() as s:
        for st in series:
            ser = str(st).strip().upper()
            if not ser or ser in seen_series:
                continue
            seen_series.add(ser)
            params: Dict[str, Any] = {"series_ticker": ser, "status": "open", "limit": 200}
            url = f"{base_url}/events"
            try:
                resp = s.get(url, params=params, timeout=20.0)
                status_code = int(resp.status_code)
                payload = resp.json() if resp.content else {}
            except Exception as exc:
                print(f"[DEBUG] /events preview failed series={ser} error={exc}")
                continue

            events = payload.get("events") if isinstance(payload, dict) else None
            event_tickers: List[str] = []
            if isinstance(events, list):
                for ev in events:
                    if not isinstance(ev, dict):
                        continue
                    et = str(ev.get("event_ticker") or ev.get("ticker") or "").strip().upper()
                    if et:
                        event_tickers.append(et)
            print(
                f"[DEBUG] /events preview series={ser} date_token={token} "
                f"url={resp.url} status={status_code} first5={event_tickers[:5]}"
            )


def _filter_candidates_with_dynamic_crypto_discovery(
    *,
    candidates: Sequence[Dict[str, str]],
    day: str,
    config: Dict[str, Any],
) -> List[Dict[str, str]]:
    enabled = str(config.get("shadow_crypto_dynamic_discovery", "1")).strip().lower() not in {"0", "false", "no"}
    if not enabled:
        print("[SHADOW][DISCOVERY] crypto dynamic discovery disabled by config")
        return list(candidates)

    series_ticker = str(config.get("shadow_crypto_series_ticker", "KXBTC")).strip().upper() or "KXBTC"
    variance_pct = max(0.0, float(config.get("shadow_crypto_variance_pct", 0.02)))
    crypto_rows = [
        row
        for row in candidates
        if str(row.get("ticker") or "").strip().upper().startswith(f"{series_ticker}-")
        or str(row.get("event_ticker") or "").strip().upper().startswith(f"{series_ticker}-")
        or str(row.get("category") or "").strip().lower() == "crypto"
    ]
    if not crypto_rows:
        print("[SHADOW][DISCOVERY] no crypto rows found to filter")
        return list(candidates)

    date_token: Optional[str] = None
    for row in crypto_rows:
        date_token = extract_kalshi_date_token_from_ticker(str(row.get("ticker") or ""))
        if date_token:
            break
        date_token = extract_kalshi_date_token_from_ticker(str(row.get("event_ticker") or ""))
        if date_token:
            break
    if not date_token:
        date_token = _kalshi_date_token_from_iso(day)
    if not date_token:
        print(f"[SHADOW][DISCOVERY] unable to derive Kalshi date token from day={day}")
        return list(candidates)

    viable: List[str] = []
    try:
        with requests.Session() as s:
            spot = fetch_btc_spot_rest_fallback(s)
            if spot is None:
                return list(candidates)
            viable = discover_viable_crypto_tickers_for_date(
                session=s,
                current_spot=float(spot),
                date_token=str(date_token).strip().upper(),
                series_ticker=series_ticker,
                variance_pct=float(variance_pct),
                base_url=str(config.get("base_url") or ""),
            )
    except Exception:
        print("[SHADOW][DISCOVERY] crypto dynamic discovery failed; using unfiltered candidates")
        return list(candidates)

    if not viable:
        print(
            f"[SHADOW][DISCOVERY] no viable crypto tickers found for date_token={date_token}; "
            "using unfiltered candidates"
        )
        return list(candidates)

    viable_set = {str(t).strip().upper() for t in viable}
    out: List[Dict[str, str]] = []
    for row in candidates:
        ticker = str(row.get("ticker") or "").strip().upper()
        is_crypto = ticker.startswith(f"{series_ticker}-") or str(row.get("category") or "").strip().lower() == "crypto"
        if is_crypto and ticker and ticker not in viable_set:
            continue
        out.append(row)
    print(
        f"[SHADOW][DISCOVERY] crypto dynamic filter date_token={date_token} "
        f"input={len(candidates)} viable={len(viable_set)} output={len(out)}"
    )
    return out


def _discover_shadow_candidates(*, day: str, tickers_override: Sequence[str], config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    research_dir = ROOT / "reports" / "research" / day
    daily_csv = ROOT / "reports" / "daily" / f"{day}_candidates.csv"

    candidates: Dict[str, Dict[str, str]] = {}

    if research_dir.exists():
        for child in sorted(research_dir.iterdir()):
            if not child.is_dir():
                continue
            cand_path = child / "candidate.json"
            if not cand_path.exists():
                continue
            data = _read_json(cand_path)
            t = str(data.get("ticker") or "").strip().upper()
            if t:
                candidates[t] = {k: str(v) for k, v in data.items()}
        print(f"[SHADOW][DISCOVERY] research candidates found={len(candidates)} from {research_dir}")

    if not candidates:
        queue_md = ROOT / "reports" / "research" / f"{day}_queue.md"
        queue_tickers = _load_queue_tickers_from_md(queue_md)
        daily_map = _load_daily_candidates(daily_csv)
        print(f"[DEBUG] Raw Discovery Output: queue_tickers={queue_tickers}")
        print(f"[DEBUG] Raw Discovery Output: daily_rows={daily_map}")
        for t in queue_tickers:
            if t in daily_map:
                candidates[t] = daily_map[t]
        print(
            f"[SHADOW][DISCOVERY] queue fallback queue_tickers={len(queue_tickers)} "
            f"daily_rows={len(daily_map)} selected={len(candidates)}"
        )

    if tickers_override:
        allow_prefixes = [str(t).strip().upper() for t in tickers_override if str(t).strip()]
        candidates = {
            t: row
            for t, row in candidates.items()
            if any(str(t).strip().upper().startswith(prefix) for prefix in allow_prefixes)
        }
        print(f"[SHADOW][DISCOVERY] --tickers override applied count={len(candidates)}")

    date_token = _kalshi_date_token_from_iso(day) or str(day).strip().upper()
    out: List[Dict[str, str]] = []
    for t, row in sorted(candidates.items()):
        action = str(row.get("action") or "").strip().lower()
        maker_or_taker = str(row.get("maker_or_taker") or "").strip().lower()
        yes_px = str(row.get("yes_price_cents") or "").strip()
        if action not in {"post_yes", "buy_yes"}:
            continue
        if maker_or_taker and maker_or_taker not in {"maker", "taker"}:
            continue
        if not yes_px:
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        if tickers_override:
            # Explicit --tickers must always be force-approved.
            out.append(row)
            continue
        if not _ticker_matches_date_token(ticker=ticker, date_token=date_token):
            print(
                f"[DEBUG] Discarded {ticker or '<missing>'} due to missing date token "
                f"{date_token} in ticker"
            )
            continue
        out.append(row)
    cfg = config if isinstance(config, dict) else {}
    if tickers_override:
        base_url = str(cfg.get("base_url") or DEFAULT_BASE).strip().rstrip("/") or DEFAULT_BASE
        existing = {str(r.get("ticker") or "").strip().upper() for r in out}
        force_count = 0
        direct_added = 0
        with requests.Session() as s:
            for event_prefix in sorted({str(x).strip().upper() for x in tickers_override if str(x).strip()}):
                if _is_direct_market_override_ticker(event_prefix):
                    ticker = event_prefix
                    if ticker in existing:
                        for row in out:
                            if str(row.get("ticker") or "").strip().upper() == ticker:
                                row["category"] = "Sports"
                                break
                    else:
                        event_ticker = ticker.rsplit("-", 1)[0]
                        category = "Sports"
                        forced = _build_candidate_row_from_market(
                            market={
                                "ticker": ticker,
                                "event_ticker": event_ticker,
                                "title": "",
                                "status": "open",
                            },
                            category=category,
                            p_true=0.5,
                            extra_notes="source=cli_tickers_override_direct_market",
                        )
                        if forced is None:
                            forced = {
                                "strategy_id": "",
                                "ticker": ticker,
                                "event_ticker": event_ticker,
                                "title": "",
                                "category": category,
                                "close_time": "",
                                "action": "post_yes",
                                "maker_or_taker": "maker",
                                "yes_price_cents": "50",
                                "yes_bid": "",
                                "yes_ask": "",
                                "no_bid": "",
                                "no_ask": "",
                                "ev_dollars": "",
                                "fees_assumed_dollars": "0.0",
                                "slippage_assumed_dollars": "0.0",
                                "rules_text_hash": "",
                                "rules_pointer": "",
                                "resolution_pointer": "",
                                "liquidity_notes": "source=cli_tickers_override_direct_market;p_true=0.500000",
                                "p_true": "0.500000",
                                "data_as_of_ts": _utc_now_iso(),
                                "market_url": "",
                            }
                        out.append(forced)
                        existing.add(ticker)
                        force_count += 1
                        direct_added += 1
                    print(
                        f"[DEBUG] --tickers direct market override accepted ticker={event_prefix} "
                        "expansion=skipped"
                    )
                    continue
                try:
                    markets = fetch_public_markets(
                        session=s,
                        event_tickers=[event_prefix],
                        base_url=base_url,
                        max_events=1,
                    )
                except Exception as exc:
                    print(
                        f"[DEBUG] --tickers event-prefix expansion failed prefix={event_prefix} "
                        f"error={exc}"
                    )
                    markets = []
                if not markets:
                    print(
                        f"[DEBUG] --tickers event-prefix expansion produced no markets "
                        f"prefix={event_prefix}"
                    )
                    continue
                prefix_added = 0
                for market in markets:
                    if not isinstance(market, dict):
                        continue
                    ticker = str(market.get("ticker") or "").strip().upper()
                    if not ticker or ticker in existing:
                        continue
                    classifier = ticker or event_prefix
                    category = "Other"
                    if classifier.startswith("KXBTC-") or classifier.startswith("KXETH-") or classifier.startswith("KXSOL"):
                        category = "Crypto"
                    elif classifier.startswith("KXHIGH") or classifier.startswith("KXLOW"):
                        category = "Climate and Weather"
                    elif classifier.startswith("KXNBA") or classifier.startswith("KXNHL"):
                        category = "Sports"
                    forced = _build_candidate_row_from_market(
                        market=market,
                        category=category,
                        p_true=0.5,
                        extra_notes="source=cli_tickers_override_event_prefix",
                    )
                    if forced is None:
                        forced = {
                            "strategy_id": "",
                            "ticker": ticker,
                            "event_ticker": str(market.get("event_ticker") or event_prefix),
                            "title": str(market.get("title") or ""),
                            "category": category,
                            "close_time": "",
                            "action": "post_yes",
                            "maker_or_taker": "maker",
                            "yes_price_cents": "50",
                            "yes_bid": "",
                            "yes_ask": "",
                            "no_bid": "",
                            "no_ask": "",
                            "ev_dollars": "",
                            "fees_assumed_dollars": "0.0",
                            "slippage_assumed_dollars": "0.0",
                            "rules_text_hash": "",
                            "rules_pointer": "",
                            "resolution_pointer": "",
                            "liquidity_notes": "source=cli_tickers_override_event_prefix;p_true=0.500000",
                            "p_true": "0.500000",
                            "data_as_of_ts": _utc_now_iso(),
                            "market_url": "",
                        }
                    out.append(forced)
                    existing.add(ticker)
                    force_count += 1
                    prefix_added += 1
                print(
                    f"[DEBUG] Force-approved override prefix={event_prefix} "
                    f"added_markets={prefix_added}"
                )
        print(
            f"[DEBUG] --tickers force-approve added={force_count} "
            f"direct_markets={direct_added} total={len(out)}"
        )

    live_rows: List[Dict[str, str]] = []
    if not tickers_override:
        live_rows = _bootstrap_candidates_from_live_discovery(day=day, config=cfg)
        if not live_rows:
            _debug_preview_event_tickers_from_api(day=day, config=cfg)
        if live_rows:
            merged: Dict[str, Dict[str, str]] = {
                str(r.get("ticker") or "").strip().upper(): r
                for r in out
                if str(r.get("ticker") or "").strip()
            }
            for row in live_rows:
                t = str(row.get("ticker") or "").strip().upper()
                if not _ticker_matches_date_token(ticker=t, date_token=date_token):
                    print(
                        f"[DEBUG] Discarded {t or '<missing>'} due to missing date token "
                        f"{date_token} in ticker"
                    )
                    continue
                if t and t not in merged:
                    merged[t] = row
            out = list(merged.values())
        print(
            f"[SHADOW][DISCOVERY] local_candidates={len(candidates)} "
            f"live_candidates={len(live_rows)} merged={len(out)}"
        )
    out = _filter_candidates_with_dynamic_crypto_discovery(candidates=out, day=day, config=cfg)
    out = _prune_parent_sports_event_candidate_rows(out)
    n_crypto = sum(1 for r in out if str(r.get("category") or "").strip().lower() == "crypto")
    n_weather = sum(1 for r in out if str(r.get("category") or "").strip().lower() == "climate and weather")
    n_sports = sum(1 for r in out if str(r.get("category") or "").strip().lower() == "sports")
    print(
        f"[SHADOW][DISCOVERY] final candidates total={len(out)} "
        f"crypto={n_crypto} weather={n_weather} sports={n_sports}"
    )
    return out


def _parse_weather_meta_from_ticker(ticker: str) -> Optional[Dict[str, str]]:
    m = WEATHER_MARKET_RE.match(str(ticker or "").strip().upper())
    if not m:
        return None
    kind = str(m.group("kind") or "").strip().upper()
    code = str(m.group("code") or "").strip().upper()
    date_token = str(m.group("date") or "").strip().upper()
    d = _parse_date(date_token)
    if not kind or not code or d is None:
        return None
    return {
        "kind": kind,
        "code": code,
        "date_token": date_token,
        "target_date": d.isoformat(),
    }


def _append_note(existing: str, note: str) -> str:
    base = str(existing or "").strip()
    n = str(note or "").strip()
    if not n:
        return base
    if not base:
        return n
    if n in base:
        return base
    return base + ";" + n


def _upsert_note_field(existing: str, *, prefix: str, value: str) -> str:
    parts = [p for p in str(existing or "").split(";") if p]
    clean_prefix = str(prefix or "").strip()
    filtered = [p for p in parts if not p.startswith(clean_prefix)]
    filtered.append(f"{clean_prefix}{value}")
    return ";".join(filtered)


def _make_shadow_order_id(day: str, ticker: str) -> str:
    return f"shadow-{day}-{str(ticker).strip().upper()}"


def _migrate_order(order: Dict[str, Any], day: str) -> Dict[str, Any]:
    out = dict(order)
    ticker = str(out.get("ticker") or "").strip().upper()
    shadow_order_id = str(out.get("shadow_order_id") or "").strip() or str(out.get("order_id") or "").strip()
    if not shadow_order_id and ticker:
        shadow_order_id = _make_shadow_order_id(day, ticker)

    # Canonical schema defaults.
    out.setdefault("shadow_order_id", shadow_order_id)
    out.setdefault("created_ts", _utc_now_iso())
    out.setdefault("strategy_id", str(out.get("strategy_id") or ""))
    out.setdefault("ticker", ticker)
    out.setdefault("event_ticker", str(out.get("event_ticker") or ""))
    out.setdefault("category", str(out.get("category") or ""))
    if (not str(out.get("category") or "").strip()) and (ticker.startswith("KXNBAGAME") or ticker.startswith("KXNHLGAME")):
        out["category"] = "Sports"
    out.setdefault("side", "yes")
    out.setdefault("action", "post_yes")
    out.setdefault("maker_or_taker", "maker")

    if "intended_price_cents" not in out:
        out["intended_price_cents"] = out.get("intended_yes_price_cents") or out.get("yes_price_cents") or ""
    if "assumed_fill_price_cents" not in out:
        out["assumed_fill_price_cents"] = out.get("fill_price_cents") or ""

    out.setdefault("size_contracts", out.get("size_contracts") or "1")
    out.setdefault("fees_assumed_dollars", out.get("fees_assumed_dollars") or out.get("fees_dollars") or "0.0")
    out.setdefault("slippage_assumed_dollars", out.get("slippage_assumed_dollars") or "0.0")
    out.setdefault("rules_text_hash", str(out.get("rules_text_hash") or ""))
    out.setdefault("rules_pointer", str(out.get("rules_pointer") or ""))
    out.setdefault("resolution_pointer", str(out.get("resolution_pointer") or ""))
    out.setdefault("status", str(out.get("status") or "open"))
    out.setdefault("resolved_ts", str(out.get("resolved_ts") or ""))
    if "resolved_payout_dollars" not in out:
        out["resolved_payout_dollars"] = out.get("payout_dollars") or ""
    out.setdefault("realized_pnl_dollars", str(out.get("realized_pnl_dollars") or ""))
    out.setdefault("notes", str(out.get("notes") or ""))

    # Runtime extensions.
    out.setdefault("_runtime_fill_trade_id", str(out.get("fill_trade_id") or ""))
    out.setdefault("_runtime_filled_ts", str(out.get("filled_ts") or ""))
    out.setdefault("_runtime_fill_count_seen", int(float(str(out.get("fill_count_seen") or "0") or "0")))
    out.setdefault("_runtime_first_fill_trade_id", str(out.get("first_fill_trade_id") or ""))
    out.setdefault("_runtime_revision_count", int(float(str(out.get("revision_count") or "0") or "0")))
    out.setdefault("_runtime_last_reprice_ts", str(out.get("last_reprice_ts") or ""))
    out.setdefault("_runtime_roi", str(out.get("roi") or ""))
    out.setdefault("_runtime_last_ev", str(out.get("last_ev") or ""))
    out.setdefault("_runtime_last_p_true", str(out.get("last_p_true") or ""))
    out.setdefault("_runtime_last_p_fill", str(out.get("last_p_fill") or ""))
    out.setdefault("_runtime_last_mu", str(out.get("last_mu") or ""))
    out.setdefault("_runtime_last_sigma", str(out.get("last_sigma") or ""))
    out.setdefault("_runtime_last_forecast_updated", str(out.get("last_forecast_updated") or ""))
    out.setdefault("_runtime_station_id", str(out.get("station_id") or ""))
    out.setdefault("_runtime_station_name", str(out.get("station_name") or ""))
    out.setdefault("_runtime_station_lat", str(out.get("station_lat") or ""))
    out.setdefault("_runtime_station_lon", str(out.get("station_lon") or ""))

    meta = _parse_weather_meta_from_ticker(ticker)
    if meta:
        out.setdefault("_weather_kind", meta["kind"])
        out.setdefault("_weather_code", meta["code"])
        out.setdefault("_weather_target_date", meta["target_date"])

    return out


def _build_order_from_candidate(*, row: Dict[str, str], day: str, size_contracts: int, execution_mode: str) -> Dict[str, Any]:
    parsed_created = _parse_ts(row.get("data_as_of_ts"))
    now_iso = (parsed_created.isoformat() if parsed_created is not None else _utc_now_iso())
    ticker = str(row.get("ticker") or "").strip().upper()
    yes_px = int(float(str(row.get("yes_price_cents") or "0") or "0"))

    initial_p_true = _p_true_from_candidate_row(row)
    initial_ev = _parse_float(row.get("ev_dollars"))

    order: Dict[str, Any] = {
        "shadow_order_id": _make_shadow_order_id(day, ticker),
        "created_ts": now_iso,
        "strategy_id": str(row.get("strategy_id") or ""),
        "ticker": ticker,
        "event_ticker": str(row.get("event_ticker") or ""),
        "category": str(row.get("category") or ""),
        "close_time": str(row.get("close_time") or ""),
        "side": "yes",
        "action": ("buy_yes" if str(execution_mode).strip().lower() == "taker" else "post_yes"),
        "maker_or_taker": ("taker" if str(execution_mode).strip().lower() == "taker" else "maker"),
        "intended_price_cents": yes_px,
        "assumed_fill_price_cents": "",
        "size_contracts": str(max(1, int(size_contracts))),
        "fees_assumed_dollars": str(row.get("fees_assumed_dollars") or "0.0"),
        "slippage_assumed_dollars": str(row.get("slippage_assumed_dollars") or "0.0"),
        "rules_text_hash": str(row.get("rules_text_hash") or ""),
        "rules_pointer": str(row.get("rules_pointer") or ""),
        "resolution_pointer": str(row.get("resolution_pointer") or ""),
        "status": "open",
        "resolved_ts": "",
        "resolved_payout_dollars": "",
        "realized_pnl_dollars": "",
        "notes": "",
        "_runtime_fill_trade_id": "",
        "_runtime_filled_ts": "",
        "_runtime_fill_count_seen": 0,
        "_runtime_first_fill_trade_id": "",
        "_runtime_revision_count": 0,
        "_runtime_last_reprice_ts": "",
        "_runtime_roi": "",
        "_runtime_last_ev": ("" if initial_ev is None else f"{float(initial_ev):.6f}"),
        "_runtime_last_p_true": ("" if initial_p_true is None else f"{float(initial_p_true):.6f}"),
        "_runtime_last_p_fill": "",
        "_runtime_last_mu": "",
        "_runtime_last_sigma": "",
        "_runtime_last_forecast_updated": "",
        "_runtime_station_id": "",
        "_runtime_station_name": "",
        "_runtime_station_lat": "",
        "_runtime_station_lon": "",
    }

    meta = _parse_weather_meta_from_ticker(ticker)
    if meta:
        order["_weather_kind"] = meta["kind"]
        order["_weather_code"] = meta["code"]
        order["_weather_target_date"] = meta["target_date"]
    if initial_p_true is not None:
        order["notes"] = _append_note(str(order.get("notes") or ""), f"p_true={float(initial_p_true):.6f}")
    return order


def _initialize_or_update_state(
    *,
    state: Dict[str, Any],
    day: str,
    candidates: Sequence[Dict[str, str]],
    size_contracts: int,
    execution_mode: str,
) -> Dict[str, Any]:
    now_iso = _utc_now_iso()
    if not isinstance(state, dict) or not state:
        state = {
            "run_date": day,
            "created_ts": now_iso,
            "updated_ts": now_iso,
            "orders": [],
            "tickers": {},
        }

    orders_in = state.get("orders")
    orders: List[Dict[str, Any]] = []
    if isinstance(orders_in, list):
        for o in orders_in:
            if isinstance(o, dict):
                orders.append(_migrate_order(o, day))

    ticker_state = state.get("tickers")
    if not isinstance(ticker_state, dict):
        ticker_state = {}

    existing_by_ticker = {str(o.get("ticker") or "").strip().upper(): o for o in orders if str(o.get("ticker") or "").strip()}

    for row in candidates:
        t = str(row.get("ticker") or "").strip().upper()
        if not t:
            continue
        if t not in existing_by_ticker:
            order = _build_order_from_candidate(
                row=row,
                day=day,
                size_contracts=size_contracts,
                execution_mode=str(execution_mode),
            )
            orders.append(order)
            existing_by_ticker[t] = order
            px = safe_int(order.get("intended_price_cents"))
            qty = safe_int(order.get("size_contracts"))
            if px is not None and qty is not None and px > 0 and qty > 0:
                log_shadow_trade(
                    oracle_type=_infer_oracle_type(
                        ticker=str(order.get("ticker") or ""),
                        category=str(order.get("category") or ""),
                    ),
                    ticker=str(order.get("ticker") or ""),
                    action=str(order.get("action") or "post_yes"),
                    price_cents=int(px),
                    quantity=int(qty),
                    spot_price=_spot_from_candidate_row(row),
                    expected_value=_parse_float(row.get("ev_dollars")),
                )

    for t in sorted(existing_by_ticker.keys()):
        ts = ticker_state.get(t)
        if not isinstance(ts, dict):
            ticker_state[t] = {
                "cursor": "",
                "seen_trade_ids": [],
                "last_checked_ts": "",
                "last_seen_trade_ts": "",
            }
        else:
            if not isinstance(ts.get("seen_trade_ids"), list):
                ts["seen_trade_ids"] = []
            ts.setdefault("cursor", "")
            ts.setdefault("last_checked_ts", "")
            ts.setdefault("last_seen_trade_ts", "")

    state["run_date"] = day
    state["orders"] = orders
    state["tickers"] = ticker_state
    state["updated_ts"] = now_iso
    return state


def _project_canonical_rows(orders: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for o in orders:
        row = {k: o.get(k, "") for k in SHADOW_LEDGER_HEADERS}
        out.append(row)
    return out


def _fetch_trades_page(*, session: requests.Session, base_url: str, ticker: str, cursor: str) -> Dict[str, Any]:
    params: Dict[str, object] = {"ticker": ticker, "limit": 200}
    if cursor:
        params["cursor"] = cursor
    resp = session.get(f"{base_url}/markets/trades", params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        return data
    return {}


def _poll_new_trades_for_ticker(
    *,
    session: requests.Session,
    base_url: str,
    ticker: str,
    created_cutoff: dt.datetime,
    ticker_state: Dict[str, Any],
    seen_cap: int,
) -> List[Dict[str, Any]]:
    seen_ids_list = [str(x) for x in (ticker_state.get("seen_trade_ids") or []) if str(x).strip()]
    seen_ids = set(seen_ids_list)

    cursor = ""
    pages = 0
    out: List[Dict[str, Any]] = []

    while pages < 30:
        try:
            payload = _fetch_trades_page(session=session, base_url=base_url, ticker=ticker, cursor=cursor)
        except Exception:
            break
        trades = payload.get("trades") if isinstance(payload.get("trades"), list) else []
        if not trades:
            break

        next_cursor = str(payload.get("cursor") or "").strip()
        stop = False
        for tr in trades:
            if not isinstance(tr, dict):
                continue
            tid = str(tr.get("trade_id") or "").strip()
            tts = _parse_ts(tr.get("created_time"))

            if tts is not None and tts < created_cutoff:
                stop = True
                break
            if tid and tid in seen_ids:
                stop = True
                break
            out.append(tr)

        cursor = next_cursor
        pages += 1
        if stop or not cursor:
            break

    for tr in out:
        tid = str(tr.get("trade_id") or "").strip()
        if tid and tid not in seen_ids:
            seen_ids_list.append(tid)
            seen_ids.add(tid)

    if len(seen_ids_list) > int(seen_cap):
        seen_ids_list = seen_ids_list[-int(seen_cap) :]

    ticker_state["cursor"] = cursor
    ticker_state["seen_trade_ids"] = seen_ids_list
    ticker_state["last_checked_ts"] = _utc_now_iso()

    latest_ts = None
    for tr in out:
        tts = _parse_ts(tr.get("created_time"))
        if tts is not None and (latest_ts is None or tts > latest_ts):
            latest_ts = tts
    if latest_ts is not None:
        ticker_state["last_seen_trade_ts"] = latest_ts.isoformat()

    return out


def _matching_fill_trades(order: Dict[str, Any], trades: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    created_ts = _parse_ts(order.get("created_ts"))
    if created_ts is None:
        return []
    try:
        target_px = int(float(str(order.get("intended_price_cents") or "0") or "0"))
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for tr in sorted(trades, key=lambda x: _parse_ts(x.get("created_time")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc)):
        if not isinstance(tr, dict):
            continue
        tts = _parse_ts(tr.get("created_time"))
        if tts is None or tts < created_ts:
            continue
        taker_side = str(tr.get("taker_side") or "").strip().lower()
        try:
            yes_px = int(float(str(tr.get("yes_price") or "")))
        except Exception:
            continue
        if taker_side == "no" and yes_px == target_px:
            out.append(tr)
    return out


def _apply_cancel_replace(
    order: Dict[str, Any],
    *,
    new_price_cents: int,
    now_iso: str,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    p_true: Optional[float] = None,
    forecast_updated: str = "",
    station_id: str = "",
    ev_dollars: Optional[float] = None,
) -> bool:
    old_price = int(float(str(order.get("intended_price_cents") or "0") or "0"))
    if int(new_price_cents) == old_price:
        return False

    rev = int(float(str(order.get("_runtime_revision_count") or "0") or "0")) + 1
    order["_runtime_revision_count"] = rev
    order["_runtime_last_reprice_ts"] = now_iso
    order["created_ts"] = now_iso
    order["intended_price_cents"] = int(new_price_cents)
    order["notes"] = _append_note(
        str(order.get("notes") or ""),
        f"reprice:{old_price}->{int(new_price_cents)}@{now_iso};rev={rev}",
    )
    order["notes"] = _append_note(
        str(order.get("notes") or ""),
        (
            f"reprice_from={old_price};mu={'' if mu is None else f'{float(mu):.4f}'};"
            f"sigma={'' if sigma is None else f'{float(sigma):.4f}'};"
            f"p_true={'' if p_true is None else f'{float(p_true):.6f}'};"
            f"forecast_updated={forecast_updated};station_id={station_id}"
        ),
    )
    qty = safe_int(order.get("size_contracts"))
    if qty is not None and qty > 0:
        log_shadow_trade(
            oracle_type=_infer_oracle_type(
                ticker=str(order.get("ticker") or ""),
                category=str(order.get("category") or ""),
            ),
            ticker=str(order.get("ticker") or ""),
            action="cancel_replace_yes",
            price_cents=int(new_price_cents),
            quantity=int(qty),
            spot_price=(None if mu is None else float(mu)),
            expected_value=(None if ev_dollars is None else float(ev_dollars)),
        )
    return True


def _fetch_market(*, session: requests.Session, base_url: str, ticker: str) -> Dict[str, Any]:
    def _safe_float_local(value: object) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _safe_int_local(value: object) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(float(value))
        except Exception:
            return None

    def _normalize_market_quotes_local(market: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(market)
        yes_bid = _safe_int_local(out.get("yes_bid"))
        if yes_bid is None or yes_bid <= 0:
            d = _safe_float_local(out.get("yes_bid_dollars"))
            if d is not None and d > 0.0:
                out["yes_bid"] = int(round(float(d) * 100.0))
        yes_ask = _safe_int_local(out.get("yes_ask"))
        if yes_ask is None or yes_ask <= 0:
            d = _safe_float_local(out.get("yes_ask_dollars"))
            if d is not None and d > 0.0:
                out["yes_ask"] = int(round(float(d) * 100.0))
        no_bid = _safe_int_local(out.get("no_bid"))
        if no_bid is None or no_bid <= 0:
            d = _safe_float_local(out.get("no_bid_dollars"))
            if d is not None and d > 0.0:
                out["no_bid"] = int(round(float(d) * 100.0))
        no_ask = _safe_int_local(out.get("no_ask"))
        if no_ask is None or no_ask <= 0:
            d = _safe_float_local(out.get("no_ask_dollars"))
            if d is not None and d > 0.0:
                out["no_ask"] = int(round(float(d) * 100.0))
        yes_bid_size = _safe_int_local(out.get("yes_bid_size"))
        if yes_bid_size is None or yes_bid_size <= 0:
            fp = _safe_float_local(out.get("yes_bid_size_fp"))
            if fp is not None and fp > 0.0:
                out["yes_bid_size"] = int(float(fp) * 100.0)
        return out

    resp = session.get(f"{base_url}/markets/{ticker}", timeout=30.0)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        return {}
    market = data.get("market")
    if isinstance(market, dict):
        return _normalize_market_quotes_local(market)
    return _normalize_market_quotes_local(data)


def _parse_market_result(market: Dict[str, Any]) -> Tuple[bool, str]:
    result_raw = market.get("result")
    status = str(market.get("status") or "").strip().lower()
    if isinstance(result_raw, bool):
        return (True, "yes" if result_raw else "no")
    result = str(result_raw or "").strip().lower()
    if result in {"yes", "no"}:
        return (True, result)
    if status in {"settled", "resolved", "finalized"}:
        return (True, "")
    return (False, "")


def _evaluate_yes_pricing(
    *,
    yes_bid: Optional[int],
    yes_ask: Optional[int],
    p_true: float,
    config: Dict[str, Any],
    join_ticks_key: str,
    slippage_cents_key: str,
) -> Dict[str, Any]:
    min_yes_bid_cents = int(config.get("min_yes_bid_cents", 1))
    max_yes_spread_cents = int(config.get("max_yes_spread_cents", 10))
    if not passes_maker_liquidity(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        min_yes_bid_cents=min_yes_bid_cents,
        max_yes_spread_cents=max_yes_spread_cents,
    ):
        return {"ok": False, "reason": "liquidity_guard_failed"}

    mode = _shadow_execution_mode(config)
    slippage_dollars = float(config.get(slippage_cents_key, 0.25)) / 100.0
    if mode == "taker":
        if yes_ask is None:
            return {"ok": False, "reason": "ask_unavailable"}
        intended = int(yes_ask)
        if intended < 1 or intended > 99:
            return {"ok": False, "reason": "ask_out_of_range"}
        p_fill = 1.0
        price = float(intended) / 100.0
        fee_rate = _shadow_fee_rate(config, execution_mode=mode)
        fees = taker_fee_dollars(contracts=1, price=price, rate=fee_rate)
        ev = float(p_true) - price - float(fees) - slippage_dollars
    else:
        join_ticks = int(config.get(join_ticks_key, 1))
        intended = intended_maker_yes_price(yes_bid=yes_bid, yes_ask=yes_ask, join_ticks=join_ticks)
        if intended is None:
            return {"ok": False, "reason": "intended_price_unavailable"}

        p_fill = maker_fill_prob(yes_bid, yes_ask, intended)
        if p_fill <= 0.0:
            return {"ok": False, "reason": "fill_prob_zero"}

        price = float(intended) / 100.0
        fee_rate = _shadow_fee_rate(config, execution_mode=mode)
        fees = maker_fee_dollars(contracts=1, price=price, rate=fee_rate)
        ev = float(p_fill) * (float(p_true) - price - float(fees) - slippage_dollars)

    return {
        "ok": True,
        "execution_mode": mode,
        "intended_price_cents": int(intended),
        "p_fill": float(p_fill),
        "fees_dollars": float(fees),
        "slippage_dollars": float(slippage_dollars),
        "ev_dollars": float(ev),
    }


def _evaluate_sports_edge(
    *,
    order: Dict[str, Any],
    market: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    strategy = str(order.get("strategy_id") or "").strip().upper()
    order_ticker = str(order.get("ticker") or "").strip().upper()
    ticker = str(market.get("ticker") or order_ticker).strip().upper()
    event_ticker = str(market.get("event_ticker") or order.get("event_ticker") or "").strip().upper()
    title = str(market.get("title") or order.get("title") or "").strip()

    if strategy.startswith("NBA_") or ticker.startswith("KXNBA"):
        live_nba = lookup_nba_live_p_true(
            ticker=ticker,
            event_ticker=event_ticker,
            title=title,
            config=config,
        )
        if live_nba is None:
            return {"ok": False, "reason": "nba_live_odds_missing"}
        p_true = float(live_nba)
    elif strategy.startswith("NHL_") or ticker.startswith("KXNHL"):
        live_nhl = lookup_nhl_live_p_true(
            ticker=ticker,
            event_ticker=event_ticker,
            title=title,
            config=config,
        )
        if live_nhl is None:
            return {"ok": False, "reason": "nhl_live_odds_missing"}
        p_true = float(live_nhl)
    else:
        p_true = _p_true_from_order(order)
    if p_true is None:
        return {"ok": False, "reason": "sports_p_true_missing"}

    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))

    if strategy.startswith("NBA_") or ticker.startswith("KXNBA"):
        join_ticks_key = "nba_join_ticks"
        slippage_key = "nba_slippage_cents_per_contract"
        min_ev = float(config.get("nba_min_ev_dollars", config.get("sports_min_ev_dollars", 0.0)))
    elif strategy.startswith("NHL_") or ticker.startswith("KXNHL"):
        join_ticks_key = "nhl_join_ticks"
        slippage_key = "nhl_slippage_cents_per_contract"
        min_ev = float(config.get("nhl_min_ev_dollars", config.get("sports_min_ev_dollars", 0.0)))
    else:
        join_ticks_key = "sports_join_ticks"
        slippage_key = "sports_slippage_cents_per_contract"
        min_ev = float(config.get("sports_min_ev_dollars", 0.0))

    if join_ticks_key not in config:
        config[join_ticks_key] = int(config.get("sports_join_ticks", 1))
    if slippage_key not in config:
        config[slippage_key] = float(config.get("sports_slippage_cents_per_contract", 0.25))

    pricing = _evaluate_yes_pricing(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        p_true=float(p_true),
        config=config,
        join_ticks_key=join_ticks_key,
        slippage_cents_key=slippage_key,
    )
    if not bool(pricing.get("ok")):
        return {"ok": False, "reason": str(pricing.get("reason") or "sports_pricing_failed")}

    mid_mu = float(p_true)
    if yes_bid is not None and yes_ask is not None and yes_bid >= 0 and yes_ask > 0:
        mid_mu = ((float(yes_bid) + float(yes_ask)) / 2.0) / 100.0

    ev = float(pricing.get("ev_dollars") or 0.0)
    if ev <= float(min_ev):
        return {
            "ok": False,
            "reason": f"sports_ev_below_min:{ev:.6f}<{float(min_ev):.6f}",
        }

    return {
        "ok": True,
        "model": "sports",
        "reason": "",
        "execution_mode": str(pricing.get("execution_mode") or _shadow_execution_mode(config)),
        "intended_price_cents": int(pricing.get("intended_price_cents") or 0),
        "p_true": float(p_true),
        "p_fill": float(pricing.get("p_fill") or 0.0),
        "mu": float(mid_mu),
        "sigma": 0.0,
        "fees_dollars": float(pricing.get("fees_dollars") or 0.0),
        "slippage_dollars": float(pricing.get("slippage_dollars") or 0.0),
        "ev_dollars": float(ev),
        "forecast_updated": _utc_now_iso(),
        "station_id": "",
        "station_name": "",
        "station_lat": 0.0,
        "station_lon": 0.0,
    }


def _evaluate_weather_edge(
    *,
    order: Dict[str, Any],
    market: Dict[str, Any],
    config: Dict[str, Any],
    weather_cache_dir: Path,
) -> Optional[Dict[str, Any]]:
    kind = str(order.get("_weather_kind") or "").strip().upper()
    code = str(order.get("_weather_code") or "").strip().upper()
    target_date_raw = str(order.get("_weather_target_date") or "").strip()
    target_date = dt.date.fromisoformat(target_date_raw) if target_date_raw else None
    if not kind or not code or target_date is None:
        return None

    yes_sub_title = str(market.get("yes_sub_title") or "").strip()
    bounds = _parse_bucket_bounds(yes_sub_title)
    if bounds is None:
        return None
    lower, upper = bounds

    bucket_kind = _bucket_kind(lower=lower, upper=upper)
    cmp_hints = _infer_comparator_hints(
        " ".join(
            [
                str(market.get("rules_primary") or ""),
                str(market.get("title") or ""),
            ]
        )
    )
    if not _bucket_matches_comparator(bucket_kind=bucket_kind, hints=cmp_hints):
        return {
            "ok": False,
            "reason": f"bucket_comparator_mismatch:{bucket_kind}:{','.join(sorted(cmp_hints))}",
        }

    cache_ttl = float(config.get("weather_cache_ttl_minutes", 30.0))
    user_agent = str(
        config.get("weather_nws_user_agent", "Chimera_v4_NightWatch (kellypclarke-maker@github.com)")
    ).strip()
    station_ctx = _resolve_station_context(
        rules_primary=str(market.get("rules_primary") or ""),
        rules_secondary=str(market.get("rules_secondary") or ""),
        title=str(market.get("title") or ""),
        ticker_code=code,
        cache_dir=weather_cache_dir,
        ttl_minutes=cache_ttl,
        user_agent=user_agent,
    )
    if station_ctx is None:
        return {"ok": False, "reason": "station_id_missing"}
    station_id = str(station_ctx.station_id).strip().upper()
    station_coords = (float(station_ctx.station_lat), float(station_ctx.station_lon))
    station_name = str(station_ctx.station_name)

    forecast = _fetch_nws_forecast(
        code=station_id,
        lat=float(station_coords[0]),
        lon=float(station_coords[1]),
        cache_dir=weather_cache_dir,
        ttl_minutes=cache_ttl,
        user_agent=user_agent,
    )
    forecast_hourly = _fetch_nws_hourly_forecast(
        code=station_id,
        lat=float(station_coords[0]),
        lon=float(station_coords[1]),
        cache_dir=weather_cache_dir,
        ttl_minutes=cache_ttl,
        user_agent=user_agent,
    )
    observations = _fetch_station_observations(
        station_id=station_id,
        cache_dir=weather_cache_dir,
        ttl_minutes=cache_ttl,
        user_agent=user_agent,
    )

    mu = _forecast_mu_for_date(
        forecast_payload=forecast or {},
        target_date=target_date,
        kind=kind,
        hourly_payload=forecast_hourly,
        observed_payload=observations,
    )
    if mu is None:
        return {"ok": False, "reason": "forecast_mu_missing"}

    forecast_as_of_ts = _parse_forecast_timestamp(
        forecast_payload=forecast,
        forecast_hourly_payload=forecast_hourly,
        observations_payload=observations,
    )
    local_tz = _resolve_local_tz(
        target_date=target_date,
        hourly_payload=forecast_hourly,
        forecast_payload=forecast or {},
    )
    sigma, sigma_meta = _effective_weather_sigma(
        kind=kind,
        code=code,
        station_id=station_id,
        target_date=target_date,
        local_tz=local_tz,
        as_of_ts=forecast_as_of_ts,
        config=config,
        cache_dir=weather_cache_dir,
        cache_ttl_minutes=cache_ttl,
    )
    p_true = _bucket_probability(lower=lower, upper=upper, mu=float(mu), sigma=float(sigma))
    if not (0.0 <= p_true <= 1.0):
        return {"ok": False, "reason": "p_true_out_of_range"}

    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))
    pricing = _evaluate_yes_pricing(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        p_true=float(p_true),
        config=config,
        join_ticks_key="weather_join_ticks",
        slippage_cents_key="weather_slippage_cents_per_contract",
    )
    if not bool(pricing.get("ok")):
        return {"ok": False, "reason": str(pricing.get("reason") or "pricing_failed")}

    forecast_updated = ""
    props = forecast.get("properties") if isinstance(forecast, dict) else {}
    props_h = forecast_hourly.get("properties") if isinstance(forecast_hourly, dict) else {}
    if isinstance(props, dict):
        forecast_updated = str(props.get("updated") or props.get("generatedAt") or "").strip()
    if not forecast_updated and isinstance(props_h, dict):
        forecast_updated = str(props_h.get("updated") or props_h.get("generatedAt") or "").strip()
    if not forecast_updated:
        forecast_updated = _latest_observation_timestamp(observations)

    return {
        "ok": True,
        "model": "weather",
        "reason": "",
        "execution_mode": str(pricing.get("execution_mode") or "maker"),
        "intended_price_cents": int(pricing.get("intended_price_cents") or 0),
        "p_true": float(p_true),
        "p_fill": float(pricing.get("p_fill") or 0.0),
        "mu": float(mu),
        "sigma": float(sigma),
        "fees_dollars": float(pricing.get("fees_dollars") or 0.0),
        "slippage_dollars": float(pricing.get("slippage_dollars") or 0.0),
        "ev_dollars": float(pricing.get("ev_dollars") or 0.0),
        "forecast_updated": forecast_updated,
        "station_id": station_id,
        "station_name": station_name,
        "station_lat": float(station_coords[0]),
        "station_lon": float(station_coords[1]),
        "station_source": str(station_ctx.source),
        "cli_code": str(station_ctx.cli_code),
        "sigma_meta": sigma_meta,
    }


def _econ_pmf_state(*, config: Dict[str, Any], econ_cache_dir: Path) -> Dict[str, Any]:
    econ_cache_dir.mkdir(parents=True, exist_ok=True)
    fred_csv_path = econ_cache_dir / "fred_CPILFESL.csv"
    fred_ttl = float(config.get("econ_cache_ttl_minutes", 1440.0))
    fred_text = _fetch_fred_csv(path=fred_csv_path, ttl_minutes=fred_ttl)
    if fred_text is None:
        return {"ok": False, "reason": "fred_fetch_failed"}
    series = _parse_fred_series(fred_text)
    lookback_years = int(config.get("econ_lookback_years", 10))
    pmf, asof, lookback_months = _mom_pmf(series=series, lookback_years=lookback_years)
    if not pmf:
        return {"ok": False, "reason": "fred_pmf_empty"}
    return {
        "ok": True,
        "pmf": pmf,
        "asof": asof,
        "lookback_months": int(lookback_months),
    }


def _evaluate_econ_edge(
    *,
    market: Dict[str, Any],
    config: Dict[str, Any],
    econ_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(econ_state.get("ok")):
        return {"ok": False, "reason": str(econ_state.get("reason") or "econ_state_unavailable")}
    if not _is_cpi_core_market(
        {
            "ticker": str(market.get("ticker") or ""),
            "event_ticker": str(market.get("event_ticker") or ""),
            "title": str(market.get("title") or ""),
            "yes_sub_title": str(market.get("yes_sub_title") or ""),
        }
    ):
        return {"ok": False, "reason": "unsupported_econ_market"}

    exact = _parse_exact_percent(str(market.get("yes_sub_title") or ""))
    if exact is None:
        return {"ok": False, "reason": "exact_bucket_parse_failed"}

    pmf = econ_state.get("pmf")
    if not isinstance(pmf, dict):
        return {"ok": False, "reason": "econ_pmf_missing"}
    p_true = float(pmf.get(float(exact), 0.0))
    if not (0.0 <= p_true <= 1.0):
        return {"ok": False, "reason": "p_true_out_of_range"}

    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))
    pricing = _evaluate_yes_pricing(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        p_true=float(p_true),
        config=config,
        join_ticks_key="econ_join_ticks",
        slippage_cents_key="econ_slippage_cents_per_contract",
    )
    if not bool(pricing.get("ok")):
        return {"ok": False, "reason": str(pricing.get("reason") or "pricing_failed")}

    return {
        "ok": True,
        "model": "econ",
        "reason": "",
        "execution_mode": str(pricing.get("execution_mode") or "maker"),
        "intended_price_cents": int(pricing.get("intended_price_cents") or 0),
        "p_true": float(p_true),
        "p_fill": float(pricing.get("p_fill") or 0.0),
        "mu": float("nan"),
        "sigma": float("nan"),
        "fees_dollars": float(pricing.get("fees_dollars") or 0.0),
        "slippage_dollars": float(pricing.get("slippage_dollars") or 0.0),
        "ev_dollars": float(pricing.get("ev_dollars") or 0.0),
        "forecast_updated": "",
        "station_id": "",
        "station_name": "",
        "station_lat": 0.0,
        "station_lon": 0.0,
        "econ_exact": float(exact),
        "econ_asof": str(econ_state.get("asof") or ""),
        "econ_lookback_months": int(econ_state.get("lookback_months") or 0),
    }


def _crypto_is_touch(market: Dict[str, Any]) -> bool:
    blob = " ".join(
        [
            str(market.get("title") or ""),
            str(market.get("rules_primary") or ""),
            str(market.get("yes_sub_title") or ""),
            str(market.get("rules_secondary") or ""),
        ]
    ).lower()
    return ("crosses" in blob) or ("at any point" in blob) or ("immediately resolves" in blob)


def _evaluate_crypto_edge(
    *,
    market: Dict[str, Any],
    config: Dict[str, Any],
    crypto_cache_dir: Path,
) -> Dict[str, Any]:
    event_ticker = str(market.get("event_ticker") or "").strip().upper()
    product_id = _crypto_product_for_event(event_ticker)
    if not product_id:
        return {"ok": False, "reason": "crypto_product_unmapped"}

    as_of = _utc_now()

    granularity_s = int(config.get("crypto_granularity_s", 60))
    lookback_hours = float(config.get("crypto_lookback_hours", 24.0))
    cache_ttl = float(config.get("crypto_cache_ttl_minutes", 10.0))

    end = as_of
    start = as_of - dt.timedelta(hours=float(lookback_hours))

    candle_cache = crypto_cache_dir / f"candles_{product_id.replace('-', '_')}_{granularity_s}.json"
    candles = _crypto_fetch_candles(
        product_id=product_id,
        granularity_s=granularity_s,
        start=start,
        end=end,
        cache_path=candle_cache,
        ttl_minutes=cache_ttl,
    )
    if candles is None:
        return {"ok": False, "reason": "crypto_candles_unavailable"}
    est = _crypto_estimate_sigma_and_spot_from_candles(candles, granularity_s=granularity_s)
    if est is None:
        return {"ok": False, "reason": "crypto_sigma_estimate_failed"}
    candle_spot, sigma_per_sqrt_s = est

    spot_sources = config.get("crypto_spot_sources") or ["coinbase", "kraken", "bitstamp"]
    if not isinstance(spot_sources, list):
        spot_sources = ["coinbase", "kraken", "bitstamp"]

    spot_cache = crypto_cache_dir / f"spot_{product_id.replace('-', '_')}.json"
    proxy_spot, used_sources = _crypto_spot_proxy(
        product_id=product_id,
        sources=[str(x) for x in spot_sources],
        cache_path=spot_cache,
        ttl_minutes=cache_ttl,
    )

    spot = float(proxy_spot) if proxy_spot is not None else float(candle_spot)

    annual_vol_floor = float(config.get("crypto_sigma_floor_annual", 0.35))
    sigma_floor = _crypto_sigma_floor_per_sqrt_s(annual_vol_floor)
    sigma_per_sqrt_s = max(float(sigma_per_sqrt_s), float(sigma_floor))

    yes_sub_title = str(market.get("yes_sub_title") or "").strip()
    bounds = _crypto_parse_bounds(yes_sub_title)
    if bounds is None:
        return {"ok": False, "reason": "crypto_bounds_parse_failed"}
    lower, upper = bounds

    rules_primary = str(market.get("rules_primary") or "")
    target_dt = _crypto_parse_rule_target_dt(rules_primary)
    if target_dt is None:
        target_dt = _crypto_parse_iso_dt(str(market.get("close_time") or ""))
    if target_dt is None:
        return {"ok": False, "reason": "crypto_target_time_missing"}

    horizon_s = (target_dt - as_of).total_seconds()
    if horizon_s <= 0.0:
        return {"ok": False, "reason": "crypto_horizon_nonpositive"}

    settle_avg_window_s = float(config.get("crypto_settlement_avg_window_s", 60.0))
    effective_horizon_s = max(1.0, float(horizon_s) - (2.0 * float(settle_avg_window_s) / 3.0))
    sigma_total = float(sigma_per_sqrt_s) * (float(effective_horizon_s) ** 0.5)

    is_touch = _crypto_is_touch(market)
    if is_touch and upper is None and lower is not None:
        p_true = _crypto_p_hit_upper(s0=spot, barrier=float(lower), sigma_total=sigma_total)
        kind = "touch_up"
    elif is_touch and lower is None and upper is not None:
        p_true = _crypto_p_hit_lower(s0=spot, barrier=float(upper), sigma_total=sigma_total)
        kind = "touch_down"
    else:
        p_true = _crypto_p_terminal_between(s0=spot, sigma_total=sigma_total, lo=lower, hi=upper)
        kind = "terminal"

    if not (0.0 <= float(p_true) <= 1.0):
        return {"ok": False, "reason": "crypto_p_true_out_of_range"}

    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))
    pricing = _evaluate_yes_pricing(
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        p_true=float(p_true),
        config=config,
        join_ticks_key="crypto_join_ticks",
        slippage_cents_key="crypto_slippage_cents_per_contract",
    )
    if not bool(pricing.get("ok")):
        return {"ok": False, "reason": str(pricing.get("reason") or "pricing_failed")}

    return {
        "ok": True,
        "model": "crypto",
        "reason": "",
        "execution_mode": str(pricing.get("execution_mode") or "maker"),
        "intended_price_cents": int(pricing.get("intended_price_cents") or 0),
        "p_true": float(p_true),
        "p_fill": float(pricing.get("p_fill") or 0.0),
        "mu": float(spot),
        "sigma": float(sigma_per_sqrt_s),
        "fees_dollars": float(pricing.get("fees_dollars") or 0.0),
        "slippage_dollars": float(pricing.get("slippage_dollars") or 0.0),
        "ev_dollars": float(pricing.get("ev_dollars") or 0.0),
        "forecast_updated": as_of.isoformat(),
        "station_id": "",
        "station_name": "",
        "station_lat": 0.0,
        "station_lon": 0.0,
        "econ_exact": float("nan"),
        "econ_asof": "",
        "econ_lookback_months": 0,
        "crypto_product": product_id,
        "crypto_kind": kind,
        "crypto_spot_sources": ",".join([str(x) for x in (used_sources or [])]),
    }


def _close_order_unfilled(order: Dict[str, Any], *, reason: str, now_iso: str, audit_note: str = "") -> None:
    order["status"] = "closed_unfilled"
    order["resolved_ts"] = now_iso
    order["resolved_payout_dollars"] = "0.000000"
    order["realized_pnl_dollars"] = "0.000000"
    order["_runtime_roi"] = "0.000000"
    order["notes"] = _append_note(str(order.get("notes") or ""), reason)
    if audit_note:
        order["notes"] = _append_note(str(order.get("notes") or ""), audit_note)


def _update_open_order_lifecycle(
    *,
    order: Dict[str, Any],
    market: Dict[str, Any],
    config: Dict[str, Any],
    weather_cache_dir: Path,
    econ_state: Dict[str, Any],
) -> None:
    if str(order.get("status") or "") != "open":
        return

    now_iso = _utc_now_iso()
    category = str(order.get("category") or "").strip()
    strategy_id = str(order.get("strategy_id") or "").strip()
    order_ticker = str(order.get("ticker") or "").strip().upper()
    ticker = str(market.get("ticker") or order_ticker).strip().upper()
    if ticker:
        order["_runtime_market_ticker"] = ticker
    yes_bid = safe_int(market.get("yes_bid"))
    yes_ask = safe_int(market.get("yes_ask"))
    print(
        f"[SHADOW][EVAL] ticker={ticker} category={category or 'NA'} strategy={strategy_id or 'NA'} "
        f"yes_bid={yes_bid} yes_ask={yes_ask}"
    )
    if category == "Climate and Weather" or strategy_id.startswith("WTHR_"):
        eval_out = _evaluate_weather_edge(order=order, market=market, config=config, weather_cache_dir=weather_cache_dir)
    elif _is_sports_order(order):
        eval_out = _evaluate_sports_edge(order=order, market=market, config=config)
    elif category == "Crypto" or strategy_id.startswith("CRYPTO_"):
        eval_out = _evaluate_crypto_edge(market=market, config=config, crypto_cache_dir=(ROOT / "data" / "external" / "crypto"))
    elif category == "Economics" or strategy_id.startswith("ECON_"):
        eval_out = _evaluate_econ_edge(market=market, config=config, econ_state=econ_state)
    else:
        eval_out = {"ok": False, "reason": "unsupported_strategy_or_category"}

    if isinstance(eval_out, dict) and bool(eval_out.get("ok")):
        print(
            f"[SHADOW][EVAL] ticker={ticker} model={str(eval_out.get('model') or '')} "
            f"p_true={float(eval_out.get('p_true') or 0.0):.6f} "
            f"ev={float(eval_out.get('ev_dollars') or 0.0):.6f} "
            f"intended={int(float(str(eval_out.get('intended_price_cents') or '0') or '0'))}"
        )
    else:
        reason = str(eval_out.get("reason") if isinstance(eval_out, dict) else "model_eval_failed")
        print(f"[SHADOW][EVAL] ticker={ticker} skipped reason={reason}")

    if not isinstance(eval_out, dict) or not bool(eval_out.get("ok")):
        reason = str(eval_out.get("reason") if isinstance(eval_out, dict) else "model_eval_failed")
        station_id = str(eval_out.get("station_id") if isinstance(eval_out, dict) else "")
        _close_order_unfilled(
            order,
            reason=f"edge_gone:{reason}",
            now_iso=now_iso,
            audit_note=f"station_id={station_id}",
        )
        return

    model = str(eval_out.get("model") or "")
    if model == "econ":
        min_ev = float(config.get("shadow_min_ev_dollars", config.get("econ_min_ev_dollars", 0.0)))
    elif model == "crypto":
        min_ev = float(config.get("shadow_min_ev_dollars", config.get("crypto_min_ev_dollars", 0.0)))
    elif model == "sports":
        min_ev = float(config.get("shadow_min_ev_dollars", config.get("sports_min_ev_dollars", 0.0)))
    else:
        min_ev = float(config.get("shadow_min_ev_dollars", config.get("weather_min_ev_dollars", 0.0)))
    exec_mode = str(eval_out.get("execution_mode") or _shadow_execution_mode(config)).strip().lower()
    if exec_mode not in {"maker", "taker"}:
        exec_mode = "maker"
    ev = float(eval_out.get("ev_dollars") or 0.0)
    order["maker_or_taker"] = exec_mode
    order["action"] = ("buy_yes" if exec_mode == "taker" else "post_yes")

    order["fees_assumed_dollars"] = f"{float(eval_out.get('fees_dollars') or 0.0):.6f}"
    order["slippage_assumed_dollars"] = f"{float(eval_out.get('slippage_dollars') or 0.0):.6f}"
    order["_runtime_last_ev"] = f"{ev:.6f}"
    new_p_true = float(eval_out.get("p_true") or 0.0)
    old_p_true = _parse_float(order.get("_runtime_last_p_true"))
    if model == "sports" and (old_p_true is None or abs(float(old_p_true) - float(new_p_true)) > 1e-9):
        print(f"[SHADOW][ORACLE] Updated p_true for {ticker}: {new_p_true:.6f}")
    order["_runtime_last_p_true"] = f"{new_p_true:.6f}"
    order["_runtime_last_p_fill"] = f"{float(eval_out.get('p_fill') or 0.0):.6f}"
    order["_runtime_last_mu"] = f"{float(eval_out.get('mu') or 0.0):.6f}"
    order["_runtime_last_sigma"] = f"{float(eval_out.get('sigma') or 0.0):.6f}"
    order["_runtime_last_forecast_updated"] = str(eval_out.get("forecast_updated") or "")
    order["_runtime_station_id"] = str(eval_out.get("station_id") or "")
    order["_runtime_station_name"] = str(eval_out.get("station_name") or "")
    order["_runtime_station_lat"] = f"{float(eval_out.get('station_lat') or 0.0):.6f}"
    order["_runtime_station_lon"] = f"{float(eval_out.get('station_lon') or 0.0):.6f}"
    order["notes"] = _upsert_note_field(
        str(order.get("notes") or ""),
        prefix="latest_eval=",
        value=(
            f"ev={ev:.6f},mu={float(eval_out.get('mu') or 0.0):.4f},"
            f"sigma={float(eval_out.get('sigma') or 0.0):.4f},"
            f"p_true={float(eval_out.get('p_true') or 0.0):.6f},"
            f"forecast_updated={str(eval_out.get('forecast_updated') or '')},"
            f"station_id={str(eval_out.get('station_id') or '')},"
            f"model={model},econ_asof={str(eval_out.get('econ_asof') or '')}"
        ),
    )

    if ev <= min_ev:
        _close_order_unfilled(
            order,
            reason=f"edge_gone:ev={ev:.6f}",
            now_iso=now_iso,
            audit_note=(
                f"mu={float(eval_out.get('mu') or 0.0):.4f};sigma={float(eval_out.get('sigma') or 0.0):.4f};"
                f"p_true={float(eval_out.get('p_true') or 0.0):.6f};"
                f"forecast_updated={str(eval_out.get('forecast_updated') or '')};"
                f"station_id={str(eval_out.get('station_id') or '')}"
            ),
        )
        return

    try:
        new_px = int(float(str(eval_out.get("intended_price_cents") or "0") or "0"))
    except Exception:
        _close_order_unfilled(
            order,
            reason="edge_gone:intended_price_invalid",
            now_iso=now_iso,
            audit_note=(
                f"mu={float(eval_out.get('mu') or 0.0):.4f};sigma={float(eval_out.get('sigma') or 0.0):.4f};"
                f"p_true={float(eval_out.get('p_true') or 0.0):.6f};"
                f"forecast_updated={str(eval_out.get('forecast_updated') or '')};"
                f"station_id={str(eval_out.get('station_id') or '')}"
            ),
        )
        return

    if exec_mode == "taker":
        order["intended_price_cents"] = int(new_px)
        order["assumed_fill_price_cents"] = int(new_px)
        order["status"] = "filled"
        order["_runtime_filled_ts"] = now_iso
        order["_runtime_fill_trade_id"] = str(order.get("_runtime_fill_trade_id") or "")
        order["_runtime_fill_count_seen"] = max(1, int(float(str(order.get("_runtime_fill_count_seen") or "0") or "0")))
        order["notes"] = _append_note(str(order.get("notes") or ""), "shadow_taker_fill=immediate")
        qty = safe_int(order.get("size_contracts"))
        if qty is not None and qty > 0:
            log_shadow_trade(
                oracle_type=_infer_oracle_type(
                    ticker=str(order.get("ticker") or ""),
                    category=str(order.get("category") or ""),
                ),
                ticker=str(order.get("ticker") or ""),
                action="buy_yes",
                price_cents=int(new_px),
                quantity=int(qty),
                spot_price=float(eval_out.get("mu") or 0.0),
                expected_value=float(ev),
            )
        return

    _apply_cancel_replace(
        order,
        new_price_cents=new_px,
        now_iso=now_iso,
        mu=float(eval_out.get("mu") or 0.0),
        sigma=float(eval_out.get("sigma") or 0.0),
        p_true=float(eval_out.get("p_true") or 0.0),
        forecast_updated=str(eval_out.get("forecast_updated") or ""),
        station_id=str(eval_out.get("station_id") or ""),
        ev_dollars=float(ev),
    )


def _prune_open_orders_by_event(*, orders: List[Dict[str, Any]], max_open_orders_per_event: int) -> None:
    cap = max(1, int(max_open_orders_per_event))
    by_event: Dict[str, List[Dict[str, Any]]] = {}
    for order in orders:
        if str(order.get("status") or "") != "open":
            continue
        if str(order.get("category") or "").strip() != "Climate and Weather":
            continue
        event_ticker = str(order.get("event_ticker") or "").strip().upper()
        if not event_ticker:
            continue
        by_event.setdefault(event_ticker, []).append(order)

    now_iso = _utc_now_iso()
    for event_ticker, event_orders in by_event.items():
        if len(event_orders) <= cap:
            continue
        ranked = sorted(
            event_orders,
            key=lambda o: (
                float(str(o.get("_runtime_last_ev") or "-1000000000") or "-1000000000"),
                str(o.get("ticker") or ""),
            ),
            reverse=True,
        )
        for loser in ranked[cap:]:
            _close_order_unfilled(
                loser,
                reason=f"event_exposure_prune:{event_ticker}",
                now_iso=now_iso,
                audit_note=f"kept_top={cap}",
            )


def _mark_fill(order: Dict[str, Any], *, matching_trades: Sequence[Dict[str, Any]]) -> None:
    if str(order.get("status") or "") != "open":
        return
    if not matching_trades:
        return

    first = sorted(matching_trades, key=lambda tr: _parse_ts(tr.get("created_time")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))[0]
    filled_ts = _parse_ts(first.get("created_time"))
    trade_id = str(first.get("trade_id") or "").strip()

    try:
        px = int(float(str(order.get("intended_price_cents") or "0") or "0"))
    except Exception:
        px = 0

    order["status"] = "filled"
    order["assumed_fill_price_cents"] = px
    order["size_contracts"] = str(order.get("size_contracts") or "1")
    order["_runtime_filled_ts"] = filled_ts.isoformat() if filled_ts is not None else _utc_now_iso()
    order["_runtime_fill_trade_id"] = trade_id
    order["_runtime_fill_count_seen"] = int(len(matching_trades))
    order["_runtime_first_fill_trade_id"] = trade_id
    order["notes"] = _append_note(order.get("notes", ""), f"fill_trade_id={trade_id}")
    order["notes"] = _append_note(order.get("notes", ""), f"fill_count_seen={len(matching_trades)}")


def _grade_resolutions(
    *,
    orders: List[Dict[str, Any]],
    market_cache: Dict[str, Dict[str, Any]],
    default_execution_mode: str,
    maker_fee_rate: float,
    taker_fee_rate: float,
) -> None:
    now_iso = _utc_now_iso()
    for order in orders:
        status = str(order.get("status") or "")
        if status not in {"open", "filled"}:
            continue
        ticker = str(order.get("ticker") or "").strip().upper()
        market = market_cache.get(ticker)
        if not isinstance(market, dict):
            continue
        settled, result = _parse_market_result(market)
        if not settled:
            continue

        if status == "open":
            _close_order_unfilled(order, reason="settled_unfilled", now_iso=now_iso)
            continue

        # Filled and settled.
        try:
            fill_cents = int(float(str(order.get("assumed_fill_price_cents") or "0") or "0"))
            size = int(float(str(order.get("size_contracts") or "1") or "1"))
        except Exception:
            continue

        payout = 1.0 if result == "yes" else 0.0
        fill_price = float(fill_cents) / 100.0
        exec_mode = str(order.get("maker_or_taker") or default_execution_mode).strip().lower()
        if exec_mode == "taker":
            fees = taker_fee_dollars(contracts=size, price=fill_price, rate=taker_fee_rate)
        else:
            fees = maker_fee_dollars(contracts=size, price=fill_price, rate=maker_fee_rate)
        pnl = float(size) * (payout - fill_price) - float(fees)
        deployed = float(size) * fill_price + float(fees)
        roi = pnl / deployed if deployed > 0.0 else 0.0

        order["status"] = "resolved"
        order["resolved_ts"] = now_iso
        order["resolved_payout_dollars"] = f"{payout:.6f}"
        order["fees_assumed_dollars"] = f"{fees:.6f}"
        order["realized_pnl_dollars"] = f"{pnl:.6f}"
        order["_runtime_roi"] = f"{roi:.6f}"
        order["notes"] = _append_note(order.get("notes", ""), f"result={result}")


def _render_summary(*, day: str, orders: Sequence[Dict[str, Any]], poll_seconds: int, max_runtime_minutes: float) -> str:
    n_total = len(orders)
    n_open = sum(1 for o in orders if str(o.get("status") or "") == "open")
    n_filled = sum(1 for o in orders if str(o.get("status") or "") == "filled")
    n_resolved = sum(1 for o in orders if str(o.get("status") or "") == "resolved")
    n_closed_unfilled = sum(1 for o in orders if str(o.get("status") or "") == "closed_unfilled")

    fills = [o for o in orders if str(o.get("_runtime_fill_trade_id") or "").strip()]
    fills_detected = len(fills)

    pnl_sum = 0.0
    deployed_sum = 0.0
    for o in orders:
        try:
            pnl_sum += float(str(o.get("realized_pnl_dollars") or "0") or "0")
        except Exception:
            pass
        if str(o.get("status") or "") == "resolved":
            try:
                size = int(float(str(o.get("size_contracts") or "1") or "1"))
                fill_cents = int(float(str(o.get("assumed_fill_price_cents") or "0") or "0"))
                fees = float(str(o.get("fees_assumed_dollars") or "0") or "0")
                deployed_sum += float(size) * (float(fill_cents) / 100.0) + fees
            except Exception:
                pass
    roi_total = pnl_sum / deployed_sum if deployed_sum > 0.0 else 0.0

    lines: List[str] = []
    lines.append(f"# Shadow Summary {day}")
    lines.append("")
    lines.append(f"- Generated: `{_utc_now_iso()}`")
    lines.append("- Mode: `paper/shadow only`")
    lines.append(f"- Poll seconds: `{poll_seconds}`")
    lines.append(f"- Max runtime minutes: `{max_runtime_minutes}`")
    lines.append(f"- Orders tracked: `{n_total}`")
    lines.append(f"- Open: `{n_open}`")
    lines.append(f"- Filled (awaiting resolution): `{n_filled}`")
    lines.append(f"- Resolved: `{n_resolved}`")
    lines.append(f"- Closed unfilled: `{n_closed_unfilled}`")
    lines.append(f"- Fills detected: `{fills_detected}`")
    lines.append(f"- Realized PnL ($): `{pnl_sum:.6f}`")
    lines.append(f"- Realized ROI: `{roi_total:.6f}`")
    lines.append("")
    lines.append("## Orders")
    lines.append("")
    lines.append("| ticker | strategy_id | category | status | yes_px | rev | fill_trade_id | pnl | roi | notes |")
    lines.append("|---|---|---|---|---:|---:|---|---:|---:|---|")
    for o in sorted(orders, key=lambda r: str(r.get("ticker") or "")):
        lines.append(
            f"| {str(o.get('ticker') or '')} | {str(o.get('strategy_id') or '')} | {str(o.get('category') or '')} | "
            f"{str(o.get('status') or '')} | {str(o.get('intended_price_cents') or '')} | "
            f"{str(o.get('_runtime_revision_count') or 0)} | {str(o.get('_runtime_fill_trade_id') or '')} | "
            f"{str(o.get('realized_pnl_dollars') or '')} | {str(o.get('_runtime_roi') or '')} | {str(o.get('notes') or '')} |"
        )
    lines.append("")
    return "\n".join(lines)


def _all_done(orders: Sequence[Dict[str, Any]]) -> bool:
    if not orders:
        return True
    for o in orders:
        if str(o.get("status") or "") in {"open", "filled"}:
            return False
    return True


def _resolve_output_path(raw: str, default_path: Path) -> Path:
    s = str(raw or "").strip()
    if not s:
        return default_path
    p = Path(s)
    if p.is_absolute():
        return p
    return ROOT / p


def _fetch_all_trades_for_ticker(*, session: requests.Session, base_url: str, ticker: str, limit: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    cursor = ""
    pages = 0
    while pages < 200:
        params: Dict[str, object] = {"ticker": ticker, "limit": int(limit)}
        if cursor:
            params["cursor"] = cursor
        try:
            resp = session.get(f"{base_url}/markets/trades", params=params, timeout=30.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            break
        if not isinstance(payload, dict):
            break
        trades = payload.get("trades")
        if not isinstance(trades, list) or not trades:
            break
        for tr in trades:
            if isinstance(tr, dict):
                out.append(tr)
        next_cursor = str(payload.get("cursor") or "").strip()
        pages += 1
        if not next_cursor:
            break
        cursor = next_cursor
    return out


def _qualifying_trades_in_window(
    *,
    order: Dict[str, Any],
    trades: Sequence[Dict[str, Any]],
    start_ts: dt.datetime,
    end_ts: dt.datetime,
) -> List[Dict[str, Any]]:
    try:
        target_px = int(float(str(order.get("intended_price_cents") or "0") or "0"))
    except Exception:
        return []
    out: List[Dict[str, Any]] = []
    for tr in trades:
        if not isinstance(tr, dict):
            continue
        tts = _parse_ts(tr.get("created_time"))
        if tts is None:
            continue
        if tts < start_ts or tts > end_ts:
            continue
        taker_side = str(tr.get("taker_side") or "").strip().lower()
        try:
            yes_px = int(float(str(tr.get("yes_price") or "")))
        except Exception:
            continue
        if taker_side == "no" and yes_px == target_px:
            out.append(tr)
    out.sort(key=lambda tr: _parse_ts(tr.get("created_time")) or dt.datetime.min.replace(tzinfo=dt.timezone.utc))
    return out


def _run_backtest_day(
    *,
    day: str,
    cfg: Dict[str, Any],
    base_url: str,
    tickers_override: Sequence[str],
    ledger_path: Path,
    summary_path: Path,
) -> Tuple[Path, Path]:
    execution_mode = _shadow_execution_mode(cfg)
    maker_fee_rate = _shadow_fee_rate(cfg, execution_mode="maker")
    taker_fee_rate = _shadow_fee_rate(cfg, execution_mode="taker")
    size_contracts = max(1, int(cfg.get("base_size_contracts", 1)))
    candidates = _discover_shadow_candidates(day=day, tickers_override=tickers_override, config=cfg)
    orders = [
        _build_order_from_candidate(
            row=row,
            day=day,
            size_contracts=size_contracts,
            execution_mode=execution_mode,
        )
        for row in candidates
    ]

    # Backtest design: deterministic single-quote simulation.
    #
    # We intentionally do NOT run the live repricing/edge-gating lifecycle here because it uses
    # wall-clock timestamps for cancel/replace (which would time-travel created_ts).
    #
    # Instead, we:
    # - treat each candidate's `data_as_of_ts` as the quote start time,
    # - keep the candidate's intended maker price fixed,
    # - detect fills conservatively via public trade prints within [created_ts, close_time].

    def _parse_kv(blob: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in str(blob or '').split(';'):
            p = part.strip()
            if not p or '=' not in p:
                continue
            k, v = p.split('=', 1)
            kk = k.strip()
            if kk:
                out[kk] = v.strip()
        return out

    with requests.Session() as s:
        for row, order in zip(candidates, orders):
            if str(order.get('status') or '') != 'open':
                continue
            ticker = str(order.get('ticker') or '').strip().upper()
            if not ticker:
                continue

            # Attach model evidence from the candidate snapshot (stable across time).
            kv = _parse_kv(str(row.get('liquidity_notes') or ''))
            if kv:
                order['_runtime_last_p_true'] = str(kv.get('p_true') or '')
                order['_runtime_last_mu'] = str(kv.get('mu') or '')
                order['_runtime_last_sigma'] = str(kv.get('sigma') or '')
                order['_runtime_last_forecast_updated'] = str(kv.get('forecast_ts') or '')
                order['_runtime_station_id'] = str(kv.get('station_id') or row.get('station_id') or '')
                order['notes'] = _append_note(
                    str(order.get('notes') or ''),
                    f"bt_mu={order['_runtime_last_mu']};bt_sigma={order['_runtime_last_sigma']};bt_p_true={order['_runtime_last_p_true']}",
                )

            created_ts = _parse_ts(order.get('created_ts')) or _utc_now()
            close_ts = _parse_ts(order.get('close_time')) or created_ts
            if close_ts < created_ts:
                close_ts = created_ts

            trades = _fetch_all_trades_for_ticker(session=s, base_url=base_url, ticker=ticker)
            matches = _qualifying_trades_in_window(order=order, trades=trades, start_ts=created_ts, end_ts=close_ts)
            if matches:
                _mark_fill(order, matching_trades=matches[:1])
            else:
                _close_order_unfilled(order, reason='backtest_no_fill', now_iso=_utc_now_iso())

        market_cache: Dict[str, Dict[str, Any]] = {}
        for order in orders:
            ticker = str(order.get('ticker') or '').strip().upper()
            if not ticker:
                continue
            try:
                market_cache[ticker] = _fetch_market(session=s, base_url=base_url, ticker=ticker)
            except Exception:
                continue

        # Add bucket bounds + final result markers for calibration.
        for order in orders:
            ticker = str(order.get('ticker') or '').strip().upper()
            market = market_cache.get(ticker)
            if not isinstance(market, dict):
                continue
            bounds = _parse_bucket_bounds(str(market.get('yes_sub_title') or ''))
            lower = '' if bounds is None or bounds[0] is None else str(bounds[0])
            upper = '' if bounds is None or bounds[1] is None else str(bounds[1])
            order['notes'] = _append_note(str(order.get('notes') or ''), f'bt_lower={lower};bt_upper={upper}')
            settled, res = _parse_market_result(market)
            if settled and res in {'yes', 'no'}:
                order['notes'] = _append_note(str(order.get('notes') or ''), f'bt_result={res}')

        _grade_resolutions(
            orders=orders,
            market_cache=market_cache,
            default_execution_mode=execution_mode,
            maker_fee_rate=float(maker_fee_rate),
            taker_fee_rate=float(taker_fee_rate),
        )

    state = {'orders': orders}
    return _save_outputs(
        day=day,
        state=state,
        poll_seconds=0,
        max_runtime_minutes=0.0,
        ledger_path=ledger_path,
        summary_path=summary_path,
    )

def _save_outputs(
    *,
    day: str,
    state: Dict[str, Any],
    poll_seconds: int,
    max_runtime_minutes: float,
    ledger_path: Path,
    summary_path: Path,
) -> Tuple[Path, Path]:
    orders = [o for o in state.get("orders", []) if isinstance(o, dict)]
    upsert_rows(ledger_path, _project_canonical_rows(orders))

    summary = _render_summary(
        day=day,
        orders=orders,
        poll_seconds=poll_seconds,
        max_runtime_minutes=max_runtime_minutes,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(summary, encoding="utf-8")
    return (ledger_path, summary_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run generic shadow quoting + fill + grading (paper only).")
    parser.add_argument("--config", default=str(ROOT / "config" / "defaults.json"), help="Path to config JSON.")
    parser.add_argument("--date", default="", help="Date YYYY-MM-DD (default UTC today).")
    parser.add_argument("--tag", default="", help="Optional output tag to avoid overwriting.")
    parser.add_argument("--poll-seconds", type=float, default=30.0, help="Polling interval seconds.")
    parser.add_argument("--max-runtime-minutes", type=float, default=180.0, help="Max runtime minutes; 0=until done.")
    parser.add_argument("--tickers", default="", help="Optional comma-separated ticker override.")
    parser.add_argument("--size-contracts", type=int, default=1, help="Contracts per shadow order (new orders only).")
    parser.add_argument("--force-size-contracts", action="store_true", help="Force size_contracts for all open/filled orders to --size-contracts.")
    parser.add_argument("--backtest", action="store_true", help="Run deterministic backtest mode (no sleep/poll loop).")
    parser.add_argument("--state-path", default="", help="Optional state JSON path.")
    parser.add_argument("--ledger-path", default="", help="Optional ledger CSV output path.")
    parser.add_argument("--summary-path", default="", help="Optional summary markdown output path.")
    args = parser.parse_args()
    print(f"[DEBUG] Entry point reached. Target Date: {args.date}")
    _load_env_list_defaults(REPO_ROOT / "env.list")

    day = str(args.date).strip() or _utc_now().date().isoformat()
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", str(args.tag or "").strip()).strip("-")
    suffix = f"_{tag}" if tag else ""
    day_key = f"{day}{suffix}"
    poll_seconds = max(0.1, float(args.poll_seconds))
    max_runtime_minutes = float(args.max_runtime_minutes)

    cfg = _load_config(Path(str(args.config)))
    _normalize_ev_thresholds_for_shadow_test(cfg)
    base_url = str(cfg.get("base_url") or DEFAULT_BASE).strip().rstrip("/") or DEFAULT_BASE
    execution_mode = _shadow_execution_mode(cfg)
    maker_fee_rate = _shadow_fee_rate(cfg, execution_mode="maker")
    taker_fee_rate = _shadow_fee_rate(cfg, execution_mode="taker")
    seen_cap = int(cfg.get("shadow_seen_trade_ids_max", 2000))

    default_state_path = ROOT / "data" / "shadow" / f"{day_key}_state.json"
    default_ledger_path = ROOT / "reports" / "shadow" / f"{day_key}_shadow_ledger.csv"
    default_summary_path = ROOT / "reports" / "shadow" / f"{day_key}_shadow_summary.md"
    state_path = _resolve_output_path(args.state_path, default_state_path)
    ledger_path = _resolve_output_path(args.ledger_path, default_ledger_path)
    summary_path = _resolve_output_path(args.summary_path, default_summary_path)

    tickers_override = _parse_ticker_override(args.tickers)
    if tickers_override:
        print(f"[SHADOW][DISCOVERY] ticker override active count={len(tickers_override)}")
    else:
        print(f"[SHADOW][DISCOVERY] --tickers omitted; running dynamic discovery for date={day}")
    if bool(args.backtest):
        out_ledger, out_summary = _run_backtest_day(
            day=day,
            cfg=cfg,
            base_url=base_url,
            tickers_override=tickers_override,
            ledger_path=ledger_path,
            summary_path=summary_path,
        )
        print(f"shadow_ledger={out_ledger}")
        print(f"shadow_summary={out_summary}")
        return 0

    candidates = _discover_shadow_candidates(day=day, tickers_override=tickers_override, config=cfg)

    state = _read_json(state_path)

    # If no state but a canonical ledger exists, bootstrap from it.
    if not state:
        ledger_path = ROOT / "reports" / "shadow" / f"{day_key}_shadow_ledger.csv"
        ledger_rows = load_ledger(ledger_path)
        if ledger_rows:
            state = {
                "run_date": day_key,
                "created_ts": _utc_now_iso(),
                "updated_ts": _utc_now_iso(),
                "orders": list(ledger_rows.values()),
                "tickers": {},
            }

    state = _initialize_or_update_state(
        state=state,
        day=day_key,
        candidates=candidates,
        size_contracts=int(args.size_contracts),
        execution_mode=execution_mode,
    )
    if bool(args.force_size_contracts):
        for o in state.get("orders") or []:
            if not isinstance(o, dict):
                continue
            if str(o.get("status") or "") in {"open", "filled"}:
                o["size_contracts"] = str(max(1, int(args.size_contracts)))
    _write_json(state_path, state)

    weather_cache_dir = ROOT / "data" / "external" / "weather"
    weather_cache_dir.mkdir(parents=True, exist_ok=True)
    econ_cache_dir = ROOT / "data" / "external" / "econ"
    econ_cache_dir.mkdir(parents=True, exist_ok=True)

    start_monotonic = time.monotonic()
    cycle = 0
    last_empty_discovery_attempt = 0.0
    crypto_series = str(cfg.get("shadow_crypto_series_ticker", "KXBTC")).strip().upper() or "KXBTC"
    btc_priority_prefix = f"{crypto_series}-"

    while True:
        cycle += 1
        orders = [o for o in state.get("orders", []) if isinstance(o, dict)]
        n_open = sum(1 for o in orders if str(o.get("status") or "") == "open")
        n_filled = sum(1 for o in orders if str(o.get("status") or "") == "filled")
        n_resolved = sum(1 for o in orders if str(o.get("status") or "") == "resolved")
        n_closed = sum(1 for o in orders if str(o.get("status") or "") == "closed_unfilled")
        elapsed_s = time.monotonic() - start_monotonic
        print(
            f"[SHADOW][HEARTBEAT] cycle={cycle} elapsed_s={elapsed_s:.1f} "
            f"orders_total={len(orders)} open={n_open} filled={n_filled} "
            f"resolved={n_resolved} closed_unfilled={n_closed}"
        )

        if not orders and not tickers_override:
            now_mono = time.monotonic()
            rediscovery_every_s = max(10.0, float(poll_seconds))
            if (now_mono - last_empty_discovery_attempt) >= rediscovery_every_s:
                last_empty_discovery_attempt = now_mono
                print("[SHADOW][DISCOVERY] state has zero orders; retrying dynamic discovery")
                fresh_candidates = _discover_shadow_candidates(day=day, tickers_override=(), config=cfg)
                if fresh_candidates:
                    state = _initialize_or_update_state(
                        state=state,
                        day=day_key,
                        candidates=fresh_candidates,
                        size_contracts=int(args.size_contracts),
                        execution_mode=execution_mode,
                    )
                    orders = [o for o in state.get("orders", []) if isinstance(o, dict)]
                    print(f"[SHADOW][DISCOVERY] rediscovery added orders={len(orders)}")
                else:
                    print("[SHADOW][DISCOVERY] rediscovery found 0 candidates")

        parent_sports_event_tickers = sorted(
            {
                str(o.get("ticker") or "").strip().upper()
                for o in orders
                if str(o.get("status") or "").strip().lower() in {"open", "filled"}
                and _is_parent_sports_event_ticker(str(o.get("ticker") or ""))
            }
        )
        parent_sports_event_markets: Dict[str, List[str]] = {}
        if parent_sports_event_tickers:
            with requests.Session() as expand_session:
                for parent_event_ticker in parent_sports_event_tickers:
                    expanded_markets = _expand_parent_sports_event_to_market_tickers(
                        session=expand_session,
                        base_url=base_url,
                        parent_event_ticker=parent_event_ticker,
                    )
                    if expanded_markets:
                        parent_sports_event_markets[parent_event_ticker] = expanded_markets

        open_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
        needed_market_tickers: set[str] = set()
        open_sports_tickers: set[str] = set()
        priority_active = True
        resolution_lookahead_minutes = float(cfg.get("shadow_resolution_lookahead_minutes", 0.0))
        resolution_cutoff = _utc_now() + dt.timedelta(minutes=float(resolution_lookahead_minutes))
        for o in orders:
            ticker = str(o.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            status = str(o.get("status") or "").strip().lower()
            runtime_ticker = ticker
            if _is_parent_sports_event_ticker(ticker):
                expanded = parent_sports_event_markets.get(ticker, [])
                runtime_ticker = (expanded[0] if expanded else "")
                if not runtime_ticker:
                    print(f"[SHADOW][DISCOVERY] unresolved parent event ticker={ticker} (no child market ticker)")

            prev_runtime_ticker = str(o.get("_runtime_market_ticker") or "").strip().upper()
            o["_runtime_market_ticker"] = runtime_ticker
            if runtime_ticker and prev_runtime_ticker != runtime_ticker:
                print(f"[SHADOW][DISCOVERY] order ticker remap order={ticker} runtime_market={runtime_ticker}")
            if not runtime_ticker:
                continue

            if status == "open":
                needed_market_tickers.add(runtime_ticker)
                open_by_ticker.setdefault(runtime_ticker, []).append(o)
                if _is_sports_order(o):
                    open_sports_tickers.add(runtime_ticker)
            elif status == "filled":
                close_ts = _parse_ts(o.get("close_time"))
                if close_ts is not None and close_ts <= resolution_cutoff:
                    needed_market_tickers.add(runtime_ticker)
        print(
            f"[SHADOW][HEARTBEAT] needed_market_tickers={len(needed_market_tickers)} "
            f"open_tickers={len(open_by_ticker)} open_sports={len(open_sports_tickers)}"
        )
        if priority_active:
            btc_needed = sum(1 for t in needed_market_tickers if str(t).startswith(btc_priority_prefix))
            if btc_needed > 0:
                print(
                    f"[SHADOW][PRIORITY] btc_priority_prefix={btc_priority_prefix} "
                    f"needed={btc_needed}"
                )

        ordered_open_tickers = _sorted_with_priority_prefix(
            open_by_ticker.keys(),
            priority_prefix=btc_priority_prefix,
            priority_active=priority_active,
        )
        ordered_needed_tickers = _sorted_with_priority_prefix(
            needed_market_tickers,
            priority_prefix=btc_priority_prefix,
            priority_active=priority_active,
        )

        market_cache: Dict[str, Dict[str, Any]] = {}
        with requests.Session() as s:
            # Fill detection first, against current active quote state.
            for ticker in ordered_open_tickers:
                ticker_orders = open_by_ticker.get(ticker, [])
                maker_orders = [
                    o
                    for o in ticker_orders
                    if str(o.get("maker_or_taker") or execution_mode).strip().lower() != "taker"
                ]
                if not maker_orders:
                    continue
                created_cutoff = min((_parse_ts(o.get("created_ts")) or _utc_now()) for o in maker_orders)
                ts = state.setdefault("tickers", {}).setdefault(
                    ticker,
                    {"cursor": "", "seen_trade_ids": [], "last_checked_ts": "", "last_seen_trade_ts": ""},
                )
                new_trades = _poll_new_trades_for_ticker(
                    session=s,
                    base_url=base_url,
                    ticker=ticker,
                    created_cutoff=created_cutoff,
                    ticker_state=ts,
                    seen_cap=seen_cap,
                )

                for order in sorted(maker_orders, key=lambda o: str(o.get("created_ts") or "")):
                    if str(order.get("status") or "") != "open":
                        continue
                    matches = _matching_fill_trades(order, new_trades)
                    if matches:
                        _mark_fill(order, matching_trades=matches)

            # Fetch fresh market snapshots for lifecycle + grading.
            sports_ws_enabled = str(cfg.get("sports_ws_enabled", "1")).strip().lower() not in {"0", "false", "no"}
            ws_sports_tickers = sorted(open_sports_tickers)
            print(
                f"[SHADOW][WS] sports_ws_enabled={sports_ws_enabled} "
                f"open_sports_tickers={len(ws_sports_tickers)}"
            )
            ws_received_sports: set[str] = set()
            if sports_ws_enabled and ws_sports_tickers:
                key_id_present = bool(str(os.environ.get("KALSHI_API_KEY_ID") or "").strip())
                private_key_present = bool(
                    str(os.environ.get("KALSHI_API_PRIVATE_KEY") or "").strip()
                    or str(os.environ.get("KALSHI_PRIVATE_KEY_PATH") or "").strip()
                )
                use_private_auth = _as_bool(cfg.get("sports_ws_use_private_auth"), default=False)
                if not use_private_auth and key_id_present and private_key_present:
                    use_private_auth = True
                    print("[SHADOW][WS] promoting private auth from env credentials")
                print(
                    f"[SHADOW][WS] private_auth_effective={use_private_auth} "
                    f"key_id_present={key_id_present} private_key_present={private_key_present}"
                )
                ws_quotes = _ws_ticker_snapshot(
                    market_tickers=ws_sports_tickers,
                    use_private_auth=use_private_auth,
                    timeout_s=float(cfg.get("sports_ws_timeout_seconds", 15.0)),
                )
                print(f"[SHADOW][WS] sports snapshot_tickers={len(ws_quotes)}")
                if not ws_quotes:
                    print("[SHADOW][WS] snapshot returned 0 sports tickers; forcing REST fallback")
                for mt, msg in ws_quotes.items():
                    if not isinstance(msg, dict):
                        continue
                    ticker = str(mt).strip().upper()
                    ws_received_sports.add(ticker)
                    market_cache[ticker] = {
                        "ticker": ticker,
                        "yes_bid": safe_int(msg.get("yes_bid")),
                        "yes_ask": safe_int(msg.get("yes_ask")),
                        "no_bid": safe_int(msg.get("no_bid")),
                        "no_ask": safe_int(msg.get("no_ask")),
                        "status": "open",
                    }
            elif not ws_sports_tickers:
                print("[SHADOW][WS] skipped sports snapshot (no open sports orders)")

            if sports_ws_enabled and ws_sports_tickers:
                for ticker in ws_sports_tickers:
                    t = str(ticker).strip().upper()
                    if not t:
                        continue
                    if t not in ws_received_sports:
                        try:
                            market_cache[t] = _fetch_market(session=s, base_url=base_url, ticker=t)
                            print(
                                f"[SHADOW][WS] REST fallback for ws-miss ticker={t} "
                                f"yes_bid={safe_int(market_cache[t].get('yes_bid'))} "
                                f"yes_ask={safe_int(market_cache[t].get('yes_ask'))}"
                            )
                        except Exception as exc:
                            print(f"[SHADOW][WS] REST fallback failed for ws-miss ticker={t} error={exc}")

            for ticker in ordered_needed_tickers:
                if ticker in market_cache:
                    continue
                try:
                    market_cache[ticker] = _fetch_market(session=s, base_url=base_url, ticker=ticker)
                except Exception:
                    continue

            if tickers_override:
                override_targets = [
                    t
                    for t in ordered_needed_tickers
                    if any(
                        str(t).startswith(str(prefix).strip().upper()) or str(prefix).strip().upper() in str(t)
                        for prefix in tickers_override
                        if str(prefix).strip()
                    )
                ]
                for ticker in override_targets:
                    if ticker in market_cache:
                        continue
                    try:
                        market_cache[ticker] = _fetch_market(session=s, base_url=base_url, ticker=ticker)
                        print(
                            f"[SHADOW][WS] override REST fallback ticker={ticker} "
                            f"yes_bid={safe_int(market_cache[ticker].get('yes_bid'))} "
                            f"yes_ask={safe_int(market_cache[ticker].get('yes_ask'))}"
                        )
                    except Exception as exc:
                        print(f"[SHADOW][WS] override REST fallback failed ticker={ticker} error={exc}")

        econ_state = _econ_pmf_state(config=cfg, econ_cache_dir=econ_cache_dir)

        # Reprice / edge-gate still-open orders.
        for order in orders:
            if str(order.get("status") or "") != "open":
                continue
            ticker = str(order.get("ticker") or "").strip().upper()
            market_lookup_ticker = str(order.get("_runtime_market_ticker") or ticker).strip().upper()
            market = market_cache.get(market_lookup_ticker)
            if not isinstance(market, dict):
                strategy_id = str(order.get("strategy_id") or "").strip()
                is_override_sports = bool(
                    _is_sports_order(order)
                    and tickers_override
                    and any(
                        market_lookup_ticker.startswith(str(prefix).strip().upper())
                        or str(prefix).strip().upper() in market_lookup_ticker
                        or ticker.startswith(str(prefix).strip().upper())
                        or str(prefix).strip().upper() in ticker
                        for prefix in tickers_override
                        if str(prefix).strip()
                    )
                )
                if is_override_sports:
                    synthetic_market = {
                        "ticker": (market_lookup_ticker or ticker),
                        "event_ticker": str(order.get("event_ticker") or "").strip().upper(),
                        "title": str(order.get("title") or ""),
                        "yes_bid": safe_int(order.get("yes_bid")),
                        "yes_ask": safe_int(order.get("yes_ask")),
                        "no_bid": safe_int(order.get("no_bid")),
                        "no_ask": safe_int(order.get("no_ask")),
                        "status": "open",
                    }
                    print(
                        f"[SHADOW][WS] synthetic sports market fallback ticker={(market_lookup_ticker or ticker)} "
                        f"strategy={strategy_id or 'NA'}"
                    )
                    _update_open_order_lifecycle(
                        order=order,
                        market=synthetic_market,
                        config=cfg,
                        weather_cache_dir=weather_cache_dir,
                        econ_state=econ_state,
                    )
                continue
            _update_open_order_lifecycle(
                order=order,
                market=market,
                config=cfg,
                weather_cache_dir=weather_cache_dir,
                econ_state=econ_state,
            )

        _prune_open_orders_by_event(
            orders=orders,
            max_open_orders_per_event=int(cfg.get("max_open_orders_per_event", 1)),
        )

        # Resolution grading for open/filled.
        _grade_resolutions(
            orders=orders,
            market_cache=market_cache,
            default_execution_mode=execution_mode,
            maker_fee_rate=maker_fee_rate,
            taker_fee_rate=taker_fee_rate,
        )

        state["orders"] = orders
        state["updated_ts"] = _utc_now_iso()
        _write_json(state_path, state)
        ledger_path, summary_path = _save_outputs(
            day=day,
            state=state,
            poll_seconds=poll_seconds,
            max_runtime_minutes=max_runtime_minutes,
            ledger_path=ledger_path,
            summary_path=summary_path,
        )

        if _all_done(orders):
            if orders or tickers_override:
                break
            print("[SHADOW][HEARTBEAT] no active orders yet; continuing discovery loop")
        if max_runtime_minutes > 0 and (time.monotonic() - start_monotonic) >= (max_runtime_minutes * 60.0):
            break
        time.sleep(poll_seconds)

    # Final persist.
    state["updated_ts"] = _utc_now_iso()
    _write_json(state_path, state)
    ledger_path, summary_path = _save_outputs(
        day=day,
        state=state,
        poll_seconds=poll_seconds,
        max_runtime_minutes=max_runtime_minutes,
        ledger_path=ledger_path,
        summary_path=summary_path,
    )

    print(f"state_json={state_path}")
    print(f"shadow_ledger={ledger_path}")
    print(f"shadow_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
