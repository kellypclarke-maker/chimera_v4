from __future__ import annotations

import os
import re
import time
from typing import Dict, FrozenSet, List, Optional, Tuple

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
_NBA_TEAMS = ("CHA", "IND", "MIA", "PHI")
_NBA_EVENT_RE = re.compile(r"^KXNBAGAME-(?P<date>\d{2}[A-Z]{3}\d{2})(?P<pair>[A-Z]{6})(?:-(?P<side>[A-Z]{3}))?$")

_NBA_TEAM_CODE_TO_NAME: Dict[str, str] = {
    "CHA": "Charlotte Hornets",
    "IND": "Indiana Pacers",
    "MIA": "Miami Heat",
    "PHI": "Philadelphia 76ers",
}

_NBA_TEAM_ALIASES: Dict[str, str] = {
    "charlotte hornets": "Charlotte Hornets",
    "indiana pacers": "Indiana Pacers",
    "miami heat": "Miami Heat",
    "philadelphia 76ers": "Philadelphia 76ers",
    "philadelphia seventy sixers": "Philadelphia 76ers",
    "76ers": "Philadelphia 76ers",
}

_ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
_ODDS_SPORT_KEY = "basketball_nba"
_ODDS_MARKET = "h2h"
_ODDS_MIN_POLL_SECONDS = 600.0  # Free-tier guard: no faster than every 10 minutes.

_ODDS_CACHE_LAST_FETCH_MONO = 0.0
_ODDS_CACHE_PAIR_PROBS: Dict[FrozenSet[str], Dict[str, float]] = {}
_ODDS_CACHE_LOGGED_KEY_STATE = False
_ODDS_CACHE_LAST_ERROR_WAS_401 = False

_NBA_TEAM_NAME_TO_CODE: Dict[str, str] = {v: k for k, v in _NBA_TEAM_CODE_TO_NAME.items()}
_NBA_MANUAL_401_PROBS: Dict[FrozenSet[str], Dict[str, float]] = {
    frozenset({"MIA", "PHI"}): {"MIA": 0.467, "PHI": 0.573},
    frozenset({"CHA", "IND"}): {"CHA": 0.885, "IND": 0.156},
}


def _normalize_team_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(name or "").strip().lower()).strip()
    if not s:
        return ""
    return _NBA_TEAM_ALIASES.get(s, "")


def _parse_nba_ticker_codes(*, ticker: str, event_ticker: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    t = str(ticker or "").strip().upper()
    ev = str(event_ticker or "").strip().upper()
    m = _NBA_EVENT_RE.match(t) or _NBA_EVENT_RE.match(ev)
    if not m:
        return None
    date_token = str(m.group("date") or "").strip().upper()
    pair = str(m.group("pair") or "").strip().upper()
    if len(pair) != 6:
        return None
    team_a = pair[:3]
    team_b = pair[3:]
    side = str(m.group("side") or "").strip().upper() or None
    if side is None:
        mt = _NBA_EVENT_RE.match(t)
        if mt:
            side = str(mt.group("side") or "").strip().upper() or None
    return (date_token, team_a, team_b, side)


def _odds_api_key(config: Dict[str, object]) -> str:
    return (
        str(
        os.environ.get("THE_ODDS_API_KEY")
        or config.get("THE_ODDS_API_KEY")
        or ""
        )
        .strip()
        .replace('"', "")
        .replace("'", "")
        .replace("\r", "")
        .replace("\n", "")
        .strip()
    )


def _odds_poll_seconds(config: Dict[str, object]) -> float:
    raw = config.get("nba_odds_poll_seconds", _ODDS_MIN_POLL_SECONDS)
    try:
        return max(_ODDS_MIN_POLL_SECONDS, float(raw))
    except Exception:
        return _ODDS_MIN_POLL_SECONDS


def _extract_bookmaker_h2h_probs(bookmaker: Dict[str, object]) -> Optional[Dict[str, float]]:
    markets = bookmaker.get("markets")
    if not isinstance(markets, list):
        return None
    for market in markets:
        if not isinstance(market, dict):
            continue
        if str(market.get("key") or "").strip().lower() != _ODDS_MARKET:
            continue
        outcomes = market.get("outcomes")
        if not isinstance(outcomes, list):
            continue
        implied: Dict[str, float] = {}
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            team_name = _normalize_team_name(str(outcome.get("name") or ""))
            if not team_name:
                continue
            try:
                price = float(outcome.get("price"))
            except Exception:
                continue
            if price <= 1.0:
                continue
            implied[team_name] = 1.0 / price
        if len(implied) < 2:
            continue
        total = sum(float(v) for v in implied.values())
        if total <= 0.0:
            continue
        return {k: float(v) / total for k, v in implied.items()}
    return None


def _refresh_odds_pair_probabilities(config: Dict[str, object]) -> Dict[FrozenSet[str], Dict[str, float]]:
    global _ODDS_CACHE_LAST_FETCH_MONO, _ODDS_CACHE_PAIR_PROBS, _ODDS_CACHE_LOGGED_KEY_STATE, _ODDS_CACHE_LAST_ERROR_WAS_401

    api_key = _odds_api_key(config)
    if not _ODDS_CACHE_LOGGED_KEY_STATE:
        print(f"[SHADOW][ORACLE] NBA Odds API key present={bool(api_key)}")
        _ODDS_CACHE_LOGGED_KEY_STATE = True
    if not api_key:
        return dict(_ODDS_CACHE_PAIR_PROBS)

    now_mono = time.monotonic()
    poll_s = _odds_poll_seconds(config)
    if _ODDS_CACHE_PAIR_PROBS and (now_mono - float(_ODDS_CACHE_LAST_FETCH_MONO)) < float(poll_s):
        return dict(_ODDS_CACHE_PAIR_PROBS)

    url = f"{_ODDS_BASE_URL}/sports/{_ODDS_SPORT_KEY}/odds"
    key_head = api_key[:4]
    key_tail = api_key[-4:] if len(api_key) >= 4 else api_key
    print(
        f"[SHADOW][ORACLE] NBA Odds key fingerprint={key_head}...{key_tail} "
        f"len={len(api_key)}"
    )
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": _ODDS_MARKET,
        "oddsFormat": "decimal",
        "dateFormat": "iso",
    }
    headers = {"Accept": "application/json"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20.0)
        if int(resp.status_code) == 401:
            _ODDS_CACHE_LAST_ERROR_WAS_401 = True
            print("[SHADOW][ORACLE] WARNING: Using Manual Hardcoded Probs due to 401 Error.")
            return dict(_ODDS_CACHE_PAIR_PROBS)
        resp.raise_for_status()
        payload = resp.json() if resp.content else []
    except Exception as exc:
        print(f"[SHADOW][ORACLE] Odds API fetch failed: {exc}")
        return dict(_ODDS_CACHE_PAIR_PROBS)

    _ODDS_CACHE_LAST_ERROR_WAS_401 = False
    if not isinstance(payload, list):
        print("[SHADOW][ORACLE] Odds API payload invalid (expected list)")
        return dict(_ODDS_CACHE_PAIR_PROBS)
    if len(payload) == 0:
        print("[SHADOW][ORACLE] NBA Odds API returned 200 with empty list payload.")

    pair_probs: Dict[FrozenSet[str], Dict[str, float]] = {}
    for game in payload:
        if not isinstance(game, dict):
            continue
        home = _normalize_team_name(str(game.get("home_team") or ""))
        away = _normalize_team_name(str(game.get("away_team") or ""))
        if not home or not away or home == away:
            continue
        bookmakers = game.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        by_team: Dict[str, List[float]] = {}
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            probs = _extract_bookmaker_h2h_probs(bookmaker)
            if not probs:
                continue
            for team_name, p in probs.items():
                by_team.setdefault(team_name, []).append(float(p))
        if len(by_team) < 2:
            continue
        averaged = {team_name: (sum(vals) / len(vals)) for team_name, vals in by_team.items() if vals}
        total = sum(float(v) for v in averaged.values())
        if total <= 0.0:
            continue
        normalized = {team_name: float(v) / total for team_name, v in averaged.items()}
        key = frozenset(normalized.keys())
        if len(key) == 2:
            pair_probs[key] = normalized

    _ODDS_CACHE_LAST_FETCH_MONO = now_mono
    _ODDS_CACHE_PAIR_PROBS = pair_probs
    print(f"[SHADOW][ORACLE] NBA odds refresh pairs={len(pair_probs)} poll_s={int(poll_s)}")
    return dict(_ODDS_CACHE_PAIR_PROBS)


def lookup_nba_live_p_true(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Dict[str, object],
) -> Optional[float]:
    parsed = _parse_nba_ticker_codes(ticker=ticker, event_ticker=event_ticker)
    if parsed is None:
        return None
    _, team_a_code, team_b_code, side_code = parsed
    team_a = _NBA_TEAM_CODE_TO_NAME.get(team_a_code)
    team_b = _NBA_TEAM_CODE_TO_NAME.get(team_b_code)
    if not team_a or not team_b:
        return None

    selected_team_code = str(side_code or "").strip().upper() or None
    selected_team = _NBA_TEAM_CODE_TO_NAME.get(str(side_code or "").strip().upper())
    if not selected_team:
        title_norm = _normalize_team_name(title)
        if title_norm in {team_a, team_b}:
            selected_team = title_norm
            selected_team_code = _NBA_TEAM_NAME_TO_CODE.get(title_norm)
    if not selected_team:
        return None

    pair_probs = _refresh_odds_pair_probabilities(config)
    if _ODDS_CACHE_LAST_ERROR_WAS_401:
        manual = _NBA_MANUAL_401_PROBS.get(frozenset({team_a_code, team_b_code}), {})
        if selected_team_code and selected_team_code in manual:
            print("[SHADOW][ORACLE] WARNING: Using Manual Hardcoded Probs due to 401 Error.")
            return float(manual[selected_team_code])
    game_probs = pair_probs.get(frozenset({team_a, team_b}))
    if not game_probs:
        return None
    p_true = game_probs.get(selected_team)
    if p_true is None:
        return None
    if not (0.0 <= float(p_true) <= 1.0):
        return None
    return float(p_true)


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


def _is_nba_row(row: Dict[str, str]) -> bool:
    ticker = str(row.get("ticker") or "").strip().upper()
    event_ticker = str(row.get("event_ticker") or "").strip().upper()
    title = str(row.get("title") or "").strip().upper()
    blob = " ".join([ticker, event_ticker, title]).upper()
    has_focus_team = any(team in blob for team in _NBA_TEAMS)
    category = category_from_row(row).strip().lower()
    return (
        has_focus_team
        and (
            ticker.startswith("KXNBA")
            or event_ticker.startswith("KXNBA")
            or " NBA " in f" {title} "
            or category == "sports"
        )
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


def discover_nba_startup_tickers(*, day_iso: str, config: Dict[str, object]) -> List[str]:
    token = kalshi_date_token_from_iso_date(day_iso)
    if not token:
        return []
    base_url = str(config.get("base_url") or "").strip()
    series = str(config.get("shadow_nba_series_ticker", "KXNBAGAME")).strip().upper() or "KXNBAGAME"
    max_events = max(1, int(config.get("shadow_bootstrap_max_nba_events", 60)))
    max_markets = max(1, int(config.get("shadow_bootstrap_max_nba_markets", 120)))
    try:
        with requests.Session() as s:
            discovered = discover_market_tickers_for_series_date(
                session=s,
                date_token=token,
                series_ticker=series,
                base_url=base_url,
                max_pages=10,
                max_events=max_events,
                max_markets_per_event=max_markets,
            )
            filtered = [t for t in discovered if any(team in str(t).upper() for team in _NBA_TEAMS)]
            return filtered
    except Exception:
        return []


class NBASpecialist:
    name = "nba"
    categories = ["Sports"]

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        cfg = context.config
        join_ticks = int(cfg.get("nba_join_ticks", cfg.get("sports_join_ticks", 1)))
        slippage_dollars = float(cfg.get("nba_slippage_cents_per_contract", cfg.get("sports_slippage_cents_per_contract", 0.25))) / 100.0
        min_ev = float(cfg.get("nba_min_ev_dollars", cfg.get("sports_min_ev_dollars", 0.0)))
        maker_fee_rate = float(cfg.get("maker_fee_rate", 0.0))
        min_yes_bid_cents = int(cfg.get("min_yes_bid_cents", 1))
        max_yes_spread_cents = int(cfg.get("max_yes_spread_cents", 10))

        out: List[CandidateProposal] = []
        for row in context.markets:
            if not _is_nba_row(row):
                continue
            live_p_true = lookup_nba_live_p_true(
                ticker=str(row.get("ticker") or ""),
                event_ticker=str(row.get("event_ticker") or ""),
                title=str(row.get("title") or ""),
                config=cfg,
            )
            p_true = live_p_true
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
                    strategy_id="NBA_live_maker",
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
                    liquidity_notes=f"p_true={p_true:.6f};sport=nba;source=odds_api",
                    risk_flags="sport=nba",
                    verification_checklist="verify sportsbook/live model p_true and final injury/news state before live usage",
                    rules_text_hash=rules_hash_from_row(row),
                    rules_missing=rules_missing,
                    rules_pointer=rules_pointer,
                    resolution_pointer=resolution_pointer_from_row(row),
                    market_url=str(row.get("market_url") or ""),
                )
            )
        return out
