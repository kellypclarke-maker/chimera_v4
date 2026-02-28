from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import os
import re
import time
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

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
from specialists.probabilities import binary_shin_devig_named_odds

_P_TRUE_RE = re.compile(r"\bp_true(?:_cal)?=([0-9]*\.?[0-9]+)")
_NHL_EVENT_RE = re.compile(r"^KXNHLGAME-(?P<date>\d{2}[A-Z]{3}\d{2})(?P<pair>[A-Z]{6})(?:-(?P<side>[A-Z]{3}))?$")

_NHL_TEAM_CODE_TO_NAME: Dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}
_NHL_TEAMS = tuple(sorted(_NHL_TEAM_CODE_TO_NAME.keys()))

_NHL_TEAM_ALIASES: Dict[str, str] = {
    "anaheim ducks": "Anaheim Ducks",
    "boston bruins": "Boston Bruins",
    "buffalo sabres": "Buffalo Sabres",
    "carolina hurricanes": "Carolina Hurricanes",
    "columbus blue jackets": "Columbus Blue Jackets",
    "calgary flames": "Calgary Flames",
    "chicago blackhawks": "Chicago Blackhawks",
    "colorado avalanche": "Colorado Avalanche",
    "dallas stars": "Dallas Stars",
    "detroit red wings": "Detroit Red Wings",
    "edmonton oilers": "Edmonton Oilers",
    "florida panthers": "Florida Panthers",
    "los angeles kings": "Los Angeles Kings",
    "minnesota wild": "Minnesota Wild",
    "montreal canadiens": "Montreal Canadiens",
    "new jersey devils": "New Jersey Devils",
    "nashville predators": "Nashville Predators",
    "new york islanders": "New York Islanders",
    "new york rangers": "New York Rangers",
    "ottawa senators": "Ottawa Senators",
    "philadelphia flyers": "Philadelphia Flyers",
    "pittsburgh penguins": "Pittsburgh Penguins",
    "seattle kraken": "Seattle Kraken",
    "san jose sharks": "San Jose Sharks",
    "st louis blues": "St. Louis Blues",
    "tampa bay lightning": "Tampa Bay Lightning",
    "toronto maple leafs": "Toronto Maple Leafs",
    "utah hockey club": "Utah Hockey Club",
    "vancouver canucks": "Vancouver Canucks",
    "vegas golden knights": "Vegas Golden Knights",
    "washington capitals": "Washington Capitals",
    "winnipeg jets": "Winnipeg Jets",
}

_ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
_ODDS_SPORT_KEY = "icehockey_nhl"
_ODDS_MARKET = "h2h"
_ODDS_MIN_POLL_SECONDS = 180.0
_ORACLE_CACHE_TTL_SECONDS = 180.0
_TICKER_DATE_TZ = timezone(timedelta(hours=-8))
_TEMPORAL_BRIDGE_SECONDS = 24.0 * 60.0 * 60.0

_ODDS_CACHE_LAST_FETCH_MONO = 0.0
_ODDS_CACHE_PAIR_PROBS: Dict[FrozenSet[str], Dict[str, float]] = {}
_ODDS_CACHE_LOGGED_KEY_STATE = False
_ODDS_CACHE_LAST_ERROR_WAS_401 = False
_ORACLE_MATCHUP_CACHE: Dict[FrozenSet[str], Tuple[float, Dict[str, float]]] = {}
_ODDS_CACHE_LAST_PAYLOAD: List[Dict[str, object]] = []
_TEMPORAL_BRIDGE_LOGGED: Set[str] = set()

_NHL_TEAM_NAME_TO_CODE: Dict[str, str] = {v: k for k, v in _NHL_TEAM_CODE_TO_NAME.items()}
_NHL_MANUAL_401_PROBS: Dict[FrozenSet[str], Dict[str, float]] = {
    frozenset({"PHI", "NYR"}): {"NYR": 0.558, "PHI": 0.488},
}


def _normalize_team_name(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(name or "").strip().lower()).strip()
    if not s:
        return ""
    return _NHL_TEAM_ALIASES.get(s, "")


def _parse_nhl_ticker_codes(*, ticker: str, event_ticker: str) -> Optional[Tuple[str, str, str, Optional[str]]]:
    t = str(ticker or "").strip().upper()
    ev = str(event_ticker or "").strip().upper()
    m = _NHL_EVENT_RE.match(t) or _NHL_EVENT_RE.match(ev)
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
        mt = _NHL_EVENT_RE.match(t)
        if mt:
            side = str(mt.group("side") or "").strip().upper() or None
    return (date_token, team_a, team_b, side)


def _odds_api_key(config: Dict[str, object]) -> str:
    return (
        str(
            os.environ.get("THE_ODDS_API_KEY_20k")
            or os.environ.get("THE_ODDS_API_KEY_20K")
            or config.get("THE_ODDS_API_KEY_20k")
            or config.get("THE_ODDS_API_KEY_20K")
            or os.environ.get("THE_ODDS_API_KEY")
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
    # Dual-mode throttle valve: hard cap live odds refresh cadence at 180s.
    return float(_ORACLE_CACHE_TTL_SECONDS)


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
        odds_by_team: Dict[str, float] = {}
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
            odds_by_team[team_name] = float(price)
        probs = binary_shin_devig_named_odds(odds_by_team)
        if not probs:
            continue
        return {k: float(v) for k, v in probs.items()}
    return None


def _parse_date_token(token: str) -> Optional[date]:
    raw = str(token or "").strip().upper()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%d%b%y").date()
    except Exception:
        return None


def _parse_commence_time(raw: object) -> Optional[datetime]:
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _extract_game_pair_and_probs(
    game: Dict[str, object],
) -> Optional[Tuple[FrozenSet[str], Dict[str, float], Optional[datetime]]]:
    home = _normalize_team_name(str(game.get("home_team") or ""))
    away = _normalize_team_name(str(game.get("away_team") or ""))
    if not home or not away or home == away:
        return None
    bookmakers = game.get("bookmakers")
    if not isinstance(bookmakers, list):
        return None
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
        return None
    averaged = {team_name: (sum(vals) / len(vals)) for team_name, vals in by_team.items() if vals}
    total = sum(float(v) for v in averaged.values())
    if total <= 0.0:
        return None
    normalized = {team_name: float(v) / total for team_name, v in averaged.items()}
    key = frozenset(normalized.keys())
    if len(key) != 2:
        return None
    commence_utc = _parse_commence_time(game.get("commence_time"))
    return key, normalized, commence_utc


def _log_temporal_bridge_once(
    *,
    ticker_date_token: str,
    api_date_token: str,
    matchup_key: FrozenSet[str],
    league_tag: str,
) -> None:
    ticker_token = str(ticker_date_token or "").strip().upper()
    api_token = str(api_date_token or "").strip().upper()
    if not ticker_token or not api_token or ticker_token == api_token:
        return
    key = f"{league_tag}:{ticker_token}->{api_token}:{'|'.join(sorted(matchup_key))}"
    if key in _TEMPORAL_BRIDGE_LOGGED:
        return
    _TEMPORAL_BRIDGE_LOGGED.add(key)
    print(
        f"[SHADOW][ORACLE][DEBUG] Matched {ticker_token} ticker to {api_token} API event via 24h bridge"
    )


def _select_temporal_matchup_probs_from_payload(
    *,
    payload: object,
    matchup_key: FrozenSet[str],
    ticker_date_token: str,
    league_tag: str,
) -> Optional[Dict[str, float]]:
    if not isinstance(payload, list):
        return None

    target_date = _parse_date_token(ticker_date_token)
    if target_date is None:
        for game in payload:
            if not isinstance(game, dict):
                continue
            extracted = _extract_game_pair_and_probs(game)
            if not extracted:
                continue
            key, probs, _ = extracted
            if key == matchup_key:
                return dict(probs)
        return None

    anchor_utc = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        tzinfo=_TICKER_DATE_TZ,
    ).astimezone(timezone.utc)

    best_probs: Optional[Dict[str, float]] = None
    best_delta: Optional[float] = None
    best_api_token = ""
    fallback_probs: Optional[Dict[str, float]] = None
    for game in payload:
        if not isinstance(game, dict):
            continue
        extracted = _extract_game_pair_and_probs(game)
        if not extracted:
            continue
        key, probs, commence_utc = extracted
        if key != matchup_key:
            continue
        if commence_utc is None:
            if fallback_probs is None:
                fallback_probs = dict(probs)
            continue
        delta_s = abs((commence_utc - anchor_utc).total_seconds())
        if delta_s > _TEMPORAL_BRIDGE_SECONDS:
            continue
        if best_delta is None or delta_s < best_delta:
            best_delta = float(delta_s)
            best_probs = dict(probs)
            best_api_token = commence_utc.strftime("%d%b%y").upper()
    if best_probs is not None:
        _log_temporal_bridge_once(
            ticker_date_token=ticker_date_token,
            api_date_token=best_api_token,
            matchup_key=matchup_key,
            league_tag=league_tag,
        )
        return best_probs
    if fallback_probs is not None:
        return fallback_probs
    return None


def _select_temporal_matchup_commence_time_from_payload(
    *,
    payload: object,
    matchup_key: FrozenSet[str],
    ticker_date_token: str,
    league_tag: str,
) -> Optional[datetime]:
    if not isinstance(payload, list):
        return None

    target_date = _parse_date_token(ticker_date_token)
    if target_date is None:
        for game in payload:
            if not isinstance(game, dict):
                continue
            extracted = _extract_game_pair_and_probs(game)
            if not extracted:
                continue
            key, _, commence_utc = extracted
            if key == matchup_key and commence_utc is not None:
                return commence_utc
        return None

    anchor_utc = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        tzinfo=_TICKER_DATE_TZ,
    ).astimezone(timezone.utc)

    best_commence: Optional[datetime] = None
    best_delta: Optional[float] = None
    best_api_token = ""
    for game in payload:
        if not isinstance(game, dict):
            continue
        extracted = _extract_game_pair_and_probs(game)
        if not extracted:
            continue
        key, _, commence_utc = extracted
        if key != matchup_key or commence_utc is None:
            continue
        delta_s = abs((commence_utc - anchor_utc).total_seconds())
        if delta_s > _TEMPORAL_BRIDGE_SECONDS:
            continue
        if best_delta is None or delta_s < best_delta:
            best_delta = float(delta_s)
            best_commence = commence_utc
            best_api_token = commence_utc.strftime("%d%b%y").upper()
    if best_commence is not None:
        _log_temporal_bridge_once(
            ticker_date_token=ticker_date_token,
            api_date_token=best_api_token,
            matchup_key=matchup_key,
            league_tag=league_tag,
        )
    return best_commence


def _refresh_odds_pair_probabilities(config: Dict[str, object]) -> Dict[FrozenSet[str], Dict[str, float]]:
    global _ODDS_CACHE_LAST_FETCH_MONO, _ODDS_CACHE_PAIR_PROBS, _ODDS_CACHE_LOGGED_KEY_STATE, _ODDS_CACHE_LAST_ERROR_WAS_401, _ORACLE_MATCHUP_CACHE, _ODDS_CACHE_LAST_PAYLOAD

    api_key = _odds_api_key(config)
    if not _ODDS_CACHE_LOGGED_KEY_STATE:
        print(f"[SHADOW][ORACLE] NHL Odds API key present={bool(api_key)}")
        _ODDS_CACHE_LOGGED_KEY_STATE = True
    if not api_key:
        return dict(_ODDS_CACHE_PAIR_PROBS)

    now_mono = time.monotonic()
    poll_s = _odds_poll_seconds(config)
    if (now_mono - float(_ODDS_CACHE_LAST_FETCH_MONO)) < float(poll_s):
        return dict(_ODDS_CACHE_PAIR_PROBS)

    url = f"{_ODDS_BASE_URL}/sports/{_ODDS_SPORT_KEY}/odds"
    key_head = api_key[:4]
    key_tail = api_key[-4:] if len(api_key) >= 4 else api_key
    print(
        f"[SHADOW][ORACLE] NHL Odds key fingerprint={key_head}...{key_tail} "
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
            _ODDS_CACHE_LAST_FETCH_MONO = now_mono
            _ODDS_CACHE_LAST_ERROR_WAS_401 = True
            print("[SHADOW][ORACLE] WARNING: Using Manual Hardcoded Probs due to 401 Error.")
            return dict(_ODDS_CACHE_PAIR_PROBS)
        resp.raise_for_status()
        payload = resp.json() if resp.content else []
    except Exception as exc:
        _ODDS_CACHE_LAST_FETCH_MONO = now_mono
        print(f"[SHADOW][ORACLE] NHL Odds API fetch failed: {exc}")
        return dict(_ODDS_CACHE_PAIR_PROBS)

    _ODDS_CACHE_LAST_ERROR_WAS_401 = False
    if not isinstance(payload, list):
        _ODDS_CACHE_LAST_FETCH_MONO = now_mono
        print("[SHADOW][ORACLE] NHL Odds API payload invalid (expected list)")
        return dict(_ODDS_CACHE_PAIR_PROBS)
    if len(payload) == 0:
        print("[SHADOW][ORACLE] NHL Odds API returned 200 with empty list payload.")
    _ODDS_CACHE_LAST_PAYLOAD = [g for g in payload if isinstance(g, dict)]

    pair_probs: Dict[FrozenSet[str], Dict[str, float]] = {}
    for game in payload:
        if not isinstance(game, dict):
            continue
        extracted = _extract_game_pair_and_probs(game)
        if not extracted:
            continue
        key, normalized, _ = extracted
        pair_probs[key] = normalized

    _ODDS_CACHE_LAST_FETCH_MONO = now_mono
    _ODDS_CACHE_PAIR_PROBS = pair_probs
    for key, probs in pair_probs.items():
        _ORACLE_MATCHUP_CACHE[key] = (now_mono, dict(probs))
    stale_cutoff = now_mono - (float(_ORACLE_CACHE_TTL_SECONDS) * 2.0)
    _ORACLE_MATCHUP_CACHE = {
        k: v for k, v in _ORACLE_MATCHUP_CACHE.items() if float(v[0]) >= stale_cutoff
    }
    print(f"[SHADOW][ORACLE] NHL odds refresh pairs={len(pair_probs)} poll_s={int(poll_s)}")
    return dict(_ODDS_CACHE_PAIR_PROBS)


def lookup_nhl_live_p_true(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Dict[str, object],
) -> Optional[float]:
    parsed = _parse_nhl_ticker_codes(ticker=ticker, event_ticker=event_ticker)
    if parsed is None:
        return None
    date_token, team_a_code, team_b_code, side_code = parsed
    team_a = _NHL_TEAM_CODE_TO_NAME.get(team_a_code)
    team_b = _NHL_TEAM_CODE_TO_NAME.get(team_b_code)
    if not team_a or not team_b:
        return None

    selected_team_code = str(side_code or "").strip().upper() or None
    selected_team = _NHL_TEAM_CODE_TO_NAME.get(str(side_code or "").strip().upper())
    if not selected_team:
        title_norm = _normalize_team_name(title)
        if title_norm in {team_a, team_b}:
            selected_team = title_norm
            selected_team_code = _NHL_TEAM_NAME_TO_CODE.get(title_norm)
    if not selected_team:
        return None

    matchup_key = frozenset({team_a, team_b})
    now_mono = time.monotonic()
    cached = _ORACLE_MATCHUP_CACHE.get(matchup_key)
    if cached is not None and (now_mono - float(cached[0])) < float(_ORACLE_CACHE_TTL_SECONDS):
        cached_probs = cached[1]
        cached_p = cached_probs.get(selected_team)
        if cached_p is not None and 0.0 <= float(cached_p) <= 1.0:
            age_s = now_mono - float(cached[0])
            print(f"[SHADOW][ORACLE] NHL cache hit matchup={team_a_code}/{team_b_code} age_s={age_s:.1f}")
            return float(cached_p)

    pair_probs = _refresh_odds_pair_probabilities(config)
    if _ODDS_CACHE_LAST_ERROR_WAS_401:
        manual = _NHL_MANUAL_401_PROBS.get(frozenset({team_a_code, team_b_code}), {})
        if selected_team_code and selected_team_code in manual:
            print("[SHADOW][ORACLE] WARNING: Using Manual Hardcoded Probs due to 401 Error.")
            return float(manual[selected_team_code])
    temporal_probs = _select_temporal_matchup_probs_from_payload(
        payload=_ODDS_CACHE_LAST_PAYLOAD,
        matchup_key=matchup_key,
        ticker_date_token=date_token,
        league_tag="NHL",
    )
    if temporal_probs is not None:
        game_probs = temporal_probs
    elif not _ODDS_CACHE_LAST_PAYLOAD:
        game_probs = pair_probs.get(matchup_key)
    else:
        game_probs = None
    if not game_probs:
        if cached is not None:
            cached_probs = cached[1]
            cached_p = cached_probs.get(selected_team)
            if cached_p is not None and 0.0 <= float(cached_p) <= 1.0:
                age_s = now_mono - float(cached[0])
                print(f"[SHADOW][ORACLE] NHL stale-cache fallback matchup={team_a_code}/{team_b_code} age_s={age_s:.1f}")
                return float(cached_p)
        return None
    _ORACLE_MATCHUP_CACHE[matchup_key] = (now_mono, dict(game_probs))
    p_true = game_probs.get(selected_team)
    if p_true is None:
        return None
    if not (0.0 <= float(p_true) <= 1.0):
        return None
    return float(p_true)


def lookup_nhl_commence_time_utc(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Dict[str, object],
) -> Optional[str]:
    parsed = _parse_nhl_ticker_codes(ticker=ticker, event_ticker=event_ticker)
    if parsed is None:
        return None
    date_token, team_a_code, team_b_code, _ = parsed
    team_a = _NHL_TEAM_CODE_TO_NAME.get(team_a_code)
    team_b = _NHL_TEAM_CODE_TO_NAME.get(team_b_code)
    if not team_a or not team_b:
        return None
    matchup_key = frozenset({team_a, team_b})
    _refresh_odds_pair_probabilities(config)
    commence = _select_temporal_matchup_commence_time_from_payload(
        payload=_ODDS_CACHE_LAST_PAYLOAD,
        matchup_key=matchup_key,
        ticker_date_token=date_token,
        league_tag="NHL",
    )
    if commence is None:
        return None
    return commence.isoformat()


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
    blob = " ".join([ticker, event_ticker, title]).upper()
    has_focus_team = any(team in blob for team in _NHL_TEAMS)
    category = category_from_row(row).strip().lower()
    return (
        has_focus_team
        and (
            ticker.startswith("KXNHL")
            or event_ticker.startswith("KXNHL")
            or " NHL " in f" {title} "
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
            discovered = discover_market_tickers_for_series_date(
                session=s,
                date_token=token,
                series_ticker=series,
                base_url=base_url,
                max_pages=10,
                max_events=max_events,
                max_markets_per_event=max_markets,
            )
            filtered = [t for t in discovered if any(team in str(t).upper() for team in _NHL_TEAMS)]
            return filtered
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
            p_true = lookup_nhl_live_p_true(
                ticker=str(row.get("ticker") or ""),
                event_ticker=str(row.get("event_ticker") or ""),
                title=str(row.get("title") or ""),
                config=cfg,
            )
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
                    liquidity_notes=f"p_true={p_true:.6f};sport=nhl;source=odds_api",
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
