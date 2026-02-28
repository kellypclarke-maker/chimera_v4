from __future__ import annotations

import asyncio
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import websockets
from specialists.probabilities import binary_shin_devig_from_decimal_odds

class ListenerState(str, Enum):
    BOOTING = "BOOTING"
    CONNECTING = "CONNECTING"
    SUBSCRIBING = "SUBSCRIBING"
    SYNCING = "SYNCING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RECONNECTING = "RECONNECTING"
    SUSPENDED = "SUSPENDED"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class MatchupOdds:
    matchup_key: str
    league: str
    team_a: str
    team_b: str
    p_true_by_team: Mapping[str, float]
    commence_time_utc: Optional[str]
    game_status: str
    source_ts_ms: int
    updated_mono: float
    stale_after_mono: float
    generation_id: int
    seq_no: int


@dataclass(frozen=True, slots=True)
class OddsSnapshot:
    generation_id: int
    listener_state: ListenerState
    state_since_mono: float
    updated_mono: float
    by_matchup: Mapping[str, MatchupOdds]


_NBA_EVENT_RE = re.compile(r"^KXNBAGAME-(?P<date>\d{2}[A-Z]{3}\d{2})(?P<pair>[A-Z]{6})(?:-(?P<side>[A-Z]{3}))?$")
_NHL_EVENT_RE = re.compile(r"^KXNHLGAME-(?P<date>\d{2}[A-Z]{3}\d{2})(?P<pair>[A-Z]{6})(?:-(?P<side>[A-Z]{3}))?$")

_NBA_TEAM_CODE_TO_NAME: Dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BKN": "Brooklyn Nets",
    "BOS": "Boston Celtics",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

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
    "LA": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NJ": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJ": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TB": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WSH": "Washington Capitals",
    "WPG": "Winnipeg Jets",
}


def _normalize_team_name(raw: object) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(raw or "").strip().lower()).strip()
    return s


def _build_alias_map(code_to_name: Mapping[str, str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    nicknames: Dict[str, list[str]] = {}
    for code, name in code_to_name.items():
        norm = _normalize_team_name(name)
        if norm:
            out[norm] = str(code).strip().upper()
            parts = norm.split()
            if len(parts) >= 2:
                nickname = " ".join(parts[1:])
                nicknames.setdefault(nickname, []).append(str(code).strip().upper())
                out.setdefault(parts[-1], str(code).strip().upper())
                out.setdefault(nickname, str(code).strip().upper())
            if norm.startswith("los angeles "):
                suffix = norm[len("los angeles ") :]
                out[f"la {suffix}"] = str(code).strip().upper()
                out[f"l a {suffix}"] = str(code).strip().upper()
            if norm.startswith("new york "):
                suffix = norm[len("new york ") :]
                out[f"ny {suffix}"] = str(code).strip().upper()
            if norm.startswith("new jersey "):
                suffix = norm[len("new jersey ") :]
                out[f"nj {suffix}"] = str(code).strip().upper()
            if norm.startswith("oklahoma city "):
                suffix = norm[len("oklahoma city ") :]
                out[f"okc {suffix}"] = str(code).strip().upper()
    for nickname, codes in nicknames.items():
        if len(set(codes)) == 1:
            out[nickname] = list(set(codes))[0]
    aliases = {
        "new jersey": "NJ",
        "new york rangers": "NYR",
        "new york islanders": "NYI",
        "new york knicks": "NYK",
        "brooklyn nets": "BKN",
        "boston celtics": "BOS",
        "cleveland cavaliers": "CLE",
        "denver nuggets": "DEN",
        "detroit pistons": "DET",
        "los angeles clippers": "LAC",
        "la clippers": "LAC",
        "los angeles lakers": "LAL",
        "la lakers": "LAL",
        "philadelphia 76ers": "PHI",
        "philadelphia seventy sixers": "PHI",
        "76ers": "PHI",
        "tampa bay": "TB",
        "los angeles kings": "LA",
        "san jose sharks": "SJ",
        "st louis blues": "STL",
        "vegas golden knights": "VGK",
        "utah hockey club": "UTA",
        "utah jazz": "UTA",
    }
    for alias, code in aliases.items():
        if code in code_to_name:
            out[_normalize_team_name(alias)] = code
    return out


_NBA_ALIASES = _build_alias_map(_NBA_TEAM_CODE_TO_NAME)
_NHL_ALIASES = _build_alias_map(_NHL_TEAM_CODE_TO_NAME)
_TEAM_NAME_MISS_LOGGED: set[tuple[str, str]] = set()
_SPORTSBOOK_SOURCE_LOGGED: set[tuple[str, str, str, int]] = set()
_PARTIAL_LINES_LOCK = threading.Lock()
_PARTIAL_LINES: Dict[Tuple[str, str, str], Dict[str, object]] = {}
BOOKMAKER_HIERARCHY: Tuple[str, ...] = ("pinnacle", "draftkings", "polymarket")
MAX_STALE_LEG_SECONDS = 300.0
MIN_BOLTODDS_STALE_SECONDS = 30.0

_SPORTSBOOK_ALIASES: Dict[str, str] = {
    "pinnacle": "pinnacle",
    "pinnacle sports": "pinnacle",
    "pinnaclesports": "pinnacle",
    "draftkings": "draftkings",
    "draft kings": "draftkings",
    "dk": "draftkings",
    "draftkings sportsbook": "draftkings",
    "draft kings sportsbook": "draftkings",
    "polymarket": "polymarket",
    "poly market": "polymarket",
}


class AtomicOddsStore:
    def __init__(self) -> None:
        now_mono = time.monotonic()
        self._lock = threading.Lock()
        self._snapshot = OddsSnapshot(
            generation_id=0,
            listener_state=ListenerState.BOOTING,
            state_since_mono=now_mono,
            updated_mono=now_mono,
            by_matchup={},
        )

    def snapshot(self) -> OddsSnapshot:
        with self._lock:
            return self._snapshot

    def set_state(self, state: ListenerState, *, generation_id: Optional[int] = None, now_mono: Optional[float] = None) -> None:
        mono = float(now_mono if now_mono is not None else time.monotonic())
        with self._lock:
            old = self._snapshot
            gen = old.generation_id if generation_id is None else int(generation_id)
            state_since = old.state_since_mono if old.listener_state == state else mono
            self._snapshot = OddsSnapshot(
                generation_id=gen,
                listener_state=state,
                state_since_mono=state_since,
                updated_mono=mono,
                by_matchup=old.by_matchup,
            )

    def publish_matchup(self, matchup: MatchupOdds) -> None:
        with self._lock:
            old = self._snapshot
            by = dict(old.by_matchup)
            by[str(matchup.matchup_key)] = matchup
            self._snapshot = OddsSnapshot(
                generation_id=int(matchup.generation_id),
                listener_state=ListenerState.READY,
                state_since_mono=(old.state_since_mono if old.listener_state == ListenerState.READY else float(matchup.updated_mono)),
                updated_mono=float(matchup.updated_mono),
                by_matchup=by,
            )

    def get_team_probability(
        self,
        *,
        league: str,
        team_a: str,
        team_b: str,
        side: str,
        now_mono: Optional[float] = None,
    ) -> Optional[float]:
        side_u = str(side or "").strip().upper()
        if not side_u:
            return None
        a = str(team_a or "").strip().upper()
        b = str(team_b or "").strip().upper()
        if not a or not b:
            return None
        lo, hi = sorted([a, b])
        key = f"{str(league or '').strip().upper()}|{lo}|{hi}"
        snap = self.snapshot()
        m = snap.by_matchup.get(key)
        if not isinstance(m, MatchupOdds):
            return None
        mono = float(now_mono if now_mono is not None else time.monotonic())
        if mono > float(m.stale_after_mono):
            return None
        p = m.p_true_by_team.get(side_u)
        if p is None:
            return None
        try:
            v = float(p)
        except Exception:
            return None
        if not (0.0 <= v <= 1.0):
            return None
        return v


_STORE = AtomicOddsStore()
_LISTENER_LOCK = threading.Lock()
_LISTENER_THREAD: Optional[threading.Thread] = None
_LISTENER_STOP = threading.Event()


def _env_or_cfg(config: Mapping[str, object], key: str, default: str = "") -> str:
    return str(config.get(key) or os.environ.get(key) or default).strip()


def _listener_enabled(config: Mapping[str, object]) -> bool:
    raw = _env_or_cfg(config, "boltodds_enabled", "1").lower()
    return raw not in {"0", "false", "no", "off"}


def _ws_url(config: Mapping[str, object]) -> str:
    return _env_or_cfg(config, "BOLTODDS_WS_URL", "wss://spro.agency/api")


def _api_key(config: Mapping[str, object]) -> str:
    return _env_or_cfg(config, "BOLTODDS_API_KEY", "")


def _ws_url_with_key(*, ws_url: str, api_key: str) -> str:
    raw = str(ws_url or "").strip()
    key = str(api_key or "").strip()
    if not raw:
        return ""
    if not key:
        return raw
    if "{key}" in raw:
        return raw.replace("{key}", key)
    parsed = urlparse(raw)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "key" not in query:
        query["key"] = key
    new_query = urlencode(query, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))


def _float_cfg(config: Mapping[str, object], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except Exception:
        return float(default)


def _allowed_sportsbooks(config: Mapping[str, object]) -> set[str]:
    raw = (
        config.get("boltodds_allowed_sportsbooks")
        or os.environ.get("BOLTODDS_ALLOWED_SPORTSBOOKS")
        or ",".join(BOOKMAKER_HIERARCHY)
    )
    vals: list[str] = []
    if isinstance(raw, str):
        vals = [str(x).strip().lower() for x in str(raw).split(",") if str(x).strip()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        vals = [str(x).strip().lower() for x in raw if str(x).strip()]
    allowed = {_normalize_sportsbook_name(v) for v in vals if v and v not in {"*", "all", "any"}}
    allowed.discard("")
    return allowed


def _normalize_sportsbook_name(raw: object) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", str(raw or "").strip().lower()).strip()
    if not s:
        return ""
    return _SPORTSBOOK_ALIASES.get(s, s.replace(" ", ""))


def _subscription_payload(config: Mapping[str, object]) -> Dict[str, object]:
    sub_payload = config.get("boltodds_subscribe_payload")
    if isinstance(sub_payload, str):
        try:
            sub_payload = json.loads(sub_payload)
        except Exception:
            sub_payload = None
    if isinstance(sub_payload, Mapping):
        action = str(sub_payload.get("action") or "subscribe").strip() or "subscribe"
        filters_obj = sub_payload.get("filters")
        if isinstance(filters_obj, Mapping):
            sports = list(filters_obj.get("sports") or ["NBA", "NHL"])
            markets = list(filters_obj.get("markets") or ["Moneyline"])
        else:
            sports = list(sub_payload.get("sports") or ["NBA", "NHL"])
            markets = list(sub_payload.get("markets") or ["Moneyline"])
        return {
            "action": action,
            "filters": {
                "sports": [str(x) for x in sports if str(x).strip()],
                "markets": [str(x) for x in markets if str(x).strip()],
            },
        }
    return {
        "action": "subscribe",
        "filters": {
            "sports": ["NBA", "NHL"],
            "markets": ["Moneyline"],
        },
    }


def _iter_payload_objects(raw_msg: object) -> Sequence[Mapping[str, object]]:
    out = []
    if isinstance(raw_msg, Mapping):
        if isinstance(raw_msg.get("payload"), Mapping):
            out.append(raw_msg.get("payload"))
        out.append(raw_msg)
    elif isinstance(raw_msg, Sequence) and not isinstance(raw_msg, (str, bytes, bytearray)):
        out.extend([x for x in raw_msg if isinstance(x, Mapping)])
    return [x for x in out if isinstance(x, Mapping)]


def _extract_error_message(raw_msg: object) -> Optional[str]:
    for payload in _iter_payload_objects(raw_msg):
        action = str(payload.get("action") or payload.get("type") or "").strip().lower()
        if action == "error":
            msg = str(payload.get("message") or payload.get("error") or "unknown BoltOdds policy error").strip()
            return msg or "unknown BoltOdds policy error"
    return None


def _american_to_decimal(raw: object) -> Optional[float]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        if s[0] in {"+", "-"}:
            line = float(s)
            if line > 0.0:
                dec = 1.0 + (line / 100.0)
                return dec if dec > 1.0 else None
            if line < 0.0:
                dec = 1.0 + (100.0 / abs(line))
                return dec if dec > 1.0 else None
            return None
        v = float(s)
    except Exception:
        return None
    if v > 1.0:
        return float(v)
    return None


def _extract_decimal_odds(team_payload: Mapping[str, object]) -> Optional[float]:
    keys = ("odds_decimal", "decimal", "price_decimal", "ml_decimal")
    for k in keys:
        if k in team_payload:
            try:
                v = float(team_payload[k])
            except Exception:
                continue
            if v > 1.0:
                return v
    # BoltOdds line_update schema uses American odds in the `odds` field (e.g. +108 / -113).
    for k in ("odds", "american_odds", "line"):
        if k in team_payload:
            dec = _american_to_decimal(team_payload.get(k))
            if dec is not None:
                return dec
    return None


def _calc_pair_probabilities(odds_a: float, odds_b: float) -> Optional[Tuple[float, float]]:
    return binary_shin_devig_from_decimal_odds(float(odds_a), float(odds_b))


def _log_name_miss_once(*, league: str, team_name: str) -> None:
    key = (str(league or "").strip().upper(), _normalize_team_name(team_name))
    if not key[1]:
        return
    if key in _TEAM_NAME_MISS_LOGGED:
        return
    _TEAM_NAME_MISS_LOGGED.add(key)
    print(f"[SHADOW][ORACLE] BoltOdds unmapped_team league={key[0]} name={team_name}")


def _hierarchy_for_allowed_books(allowed_books: set[str]) -> Tuple[str, ...]:
    ranked = [book for book in BOOKMAKER_HIERARCHY if book in allowed_books or not allowed_books]
    if ranked:
        return tuple(ranked)
    return tuple(BOOKMAKER_HIERARCHY)


def _log_book_source_once(*, league: str, matchup_key: str, sportsbook: str, fallback: bool) -> None:
    key = (
        str(league or "").strip().upper(),
        str(matchup_key or "").strip().upper(),
        str(sportsbook or "").strip().lower(),
        1 if fallback else 0,
    )
    if not sportsbook or key in _SPORTSBOOK_SOURCE_LOGGED:
        return
    _SPORTSBOOK_SOURCE_LOGGED.add(key)
    if fallback:
        print(
            f"[SHADOW][ORACLE][DEBUG] BoltOdds fallback book selected league={league} "
            f"matchup={matchup_key} sportsbook={sportsbook}"
        )
    else:
        print(
            f"[SHADOW][ORACLE][DEBUG] BoltOdds preferred book selected league={league} "
            f"matchup={matchup_key} sportsbook={sportsbook}"
        )


def _update_partial_line(
    *,
    league: str,
    sportsbook: str,
    home_code: str,
    away_code: str,
    home_odds: Optional[float],
    away_odds: Optional[float],
    commence_time_utc: Optional[str],
    game_status: str,
    source_ts_ms: int,
    leg_updated_epoch_s: float,
) -> None:
    lo, hi = sorted([str(home_code).strip().upper(), str(away_code).strip().upper()])
    cache_key = (str(league).strip().upper(), f"{lo}|{hi}", str(sportsbook or "").strip().lower())
    with _PARTIAL_LINES_LOCK:
        prev = _PARTIAL_LINES.get(cache_key, {})
        merged: Dict[str, object] = dict(prev)
        merged["league"] = str(league).strip().upper()
        odds_by_team = dict(merged.get("odds_by_team") or {})
        updated_by_team = dict(merged.get("updated_epoch_s_by_team") or {})
        home_code_u = str(home_code).strip().upper()
        away_code_u = str(away_code).strip().upper()
        if home_odds is not None:
            odds_by_team[home_code_u] = float(home_odds)
            updated_by_team[home_code_u] = float(leg_updated_epoch_s)
        if away_odds is not None:
            odds_by_team[away_code_u] = float(away_odds)
            updated_by_team[away_code_u] = float(leg_updated_epoch_s)
        merged["odds_by_team"] = odds_by_team
        merged["updated_epoch_s_by_team"] = updated_by_team
        if commence_time_utc:
            merged["commence_time_utc"] = str(commence_time_utc)
        merged["game_status"] = str(game_status or merged.get("game_status") or "unknown").strip().lower() or "unknown"
        merged["source_ts_ms"] = max(int(source_ts_ms or 0), int(merged.get("source_ts_ms") or 0))
        _PARTIAL_LINES[cache_key] = merged


def _select_best_complete_line(
    *,
    league: str,
    home_code: str,
    away_code: str,
    allowed_books: set[str],
    max_stale_leg_seconds: float,
    allow_hierarchy_fallback: bool,
) -> Optional[Dict[str, object]]:
    lo, hi = sorted([str(home_code).strip().upper(), str(away_code).strip().upper()])
    pair_key = f"{lo}|{hi}"
    preferred_hierarchy = _hierarchy_for_allowed_books(allowed_books)
    fallback_hierarchy = tuple(book for book in BOOKMAKER_HIERARCHY if book not in preferred_hierarchy)

    def _candidate_for_book(sportsbook: str) -> Optional[Dict[str, object]]:
        cache_key = (str(league).strip().upper(), pair_key, str(sportsbook).strip().lower())
        merged = _PARTIAL_LINES.get(cache_key)
        if not isinstance(merged, dict):
            return None
        odds_by_team = merged.get("odds_by_team")
        updated_by_team = merged.get("updated_epoch_s_by_team")
        if not isinstance(odds_by_team, Mapping) or not isinstance(updated_by_team, Mapping):
            return None
        home_code_u = str(home_code).strip().upper()
        away_code_u = str(away_code).strip().upper()
        home_odds = odds_by_team.get(home_code_u)
        away_odds = odds_by_team.get(away_code_u)
        home_updated = updated_by_team.get(home_code_u)
        away_updated = updated_by_team.get(away_code_u)
        try:
            home_odds_f = float(home_odds)
            away_odds_f = float(away_odds)
            home_updated_f = float(home_updated)
            away_updated_f = float(away_updated)
        except Exception:
            return None
        if abs(home_updated_f - away_updated_f) > max(1.0, float(max_stale_leg_seconds)):
            return None
        return {
            "sportsbook": sportsbook,
            "team_odds": {
                home_code_u: home_odds_f,
                away_code_u: away_odds_f,
            },
            "commence_time_utc": (str(merged.get("commence_time_utc") or "") or None),
            "game_status": str(merged.get("game_status") or "unknown").strip().lower() or "unknown",
            "source_ts_ms": int(merged.get("source_ts_ms") or 0),
        }

    with _PARTIAL_LINES_LOCK:
        for sportsbook in preferred_hierarchy:
            candidate = _candidate_for_book(sportsbook)
            if candidate is not None:
                return candidate
        if allow_hierarchy_fallback:
            for sportsbook in fallback_hierarchy:
                candidate = _candidate_for_book(sportsbook)
                if candidate is not None:
                    candidate["fallback_book"] = True
                    return candidate
    return None


def _reset_partial_lines() -> None:
    with _PARTIAL_LINES_LOCK:
        _PARTIAL_LINES.clear()


def _parse_msg_to_matchup(
    *,
    payload: Mapping[str, object],
    config: Mapping[str, object],
    generation_id: int,
    seq_no: int,
) -> Optional[MatchupOdds]:
    action = str(payload.get("action") or "").strip().lower()
    if action and action != "line_update":
        return None

    data = payload.get("data") if isinstance(payload.get("data"), Mapping) else payload
    if not isinstance(data, Mapping):
        return None

    sportsbook = _normalize_sportsbook_name(data.get("sportsbook"))
    allowed_books = _allowed_sportsbooks(config)

    league_raw = str(data.get("league") or data.get("sport") or "").strip().upper()
    if "NBA" in league_raw or league_raw == "BASKETBALL_NBA":
        league = "NBA"
        aliases = _NBA_ALIASES
    elif "NHL" in league_raw or league_raw == "ICEHOCKEY_NHL":
        league = "NHL"
        aliases = _NHL_ALIASES
    else:
        return None

    home_name = _normalize_team_name(data.get("home_team") or data.get("home") or data.get("team_home"))
    away_name = _normalize_team_name(data.get("away_team") or data.get("away") or data.get("team_away"))
    if not home_name or not away_name:
        return None

    home_code = aliases.get(home_name)
    away_code = aliases.get(away_name)
    if not home_code:
        _log_name_miss_once(league=league, team_name=str(data.get("home_team") or data.get("home") or data.get("team_home") or ""))
    if not away_code:
        _log_name_miss_once(league=league, team_name=str(data.get("away_team") or data.get("away") or data.get("team_away") or ""))
    if not home_code or not away_code:
        return None

    home_odds = None
    away_odds = None

    if isinstance(data.get("home"), Mapping):
        home_odds = _extract_decimal_odds(data.get("home"))
    if isinstance(data.get("away"), Mapping):
        away_odds = _extract_decimal_odds(data.get("away"))

    if home_odds is None:
        for key in ("home_odds_decimal", "home_decimal", "home_price_decimal"):
            if key in data:
                try:
                    v = float(data[key])
                except Exception:
                    continue
                if v > 1.0:
                    home_odds = v
                    break

    if away_odds is None:
        for key in ("away_odds_decimal", "away_decimal", "away_price_decimal"):
            if key in data:
                try:
                    v = float(data[key])
                except Exception:
                    continue
                if v > 1.0:
                    away_odds = v
                    break

    if home_odds is None or away_odds is None:
        outcomes = data.get("outcomes")
        if isinstance(outcomes, Mapping):
            for row in outcomes.values():
                if not isinstance(row, Mapping):
                    continue
                team_name = _normalize_team_name(
                    row.get("outcome_target") or row.get("name") or row.get("team")
                )
                code = aliases.get(team_name)
                dec = _extract_decimal_odds(row)
                if dec is None:
                    continue
                if code == home_code:
                    home_odds = dec
                elif code == away_code:
                    away_odds = dec
        elif isinstance(outcomes, Sequence):
            for row in outcomes:
                if not isinstance(row, Mapping):
                    continue
                team_name = _normalize_team_name(
                    row.get("outcome_target") or row.get("name") or row.get("team")
                )
                code = aliases.get(team_name)
                dec = _extract_decimal_odds(row)
                if dec is None:
                    continue
                if code == home_code:
                    home_odds = dec
                elif code == away_code:
                    away_odds = dec

    commence_raw = str(
        data.get("commence_time")
        or data.get("start_time")
        or (
            data.get("info", {}).get("when")
            if isinstance(data.get("info"), Mapping)
            else ""
        )
        or ""
    ).strip()
    commence_iso = None
    if commence_raw:
        s = commence_raw
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            d = datetime.fromisoformat(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            commence_iso = d.astimezone(timezone.utc).isoformat()
        except Exception:
            try:
                # Example: "2026-02-28, 07:00 PM" from info.when
                d_local = datetime.strptime(commence_raw, "%Y-%m-%d, %I:%M %p")
                d_local = d_local.replace(tzinfo=timezone.utc)
                commence_iso = d_local.isoformat()
            except Exception:
                commence_iso = None

    source_ts_ms = 0
    ts_raw = str(payload.get("timestamp") or "").strip()
    if ts_raw:
        try:
            ts_norm = ts_raw[:-1] + "+00:00" if ts_raw.endswith("Z") else ts_raw
            ts_dt = datetime.fromisoformat(ts_norm)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            source_ts_ms = int(ts_dt.timestamp() * 1000.0)
        except Exception:
            source_ts_ms = 0
    for ts_key in ("timestamp_ms", "ts", "updated_ms"):
        if ts_key in data:
            try:
                source_ts_ms = int(float(data[ts_key]))
                break
            except Exception:
                pass

    now_mono = time.monotonic()
    leg_updated_epoch_s = time.time()
    stale_seconds = max(
        MIN_BOLTODDS_STALE_SECONDS,
        _float_cfg(config, "boltodds_stale_seconds", MIN_BOLTODDS_STALE_SECONDS),
    )
    max_stale_leg_seconds = _float_cfg(config, "boltodds_max_stale_leg_seconds", MAX_STALE_LEG_SECONDS)
    allow_hierarchy_fallback = str(
        config.get("boltodds_allow_hierarchy_fallback")
        or os.environ.get("BOLTODDS_ALLOW_HIERARCHY_FALLBACK")
        or "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    _update_partial_line(
        league=league,
        sportsbook=sportsbook,
        home_code=home_code,
        away_code=away_code,
        home_odds=home_odds,
        away_odds=away_odds,
        commence_time_utc=commence_iso,
        game_status=str(data.get("game_status") or data.get("status") or "unknown").strip().lower() or "unknown",
        source_ts_ms=source_ts_ms,
        leg_updated_epoch_s=leg_updated_epoch_s,
    )
    selected_line = _select_best_complete_line(
        league=league,
        home_code=home_code,
        away_code=away_code,
        allowed_books=allowed_books,
        max_stale_leg_seconds=max_stale_leg_seconds,
        allow_hierarchy_fallback=allow_hierarchy_fallback,
    )
    if not isinstance(selected_line, dict):
        return None
    team_odds = selected_line.get("team_odds")
    if not isinstance(team_odds, Mapping):
        return None
    try:
        home_odds = float(team_odds[str(home_code).strip().upper()])
        away_odds = float(team_odds[str(away_code).strip().upper()])
    except Exception:
        return None
    commence_iso = selected_line.get("commence_time_utc")
    game_status = str(selected_line.get("game_status") or "unknown").strip().lower() or "unknown"
    source_ts_ms = int(selected_line.get("source_ts_ms") or 0)
    lo, hi = sorted([home_code, away_code])
    _log_book_source_once(
        league=league,
        matchup_key=f"{lo}|{hi}",
        sportsbook=str(selected_line.get("sportsbook") or ""),
        fallback=bool(selected_line.get("fallback_book")),
    )

    probs = _calc_pair_probabilities(home_odds, away_odds)
    if probs is None:
        return None

    p_home, p_away = probs
    p_true_by_team: Dict[str, float] = {
        str(home_code): float(p_home),
        str(away_code): float(p_away),
    }

    matchup_key = f"{league}|{lo}|{hi}"

    return MatchupOdds(
        matchup_key=matchup_key,
        league=league,
        team_a=home_code,
        team_b=away_code,
        p_true_by_team=p_true_by_team,
        commence_time_utc=commence_iso,
        game_status=game_status,
        source_ts_ms=source_ts_ms,
        updated_mono=now_mono,
        stale_after_mono=now_mono + max(1.0, float(stale_seconds)),
        generation_id=int(generation_id),
        seq_no=int(seq_no),
    )


async def _listener_loop(config: Mapping[str, object], stop_event: threading.Event) -> None:
    generation = 0
    backoff = 1.0
    while not stop_event.is_set():
        generation += 1
        _reset_partial_lines()
        ws_url = _ws_url(config)
        api_key = _api_key(config)
        if not ws_url or not api_key:
            _STORE.set_state(ListenerState.SUSPENDED, generation_id=generation)
            await asyncio.sleep(5.0)
            continue
        ws_connect_url = _ws_url_with_key(ws_url=ws_url, api_key=api_key)

        _STORE.set_state(ListenerState.CONNECTING, generation_id=generation)
        sub_payload = _subscription_payload(config)

        try:
            _STORE.set_state(ListenerState.SUBSCRIBING, generation_id=generation)
            connect_kwargs = {
                "ping_interval": 20,
                "close_timeout": 5,
                "open_timeout": 10,
            }
            ws = await websockets.connect(ws_connect_url, **connect_kwargs)

            async with ws:
                # BoltOdds emits an immediate payload after connect; wait for it before subscribe.
                ack_raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                if isinstance(ack_raw, (bytes, bytearray)):
                    ack_raw = ack_raw.decode("utf-8", errors="ignore")
                try:
                    ack_msg = json.loads(str(ack_raw))
                except Exception:
                    ack_msg = {"raw": str(ack_raw)}
                err = _extract_error_message(ack_msg)
                if err:
                    print(
                        f"[SHADOW][ORACLE][CRITICAL] BoltOdds policy/auth error during connect handshake: {err}"
                    )
                    _STORE.set_state(ListenerState.SUSPENDED, generation_id=generation)
                    stop_event.set()
                    return
                print(
                    f"[SHADOW][ORACLE] BoltOdds ack received generation={generation} "
                    f"payload_type={type(ack_msg).__name__}"
                )

                await ws.send(json.dumps(sub_payload, separators=(",", ":"), sort_keys=True))
                print(f"[SHADOW][ORACLE] BoltOdds subscribe sent payload={sub_payload}")
                _STORE.set_state(ListenerState.SYNCING, generation_id=generation)
                seq_no = 0
                backoff = 1.0

                while not stop_event.is_set():
                    raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8", errors="ignore")
                    try:
                        msg = json.loads(str(raw))
                    except Exception:
                        continue

                    err = _extract_error_message(msg)
                    if err:
                        print(
                            f"[SHADOW][ORACLE][CRITICAL] BoltOdds stream error received: {err}. "
                            f"Listener entering SUSPENDED and stopping."
                        )
                        _STORE.set_state(ListenerState.SUSPENDED, generation_id=generation)
                        stop_event.set()
                        return

                    payloads = []
                    payloads.extend(_iter_payload_objects(msg))

                    accepted = 0
                    for payload in payloads:
                        seq_no += 1
                        matchup = _parse_msg_to_matchup(
                            payload=payload,
                            config=config,
                            generation_id=generation,
                            seq_no=seq_no,
                        )
                        if matchup is None:
                            continue
                        accepted += 1
                        _STORE.publish_matchup(matchup)

                    if accepted == 0:
                        snap = _STORE.snapshot()
                        idle_seconds = time.monotonic() - float(snap.updated_mono)
                        stale_seconds = max(1.0, _float_cfg(config, "boltodds_stale_seconds", 5.0))
                        if idle_seconds > stale_seconds:
                            _STORE.set_state(ListenerState.DEGRADED, generation_id=generation)
                        continue

                    _STORE.set_state(ListenerState.READY, generation_id=generation)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            print(f"[SHADOW][ORACLE] BoltOdds listener reconnecting error={type(exc).__name__}: {exc}")
            _STORE.set_state(ListenerState.RECONNECTING, generation_id=generation)
            jitter = random.random() * 0.2 * backoff
            await asyncio.sleep(min(30.0, backoff + jitter))
            backoff = min(30.0, backoff * 2.0)

    _reset_partial_lines()
    _STORE.set_state(ListenerState.STOPPED)


def _listener_main(config: Mapping[str, object], stop_event: threading.Event) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_listener_loop(config, stop_event))
    finally:
        pending = asyncio.all_tasks(loop=loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def ensure_bolt_listener(config: Mapping[str, object]) -> None:
    if not _listener_enabled(config):
        return
    global _LISTENER_THREAD
    with _LISTENER_LOCK:
        if _LISTENER_THREAD is not None and _LISTENER_THREAD.is_alive():
            return
        _LISTENER_STOP.clear()
        thread = threading.Thread(
            target=_listener_main,
            args=(dict(config), _LISTENER_STOP),
            name="boltodds-listener",
            daemon=True,
        )
        thread.start()
        _LISTENER_THREAD = thread


def stop_bolt_listener() -> None:
    global _LISTENER_THREAD
    with _LISTENER_LOCK:
        _LISTENER_STOP.set()
        if _LISTENER_THREAD is not None and _LISTENER_THREAD.is_alive():
            _LISTENER_THREAD.join(timeout=2.0)
        _LISTENER_THREAD = None


def _parse_codes_for_ticker(ticker: str, event_ticker: str, nba: bool) -> Optional[Tuple[str, str, str, Optional[str]]]:
    t = str(ticker or "").strip().upper()
    ev = str(event_ticker or "").strip().upper()
    pat = _NBA_EVENT_RE if nba else _NHL_EVENT_RE
    m = pat.match(t) or pat.match(ev)
    if not m:
        return None
    pair = str(m.group("pair") or "").strip().upper()
    if len(pair) != 6:
        return None
    a = pair[:3]
    b = pair[3:]
    side = str(m.group("side") or "").strip().upper() or None
    return (str(m.group("date") or "").strip().upper(), a, b, side)


def _selected_side_code(parsed: Tuple[str, str, str, Optional[str]], title: str) -> Optional[str]:
    _, team_a, team_b, side = parsed
    if side in {team_a, team_b}:
        return side
    title_up = str(title or "").strip().upper()
    if team_a and team_a in title_up:
        return team_a
    if team_b and team_b in title_up:
        return team_b
    return None


def lookup_bolt_nba_live_p_true(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Mapping[str, object],
) -> Optional[float]:
    ensure_bolt_listener(config)
    parsed = _parse_codes_for_ticker(ticker=ticker, event_ticker=event_ticker, nba=True)
    if parsed is None:
        return None
    _, team_a, team_b, _ = parsed
    side = _selected_side_code(parsed, title)
    if side is None:
        return None
    return _STORE.get_team_probability(league="NBA", team_a=team_a, team_b=team_b, side=side)


def lookup_bolt_nhl_live_p_true(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Mapping[str, object],
) -> Optional[float]:
    ensure_bolt_listener(config)
    parsed = _parse_codes_for_ticker(ticker=ticker, event_ticker=event_ticker, nba=False)
    if parsed is None:
        return None
    _, team_a, team_b, _ = parsed
    side = _selected_side_code(parsed, title)
    if side is None:
        return None
    return _STORE.get_team_probability(league="NHL", team_a=team_a, team_b=team_b, side=side)


def lookup_bolt_nba_commence_time_utc(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Mapping[str, object],
) -> Optional[str]:
    ensure_bolt_listener(config)
    parsed = _parse_codes_for_ticker(ticker=ticker, event_ticker=event_ticker, nba=True)
    if parsed is None:
        return None
    _, team_a, team_b, _ = parsed
    lo, hi = sorted([team_a, team_b])
    key = f"NBA|{lo}|{hi}"
    snap = _STORE.snapshot()
    m = snap.by_matchup.get(key)
    if not isinstance(m, MatchupOdds):
        return None
    if time.monotonic() > float(m.stale_after_mono):
        return None
    return str(m.commence_time_utc or "") or None


def lookup_bolt_nhl_commence_time_utc(
    *,
    ticker: str,
    event_ticker: str,
    title: str,
    config: Mapping[str, object],
) -> Optional[str]:
    ensure_bolt_listener(config)
    parsed = _parse_codes_for_ticker(ticker=ticker, event_ticker=event_ticker, nba=False)
    if parsed is None:
        return None
    _, team_a, team_b, _ = parsed
    lo, hi = sorted([team_a, team_b])
    key = f"NHL|{lo}|{hi}"
    snap = _STORE.snapshot()
    m = snap.by_matchup.get(key)
    if not isinstance(m, MatchupOdds):
        return None
    if time.monotonic() > float(m.stale_after_mono):
        return None
    return str(m.commence_time_utc or "") or None
