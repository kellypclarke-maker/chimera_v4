from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from chimera.clients.http import get_json
from chimera.clients.http import get_with_retries
from chimera.teams import normalize_team_code


ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEY_BY_LEAGUE = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
    "nfl": "americanfootball_nfl",
}


_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _name_key(name: str) -> str:
    return _NON_ALNUM.sub("", str(name or "").strip().upper())


def _build_name_map() -> Dict[str, Dict[str, str]]:
    # Minimal, explicit mappings for Odds API team name strings.
    nhl = {
        "ANAHEIMDUCKS": "ANA",
        "BOSTONBRUINS": "BOS",
        "BUFFALOSABRES": "BUF",
        "CAROLINAHURRICANES": "CAR",
        "COLUMBUSBLUEJACKETS": "CBJ",
        "CALGARYFLAMES": "CGY",
        "CHICAGOBLACKHAWKS": "CHI",
        "COLORADOAVALANCHE": "COL",
        "DALLASSTARS": "DAL",
        "DETROITREDWINGS": "DET",
        "EDMONTONOILERS": "EDM",
        "FLORIDAPANTHERS": "FLA",
        "LOSANGELESKINGS": "LAK",
        "MINNESOTAWILD": "MIN",
        "MONTREALCANADIENS": "MTL",
        "NEWJERSEYDEVILS": "NJD",
        "NASHVILLEPREDATORS": "NSH",
        "NEWYORKISLANDERS": "NYI",
        "NEWYORKRANGERS": "NYR",
        "OTTAWASENATORS": "OTT",
        "PHILADELPHIAFLYERS": "PHI",
        "PITTSBURGHPENGUINS": "PIT",
        "SEATTLEKRAKEN": "SEA",
        "SANJOSESHARKS": "SJS",
        "STLOUISBLUES": "STL",
        "STLOUIS": "STL",
        "TAMPABAYLIGHTNING": "TBL",
        "TORONTOMAPLELEAFS": "TOR",
        "UTAHHOCKEYCLUB": "UTA",
        "UTAHMAMMOTH": "UTA",
        "UTAH": "UTA",
        "VANCOUVERCANUCKS": "VAN",
        "VEGASGOLDENKNIGHTS": "VGK",
        "WINNIPEGJETS": "WPG",
        "WASHINGTONCAPITALS": "WSH",
        "ARIZONACOYOTES": "UTA",
    }
    nba = {
        "ATLANTAHAWKS": "ATL",
        "BOSTONCELTICS": "BOS",
        "BROOKLYNNETS": "BKN",
        "CHARLOTTEHORNETS": "CHA",
        "CHICAGOBULLS": "CHI",
        "CLEVELANDCAVALIERS": "CLE",
        "DALLASMAVERICKS": "DAL",
        "DENVERNUGGETS": "DEN",
        "DETROITPISTONS": "DET",
        "GOLDENSTATEWARRIORS": "GSW",
        "HOUSTONROCKETS": "HOU",
        "INDIANAPACERS": "IND",
        "LOSANGELESCLIPPERS": "LAC",
        "LOSANGELESLAKERS": "LAL",
        "MEMPHISGRIZZLIES": "MEM",
        "MIAMIHEAT": "MIA",
        "MILWAUKEEBUCKS": "MIL",
        "MINNESOTATIMBERWOLVES": "MIN",
        "NEWORLEANSPELICANS": "NOP",
        "NEWYORKKNICKS": "NYK",
        "OKLAHOMACITYTHUNDER": "OKC",
        "ORLANDOMAGIC": "ORL",
        "PHILADELPHIA76ERS": "PHI",
        "PHILADELPHIASEVENTYSIXERS": "PHI",
        "PHOENIXSUNS": "PHX",
        "PORTLANDTRAILBLAZERS": "POR",
        "SACRAMENTOKINGS": "SAC",
        "SANANTONIOSPURS": "SAS",
        "TORONTORAPTORS": "TOR",
        "UTAHJAZZ": "UTA",
        "WASHINGTONWIZARDS": "WAS",
    }
    nfl = {
        "ARIZONACARDINALS": "ARI",
        "ATLANTAFALCONS": "ATL",
        "BALTIMORERAVENS": "BAL",
        "BUFFALOBILLS": "BUF",
        "CAROLINAPANTHERS": "CAR",
        "CHICAGOBEARS": "CHI",
        "CINCINNATIBENGALS": "CIN",
        "CLEVELANDBROWNS": "CLE",
        "DALLASCOWBOYS": "DAL",
        "DENVERBRONCOS": "DEN",
        "DETROITLIONS": "DET",
        "GREENBAYPACKERS": "GB",
        "HOUSTONTEXANS": "HOU",
        "INDIANAPOLISCOLTS": "IND",
        "JACKSONVILLEJAGUARS": "JAX",
        "KANSASCITYCHIEFS": "KC",
        "LASVEGASRAIDERS": "LV",
        "LOSANGELESCHARGERS": "LAC",
        "LOSANGELESRAMS": "LAR",
        "MIAMIDOLPHINS": "MIA",
        "MINNESOTAVIKINGS": "MIN",
        "NEWENGLANDPATRIOTS": "NE",
        "NEWORLEANSSAINTS": "NO",
        "NEWYORKGIANTS": "NYG",
        "NEWYORKJETS": "NYJ",
        "PHILADELPHIAEAGLES": "PHI",
        "PITTSBURGHSTEELERS": "PIT",
        "SEATTLESEAHAWKS": "SEA",
        "SANFRANCISCO49ERS": "SF",
        "TAMPABAYBUCCANEERS": "TB",
        "TENNESSEETITANS": "TEN",
        "WASHINGTONCOMMANDERS": "WAS",
        "WASHINGTONFOOTBALLTEAM": "WAS",
        "WASHINGTONREDSKINS": "WAS",
    }
    return {"nhl": nhl, "nba": nba, "nfl": nfl}


_NAME_MAP = _build_name_map()


def team_name_to_code(name: str, league: str) -> Optional[str]:
    lg = str(league or "").strip().lower()
    if lg not in _NAME_MAP:
        return None
    k = _name_key(name)
    if not k:
        return None
    code = _NAME_MAP[lg].get(k)
    if code:
        return normalize_team_code(code, lg) or code
    # Fall back: if Odds API happens to return abbreviations sometimes.
    return normalize_team_code(str(name).strip().upper(), lg)


def american_to_implied_prob(price_american: int) -> float:
    a = int(price_american)
    if a == 0:
        raise ValueError("american odds cannot be 0")
    if a > 0:
        return 100.0 / (a + 100.0)
    return (-a) / ((-a) + 100.0)


def devig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    s = float(p1) + float(p2)
    if s <= 0.0:
        return 0.5, 0.5
    return float(p1) / s, float(p2) / s


@dataclass(frozen=True)
class H2HConsensus:
    matchup: str  # AWAY@HOME (canonical codes)
    commence_time_utc: dt.datetime
    p_home: float
    p_away: float
    books_used: int


def _parse_utc_iso(text: str) -> Optional[dt.datetime]:
    s = str(text or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        out = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt.timezone.utc)
    return out.astimezone(dt.timezone.utc)


def _cache_key(url: str, params: Dict[str, object]) -> str:
    raw = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params.keys()))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def fetch_odds(
    *,
    league: str,
    markets: Sequence[str] = ("h2h",),
    regions: str = "us",
    bookmakers: Sequence[str] = (),
    odds_format: str = "american",
    date_format: str = "iso",
    api_key: Optional[str] = None,
    base_url: str = ODDS_API_BASE,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch live odds snapshot from The Odds API.
    """
    lg = str(league or "").strip().lower()
    sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
    if not sport_key:
        raise ValueError(f"unsupported league: {league}")
    # Prefer explicit key, otherwise try live key then history key (history plans can access live endpoints too).
    candidates: List[str] = []
    if api_key and str(api_key).strip():
        candidates.append(str(api_key).strip())
    else:
        for k in ("THE_ODDS_API_KEY", "THE_ODDS_API_HISTORY_KEY"):
            v = str(os.environ.get(k) or "").strip()
            if v and v not in candidates:
                candidates.append(v)
    if not candidates:
        raise RuntimeError("missing THE_ODDS_API_KEY / THE_ODDS_API_HISTORY_KEY (or pass api_key=...)")

    url = f"{str(base_url).rstrip('/')}/sports/{sport_key}/odds"
    last_http: Optional[requests.HTTPError] = None
    for key in candidates:
        params: Dict[str, object] = {
            "apiKey": str(key),
            "regions": str(regions).strip().lower() or "us",
            "markets": ",".join([m.strip().lower() for m in markets if str(m).strip()]) or "h2h",
            "oddsFormat": str(odds_format).strip().lower() or "american",
            "dateFormat": str(date_format).strip().lower() or "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(sorted({b.strip().lower() for b in bookmakers if str(b).strip()}))

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            k = _cache_key(url, params)
            cache_path = Path(cache_dir) / "odds_api" / lg / f"live_{k}.json"
            if cache_path.exists() and not force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    cached = None
                if isinstance(cached, list):
                    return cached

        s = session or requests.Session()
        try:
            resp = get_with_retries(s, url, params=params, timeout_s=30.0)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_http = exc
            r = getattr(exc, "response", None)
            status = getattr(r, "status_code", None)
            if status == 401 and len(candidates) > 1:
                continue
            raise

        data = resp.json()
        if not isinstance(data, list):
            raise TypeError(f"unexpected odds payload shape from {url} (expected list)")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    if last_http is not None:
        raise last_http
    raise RuntimeError("odds request failed")


def fetch_events(
    *,
    league: str,
    api_key: Optional[str] = None,
    base_url: str = ODDS_API_BASE,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch the list of events for a league (no odds, free endpoint).
    """
    lg = str(league or "").strip().lower()
    sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
    if not sport_key:
        raise ValueError(f"unsupported league: {league}")
    candidates: List[str] = []
    if api_key and str(api_key).strip():
        candidates.append(str(api_key).strip())
    else:
        for k in ("THE_ODDS_API_KEY", "THE_ODDS_API_HISTORY_KEY"):
            v = str(os.environ.get(k) or "").strip()
            if v and v not in candidates:
                candidates.append(v)
    if not candidates:
        raise RuntimeError("missing THE_ODDS_API_KEY / THE_ODDS_API_HISTORY_KEY (or pass api_key=...)")

    url = f"{str(base_url).rstrip('/')}/sports/{sport_key}/events"
    last_http: Optional[requests.HTTPError] = None
    for key in candidates:
        params: Dict[str, object] = {"apiKey": str(key)}

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            k = _cache_key(url, {**params, "apiKey": "REDACTED"})
            cache_path = Path(cache_dir) / "odds_api" / lg / f"events_{k}.json"
            if cache_path.exists() and not force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    cached = None
                if isinstance(cached, list):
                    return cached

        s = session or requests.Session()
        try:
            resp = get_with_retries(s, url, params=params, timeout_s=30.0)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_http = exc
            r = getattr(exc, "response", None)
            status = getattr(r, "status_code", None)
            if status == 401 and len(candidates) > 1:
                continue
            raise

        data = resp.json()
        if not isinstance(data, list):
            raise TypeError(f"unexpected events payload shape from {url} (expected list)")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    if last_http is not None:
        raise last_http
    raise RuntimeError("events request failed")


def fetch_event_odds(
    *,
    league: str,
    event_id: str,
    markets: Sequence[str] = ("h2h",),
    regions: str = "us",
    bookmakers: Sequence[str] = (),
    odds_format: str = "american",
    date_format: str = "iso",
    api_key: Optional[str] = None,
    base_url: str = ODDS_API_BASE,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Fetch odds for a specific event.

    This endpoint supports additional markets not available via the bulk `/odds`
    endpoint (e.g., `alternate_totals`, `alternate_spreads`).
    """
    lg = str(league or "").strip().lower()
    sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
    if not sport_key:
        raise ValueError(f"unsupported league: {league}")
    # Prefer explicit key, otherwise try live key then history key.
    candidates: List[str] = []
    if api_key and str(api_key).strip():
        candidates.append(str(api_key).strip())
    else:
        for k in ("THE_ODDS_API_KEY", "THE_ODDS_API_HISTORY_KEY"):
            v = str(os.environ.get(k) or "").strip()
            if v and v not in candidates:
                candidates.append(v)
    if not candidates:
        raise RuntimeError("missing THE_ODDS_API_KEY / THE_ODDS_API_HISTORY_KEY (or pass api_key=...)")
    eid = str(event_id or "").strip()
    if not eid:
        raise ValueError("missing event_id")

    url = f"{str(base_url).rstrip('/')}/sports/{sport_key}/events/{eid}/odds"
    last_http: Optional[requests.HTTPError] = None
    for key in candidates:
        params: Dict[str, object] = {
            "apiKey": str(key),
            "regions": str(regions).strip().lower() or "us",
            "markets": ",".join([m.strip().lower() for m in markets if str(m).strip()]) or "h2h",
            "oddsFormat": str(odds_format).strip().lower() or "american",
            "dateFormat": str(date_format).strip().lower() or "iso",
        }
        if bookmakers:
            params["bookmakers"] = ",".join(sorted({b.strip().lower() for b in bookmakers if str(b).strip()}))

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            k = _cache_key(url, params)
            cache_path = Path(cache_dir) / "odds_api" / lg / f"event_{eid}_{k}.json"
            if cache_path.exists() and not force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    cached = None
                if isinstance(cached, dict):
                    return cached

        s = session or requests.Session()
        try:
            resp = get_with_retries(s, url, params=params, timeout_s=30.0)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_http = exc
            r = getattr(exc, "response", None)
            status = getattr(r, "status_code", None)
            if status == 401 and len(candidates) > 1:
                continue
            raise

        data = resp.json()
        if not isinstance(data, dict):
            raise TypeError(f"unexpected event-odds payload shape from {url} (expected dict)")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        return data

    if last_http is not None:
        raise last_http
    raise RuntimeError("event-odds request failed")


def fetch_odds_history(
    *,
    league: str,
    snapshot_iso: str,
    markets: Sequence[str] = ("h2h",),
    regions: str = "us",
    bookmakers: Sequence[str] = (),
    odds_format: str = "american",
    date_format: str = "iso",
    api_key: Optional[str] = None,
    base_url: str = ODDS_API_BASE,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch a historical odds snapshot.

    Note: The Odds API history endpoint expects `date` in ISO format.
    """
    lg = str(league or "").strip().lower()
    sport_key = SPORT_KEY_BY_LEAGUE.get(lg)
    if not sport_key:
        raise ValueError(f"unsupported league: {league}")

    # Prefer an explicit key, otherwise try history key then live key.
    candidates: List[str] = []
    if api_key and str(api_key).strip():
        candidates.append(str(api_key).strip())
    else:
        for k in ("THE_ODDS_API_HISTORY_KEY", "THE_ODDS_API_KEY"):
            v = str(os.environ.get(k) or "").strip()
            if v and v not in candidates:
                candidates.append(v)
    if not candidates:
        raise RuntimeError("missing THE_ODDS_API_HISTORY_KEY / THE_ODDS_API_KEY (or pass api_key=...)")

    url = f"{str(base_url).rstrip('/')}/sports/{sport_key}/odds-history"
    last_http: Optional[requests.HTTPError] = None
    for key in candidates:
        params: Dict[str, object] = {
            "apiKey": key,
            "regions": str(regions).strip().lower() or "us",
            "markets": ",".join([m.strip().lower() for m in markets if str(m).strip()]) or "h2h",
            "oddsFormat": str(odds_format).strip().lower() or "american",
            "dateFormat": str(date_format).strip().lower() or "iso",
            "date": str(snapshot_iso).strip(),
        }
        if bookmakers:
            params["bookmakers"] = ",".join(sorted({b.strip().lower() for b in bookmakers if str(b).strip()}))

        cache_path: Optional[Path] = None
        if cache_dir is not None:
            # Use the snapshot timestamp itself as an index-friendly filename.
            safe_ts = re.sub(r"[^0-9TZ]", "", str(snapshot_iso).upper())
            k = _cache_key(url, {**params, "apiKey": "REDACTED"})
            cache_path = Path(cache_dir) / "odds_api" / lg / "history" / f"{safe_ts}_{k}.json"
            if cache_path.exists() and not force:
                try:
                    cached = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:
                    cached = None
                if isinstance(cached, dict) and isinstance(cached.get("data"), list):
                    return list(cached.get("data") or [])
                if isinstance(cached, list):
                    return cached

        s = session or requests.Session()
        try:
            resp = get_with_retries(s, url, params=params, timeout_s=60.0)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            last_http = exc
            r = getattr(exc, "response", None)
            status = getattr(r, "status_code", None)
            # If the key is invalid or lacks history access, try the next candidate key.
            if status == 401 and len(candidates) > 1:
                try:
                    payload = r.json() if r is not None else {}
                except Exception:
                    payload = {}
                err_code = str(payload.get("error_code") or payload.get("code") or "") if isinstance(payload, dict) else ""
                if err_code in {"INVALID_KEY", "DEACTIVATED_KEY", "HISTORICAL_UNAVAILABLE_ON_FREE_USAGE_PLAN"}:
                    continue
            raise

        payload = resp.json()
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            data = list(payload.get("data") or [])
        elif isinstance(payload, list):
            data = payload
        else:
            raise TypeError(f"unexpected odds-history payload shape from {url}")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return data

    if last_http is not None:
        raise last_http
    raise RuntimeError("odds-history request failed")


def consensus_h2h_probs(events: Iterable[Dict[str, Any]], *, league: str) -> List[H2HConsensus]:
    """
    Build a consensus de-vig'd H2H probability per event.

    Strategy:
      - For each bookmaker, compute de-vig probs from the two H2H prices.
      - Consensus = simple average across bookmakers.
    """
    lg = str(league or "").strip().lower()
    out: List[H2HConsensus] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        home_name = str(ev.get("home_team") or "").strip()
        away_name = str(ev.get("away_team") or "").strip()
        home = team_name_to_code(home_name, lg)
        away = team_name_to_code(away_name, lg)
        if not home or not away or home == away:
            continue
        commence = _parse_utc_iso(str(ev.get("commence_time") or "")) or None
        if commence is None:
            continue

        book_probs: List[float] = []
        for bm in ev.get("bookmakers") or []:
            if not isinstance(bm, dict):
                continue
            markets = bm.get("markets") or []
            h2h = next((m for m in markets if isinstance(m, dict) and str(m.get("key") or "").lower() == "h2h"), None)
            if not isinstance(h2h, dict):
                continue
            outcomes = h2h.get("outcomes") or []
            if not isinstance(outcomes, list) or len(outcomes) < 2:
                continue
            prices: Dict[str, int] = {}
            for o in outcomes:
                if not isinstance(o, dict):
                    continue
                name = str(o.get("name") or "").strip()
                try:
                    price = int(o.get("price"))
                except Exception:
                    continue
                code = team_name_to_code(name, lg)
                if code:
                    prices[code] = price
            if home not in prices or away not in prices:
                continue
            try:
                p_home_raw = american_to_implied_prob(prices[home])
                p_away_raw = american_to_implied_prob(prices[away])
            except Exception:
                continue
            p_home, _ = devig_two_way(p_home_raw, p_away_raw)
            if 0.0 <= p_home <= 1.0:
                book_probs.append(p_home)

        if not book_probs:
            continue
        p_home = sum(book_probs) / len(book_probs)
        p_home = max(0.0001, min(0.9999, float(p_home)))
        out.append(
            H2HConsensus(
                matchup=f"{away}@{home}",
                commence_time_utc=commence,
                p_home=p_home,
                p_away=1.0 - p_home,
                books_used=len(book_probs),
            )
        )
    return out

