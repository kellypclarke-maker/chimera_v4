from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from chimera.clients.http import get_json
from chimera.teams import normalize_team_code


_SPORT_BY_LEAGUE = {"nba": "basketball", "nhl": "hockey", "nfl": "football"}


def _date_token(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


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


@dataclass(frozen=True)
class EspnGame:
    league: str
    event_id: str
    start_time_utc: dt.datetime
    matchup: str  # AWAY@HOME (canonical codes)
    away: str
    home: str
    state: str  # pre|in|post (lower)
    home_score: Optional[float]
    away_score: Optional[float]
    home_win: Optional[float]  # 1.0 home win, 0.0 away win, 0.5 tie/unknown


def scoreboard_url(*, league: str) -> str:
    lg = str(league or "").strip().lower()
    sport = _SPORT_BY_LEAGUE.get(lg)
    if not sport:
        raise ValueError(f"unsupported league: {league}")
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/scoreboard"


def fetch_scoreboard(
    *,
    league: str,
    day: dt.date,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    session: Optional[requests.Session] = None,
) -> Dict[str, Any]:
    """
    Fetch ESPN scoreboard for a given day.

    If `cache_dir` is provided, caches the JSON to:
      <cache_dir>/espn/scoreboard/<league>/<YYYYMMDD>.json
    """
    lg = str(league or "").strip().lower()
    if lg not in _SPORT_BY_LEAGUE:
        raise ValueError(f"unsupported league: {league}")

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "espn" / "scoreboard" / lg / f"{_date_token(day)}.json"
        if cache_path.exists() and not force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if isinstance(cached, dict):
                return cached

    s = session or requests.Session()
    url = scoreboard_url(league=lg)
    data = get_json(s, url, params={"dates": _date_token(day)}, timeout_s=20.0)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data


def _extract_teams(comp: Dict[str, Any], league: str) -> Optional[Tuple[str, str]]:
    competitors = comp.get("competitors") or []
    if not isinstance(competitors, list) or len(competitors) < 2:
        return None

    home = next((c for c in competitors if (c.get("homeAway") or "").lower() == "home"), None)
    away = next((c for c in competitors if (c.get("homeAway") or "").lower() == "away"), None)
    if not isinstance(home, dict) or not isinstance(away, dict):
        return None

    def _abbr(entry: Dict[str, Any]) -> str:
        team = entry.get("team") if isinstance(entry.get("team"), dict) else {}
        return str(team.get("abbreviation") or team.get("shortDisplayName") or "").upper()

    away_code = normalize_team_code(_abbr(away), league)
    home_code = normalize_team_code(_abbr(home), league)
    if not away_code or not home_code or away_code == home_code:
        return None
    return away_code, home_code


def extract_games(*, league: str, scoreboard: Dict[str, Any]) -> List[EspnGame]:
    lg = str(league or "").strip().lower()
    out: List[EspnGame] = []
    events = scoreboard.get("events") or []
    if not isinstance(events, list):
        return out
    for ev in events:
        if not isinstance(ev, dict):
            continue
        event_id = str(ev.get("id") or "").strip()
        competitions = ev.get("competitions") or []
        if not isinstance(competitions, list) or not competitions:
            continue
        comp = competitions[0] if isinstance(competitions[0], dict) else {}
        start_iso = ev.get("date") or comp.get("date") or ""
        start_dt = _parse_utc_iso(str(start_iso))
        if start_dt is None:
            continue
        teams = _extract_teams(comp, lg)
        if teams is None:
            continue
        away, home = teams

        status = comp.get("status") if isinstance(comp.get("status"), dict) else {}
        state = (status.get("type") or {}).get("state") if isinstance((status.get("type") or {}), dict) else None
        state_norm = str(state or "").strip().lower() or "unknown"

        # Scores
        away_score: Optional[float] = None
        home_score: Optional[float] = None
        try:
            competitors = comp.get("competitors") or []
            home_entry = next((c for c in competitors if (c.get("homeAway") or "").lower() == "home"), None) or {}
            away_entry = next((c for c in competitors if (c.get("homeAway") or "").lower() == "away"), None) or {}
            home_score = float(home_entry.get("score")) if str(home_entry.get("score") or "").strip() else None
            away_score = float(away_entry.get("score")) if str(away_entry.get("score") or "").strip() else None
        except Exception:
            away_score, home_score = None, None

        home_win: Optional[float] = None
        if state_norm == "post" and home_score is not None and away_score is not None:
            if home_score > away_score:
                home_win = 1.0
            elif home_score < away_score:
                home_win = 0.0
            else:
                home_win = 0.5

        out.append(
            EspnGame(
                league=lg,
                event_id=event_id,
                start_time_utc=start_dt,
                matchup=f"{away}@{home}",
                away=away,
                home=home,
                state=state_norm,
                home_score=home_score,
                away_score=away_score,
                home_win=home_win,
            )
        )
    return out


