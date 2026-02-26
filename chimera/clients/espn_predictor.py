from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from chimera.clients.http import get_json


_SPORT_BY_LEAGUE = {"nba": "basketball", "nfl": "football"}
_BASE = "http://sports.core.api.espn.com/v2"
_UA = "project-chimera/espn-predictor/1.0"


def _competition_url(*, league: str, event_id: str) -> str:
    lg = str(league or "").strip().lower()
    sport = _SPORT_BY_LEAGUE.get(lg)
    if not sport:
        raise ValueError(f"ESPN predictor unsupported league: {league}")
    eid = str(event_id).strip()
    if not eid:
        raise ValueError("event_id required")
    return f"{_BASE}/sports/{sport}/leagues/{lg}/events/{eid}/competitions/{eid}"


def _stat_value(team_payload: Dict[str, Any], *, name: str) -> Optional[float]:
    stats = team_payload.get("statistics") or []
    if not isinstance(stats, list):
        return None
    for s in stats:
        if not isinstance(s, dict):
            continue
        if str(s.get("name") or "").strip() != name:
            continue
        try:
            return float(s.get("value"))
        except Exception:
            return None
    return None


@dataclass(frozen=True)
class EspnPredictor:
    league: str
    event_id: str
    p_home: Optional[float]
    last_modified: Optional[str]
    source_url: str


def fetch_predictor_for_event(
    session: requests.Session,
    *,
    league: str,
    event_id: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> EspnPredictor:
    """
    Fetch ESPN's pregame "Matchup Predictor" and return home win probability.

    ESPN does not provide a reliable historical time series at exact T-minus anchors;
    we include `lastModified` for provenance.
    """
    lg = str(league or "").strip().lower()
    eid = str(event_id or "").strip()
    if lg not in _SPORT_BY_LEAGUE:
        return EspnPredictor(league=lg, event_id=eid, p_home=None, last_modified=None, source_url="")

    cache_path: Optional[Path] = None
    if cache_dir is not None and eid:
        cache_path = Path(cache_dir) / "espn" / "predictor" / lg / f"{eid}.json"
        if cache_path.exists() and not force:
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                raw = None
            if isinstance(raw, dict):
                return _parse_predictor_json(lg=lg, eid=eid, pred=raw, source_url=str(raw.get("$ref") or ""))

    comp_url = _competition_url(league=lg, event_id=eid)
    comp = get_json(
        session,
        comp_url,
        params={"lang": "en", "region": "us"},
        headers={"User-Agent": _UA},
        timeout_s=20.0,
    )
    predictor_ref = comp.get("predictor") if isinstance(comp.get("predictor"), dict) else {}
    ref_url = str(predictor_ref.get("$ref") or "").strip()
    if not ref_url:
        return EspnPredictor(league=lg, event_id=eid, p_home=None, last_modified=None, source_url="")

    pred = get_json(
        session,
        ref_url,
        params=None,
        headers={"User-Agent": _UA},
        timeout_s=20.0,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(pred, indent=2, sort_keys=True), encoding="utf-8")

    return _parse_predictor_json(lg=lg, eid=eid, pred=pred, source_url=ref_url)


def _parse_predictor_json(*, lg: str, eid: str, pred: Dict[str, Any], source_url: str) -> EspnPredictor:
    last_modified = str(pred.get("lastModified") or "").strip() or None
    home_team = pred.get("homeTeam") if isinstance(pred.get("homeTeam"), dict) else {}
    away_team = pred.get("awayTeam") if isinstance(pred.get("awayTeam"), dict) else {}

    p_home: Optional[float] = None

    gp_home = _stat_value(home_team, name="gameProjection")
    if gp_home is not None:
        p_home = gp_home / 100.0
    else:
        # Some NBA predictor payloads only include statistics on one side.
        # If away team has 'teamChanceLoss', that's effectively home win prob (assuming no ties).
        away_loss = _stat_value(away_team, name="teamChanceLoss")
        if away_loss is not None:
            p_home = away_loss / 100.0
        else:
            gp_away = _stat_value(away_team, name="gameProjection")
            if gp_away is not None:
                p_home = 1.0 - (gp_away / 100.0)

    if p_home is not None and not (0.0 <= float(p_home) <= 1.0):
        p_home = None

    return EspnPredictor(
        league=str(lg),
        event_id=str(eid),
        p_home=None if p_home is None else float(p_home),
        last_modified=last_modified,
        source_url=str(source_url or ""),
    )


