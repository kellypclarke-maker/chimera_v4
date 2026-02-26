from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

from chimera.clients.http import get_json


_SPORT_BY_LEAGUE = {"nba": "basketball", "nhl": "hockey", "nfl": "football"}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def summary_url(*, league: str) -> str:
    lg = str(league or "").strip().lower()
    sport = _SPORT_BY_LEAGUE.get(lg)
    if not sport:
        raise ValueError(f"unsupported league: {league}")
    return f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{lg}/summary"


def _to_prob01(x: object) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v > 1.0 + 1e-9:
        v = v / 100.0
    if v < 0.0 or v > 1.0:
        return None
    return float(v)


@dataclass(frozen=True)
class EspnLiveProb:
    event_id: str
    home_win_prob: Optional[float]
    away_win_prob: Optional[float]
    source: str  # espn_winprob | espn_predictor_fallback
    fetched_at_utc: str


def fetch_live_prob(
    session: requests.Session,
    *,
    league: str,
    event_id: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> EspnLiveProb:
    """
    Fetch ESPN summary payload and extract the best-available HOME win probability.

    - In-play: uses `winprobability` if present (takes the last snapshot).
    - Pregame fallback: uses `predictor` if present.
    """
    lg = str(league or "").strip().lower()
    eid = str(event_id or "").strip()
    if not eid:
        raise ValueError("missing event_id")

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "espn" / "summary" / lg / f"{eid}.json"
        if cache_path.exists() and not force:
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cached = None
            if isinstance(cached, dict):
                payload = cached
            else:
                payload = None
        else:
            payload = None
    else:
        payload = None

    if payload is None:
        payload = get_json(session, summary_url(league=lg), params={"event": eid}, timeout_s=20.0)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    home_p: Optional[float] = None
    away_p: Optional[float] = None
    source = ""

    wp = payload.get("winprobability")
    if isinstance(wp, list) and wp:
        last = wp[-1] if isinstance(wp[-1], dict) else {}
        home_p = _to_prob01(last.get("homeWinPercentage"))
        away_p = _to_prob01(last.get("awayWinPercentage"))
        if home_p is None and away_p is not None:
            home_p = 1.0 - float(away_p)
        if away_p is None and home_p is not None:
            away_p = 1.0 - float(home_p)
        source = "espn_winprob"

    if home_p is None:
        pred = payload.get("predictor") if isinstance(payload.get("predictor"), dict) else {}
        home_team = pred.get("homeTeam") if isinstance(pred.get("homeTeam"), dict) else {}
        # ESPN sometimes provides `teamChanceLoss` in percent.
        loss = _to_prob01(home_team.get("teamChanceLoss"))
        if loss is not None:
            home_p = 1.0 - float(loss)
            away_p = 1.0 - float(home_p)
            source = "espn_predictor_fallback"

    return EspnLiveProb(
        event_id=eid,
        home_win_prob=home_p,
        away_win_prob=away_p,
        source=source or "unknown",
        fetched_at_utc=_utc_now_iso(),
    )


