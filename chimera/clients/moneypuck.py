from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

from chimera.teams import normalize_team_code


TEAM_METRICS_BASE = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
CURRENT_INJURIES_URL = "https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv"
SCHEDULE_BASE = "https://moneypuck.com/moneypuck/OldSeasonScheduleJson"
PREDICTIONS_BASE = "https://moneypuck.com/moneypuck/predictions"
GAMEDATA_BASE = "https://moneypuck.com/moneypuck/gameData"
_DEFAULT_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _season_start_year(d: dt.date) -> int:
    # NHL season generally begins in early Oct; treat Jul+ as the new season-year.
    return d.year if d.month >= 7 else d.year - 1


def season_string_for_date(d: dt.date) -> str:
    y0 = _season_start_year(d)
    return f"{y0}{y0 + 1}"


def _download_text(url: str, *, timeout_s: float = 30.0) -> str:
    # MoneyPuck commonly blocks requests without a browser-like User-Agent.
    ua = str(os.environ.get("MONEYPUCK_USER_AGENT") or _DEFAULT_UA).strip() or _DEFAULT_UA
    r = requests.get(url, timeout=float(timeout_s), headers={"User-Agent": ua})
    r.raise_for_status()
    return str(r.text or "")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def fetch_team_and_goalie_metrics(
    *,
    season_start_year: int,
    game_type: str = "regular",
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch MoneyPuck teams.csv + goalies.csv for a season and return:
      (team_metrics_df, goalie_metrics_df)
    """
    season = int(season_start_year)
    game_type = str(game_type).strip() or "regular"

    teams_url = f"{TEAM_METRICS_BASE}/{season}/{game_type}/teams.csv"
    goalies_url = f"{TEAM_METRICS_BASE}/{season}/{game_type}/goalies.csv"

    def _cached_csv(url: str, name: str) -> pd.DataFrame:
        if cache_dir is None:
            return pd.read_csv(StringIO(_download_text(url)))
        path = Path(cache_dir) / "moneypuck" / str(season) / game_type / name
        if path.exists() and not force:
            return pd.read_csv(path)
        text = _download_text(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return pd.read_csv(StringIO(text))

    teams_df = _cached_csv(teams_url, "teams.csv")
    goalies_df = _cached_csv(goalies_url, "goalies.csv")

    # Normalize team code column if present.
    if "team" in teams_df.columns:
        teams_df["team"] = teams_df["team"].astype(str).str.strip().str.upper().map(lambda x: normalize_team_code(x, "nhl") or x)
    if "team" in goalies_df.columns:
        goalies_df["team"] = goalies_df["team"].astype(str).str.strip().str.upper().map(lambda x: normalize_team_code(x, "nhl") or x)

    return teams_df, goalies_df


def build_simple_team_metrics(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MoneyPuck teams.csv into a compact, modeling-friendly table.
    """
    rows: List[Dict[str, object]] = []
    for _, row in teams_df.iterrows():
        team = str(row.get("team") or "").strip().upper()
        if not team:
            continue
        team = normalize_team_code(team, "nhl") or team
        shots = row.get("shotsOnGoalFor")
        goals = row.get("goalsFor")
        oi_sh_pct: Optional[float] = None
        try:
            if shots is not None and float(shots) > 0:
                oi_sh_pct = float(goals) / float(shots)
        except Exception:
            oi_sh_pct = None
        rows.append(
            {
                "team": team,
                "xgf_for": row.get("xGoalsFor"),
                "xgf_against": row.get("xGoalsAgainst"),
                "hdcf_for": row.get("highDangerShotsFor"),
                "hdca": row.get("highDangerShotsAgainst"),
                "oi_sh_pct": oi_sh_pct,
            }
        )
    return pd.DataFrame(rows)


def build_goalie_gsax60(goalies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a team-level goalie_gsax60 from MoneyPuck goalies.csv.
    """
    team_vals: Dict[str, List[float]] = {}
    for _, row in goalies_df.iterrows():
        team = str(row.get("team") or "").strip().upper()
        if not team:
            continue
        team = normalize_team_code(team, "nhl") or team
        try:
            xg = float(row.get("xGoals"))
            goals = float(row.get("goals"))
            icetime = float(row.get("icetime"))
        except Exception:
            continue
        if icetime <= 0:
            continue
        gsax60 = (xg - goals) / icetime * 3600.0
        team_vals.setdefault(team, []).append(gsax60)
    out_rows = [{"team": team, "goalie_gsax60": sum(vals) / len(vals)} for team, vals in team_vals.items() if vals]
    return pd.DataFrame(out_rows)


@dataclass(frozen=True)
class InjurySnapshot:
    fetched_at_utc: str
    sha256: str
    rows: List[Dict[str, str]]


def fetch_current_injuries(
    *,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> InjurySnapshot:
    """
    Fetch MoneyPuck's public NHL injuries CSV and return normalized rows.

    Source: https://moneypuck.com/moneypuck/playerData/playerNews/current_injuries.csv
    """
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "moneypuck" / "injuries" / "current_injuries.csv"
        if cache_path.exists() and not force:
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
        else:
            text = _download_text(CURRENT_INJURIES_URL)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
    else:
        text = _download_text(CURRENT_INJURIES_URL)

    reader = csv.DictReader(StringIO(text))
    rows: List[Dict[str, str]] = []
    for r in reader:
        if not isinstance(r, dict):
            continue
        team_raw = str(r.get("team") or "").strip().upper()
        team = normalize_team_code(team_raw, "nhl") or team_raw
        row = {str(k): str(v or "").strip() for k, v in r.items()}
        row["team"] = team
        rows.append(row)

    fetched_at_utc = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return InjurySnapshot(
        fetched_at_utc=fetched_at_utc,
        sha256=_sha256(text),
        rows=rows,
    )


def write_injuries_snapshot(snapshot: InjurySnapshot, *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "fetched_at_utc": snapshot.fetched_at_utc,
                "sha256": snapshot.sha256,
                "rows": snapshot.rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return out_path


@dataclass(frozen=True)
class ScheduledGame:
    game_id: int
    away: str
    home: str
    est: str  # "YYYYMMDD HH:MM:SS" local (EST/EDT)

    @property
    def date_iso(self) -> str:
        raw = str(self.est or "").split(" ", 1)[0]
        if len(raw) == 8 and raw.isdigit():
            return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
        return ""


def fetch_schedule(
    *,
    season: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> List[ScheduledGame]:
    """
    Fetch MoneyPuck season schedule JSON.

    Source: https://moneypuck.com/moneypuck/OldSeasonScheduleJson/SeasonSchedule-<season>.json
    where season looks like '20252026'.
    """
    season_s = str(season).strip()
    if not season_s or not season_s.isdigit() or len(season_s) != 8:
        raise ValueError(f"invalid season string: {season!r} (expected 'YYYYYYYY')")
    url = f"{SCHEDULE_BASE}/SeasonSchedule-{season_s}.json"

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "moneypuck" / "schedule" / f"SeasonSchedule-{season_s}.json"
        if cache_path.exists() and not force:
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                raw = None
            if isinstance(raw, list):
                return _parse_schedule_rows(raw)

    raw_text = _download_text(url, timeout_s=30.0)
    raw = json.loads(raw_text)
    if not isinstance(raw, list):
        raise TypeError("unexpected schedule JSON shape (expected list)")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(raw_text, encoding="utf-8")

    return _parse_schedule_rows(raw)


def _parse_schedule_rows(rows: List[Dict[str, Any]]) -> List[ScheduledGame]:
    out: List[ScheduledGame] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            game_id = int(r.get("id"))
        except Exception:
            continue
        away_raw = str(r.get("a") or "").strip().upper()
        home_raw = str(r.get("h") or "").strip().upper()
        est = str(r.get("est") or "").strip()
        away = normalize_team_code(away_raw, "nhl") or away_raw
        home = normalize_team_code(home_raw, "nhl") or home_raw
        if not away or not home or away == home or not est:
            continue
        out.append(ScheduledGame(game_id=game_id, away=away, home=home, est=est))
    return out


def find_game_id(
    *,
    schedule: Sequence[ScheduledGame],
    away: str,
    home: str,
    start_time_utc: Optional[dt.datetime] = None,
) -> Optional[int]:
    """
    Best-effort mapping from matchup to MoneyPuck `game_id`.
    """
    away_can = normalize_team_code(away, "nhl") or str(away).strip().upper()
    home_can = normalize_team_code(home, "nhl") or str(home).strip().upper()
    if not away_can or not home_can:
        return None

    date_candidates: Optional[set[str]] = None
    if start_time_utc is not None:
        ts = start_time_utc.astimezone(dt.timezone.utc)
        date_candidates = {
            ts.date().isoformat(),
            (ts.date() + dt.timedelta(days=1)).isoformat(),
            (ts.date() - dt.timedelta(days=1)).isoformat(),
        }

    matches: List[ScheduledGame] = []
    for g in schedule:
        if g.away != away_can or g.home != home_can:
            continue
        if date_candidates is not None and g.date_iso and g.date_iso not in date_candidates:
            continue
        matches.append(g)
    if not matches:
        # Fallback without date filtering.
        for g in schedule:
            if g.away == away_can and g.home == home_can:
                matches.append(g)
        if not matches:
            return None
    # Prefer the one closest in date if we have a start_time.
    if start_time_utc is not None:
        target = start_time_utc.astimezone(dt.timezone.utc).date()
        matches.sort(key=lambda g: abs((dt.date.fromisoformat(g.date_iso) - target).days) if g.date_iso else 999)
    return int(matches[0].game_id)


@dataclass(frozen=True)
class PregamePrediction:
    game_id: int
    home: str
    away: str
    moneypuck_home_win: Optional[float]
    betting_odds_home_win: Optional[float]


def fetch_pregame_prediction(
    *,
    game_id: int,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> PregamePrediction:
    """
    Fetch MoneyPuck pregame prediction CSV for a game id.
    """
    gid = int(game_id)
    url = f"{PREDICTIONS_BASE}/{gid}.csv"

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "moneypuck" / "predictions" / f"{gid}.csv"
        if cache_path.exists() and not force:
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
            return _parse_prediction_csv(gid, text)

    text = _download_text(url, timeout_s=30.0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
    return _parse_prediction_csv(gid, text)


def _parse_prediction_csv(game_id: int, text: str) -> PregamePrediction:
    reader = csv.DictReader(StringIO(text))
    row = next(reader, None) or {}
    home_raw = str(row.get("homeTeamCode") or "").strip().upper()
    away_raw = str(row.get("roadTeamCode") or "").strip().upper()
    home = normalize_team_code(home_raw, "nhl") or home_raw
    away = normalize_team_code(away_raw, "nhl") or away_raw

    def _f(k: str) -> Optional[float]:
        try:
            v = row.get(k)
            if v is None or str(v).strip() == "":
                return None
            return float(v)
        except Exception:
            return None

    return PregamePrediction(
        game_id=int(game_id),
        home=home,
        away=away,
        moneypuck_home_win=_f("preGameMoneyPuckHomeWinPrediction"),
        betting_odds_home_win=_f("preGameBettingOddsHomeWinPrediction"),
    )


@dataclass(frozen=True)
class LiveWinProb:
    game_id: int
    home_win_prob: Optional[float]
    away_win_prob: Optional[float]
    fetched_at_utc: str
    source: str = "moneypuck_live"


def _moneypuck_game_data_url(game_id: int) -> str:
    season_start = int(str(int(game_id))[:4])
    season_dir = f"{season_start}{season_start + 1}"
    return f"{GAMEDATA_BASE}/{season_dir}/{int(game_id)}.csv"


def _safe_float(x: object) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return float(v)


def _pick_prob_col(row: dict, *, prefer_home: bool) -> Optional[float]:
    keys = [str(k) for k in row.keys()]
    if prefer_home:
        candidates = [
            "homeWinProbability",
            "homeWinProb",
            "homeWinProbabilityAdj",
            "homeWinProbabilityAdjusted",
            "homeWinProbabilityOT",
        ]
    else:
        candidates = [
            "awayWinProbability",
            "awayWinProb",
            "awayWinProbabilityAdj",
            "awayWinProbabilityAdjusted",
            "awayWinProbabilityOT",
        ]
    for c in candidates:
        if c in row:
            v = _safe_float(row.get(c))
            if v is not None and 0.0 <= v <= 1.0:
                return v
    target = "home" if prefer_home else "away"
    for k in keys:
        lk = k.lower()
        if target in lk and "win" in lk and "prob" in lk:
            v = _safe_float(row.get(k))
            if v is not None and 0.0 <= v <= 1.0:
                return v
    return None


def fetch_live_win_prob(
    *,
    game_id: int,
    cache_dir: Optional[Path] = None,
    force: bool = False,
) -> LiveWinProb:
    """
    Fetch MoneyPuck live win probabilities for an NHL game.
    """
    gid = int(game_id)
    url = _moneypuck_game_data_url(gid)

    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / "moneypuck" / "gamedata" / f"{gid}.csv"
        if cache_path.exists() and not force:
            text = cache_path.read_text(encoding="utf-8", errors="ignore")
        else:
            text = _download_text(url, timeout_s=30.0)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
    else:
        text = _download_text(url, timeout_s=30.0)

    reader = csv.DictReader(StringIO(text))
    last: Optional[dict] = None
    for r in reader:
        if isinstance(r, dict):
            last = r

    home_p = _pick_prob_col(last or {}, prefer_home=True) if last else None
    away_p = _pick_prob_col(last or {}, prefer_home=False) if last else None
    if home_p is None and away_p is not None:
        home_p = 1.0 - float(away_p)
    if away_p is None and home_p is not None:
        away_p = 1.0 - float(home_p)

    fetched_at_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return LiveWinProb(game_id=gid, home_win_prob=home_p, away_win_prob=away_p, fetched_at_utc=fetched_at_utc)

