from __future__ import annotations

import asyncio
import csv
import datetime as dt
import math
import os
import re
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests

from chimera.clients.espn import EspnGame, extract_games, fetch_scoreboard
from chimera.clients.espn_live import fetch_live_prob
from chimera.clients.kalshi import fetch_market, resolve_game_markets, resolve_matchup_event
from chimera.clients.kalshi_ws import KalshiWsClient
from chimera.clients.moneypuck import fetch_live_win_prob, fetch_pregame_prediction, fetch_schedule, find_game_id, season_string_for_date
from chimera.clients.odds_api import (
    american_to_implied_prob,
    consensus_h2h_probs,
    devig_two_way,
    fetch_event_odds,
    fetch_odds,
    team_name_to_code,
)
from chimera.fees import expected_value_yes, maker_fee_dollars, taker_fee_dollars
from chimera.model.ensemble import EnsembleModel, load_model
from chimera.teams import normalize_team_code
from chimera.trading.ledger import LedgerWriter


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_int(x: object) -> Optional[int]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def _safe_float(x: object) -> Optional[float]:
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(str(x).strip())
    except Exception:
        return None


def _ask_minus_1_limit_cents(*, ask: Optional[int]) -> Optional[int]:
    if ask is None:
        return None
    a = int(ask)
    if a <= 1:
        return None
    px = a - 1
    if px < 1 or px > 99:
        return None
    return int(px)


def _ask_price_cents(*, ask: Optional[int]) -> Optional[int]:
    if ask is None:
        return None
    a = int(ask)
    if a < 1 or a > 99:
        return None
    return int(a)


def _desired_maker_limit_cents(*, bid: Optional[int], ask: Optional[int], improve_cents: int = 1) -> Optional[int]:
    if ask is None:
        return None
    ask = int(ask)
    if ask <= 1:
        return None
    bid = None if bid is None else int(bid)
    if bid is None or bid < 1:
        px = ask - 1
    else:
        desired = bid + int(max(0, improve_cents))
        px = (ask - 1) if desired >= ask else desired
    if px < 1:
        px = 1
    if px >= ask:
        px = ask - 1
    if px < 1 or px > 99:
        return None
    return int(px)


def _edge_capped_limit_cents(
    *,
    p_true: float,
    edge_threshold: float,
    bid: Optional[int],
    ask: Optional[int],
    improve_cents: int,
) -> Optional[int]:
    if ask is None:
        return None
    ask = int(ask)
    if ask <= 1:
        return None
    cap = int(math.floor((float(p_true) - float(edge_threshold)) * 100.0 + 1e-9))
    if cap < 1:
        return None
    quote = _desired_maker_limit_cents(bid=bid, ask=ask, improve_cents=improve_cents)
    if quote is None:
        return None
    limit = int(min(int(quote), int(cap), int(ask) - 1))
    if limit < 1 or limit > 99:
        return None
    return int(limit)


_TOTAL_SUFFIX_RE = re.compile(r"^(?P<strike>\d{1,4})$")
_SPREAD_SUFFIX_RE = re.compile(r"^(?P<team>[A-Z]{2,4})(?P<strike>\d{1,3})$")


@dataclass(frozen=True)
class SpreadMarket:
    ticker: str
    team: str
    matchup_side: str  # home|away
    strike_int: int  # Kalshi encoded integer (X => wins by > X.5)
    threshold: float  # X.5


@dataclass(frozen=True)
class TotalMarket:
    ticker: str
    strike_int: int  # Kalshi encoded integer (X => total > X.5)
    threshold: float  # X.5


@dataclass
class OddsAltCache:
    fetched_ts: float
    totals_p_over_by_strike: Dict[int, float]
    totals_books_by_strike: Dict[int, int]
    spreads_p_cover_by_team_strike: Dict[Tuple[str, int], float]
    spreads_books_by_team_strike: Dict[Tuple[str, int], int]


@dataclass
class H2HCache:
    fetched_ts: float
    primary_by_matchup: Dict[str, object]
    sharp_by_matchup: Dict[str, object]


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


def _is_half_point(x: float) -> bool:
    # True for values like 220.5, 3.5, 19.5 (within tolerance).
    xf = float(x)
    base = math.floor(xf)
    return abs(xf - (float(base) + 0.5)) <= 1e-6


def _totals_probs_from_event(payload: Dict[str, object], league: str) -> Tuple[Dict[int, float], Dict[int, int]]:
    by_strike: Dict[int, List[float]] = {}
    for bm in payload.get("bookmakers") or []:
        if not isinstance(bm, dict):
            continue
        markets = bm.get("markets") or []
        mkt = next((m for m in markets if isinstance(m, dict) and str(m.get("key") or "").strip().lower() == "alternate_totals"), None)
        if not isinstance(mkt, dict):
            continue
        outcomes = mkt.get("outcomes") or []
        if not isinstance(outcomes, list):
            continue
        by_point: Dict[float, Dict[str, float]] = {}
        for o in outcomes:
            if not isinstance(o, dict):
                continue
            nm = str(o.get("name") or "").strip().lower()
            if nm not in {"over", "under"}:
                continue
            try:
                point = float(o.get("point"))
                price = int(o.get("price"))
            except Exception:
                continue
            if not _is_half_point(point):
                continue
            try:
                p_raw = american_to_implied_prob(int(price))
            except Exception:
                continue
            by_point.setdefault(point, {})[nm] = float(p_raw)
        for point, probs in by_point.items():
            if "over" not in probs or "under" not in probs:
                continue
            p_over, _ = devig_two_way(float(probs["over"]), float(probs["under"]))
            strike_int = int(math.floor(float(point)))
            if abs(float(point) - (float(strike_int) + 0.5)) > 1e-6:
                continue
            by_strike.setdefault(int(strike_int), []).append(float(p_over))
    p_out: Dict[int, float] = {}
    books_out: Dict[int, int] = {}
    for k, vals in by_strike.items():
        if not vals:
            continue
        p_out[int(k)] = float(sum(vals) / len(vals))
        books_out[int(k)] = int(len(vals))
    return p_out, books_out


def _spreads_probs_from_event(payload: Dict[str, object], league: str) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
    """
    Convert Odds API `alternate_spreads` into p(team wins by > X.5).

    Odds API often includes 4 outcomes at the same abs point:
      - TeamA +X.5, TeamA -X.5, TeamB +X.5, TeamB -X.5
    We must pair the correct 2-way market:
      p(TeamA wins by > X.5) uses (TeamA -X.5) vs (TeamB +X.5)
      p(TeamB wins by > X.5) uses (TeamB -X.5) vs (TeamA +X.5)
    """
    by_team_strike: Dict[Tuple[str, int], List[float]] = {}
    lg = str(league or "").strip().lower()
    for bm in payload.get("bookmakers") or []:
        if not isinstance(bm, dict):
            continue
        markets = bm.get("markets") or []
        mkt = next((m for m in markets if isinstance(m, dict) and str(m.get("key") or "").strip().lower() == "alternate_spreads"), None)
        if not isinstance(mkt, dict):
            continue
        outcomes = mkt.get("outcomes") or []
        if not isinstance(outcomes, list):
            continue

        # abs_point -> team_code -> {'pos': p_raw, 'neg': p_raw}
        by_abs_point: Dict[float, Dict[str, Dict[str, float]]] = {}
        for o in outcomes:
            if not isinstance(o, dict):
                continue
            name = str(o.get("name") or "").strip()
            code = team_name_to_code(name, lg) or normalize_team_code(name, lg) or ""
            if not code:
                continue
            try:
                point = float(o.get("point"))
                price = int(o.get("price"))
            except Exception:
                continue
            abs_point = abs(float(point))
            if not _is_half_point(abs_point):
                continue
            try:
                p_raw = american_to_implied_prob(int(price))
            except Exception:
                continue
            side_key = "neg" if float(point) < 0 else "pos"
            by_abs_point.setdefault(float(abs_point), {}).setdefault(str(code), {})[side_key] = float(p_raw)

        for abs_point, teams in by_abs_point.items():
            if not _is_half_point(abs_point):
                continue
            if not isinstance(teams, dict) or len(teams) != 2:
                continue
            team_codes = list(teams.keys())
            a, b = team_codes[0], team_codes[1]
            a_map = teams.get(a, {}) if isinstance(teams.get(a), dict) else {}
            b_map = teams.get(b, {}) if isinstance(teams.get(b), dict) else {}

            # Map abs_point to Kalshi strike integer.
            strike_int = int(math.floor(float(abs_point)))
            if abs(float(abs_point) - (float(strike_int) + 0.5)) > 1e-6:
                continue

            # p(A wins by > abs_point): A neg vs B pos
            if "neg" in a_map and "pos" in b_map:
                p_a, _ = devig_two_way(float(a_map["neg"]), float(b_map["pos"]))
                by_team_strike.setdefault((str(a), int(strike_int)), []).append(float(p_a))

            # p(B wins by > abs_point): B neg vs A pos
            if "neg" in b_map and "pos" in a_map:
                p_b, _ = devig_two_way(float(b_map["neg"]), float(a_map["pos"]))
                by_team_strike.setdefault((str(b), int(strike_int)), []).append(float(p_b))

    p_out: Dict[Tuple[str, int], float] = {}
    books_out: Dict[Tuple[str, int], int] = {}
    for k, vals in by_team_strike.items():
        if not vals:
            continue
        p_out[(str(k[0]), int(k[1]))] = float(sum(vals) / len(vals))
        books_out[(str(k[0]), int(k[1]))] = int(len(vals))
    return p_out, books_out


@dataclass(frozen=True)
class ShadowRow:
    snapshot_ts_utc: str
    league: str
    date: str
    event_id: str
    matchup: str
    state: str
    home_score: str
    away_score: str
    market_type: str
    strike: str
    kalshi_side: str
    p_home: str
    p_home_source: str
    p_yes: str
    p_yes_source: str
    p_yes_books_used: str
    side: str
    team: str
    kalshi_ticker: str
    yes_bid_cents: str
    yes_ask_cents: str
    no_bid_cents: str
    no_ask_cents: str
    maker_limit_cents: str
    edge_at_limit: str
    ev_at_limit: str
    edge_at_ask: str
    ev_at_ask: str
    kalshi_quote_age_seconds: str
    would_trade: str
    best_for_matchup: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "snapshot_ts_utc": self.snapshot_ts_utc,
            "league": self.league,
            "date": self.date,
            "event_id": self.event_id,
            "matchup": self.matchup,
            "state": self.state,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "market_type": self.market_type,
            "strike": self.strike,
            "kalshi_side": self.kalshi_side,
            "p_home": self.p_home,
            "p_home_source": self.p_home_source,
            "p_yes": self.p_yes,
            "p_yes_source": self.p_yes_source,
            "p_yes_books_used": self.p_yes_books_used,
            "side": self.side,
            "team": self.team,
            "kalshi_ticker": self.kalshi_ticker,
            "yes_bid_cents": self.yes_bid_cents,
            "yes_ask_cents": self.yes_ask_cents,
            "no_bid_cents": self.no_bid_cents,
            "no_ask_cents": self.no_ask_cents,
            "maker_limit_cents": self.maker_limit_cents,
            "edge_at_limit": self.edge_at_limit,
            "ev_at_limit": self.ev_at_limit,
            "edge_at_ask": self.edge_at_ask,
            "ev_at_ask": self.ev_at_ask,
            "kalshi_quote_age_seconds": self.kalshi_quote_age_seconds,
            "would_trade": self.would_trade,
            "best_for_matchup": self.best_for_matchup,
        }


def _append_csv(path: Path, rows: List[ShadowRow]) -> Path:
    if not rows:
        return path
    desired_fields = list(rows[0].to_dict().keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    if path.exists():
        try:
            first = path.open("r", encoding="utf-8", errors="ignore").readline().strip()
            existing = [c.strip() for c in first.split(",")] if first else []
        except Exception:
            existing = []
        if existing and existing != desired_fields:
            tag = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            path = path.with_name(f"{path.stem}_{tag}.csv")
            write_header = True
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=desired_fields)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
    return path


_SHADOW_CSV_FIELDS: List[str] = [
    "snapshot_ts_utc",
    "league",
    "date",
    "event_id",
    "matchup",
    "state",
    "home_score",
    "away_score",
    "market_type",
    "strike",
    "kalshi_side",
    "p_home",
    "p_home_source",
    "p_yes",
    "p_yes_source",
    "p_yes_books_used",
    "side",
    "team",
    "kalshi_ticker",
    "yes_bid_cents",
    "yes_ask_cents",
    "no_bid_cents",
    "no_ask_cents",
    "maker_limit_cents",
    "edge_at_limit",
    "ev_at_limit",
    "edge_at_ask",
    "ev_at_ask",
    "kalshi_quote_age_seconds",
    "would_trade",
    "best_for_matchup",
]


def _ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SHADOW_CSV_FIELDS)
        w.writeheader()


async def run_shadow(
    *,
    league: str,
    date_iso: str,
    edge_threshold: float = 0.05,
    improve_cents: int = 1,
    refresh_seconds: int = 15,
    max_runtime_seconds: int = 0,
    out_path: Optional[Path] = None,
    ledger_path: Optional[Path] = None,
) -> Path:
    """
    Shadow-mode logger:
      - No trading.
      - Logs win probabilities + Kalshi quotes + hypothetical maker/taker entries.
      - NBA/NFL anchor: Pinnacle + primary US book via Odds API.
      - NHL anchor: MoneyPuck pregame/live win prob.
      - Subscribes to Kalshi WS ticker updates and logs hypothetical entries.
    """
    lg = str(league).strip().lower()
    day = dt.date.fromisoformat(str(date_iso))
    execution_mode = str(os.environ.get("SHADOW_EXECUTION_MODE", "maker")).strip().lower()
    if execution_mode not in {"maker", "taker"}:
        execution_mode = "maker"
    fee_mode_default = "taker" if execution_mode == "taker" else "maker"
    fee_mode = str(os.environ.get("SHADOW_FEE_MODE", fee_mode_default)).strip().lower() or fee_mode_default

    out = out_path or (Path("reports/shadow") / f"{day.isoformat()}_{lg}_shadow.csv")
    _ensure_csv_header(out)
    s = requests.Session()
    ledger = LedgerWriter.create(cmd="shadow", league=lg, date_iso=str(date_iso), path=ledger_path)
    ledger.write(
        {
            "event": "run_start",
            "cmd": "shadow",
            "league": lg,
            "date": str(date_iso),
            "edge_threshold": float(edge_threshold),
            "improve_cents": int(improve_cents),
            "refresh_seconds": int(refresh_seconds),
            "max_runtime_seconds": int(max_runtime_seconds),
            "execution_mode": execution_mode,
            "fee_mode": fee_mode,
            "out_csv": str(out),
        }
    )

    # Track tickers for all games on the slate (pre + in). We'll log state each snapshot.
    sb0 = fetch_scoreboard(league=lg, day=day, cache_dir=Path("data/cache"), session=s)
    games0 = extract_games(league=lg, scoreboard=sb0)
    if not games0:
        ledger.write({"event": "info", "cmd": "shadow", "msg": f"no_games_for_slate: {lg} {day.isoformat()}"})
        return out

    mt_env = str(os.environ.get("SHADOW_MARKET_TYPES", "ml,spread,total")).strip().lower()
    mt_tokens = {t.strip().lower() for t in mt_env.split(",") if t.strip()}
    if not mt_tokens:
        mt_tokens = {"ml", "spread", "total"}
    mt_norm: set[str] = set()
    for t in mt_tokens:
        if t in {"ml", "moneyline", "game"}:
            mt_norm.add("ml")
        elif t in {"spread", "spreads"}:
            mt_norm.add("spread")
        elif t in {"total", "totals"}:
            mt_norm.add("total")
    if not mt_norm:
        mt_norm = {"ml", "spread", "total"}
    include_ml = "ml" in mt_norm
    include_spread = "spread" in mt_norm
    include_total = "total" in mt_norm

    # Resolve Kalshi tickers.
    kalshi_by_event: Dict[str, Dict[str, str]] = {}
    spreads_by_event: Dict[str, List[SpreadMarket]] = {}
    totals_by_event: Dict[str, List[TotalMarket]] = {}
    for g in games0:
        resolved = resolve_game_markets(league=lg, day=day, away=g.away, home=g.home, session=s)
        if resolved is None:
            for alt in (day - dt.timedelta(days=1), day + dt.timedelta(days=1)):
                resolved = resolve_game_markets(league=lg, day=alt, away=g.away, home=g.home, session=s)
                if resolved is not None:
                    break
        if resolved is None:
            continue
        kalshi_by_event[str(g.event_id)] = {
            "home": str(resolved.market_ticker_home_yes).strip().upper(),
            "away": str(resolved.market_ticker_away_yes).strip().upper(),
        }

        if not (include_spread or include_total):
            continue

        # Resolve spread/total events (if available).
        def _resolve_event(mt: str) -> Optional[ResolvedEvent]:
            r = resolve_matchup_event(league=lg, market_type=mt, day=day, away=g.away, home=g.home, session=s)
            if r is not None:
                return r
            for alt_day in (day - dt.timedelta(days=1), day + dt.timedelta(days=1)):
                r2 = resolve_matchup_event(league=lg, market_type=mt, day=alt_day, away=g.away, home=g.home, session=s)
                if r2 is not None:
                    return r2
            return None

        if include_spread:
            spread_ev = _resolve_event("spread")
            if spread_ev is not None:
                parsed: List[SpreadMarket] = []
                for m in spread_ev.markets:
                    ticker = str(m.get("ticker") or "").strip().upper()
                    if not ticker:
                        continue
                    suffix = ticker.split("-")[-1].strip().upper()
                    ms = _SPREAD_SUFFIX_RE.match(suffix)
                    if not ms:
                        continue
                    team_tok = str(ms.group("team")).upper()
                    strike_s = str(ms.group("strike"))
                    try:
                        strike_int = int(strike_s)
                    except Exception:
                        continue
                    team = normalize_team_code(team_tok, lg) or team_tok
                    matchup_side = "home" if team == g.home else ("away" if team == g.away else "")
                    if matchup_side not in {"home", "away"}:
                        continue
                    parsed.append(
                        SpreadMarket(
                            ticker=ticker,
                            team=str(team),
                            matchup_side=str(matchup_side),
                            strike_int=int(strike_int),
                            threshold=float(strike_int) + 0.5,
                        )
                    )
                if parsed:
                    spreads_by_event[str(g.event_id)] = parsed

        if include_total:
            total_ev = _resolve_event("total")
            if total_ev is not None:
                parsed_t: List[TotalMarket] = []
                for m in total_ev.markets:
                    ticker = str(m.get("ticker") or "").strip().upper()
                    if not ticker:
                        continue
                    suffix = ticker.split("-")[-1].strip().upper()
                    mt = _TOTAL_SUFFIX_RE.match(suffix)
                    if not mt:
                        continue
                    try:
                        strike_int = int(str(mt.group("strike")))
                    except Exception:
                        continue
                    parsed_t.append(
                        TotalMarket(
                            ticker=ticker,
                            strike_int=int(strike_int),
                            threshold=float(strike_int) + 0.5,
                        )
                    )
                if parsed_t:
                    totals_by_event[str(g.event_id)] = parsed_t

    tickers: List[str] = sorted(
        ({t for m in kalshi_by_event.values() for t in m.values() if t} if include_ml else set())
        | ({sm.ticker for sms in spreads_by_event.values() for sm in sms if sm.ticker} if include_spread else set())
        | ({tm.ticker for tms in totals_by_event.values() for tm in tms if tm.ticker} if include_total else set())
    )
    if not tickers:
        raise RuntimeError("no Kalshi tickers resolved for this slate")

    # Optional NHL MoneyPuck mapping.
    mp_schedule = None
    mp_game_id_by_event: Dict[str, int] = {}
    if lg == "nhl":
        season = season_string_for_date(day)
        mp_schedule = fetch_schedule(season=season, cache_dir=Path("data/cache"))
        for g in games0:
            gid = find_game_id(schedule=mp_schedule, away=g.away, home=g.home, start_time_utc=g.start_time_utc)
            if gid is not None:
                mp_game_id_by_event[str(g.event_id)] = int(gid)

    quotes: Dict[str, Dict[str, object]] = {t: {"yes_bid": None, "yes_ask": None, "no_bid": None, "no_ask": None, "last_update_ts": None} for t in tickers}
    quotes_event = asyncio.Event()

    # Map Odds API event_id per ESPN event_id (used for spread/total consensus).
    odds_event_id_by_event: Dict[str, str] = {}
    if include_spread or include_total:
        try:
            odds_events = fetch_odds(
                league=lg,
                markets=("h2h",),
                regions="us",
                cache_dir=Path("data/cache"),
                session=s,
            )
            by_matchup: Dict[str, List[Tuple[dt.datetime, str]]] = {}
            for ev in odds_events:
                if not isinstance(ev, dict):
                    continue
                eid = str(ev.get("id") or "").strip()
                if not eid:
                    continue
                home_name = str(ev.get("home_team") or "").strip()
                away_name = str(ev.get("away_team") or "").strip()
                home = team_name_to_code(home_name, lg)
                away = team_name_to_code(away_name, lg)
                if not home or not away:
                    continue
                commence = _parse_utc_iso(str(ev.get("commence_time") or ""))
                if commence is None:
                    continue
                matchup = f"{away}@{home}"
                by_matchup.setdefault(matchup, []).append((commence, eid))
            for g in games0:
                cands = by_matchup.get(g.matchup) or []
                if not cands:
                    continue
                # pick commence_time closest to ESPN start
                best = min(cands, key=lambda t: abs((t[0] - g.start_time_utc).total_seconds()))
                odds_event_id_by_event[str(g.event_id)] = str(best[1])
        except Exception as exc:
            ledger.write({"event": "warn", "cmd": "shadow", "msg": f"odds_api_mapping_failed: {exc}"})

    alt_cache: Dict[str, OddsAltCache] = {}
    h2h_cache: Optional[H2HCache] = None
    primary_books = tuple([b.strip().lower() for b in str(os.environ.get("SHADOW_PRIMARY_BOOKS", "draftkings")).split(",") if b.strip()])
    if not primary_books:
        primary_books = ("draftkings",)
    require_agreement = str(os.environ.get("SHADOW_REQUIRE_AGREE", "1")).strip().lower() not in {"0", "false", "no"}
    # ESPN live win-prob is convenient but has historically been a weak "edge" anchor.
    # Default off; enable explicitly with SHADOW_ALLOW_ESPN_LIVE=1 if desired.
    allow_espn_live = str(os.environ.get("SHADOW_ALLOW_ESPN_LIVE", "0")).strip().lower() not in {"0", "false", "no"}

    nhl_pre_prob_source = str(os.environ.get("SHADOW_NHL_PRE_PROB_SOURCE", "ensemble")).strip().lower() or "ensemble"
    nhl_live_prob_source = str(os.environ.get("SHADOW_NHL_LIVE_PROB_SOURCE", "moneypuck")).strip().lower() or "moneypuck"
    nhl_ensemble_model_path = Path(str(os.environ.get("SHADOW_NHL_ENSEMBLE_MODEL", "data/models/nhl_ensemble.json"))).expanduser()
    nhl_ensemble_model: Optional[EnsembleModel] = None
    if lg == "nhl" and nhl_pre_prob_source == "ensemble" and nhl_ensemble_model_path.exists():
        try:
            nhl_ensemble_model = load_model(nhl_ensemble_model_path)
            ledger.write({"event": "info", "cmd": "shadow", "msg": f"loaded_nhl_ensemble_model: {nhl_ensemble_model_path}"})
        except Exception as exc:
            ledger.write({"event": "warn", "cmd": "shadow", "msg": f"failed_to_load_nhl_ensemble_model: {exc}"})
            nhl_ensemble_model = None

    pre_filter_enable = str(os.environ.get("SHADOW_PRE_FILTER_ENABLE", "0")).strip().lower() not in {"0", "false", "no"}
    pre_skip_limit_ge_cents = _safe_int(os.environ.get("SHADOW_PRE_SKIP_LIMIT_GE_CENTS"))
    pre_skip_longshot_lt_cents = _safe_int(os.environ.get("SHADOW_PRE_SKIP_LONGSHOT_LT_CENTS"))
    pre_skip_longshot_edge_lo = _safe_float(os.environ.get("SHADOW_PRE_SKIP_LONGSHOT_EDGE_LO"))
    pre_skip_longshot_edge_hi = _safe_float(os.environ.get("SHADOW_PRE_SKIP_LONGSHOT_EDGE_HI"))
    pre_require_edge_at_ask = str(os.environ.get("SHADOW_PRE_REQUIRE_EDGE_AT_ASK", "0")).strip().lower() not in {"0", "false", "no"}
    pre_edge_at_ask_min = _safe_float(os.environ.get("SHADOW_PRE_EDGE_AT_ASK_MIN"))
    if pre_edge_at_ask_min is None:
        pre_edge_at_ask_min = 0.0
    maker_limit_mode = str(os.environ.get("SHADOW_MAKER_LIMIT_MODE", "edge_capped")).strip().lower() or "edge_capped"
    maker_fee_rate = _safe_float(os.environ.get("KALSHI_MAKER_FEE_RATE")) or 0.0
    taker_fee_rate = _safe_float(os.environ.get("KALSHI_TAKER_FEE_RATE")) or 0.07

    def _fee_for_price(*, price: float) -> float:
        mode = str(fee_mode).strip().lower()
        if mode in {"none", "0", "off"}:
            return 0.0
        if mode in {"taker"}:
            return taker_fee_dollars(contracts=1, price=float(price), rate=float(taker_fee_rate))
        return maker_fee_dollars(contracts=1, price=float(price), rate=float(maker_fee_rate))

    def _get_h2h_consensus(*, ttl_seconds: float) -> Optional[H2HCache]:
        nonlocal h2h_cache
        now = time.time()
        if h2h_cache is not None and (now - float(h2h_cache.fetched_ts)) < float(ttl_seconds):
            return h2h_cache
        try:
            odds_events_primary = fetch_odds(
                league=lg,
                markets=("h2h",),
                regions="us",
                bookmakers=primary_books,
                cache_dir=Path("data/cache"),
                force=True,
                session=s,
            )
            odds_events_sharp = fetch_odds(
                league=lg,
                markets=("h2h",),
                regions="eu",
                bookmakers=("pinnacle",),
                cache_dir=Path("data/cache"),
                force=True,
                session=s,
            )
            primary_map = {c.matchup: c for c in consensus_h2h_probs(odds_events_primary, league=lg)}
            sharp_map = {c.matchup: c for c in consensus_h2h_probs(odds_events_sharp, league=lg)}
            h2h_cache = H2HCache(fetched_ts=float(now), primary_by_matchup=primary_map, sharp_by_matchup=sharp_map)
        except Exception:
            return h2h_cache
        return h2h_cache

    async def _ws_listener() -> None:
        ws = KalshiWsClient(use_private_auth=True)
        try:
            await ws.connect()
        except Exception as exc:
            print(f"[warn] WS connect failed; falling back to REST polling only: {exc}")
            return
        try:
            await ws.subscribe(channels=["ticker"], market_tickers=tickers, send_initial_snapshot=True)
            while True:
                msg = await ws.recv_json()
                t = str(msg.get("type") or "").strip().lower()
                if t != "ticker":
                    continue
                payload = msg.get("msg") if isinstance(msg.get("msg"), dict) else {}
                mt = str(payload.get("market_ticker") or "").strip().upper()
                if mt not in quotes:
                    continue
                bid = _safe_int(payload.get("yes_bid"))
                ask = _safe_int(payload.get("yes_ask"))
                nbid = _safe_int(payload.get("no_bid"))
                nask = _safe_int(payload.get("no_ask"))
                quotes[mt] = {"yes_bid": bid, "yes_ask": ask, "no_bid": nbid, "no_ask": nask, "last_update_ts": time.time()}
                quotes_event.set()
        finally:
            await ws.close()

    ws_task = asyncio.create_task(_ws_listener())

    def _seed_quotes_rest() -> None:
        for t in tickers:
            if quotes[t]["yes_ask"] is not None:
                continue
            m = fetch_market(ticker=t, session=s)
            quotes[t]["yes_bid"] = _safe_int(m.get("yes_bid"))
            quotes[t]["yes_ask"] = _safe_int(m.get("yes_ask"))
            quotes[t]["no_bid"] = _safe_int(m.get("no_bid"))
            quotes[t]["no_ask"] = _safe_int(m.get("no_ask"))
            quotes[t]["last_update_ts"] = time.time()

    t0 = time.time()
    last_signal_key_by_matchup: Dict[str, str] = {}
    persist_state_by_matchup: Dict[str, Tuple[str, int]] = {}
    try:
        await asyncio.sleep(0.75)
        seed_rest = str(os.environ.get("SHADOW_SEED_QUOTES_REST", "0")).strip().lower() not in {"0", "false", "no"}
        if seed_rest:
            _seed_quotes_rest()

        live_require_stale = str(os.environ.get("SHADOW_LIVE_REQUIRE_STALE", "0")).strip().lower() not in {"0", "false", "no"}
        live_stale_seconds = float(os.environ.get("SHADOW_LIVE_STALE_SECONDS", "15") or 15)
        live_persist_required = max(1, int(float(os.environ.get("SHADOW_LIVE_PERSIST_COUNT", "1") or 1)))
        live_odds_ttl_seconds = float(os.environ.get("SHADOW_LIVE_ODDS_TTL_SECONDS", "5") or 5)
        pre_odds_ttl_seconds = float(os.environ.get("SHADOW_PRE_ODDS_TTL_SECONDS", "60") or 60)

        while True:
            if int(max_runtime_seconds) > 0 and (time.time() - t0) >= float(max_runtime_seconds):
                return out

            sb = fetch_scoreboard(league=lg, day=day, cache_dir=Path("data/cache"), force=True, session=s)
            games = extract_games(league=lg, scoreboard=sb)
            by_event: Dict[str, EspnGame] = {str(g.event_id): g for g in games}

            rows: List[ShadowRow] = []
            snap_ts = _utc_now_iso()
            snap_now_ts = time.time()
            state_env = str(os.environ.get("SHADOW_STATES", "pre,in"))
            active_states = {s.strip().lower() for s in state_env.split(",") if s.strip()}
            if not active_states:
                active_states = {"pre", "in"}

            def _get_alt_probs(*, odds_event_id: str, ttl_seconds: float) -> Optional[OddsAltCache]:
                oid = str(odds_event_id or "").strip()
                if not oid:
                    return None
                now = time.time()
                cached = alt_cache.get(oid)
                if cached is not None and (now - float(cached.fetched_ts)) < float(ttl_seconds):
                    return cached
                try:
                    payload = fetch_event_odds(
                        league=lg,
                        event_id=oid,
                        markets=("alternate_totals", "alternate_spreads"),
                        regions="us",
                        cache_dir=Path("data/cache"),
                        force=True,
                        session=s,
                    )
                    totals_p, totals_books = _totals_probs_from_event(payload, lg)
                    spreads_p, spreads_books = _spreads_probs_from_event(payload, lg)
                    entry = OddsAltCache(
                        fetched_ts=float(now),
                        totals_p_over_by_strike=totals_p,
                        totals_books_by_strike=totals_books,
                        spreads_p_cover_by_team_strike=spreads_p,
                        spreads_books_by_team_strike=spreads_books,
                    )
                    alt_cache[oid] = entry
                    return entry
                except Exception:
                    return cached

            for eid, tick_map in kalshi_by_event.items():
                g = by_event.get(eid)
                if g is None:
                    continue
                if str(g.state).strip().lower() not in active_states:
                    continue

                p_home: Optional[float] = None
                p_books_home: Optional[float] = None
                p_sharp_home: Optional[float] = None
                books_used_primary: Optional[int] = None
                books_used_sharp: Optional[int] = None
                p_home_source = ""
                is_live_prob = False
                if lg in {"nba", "nfl"}:
                    if str(g.state).strip().lower() == "in" and allow_espn_live:
                        try:
                            lp = fetch_live_prob(s, league=lg, event_id=str(eid), cache_dir=Path("data/cache"), force=True)
                            if lp.home_win_prob is not None:
                                p_home = float(lp.home_win_prob)
                                p_home_source = str(lp.source or "espn_live").strip() or "espn_live"
                                is_live_prob = True
                        except Exception:
                            pass
                    if p_home is None:
                        ttl = float(live_odds_ttl_seconds) if str(g.state).strip().lower() == "in" else float(pre_odds_ttl_seconds)
                        h2h = _get_h2h_consensus(ttl_seconds=float(ttl))
                        if h2h is not None:
                            c_primary = h2h.primary_by_matchup.get(g.matchup) if isinstance(h2h.primary_by_matchup, dict) else None
                            c_sharp = h2h.sharp_by_matchup.get(g.matchup) if isinstance(h2h.sharp_by_matchup, dict) else None
                            if c_primary is not None and c_sharp is not None:
                                p_books_home = float(getattr(c_primary, "p_home", 0.0))
                                try:
                                    books_used_primary = int(getattr(c_primary, "books_used", 0))
                                except Exception:
                                    books_used_primary = None
                                p_sharp_home = float(getattr(c_sharp, "p_home", 0.0))
                                try:
                                    books_used_sharp = int(getattr(c_sharp, "books_used", 0))
                                except Exception:
                                    books_used_sharp = None
                                p_home = (float(p_books_home) + float(p_sharp_home)) / 2.0
                                p_home_source = "pin_books_agree" if require_agreement else "pin_books_avg"
                            elif c_primary is not None:
                                p_books_home = float(getattr(c_primary, "p_home", 0.0))
                                try:
                                    books_used_primary = int(getattr(c_primary, "books_used", 0))
                                except Exception:
                                    books_used_primary = None
                                p_home = float(p_books_home)
                                p_home_source = "primary_books"
                            elif c_sharp is not None:
                                p_sharp_home = float(getattr(c_sharp, "p_home", 0.0))
                                try:
                                    books_used_sharp = int(getattr(c_sharp, "books_used", 0))
                                except Exception:
                                    books_used_sharp = None
                                p_home = float(p_sharp_home)
                                p_home_source = "pinnacle"
                elif lg == "nhl":
                    gid = mp_game_id_by_event.get(eid)
                    ttl = float(live_odds_ttl_seconds) if str(g.state).strip().lower() == "in" else float(pre_odds_ttl_seconds)
                    h2h = _get_h2h_consensus(ttl_seconds=float(ttl))
                    if h2h is not None:
                        c_primary = h2h.primary_by_matchup.get(g.matchup) if isinstance(h2h.primary_by_matchup, dict) else None
                        c_sharp = h2h.sharp_by_matchup.get(g.matchup) if isinstance(h2h.sharp_by_matchup, dict) else None
                        if c_primary is not None:
                            p_books_home = float(getattr(c_primary, "p_home", 0.0))
                            try:
                                books_used_primary = int(getattr(c_primary, "books_used", 0))
                            except Exception:
                                books_used_primary = None
                        if c_sharp is not None:
                            p_sharp_home = float(getattr(c_sharp, "p_home", 0.0))
                            try:
                                books_used_sharp = int(getattr(c_sharp, "books_used", 0))
                            except Exception:
                                books_used_sharp = None

                    if gid is not None:
                        try:
                            if g.state == "in":
                                if nhl_live_prob_source in {"books", "primary", "primary_books"} and p_books_home is not None:
                                    p_home = float(p_books_home)
                                    p_home_source = "primary_books"
                                elif nhl_live_prob_source in {"pinnacle", "sharp"} and p_sharp_home is not None:
                                    p_home = float(p_sharp_home)
                                    p_home_source = "pinnacle"
                                else:
                                    lp2 = fetch_live_win_prob(game_id=int(gid), cache_dir=Path("data/cache"), force=True)
                                    p_home = lp2.home_win_prob
                                    p_home_source = lp2.source
                                    is_live_prob = True
                            else:
                                pre = fetch_pregame_prediction(game_id=int(gid), cache_dir=Path("data/cache"))
                                p_mp = pre.moneypuck_home_win
                                if nhl_pre_prob_source == "ensemble" and nhl_ensemble_model is not None and p_books_home is not None and p_mp is not None:
                                    raw = float(nhl_ensemble_model.predict_p_home(p_books=float(p_books_home), p_moneypuck=float(p_mp)))
                                    lo, hi = sorted([float(p_books_home), float(p_mp)])
                                    p_home = max(float(lo), min(float(hi), float(raw)))
                                    p_home_source = "ensemble"
                                elif nhl_pre_prob_source in {"books", "primary", "primary_books"} and p_books_home is not None:
                                    p_home = float(p_books_home)
                                    p_home_source = "primary_books"
                                elif nhl_pre_prob_source in {"pinnacle", "sharp"} and p_sharp_home is not None:
                                    p_home = float(p_sharp_home)
                                    p_home_source = "pinnacle"
                                else:
                                    p_home = p_mp
                                    p_home_source = "moneypuck_pregame"
                        except Exception:
                            p_home, p_home_source = None, ""

                if p_home is None:
                    continue
                p_away = 1.0 - float(p_home)

                side_rows: List[Tuple[Optional[float], ShadowRow]] = []
                # Moneyline (YES only on each team's market).
                for side in (("home", "away") if include_ml else ()):
                    ticker = str(tick_map.get(side) or "").strip().upper()
                    if not ticker:
                        continue
                    q = quotes.get(ticker) or {}
                    yes_bid = q.get("yes_bid")
                    yes_ask = q.get("yes_ask")
                    no_bid = q.get("no_bid")
                    no_ask = q.get("no_ask")
                    last_upd = q.get("last_update_ts")
                    quote_age_s: Optional[int] = None
                    try:
                        if last_upd is not None:
                            quote_age_s = max(0, int(float(snap_now_ts) - float(last_upd)))
                    except Exception:
                        quote_age_s = None
                    if yes_ask is None or no_ask is None:
                        m = fetch_market(ticker=ticker, session=s)
                        yes_bid = _safe_int(m.get("yes_bid"))
                        yes_ask = _safe_int(m.get("yes_ask"))
                        no_bid = _safe_int(m.get("no_bid"))
                        no_ask = _safe_int(m.get("no_ask"))
                        quotes[ticker] = {"yes_bid": yes_bid, "yes_ask": yes_ask, "no_bid": no_bid, "no_ask": no_ask, "last_update_ts": time.time()}
                        quote_age_s = 0

                    team = g.home if side == "home" else g.away
                    p_yes_event = float(p_home) if side == "home" else float(p_away)

                    if execution_mode == "taker":
                        limit = _ask_price_cents(ask=yes_ask)
                    else:
                        if maker_limit_mode in {"ask_minus_1", "ask-1"}:
                            limit = _ask_minus_1_limit_cents(ask=yes_ask)
                        elif maker_limit_mode in {"bid_plus_improve", "bid+improve", "desired"}:
                            limit = _desired_maker_limit_cents(bid=yes_bid, ask=yes_ask, improve_cents=int(improve_cents))
                        else:
                            limit = _edge_capped_limit_cents(
                                p_true=p_yes_event,
                                edge_threshold=float(edge_threshold),
                                bid=yes_bid,
                                ask=yes_ask,
                                improve_cents=int(improve_cents),
                            )
                    edge = None if limit is None else (p_yes_event - float(limit) / 100.0)
                    ev = None
                    if limit is not None:
                        try:
                            px = float(limit) / 100.0
                            ev = expected_value_yes(p_true=float(p_yes_event), price=float(px), fee=_fee_for_price(price=float(px)))
                        except Exception:
                            ev = None
                    edge_at_ask = None
                    ev_at_ask = None
                    if yes_ask is not None:
                        try:
                            ask_px = float(yes_ask) / 100.0
                            edge_at_ask = float(p_yes_event) - float(ask_px)
                            ev_at_ask = expected_value_yes(p_true=float(p_yes_event), price=float(ask_px), fee=_fee_for_price(price=float(ask_px)))
                        except Exception:
                            edge_at_ask, ev_at_ask = None, None
                    agree_ok = True
                    if require_agreement and lg in {"nba", "nfl"} and not bool(is_live_prob):
                        if p_books_home is None or p_sharp_home is None or limit is None:
                            agree_ok = False
                        else:
                            limit_price = float(limit) / 100.0
                            if side == "home":
                                edge_books = float(p_books_home) - float(limit_price)
                                edge_sharp = float(p_sharp_home) - float(limit_price)
                            else:
                                edge_books = (1.0 - float(p_books_home)) - float(limit_price)
                                edge_sharp = (1.0 - float(p_sharp_home)) - float(limit_price)
                            agree_ok = bool(edge_books >= float(edge_threshold) and edge_sharp >= float(edge_threshold))
                    would = bool(agree_ok and limit is not None and edge is not None and edge >= float(edge_threshold) and (ev or 0.0) > 0.0)
                    # Optional pre-game filter (NBA-focused): skip heavy favorites and a low-confidence longshot edge band.
                    # This is applied to the tradability decision AND to best-per-matchup selection via rank_ev below.
                    if str(g.state).strip().lower() == "pre" and bool(would) and bool(pre_filter_enable):
                        if pre_require_edge_at_ask:
                            if edge_at_ask is None or float(edge_at_ask) <= float(pre_edge_at_ask_min):
                                would = False
                        if pre_skip_limit_ge_cents is not None and limit is not None and int(limit) >= int(pre_skip_limit_ge_cents):
                            would = False
                        if (
                            bool(would)
                            and pre_skip_longshot_lt_cents is not None
                            and limit is not None
                            and int(limit) < int(pre_skip_longshot_lt_cents)
                            and pre_skip_longshot_edge_lo is not None
                            and pre_skip_longshot_edge_hi is not None
                            and edge is not None
                            and float(pre_skip_longshot_edge_lo) <= float(edge) < float(pre_skip_longshot_edge_hi)
                        ):
                            would = False
                    # Optional live-only gating: require Kalshi quote staleness and/or persistence across multiple snapshots.
                    if str(g.state).strip().lower() == "in" and bool(would):
                        if live_require_stale:
                            if quote_age_s is None or float(quote_age_s) < float(live_stale_seconds):
                                would = False

                        key0 = f"{ticker}|yes|{'' if limit is None else str(int(limit))}"
                        prev_key, prev_count = persist_state_by_matchup.get(g.matchup, ("", 0))
                        if prev_key == key0:
                            prev_count += 1
                        else:
                            prev_key, prev_count = key0, 1
                        persist_state_by_matchup[g.matchup] = (prev_key, prev_count)
                        if prev_count < int(live_persist_required):
                                    would = False

                    books_used_for_row: Optional[int] = None
                    if p_home_source in {"primary_books", "pin_books_agree", "pin_books_avg", "ensemble"}:
                        books_used_for_row = books_used_primary
                    elif p_home_source in {"pinnacle"}:
                        books_used_for_row = books_used_sharp

                    side_rows.append(
                        (
                            None if (not bool(would) or ev is None) else float(ev),
                            ShadowRow(
                                snapshot_ts_utc=snap_ts,
                                league=lg,
                                date=day.isoformat(),
                                event_id=eid,
                                matchup=g.matchup,
                                state=str(g.state),
                                home_score="" if g.home_score is None else str(g.home_score),
                                away_score="" if g.away_score is None else str(g.away_score),
                                market_type="ml",
                                strike="",
                                kalshi_side="yes",
                                p_home=f"{float(p_home):.6f}",
                                p_home_source=str(p_home_source or ""),
                                p_yes=f"{float(p_yes_event):.6f}",
                                p_yes_source=str(p_home_source or ""),
                                p_yes_books_used="" if books_used_for_row is None else str(int(books_used_for_row)),
                                side=side,
                                team=str(team),
                                kalshi_ticker=ticker,
                                yes_bid_cents="" if yes_bid is None else str(int(yes_bid)),
                                yes_ask_cents="" if yes_ask is None else str(int(yes_ask)),
                                no_bid_cents="" if no_bid is None else str(int(no_bid)),
                                no_ask_cents="" if no_ask is None else str(int(no_ask)),
                                maker_limit_cents="" if limit is None else str(int(limit)),
                                edge_at_limit="" if edge is None else f"{float(edge):.6f}",
                                ev_at_limit="" if ev is None else f"{float(ev):.6f}",
                                edge_at_ask="" if edge_at_ask is None else f"{float(edge_at_ask):.6f}",
                                ev_at_ask="" if ev_at_ask is None else f"{float(ev_at_ask):.6f}",
                                kalshi_quote_age_seconds="" if quote_age_s is None else str(int(quote_age_s)),
                                would_trade=str(bool(would)),
                                best_for_matchup="False",
                            ),
                        )
                    )

                # Spread/total: pick best EV strike+side per market type.
                odds_eid = odds_event_id_by_event.get(eid, "")
                ttl = 60.0 if str(g.state).strip().lower() == "in" else 300.0
                alt = None
                if (include_spread or include_total) and odds_eid:
                    alt = _get_alt_probs(odds_event_id=str(odds_eid), ttl_seconds=float(ttl))

                def _best_trade_for_ticker(
                    *,
                    p_yes_event: float,
                    yes_bid: Optional[int],
                    yes_ask: Optional[int],
                    no_bid: Optional[int],
                    no_ask: Optional[int],
                ) -> Tuple[Optional[str], Optional[int], Optional[float], Optional[float], Optional[bool]]:
                    best_side: Optional[str] = None
                    best_limit: Optional[int] = None
                    best_edge: Optional[float] = None
                    best_ev: Optional[float] = None
                    best_would: Optional[bool] = None

                    # YES
                    if execution_mode == "taker":
                        lim_y = _ask_price_cents(ask=yes_ask)
                    else:
                        lim_y = _edge_capped_limit_cents(
                            p_true=float(p_yes_event),
                            edge_threshold=float(edge_threshold),
                            bid=yes_bid,
                            ask=yes_ask,
                            improve_cents=int(improve_cents),
                        )
                    if lim_y is not None:
                        price = float(lim_y) / 100.0
                        edge_y = float(p_yes_event) - price
                        ev_y = expected_value_yes(p_true=float(p_yes_event), price=float(price), fee=_fee_for_price(price=float(price)))
                        would_y = bool(edge_y >= float(edge_threshold) and ev_y > 0.0)
                        best_side, best_limit, best_edge, best_ev, best_would = "yes", int(lim_y), float(edge_y), float(ev_y), bool(would_y)

                    # NO
                    p_no = 1.0 - float(p_yes_event)
                    if execution_mode == "taker":
                        lim_n = _ask_price_cents(ask=no_ask)
                    else:
                        lim_n = _edge_capped_limit_cents(
                            p_true=float(p_no),
                            edge_threshold=float(edge_threshold),
                            bid=no_bid,
                            ask=no_ask,
                            improve_cents=int(improve_cents),
                        )
                    if lim_n is not None:
                        price = float(lim_n) / 100.0
                        edge_n = float(p_no) - price
                        ev_n = expected_value_yes(p_true=float(p_no), price=float(price), fee=_fee_for_price(price=float(price)))
                        would_n = bool(edge_n >= float(edge_threshold) and ev_n > 0.0)
                        if best_ev is None or float(ev_n) > float(best_ev):
                            best_side, best_limit, best_edge, best_ev, best_would = "no", int(lim_n), float(edge_n), float(ev_n), bool(would_n)

                    return best_side, best_limit, best_edge, best_ev, best_would

                # Spread best
                spread_markets = spreads_by_event.get(eid) or []
                if include_spread and alt is not None and spread_markets:
                    best_spread: Optional[Tuple[float, ShadowRow]] = None
                    for sm in spread_markets:
                        p_yes_ev = alt.spreads_p_cover_by_team_strike.get((sm.team, int(sm.strike_int)))
                        if p_yes_ev is None:
                            continue
                        q = quotes.get(sm.ticker) or {}
                        yb, ya = q.get("yes_bid"), q.get("yes_ask")
                        nb, na = q.get("no_bid"), q.get("no_ask")
                        last_upd = q.get("last_update_ts")
                        quote_age_s: Optional[int] = None
                        try:
                            if last_upd is not None:
                                quote_age_s = max(0, int(float(snap_now_ts) - float(last_upd)))
                        except Exception:
                            quote_age_s = None
                        if ya is None or na is None:
                            m = fetch_market(ticker=sm.ticker, session=s)
                            yb = _safe_int(m.get("yes_bid"))
                            ya = _safe_int(m.get("yes_ask"))
                            nb = _safe_int(m.get("no_bid"))
                            na = _safe_int(m.get("no_ask"))
                            quotes[sm.ticker] = {"yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na, "last_update_ts": time.time()}
                            quote_age_s = 0

                        kside, limit, edge, ev, would = _best_trade_for_ticker(
                            p_yes_event=float(p_yes_ev),
                            yes_bid=yb,
                            yes_ask=ya,
                            no_bid=nb,
                            no_ask=na,
                        )
                        if kside is None or limit is None or ev is None:
                            continue
                        sr = ShadowRow(
                            snapshot_ts_utc=snap_ts,
                            league=lg,
                            date=day.isoformat(),
                            event_id=eid,
                            matchup=g.matchup,
                            state=str(g.state),
                            home_score="" if g.home_score is None else str(g.home_score),
                            away_score="" if g.away_score is None else str(g.away_score),
                            market_type="spread",
                            strike=f"{float(sm.threshold):.1f}",
                            kalshi_side=str(kside),
                            p_home=f"{float(p_home):.6f}",
                            p_home_source=str(p_home_source or ""),
                            p_yes=f"{float(p_yes_ev):.6f}",
                            p_yes_source="odds_api_alternate_spreads",
                            p_yes_books_used=str(int(alt.spreads_books_by_team_strike.get((sm.team, int(sm.strike_int)), 0) or 0)),
                            side=str(sm.matchup_side),
                            team=str(sm.team),
                            kalshi_ticker=str(sm.ticker),
                            yes_bid_cents="" if yb is None else str(int(yb)),
                            yes_ask_cents="" if ya is None else str(int(ya)),
                            no_bid_cents="" if nb is None else str(int(nb)),
                            no_ask_cents="" if na is None else str(int(na)),
                            maker_limit_cents=str(int(limit)),
                            edge_at_limit="" if edge is None else f"{float(edge):.6f}",
                            ev_at_limit="" if ev is None else f"{float(ev):.6f}",
                            edge_at_ask="",
                            ev_at_ask="",
                            kalshi_quote_age_seconds="" if quote_age_s is None else str(int(quote_age_s)),
                            would_trade=str(bool(would)),
                            best_for_matchup="False",
                        )
                        candidate = (float(ev), sr)
                        if best_spread is None or float(candidate[0]) > float(best_spread[0]):
                            best_spread = candidate
                    if best_spread is not None:
                        side_rows.append((float(best_spread[0]), best_spread[1]))

                # Total best
                total_markets = totals_by_event.get(eid) or []
                if include_total and alt is not None and total_markets:
                    best_total: Optional[Tuple[float, ShadowRow]] = None
                    for tm in total_markets:
                        p_over = alt.totals_p_over_by_strike.get(int(tm.strike_int))
                        if p_over is None:
                            continue
                        q = quotes.get(tm.ticker) or {}
                        yb, ya = q.get("yes_bid"), q.get("yes_ask")
                        nb, na = q.get("no_bid"), q.get("no_ask")
                        last_upd = q.get("last_update_ts")
                        quote_age_s: Optional[int] = None
                        try:
                            if last_upd is not None:
                                quote_age_s = max(0, int(float(snap_now_ts) - float(last_upd)))
                        except Exception:
                            quote_age_s = None
                        if ya is None or na is None:
                            m = fetch_market(ticker=tm.ticker, session=s)
                            yb = _safe_int(m.get("yes_bid"))
                            ya = _safe_int(m.get("yes_ask"))
                            nb = _safe_int(m.get("no_bid"))
                            na = _safe_int(m.get("no_ask"))
                            quotes[tm.ticker] = {"yes_bid": yb, "yes_ask": ya, "no_bid": nb, "no_ask": na, "last_update_ts": time.time()}
                            quote_age_s = 0

                        kside, limit, edge, ev, would = _best_trade_for_ticker(
                            p_yes_event=float(p_over),
                            yes_bid=yb,
                            yes_ask=ya,
                            no_bid=nb,
                            no_ask=na,
                        )
                        if kside is None or limit is None or ev is None:
                            continue
                        sr = ShadowRow(
                            snapshot_ts_utc=snap_ts,
                            league=lg,
                            date=day.isoformat(),
                            event_id=eid,
                            matchup=g.matchup,
                            state=str(g.state),
                            home_score="" if g.home_score is None else str(g.home_score),
                            away_score="" if g.away_score is None else str(g.away_score),
                            market_type="total",
                            strike=f"{float(tm.threshold):.1f}",
                            kalshi_side=str(kside),
                            p_home=f"{float(p_home):.6f}",
                            p_home_source=str(p_home_source or ""),
                            p_yes=f"{float(p_over):.6f}",
                            p_yes_source="odds_api_alternate_totals",
                            p_yes_books_used=str(int(alt.totals_books_by_strike.get(int(tm.strike_int), 0) or 0)),
                            side="",
                            team="TOTAL",
                            kalshi_ticker=str(tm.ticker),
                            yes_bid_cents="" if yb is None else str(int(yb)),
                            yes_ask_cents="" if ya is None else str(int(ya)),
                            no_bid_cents="" if nb is None else str(int(nb)),
                            no_ask_cents="" if na is None else str(int(na)),
                            maker_limit_cents=str(int(limit)),
                            edge_at_limit="" if edge is None else f"{float(edge):.6f}",
                            ev_at_limit="" if ev is None else f"{float(ev):.6f}",
                            edge_at_ask="",
                            ev_at_ask="",
                            kalshi_quote_age_seconds="" if quote_age_s is None else str(int(quote_age_s)),
                            would_trade=str(bool(would)),
                            best_for_matchup="False",
                        )
                        candidate = (float(ev), sr)
                        if best_total is None or float(candidate[0]) > float(best_total[0]):
                            best_total = candidate
                    if best_total is not None:
                        side_rows.append((float(best_total[0]), best_total[1]))

                # Mark best per matchup (max EV at the proposed execution price).
                best_idx: Optional[int] = None
                best_ev = float("-inf")
                for i, (ev, _) in enumerate(side_rows):
                    if ev is None:
                        continue
                    if float(ev) > float(best_ev):
                        best_ev = float(ev)
                        best_idx = int(i)
                if best_idx is not None:
                    ev, r0 = side_rows[best_idx]
                    side_rows[best_idx] = (ev, replace(r0, best_for_matchup="True"))

                for _, sr in side_rows:
                    rows.append(sr)
                    if str(sr.best_for_matchup).strip().lower() in {"true", "1", "yes", "y"} and str(sr.would_trade).strip().lower() in {
                        "true",
                        "1",
                        "yes",
                        "y",
                    }:
                        key = f"{sr.market_type}|{sr.kalshi_ticker}|{sr.kalshi_side}|{sr.maker_limit_cents}"
                        prev = last_signal_key_by_matchup.get(sr.matchup)
                        if prev != key:
                            last_signal_key_by_matchup[sr.matchup] = key
                            ledger.write(
                                {
                                    "event": "shadow_signal",
                                    "cmd": "shadow",
                                    "league": lg,
                                    "category": ("NBA" if lg == "nba" else "NHL" if lg == "nhl" else "WEATHER"),
                                    "date": day.isoformat(),
                                    "snapshot_ts_utc": snap_ts,
                                    "matchup": sr.matchup,
                                    "state": sr.state,
                                    "market_type": sr.market_type,
                                    "strike": sr.strike,
                                    "kalshi_side": sr.kalshi_side,
                                    "team": sr.team,
                                    "ticker": sr.kalshi_ticker,
                                    "p_yes": sr.p_yes,
                                    "maker_limit_cents": sr.maker_limit_cents,
                                    "yes_bid_cents": sr.yes_bid_cents,
                                    "yes_ask_cents": sr.yes_ask_cents,
                                    "no_bid_cents": sr.no_bid_cents,
                                    "no_ask_cents": sr.no_ask_cents,
                                    "edge_at_limit": sr.edge_at_limit,
                                    "ev_at_limit": sr.ev_at_limit,
                                }
                            )

            out = _append_csv(out, rows)

            # Stop conditions:
            # - If slate is over (no games remain in pre/in), stop for all modes.
            # - Otherwise, only stop early if this shadow run includes "pre" and there are no more "pre" games.
            slate_active = any(str(g.state).strip().lower() in {"pre", "in"} for g in by_event.values())
            if not slate_active:
                return out
            if "pre" in active_states and not any(str(g.state).strip().lower() in active_states for g in by_event.values()):
                return out

            try:
                quotes_event.clear()
                await asyncio.wait_for(quotes_event.wait(), timeout=float(refresh_seconds))
            except asyncio.TimeoutError:
                pass
    finally:
        ws_task.cancel()
        try:
            await ws_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
