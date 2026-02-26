from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from chimera.clients.espn import EspnGame, extract_games, fetch_scoreboard
from chimera.clients.kalshi import fetch_market, mid_from_bidask, resolve_game_markets
from chimera.clients.espn_predictor import fetch_predictor_for_event
from chimera.clients.moneypuck import fetch_pregame_prediction, fetch_schedule, find_game_id, season_string_for_date
from chimera.clients.odds_api import H2HConsensus, consensus_h2h_probs, fetch_odds
from chimera.fees import expected_value_yes, maker_fee_dollars, taker_fee_dollars
from chimera.model.ensemble import EnsembleModel, load_model


def _to_iso_z(ts: dt.datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    ts = ts.astimezone(dt.timezone.utc).replace(microsecond=0)
    return ts.isoformat().replace("+00:00", "Z")


def _safe_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def _ask_minus_1_limit_cents(*, yes_ask: Optional[int]) -> Optional[int]:
    if yes_ask is None:
        return None
    ask = int(yes_ask)
    if ask <= 1:
        return None
    px = int(ask) - 1
    if px < 1 or px > 99:
        return None
    return int(px)


def _desired_maker_limit_cents(*, yes_bid: Optional[int], yes_ask: Optional[int], improve_cents: int = 1) -> Optional[int]:
    """
    Best-effort maker limit that joins/improves the bid without crossing.
    """
    if yes_ask is None:
        return None
    ask = int(yes_ask)
    if ask <= 1:
        return None
    bid = None if yes_bid is None else int(yes_bid)
    if bid is None or bid < 1:
        px = ask - 1
    else:
        desired = int(bid) + int(max(0, improve_cents))
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
    p_yes: float,
    edge_threshold: float,
    yes_bid: Optional[int],
    yes_ask: Optional[int],
    improve_cents: int = 1,
) -> Optional[int]:
    """
    Maker limit capped so the strategy still clears `edge_threshold` if filled.
    """
    if yes_ask is None:
        return None
    ask = int(yes_ask)
    if ask <= 1:
        return None
    cap = int((float(p_yes) - float(edge_threshold)) * 100.0)
    if cap < 1:
        return None
    quote = _desired_maker_limit_cents(yes_bid=yes_bid, yes_ask=yes_ask, improve_cents=int(improve_cents))
    if quote is None:
        return None
    limit = int(min(int(quote), int(cap), int(ask) - 1))
    if limit < 1 or limit > 99:
        return None
    return int(limit)


def _merge_consensus(lists: Iterable[List[H2HConsensus]]) -> Dict[str, H2HConsensus]:
    merged: Dict[str, Tuple[float, int, dt.datetime]] = {}
    for cons_list in lists:
        for c in cons_list:
            if not isinstance(c, H2HConsensus):
                continue
            w = max(1, int(c.books_used))
            entry = merged.get(c.matchup)
            if entry is None:
                merged[c.matchup] = (float(c.p_home) * w, w, c.commence_time_utc)
            else:
                p_sum, w_sum, commence = entry
                merged[c.matchup] = (p_sum + float(c.p_home) * w, w_sum + w, commence)
    out: Dict[str, H2HConsensus] = {}
    for matchup, (p_sum, w_sum, commence) in merged.items():
        if w_sum <= 0:
            continue
        p_home = max(0.0001, min(0.9999, float(p_sum) / float(w_sum)))
        out[matchup] = H2HConsensus(
            matchup=matchup,
            commence_time_utc=commence,
            p_home=p_home,
            p_away=1.0 - p_home,
            books_used=w_sum,
        )
    return out


@dataclass(frozen=True)
class EdgeRow:
    date: str
    league: str
    matchup: str
    start_time_utc: str
    event_id: str
    side: str  # home|away (YES on that team)
    team: str
    kalshi_ticker: str
    kalshi_yes_bid_cents: Optional[int]
    kalshi_yes_ask_cents: Optional[int]
    kalshi_yes_limit_cents: Optional[int]
    kalshi_yes_exec_cents: Optional[int]
    kalshi_mid: Optional[float]
    # Diagnostics: home-team probabilities behind `p_yes`
    p_home: Optional[float]
    p_home_model: Optional[float]
    p_home_clamped: bool
    p_books_home: Optional[float]
    books_used: Optional[int]
    p_sharp_home: Optional[float]
    sharp_books_used: Optional[int]
    p_moneypuck_home: Optional[float]
    moneypuck_game_id: Optional[int]
    p_espn_home: Optional[float]
    espn_predictor_last_modified: Optional[str]
    p_yes: Optional[float]
    p_source: str  # books_us_consensus | moneypuck
    p_source_detail: str
    edge_vs_ask: Optional[float]
    edge_vs_limit: Optional[float]
    edge_vs_exec: Optional[float]
    execution: str  # maker|taker
    fee_mode: str  # maker|taker|none
    fee_per_contract: Optional[float]
    ev_net_per_contract: Optional[float]
    ev_net_at_ask_per_contract: Optional[float]
    recommend: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "date": self.date,
            "league": self.league,
            "matchup": self.matchup,
            "start_time_utc": self.start_time_utc,
            "event_id": self.event_id,
            "side": self.side,
            "team": self.team,
            "kalshi_ticker": self.kalshi_ticker,
            "kalshi_yes_bid_cents": "" if self.kalshi_yes_bid_cents is None else int(self.kalshi_yes_bid_cents),
            "kalshi_yes_ask_cents": "" if self.kalshi_yes_ask_cents is None else int(self.kalshi_yes_ask_cents),
            "kalshi_yes_limit_cents": "" if self.kalshi_yes_limit_cents is None else int(self.kalshi_yes_limit_cents),
            "kalshi_yes_exec_cents": "" if self.kalshi_yes_exec_cents is None else int(self.kalshi_yes_exec_cents),
            "kalshi_mid": "" if self.kalshi_mid is None else round(float(self.kalshi_mid), 6),
            "p_home": "" if self.p_home is None else round(float(self.p_home), 6),
            "p_home_model": "" if self.p_home_model is None else round(float(self.p_home_model), 6),
            "p_home_clamped": bool(self.p_home_clamped),
            "p_books_home": "" if self.p_books_home is None else round(float(self.p_books_home), 6),
            "books_used": "" if self.books_used is None else int(self.books_used),
            "p_sharp_home": "" if self.p_sharp_home is None else round(float(self.p_sharp_home), 6),
            "sharp_books_used": "" if self.sharp_books_used is None else int(self.sharp_books_used),
            "p_moneypuck_home": "" if self.p_moneypuck_home is None else round(float(self.p_moneypuck_home), 6),
            "moneypuck_game_id": "" if self.moneypuck_game_id is None else int(self.moneypuck_game_id),
            "p_espn_home": "" if self.p_espn_home is None else round(float(self.p_espn_home), 6),
            "espn_predictor_last_modified": "" if not self.espn_predictor_last_modified else str(self.espn_predictor_last_modified),
            "p_yes": "" if self.p_yes is None else round(float(self.p_yes), 6),
            "p_source": self.p_source,
            "p_source_detail": self.p_source_detail,
            "edge_vs_ask": "" if self.edge_vs_ask is None else round(float(self.edge_vs_ask), 6),
            "edge_vs_limit": "" if self.edge_vs_limit is None else round(float(self.edge_vs_limit), 6),
            "edge_vs_exec": "" if self.edge_vs_exec is None else round(float(self.edge_vs_exec), 6),
            "execution": str(self.execution),
            "fee_mode": str(self.fee_mode),
            "fee_per_contract": "" if self.fee_per_contract is None else round(float(self.fee_per_contract), 6),
            "ev_net_per_contract": "" if self.ev_net_per_contract is None else round(float(self.ev_net_per_contract), 6),
            "ev_net_at_ask_per_contract": ""
            if self.ev_net_at_ask_per_contract is None
            else round(float(self.ev_net_at_ask_per_contract), 6),
            "recommend": bool(self.recommend),
        }


def compute_edges(
    *,
    league: str,
    date_iso: str,
    odds_regions: str = "us",
    odds_bookmakers: Sequence[str] = (),
    prob_source: str = "books",
    ensemble_model_path: Optional[Path] = None,
    edge_threshold: float = 0.05,
    min_books_used: int = 1,
    allow_inplay: bool = False,
    fee_rate: float = 0.07,
    execution: str = "maker",
    fee_mode: str = "maker",
    maker_limit_mode: str = "ask_minus_1",
    maker_improve_cents: int = 1,
    maker_fee_rate: Optional[float] = None,
    taker_fee_rate: Optional[float] = None,
) -> List[EdgeRow]:
    lg = str(league).strip().lower()
    day = dt.date.fromisoformat(str(date_iso))

    s = requests.Session()
    sb = fetch_scoreboard(league=lg, day=day, cache_dir=Path("data/cache"), session=s)
    games = extract_games(league=lg, scoreboard=sb)
    if not allow_inplay:
        games = [g for g in games if g.state == "pre"]

    prob_source_norm = str(prob_source or "books").strip().lower()
    cons_by_matchup: Dict[str, H2HConsensus] = {}
    cons_sharp_by_matchup: Dict[str, H2HConsensus] = {}
    cons_primary_by_matchup: Dict[str, H2HConsensus] = {}
    primary_books: Tuple[str, ...] = tuple()
    mp_schedule = None
    ensemble_model: Optional[EnsembleModel] = None
    if prob_source_norm == "books":
        def _load_consensus(force: bool) -> Dict[str, H2HConsensus]:
            regions = [r.strip().lower() for r in str(odds_regions or "us").split(",") if r.strip()]
            if not regions:
                regions = ["us"]
            cons_lists: List[List[H2HConsensus]] = []
            for reg in regions:
                odds_events = fetch_odds(
                    league=lg,
                    markets=("h2h",),
                    regions=reg,
                    bookmakers=odds_bookmakers,
                    cache_dir=Path("data/cache"),
                    force=force,
                    session=s,
                )
                cons_lists.append(consensus_h2h_probs(odds_events, league=lg))
            return _merge_consensus(cons_lists)

        cons_by_matchup = _load_consensus(False)
        if games and not any(g.matchup in cons_by_matchup for g in games):
            cons_by_matchup = _load_consensus(True)
    elif prob_source_norm in {"pin_books_avg", "pin_books_agree", "pin_dk_avg", "pin_dk_agree"}:
        # Primary US book(s) (default to DraftKings).
        primary_books = tuple([b.strip().lower() for b in odds_bookmakers if str(b).strip()]) or ("draftkings",)

        def _load_primary(force: bool) -> Dict[str, H2HConsensus]:
            odds_events = fetch_odds(
                league=lg,
                markets=("h2h",),
                regions="us",
                bookmakers=primary_books,
                cache_dir=Path("data/cache"),
                force=force,
                session=s,
            )
            return {c.matchup: c for c in consensus_h2h_probs(odds_events, league=lg)}

        def _load_pinnacle(force: bool) -> Dict[str, H2HConsensus]:
            odds_events = fetch_odds(
                league=lg,
                markets=("h2h",),
                regions="eu",
                bookmakers=("pinnacle",),
                cache_dir=Path("data/cache"),
                force=force,
                session=s,
            )
            return {c.matchup: c for c in consensus_h2h_probs(odds_events, league=lg)}

        cons_primary_by_matchup = _load_primary(False)
        cons_sharp_by_matchup = _load_pinnacle(False)
        if games and not any(g.matchup in cons_primary_by_matchup for g in games):
            cons_primary_by_matchup = _load_primary(True)
        if games and not any(g.matchup in cons_sharp_by_matchup for g in games):
            cons_sharp_by_matchup = _load_pinnacle(True)
    elif prob_source_norm == "moneypuck":
        if lg != "nhl":
            raise SystemExit("[error] --prob-source moneypuck is only supported for NHL")
        season = season_string_for_date(day)
        mp_schedule = fetch_schedule(season=season, cache_dir=Path("data/cache"))
    elif prob_source_norm == "espn":
        if lg not in {"nba", "nfl"}:
            raise SystemExit("[error] --prob-source espn is only supported for NBA/NFL")
    elif prob_source_norm == "ensemble":
        # Needs books + an auxiliary model (NHL=MoneyPuck, NBA/NFL=ESPN predictor).
        def _load_consensus(force: bool) -> Dict[str, H2HConsensus]:
            regions = [r.strip().lower() for r in str(odds_regions or "us").split(",") if r.strip()]
            if not regions:
                regions = ["us"]
            cons_lists: List[List[H2HConsensus]] = []
            for reg in regions:
                odds_events = fetch_odds(
                    league=lg,
                    markets=("h2h",),
                    regions=reg,
                    bookmakers=odds_bookmakers,
                    cache_dir=Path("data/cache"),
                    force=force,
                    session=s,
                )
                cons_lists.append(consensus_h2h_probs(odds_events, league=lg))
            return _merge_consensus(cons_lists)

        if lg == "nfl":
            def _load_books(force: bool) -> Dict[str, H2HConsensus]:
                odds_events = fetch_odds(
                    league=lg,
                    markets=("h2h",),
                    regions="us",
                    bookmakers=odds_bookmakers,
                    cache_dir=Path("data/cache"),
                    force=force,
                    session=s,
                )
                return {c.matchup: c for c in consensus_h2h_probs(odds_events, league=lg)}

            def _load_sharp(force: bool) -> Dict[str, H2HConsensus]:
                odds_events = fetch_odds(
                    league=lg,
                    markets=("h2h",),
                    regions="eu",
                    bookmakers=("pinnacle",),
                    cache_dir=Path("data/cache"),
                    force=force,
                    session=s,
                )
                return {c.matchup: c for c in consensus_h2h_probs(odds_events, league=lg)}

            cons_by_matchup = _load_books(False)
            cons_sharp_by_matchup = _load_sharp(False)
            if games and not any(g.matchup in cons_by_matchup for g in games):
                cons_by_matchup = _load_books(True)
            if games and not any(g.matchup in cons_sharp_by_matchup for g in games):
                cons_sharp_by_matchup = _load_sharp(True)
        else:
            cons_by_matchup = _load_consensus(False)
            if games and not any(g.matchup in cons_by_matchup for g in games):
                cons_by_matchup = _load_consensus(True)
        if lg == "nhl":
            season = season_string_for_date(day)
            mp_schedule = fetch_schedule(season=season, cache_dir=Path("data/cache"))
        model_path = ensemble_model_path or (Path("data/models") / f"{lg}_ensemble.json")
        if not Path(model_path).exists():
            raise SystemExit(f"[error] missing ensemble model: {model_path} (run `fit-ensemble` first)")
        ensemble_model = load_model(Path(model_path))
    else:
        raise SystemExit(f"[error] unknown --prob-source: {prob_source}")

    exec_mode = str(execution or "maker").strip().lower() or "maker"
    fee_mode_norm = str(fee_mode or "").strip().lower() or ("maker" if exec_mode == "maker" else "taker")
    maker_limit_mode_norm = str(maker_limit_mode or "ask_minus_1").strip().lower() or "ask_minus_1"
    try:
        maker_fee_rate_f = float(os.environ.get("KALSHI_MAKER_FEE_RATE") or 0.0) if maker_fee_rate is None else float(maker_fee_rate)
    except Exception:
        maker_fee_rate_f = 0.0
    try:
        taker_fee_rate_f = (
            float(os.environ.get("KALSHI_TAKER_FEE_RATE") or 0.07) if taker_fee_rate is None else float(taker_fee_rate)
        )
    except Exception:
        taker_fee_rate_f = float(fee_rate)

    out: List[EdgeRow] = []
    require_agreement = prob_source_norm in {"pin_books_agree", "pin_dk_agree"}
    for g in games:
        p_home: Optional[float] = None  # used downstream for p_yes/edge
        p_home_model: Optional[float] = None  # raw model output (before clamps)
        p_home_clamped = False
        p_books_home: Optional[float] = None
        books_used: Optional[int] = None
        p_moneypuck_home: Optional[float] = None
        moneypuck_game_id: Optional[int] = None
        p_espn_home: Optional[float] = None
        espn_last_modified: Optional[str] = None
        p_sharp_home: Optional[float] = None
        sharp_books_used: Optional[int] = None
        p_source_label = ""
        p_source_detail = ""
        if prob_source_norm == "books":
            c = cons_by_matchup.get(g.matchup)
            if c is None or c.books_used < int(min_books_used):
                continue
            p_home = float(c.p_home)
            p_home_model = float(p_home)
            p_books_home = float(c.p_home)
            books_used = int(c.books_used)
            p_source_label = "books_us_consensus"
            p_source_detail = f"books_used={int(c.books_used)}"
        elif prob_source_norm == "moneypuck":
            assert mp_schedule is not None
            gid = find_game_id(schedule=mp_schedule, away=g.away, home=g.home, start_time_utc=g.start_time_utc)
            if gid is None:
                continue
            pre = fetch_pregame_prediction(game_id=int(gid), cache_dir=Path("data/cache"))
            if pre.moneypuck_home_win is None:
                continue
            p_home = float(pre.moneypuck_home_win)
            p_home_model = float(p_home)
            p_moneypuck_home = float(pre.moneypuck_home_win)
            moneypuck_game_id = int(gid)
            p_source_label = "moneypuck"
            p_source_detail = f"game_id={int(gid)}"
        elif prob_source_norm == "espn":
            pred = fetch_predictor_for_event(s, league=lg, event_id=str(g.event_id), cache_dir=Path("data/cache"))
            if pred.p_home is None:
                continue
            p_home = float(pred.p_home)
            p_home_model = float(p_home)
            p_espn_home = float(pred.p_home)
            espn_last_modified = pred.last_modified
            p_source_label = "espn_predictor"
            lm = pred.last_modified or ""
            p_source_detail = f"last_modified={lm}" if lm else ""
        elif prob_source_norm in {"pin_books_avg", "pin_books_agree", "pin_dk_avg", "pin_dk_agree"}:
            c_primary = cons_primary_by_matchup.get(g.matchup)
            c_sharp = cons_sharp_by_matchup.get(g.matchup)
            if c_primary is None or c_sharp is None:
                continue
            p_books_home = float(c_primary.p_home)
            books_used = int(c_primary.books_used)
            p_sharp_home = float(c_sharp.p_home)
            sharp_books_used = int(c_sharp.books_used)
            p_home = (float(p_books_home) + float(p_sharp_home)) / 2.0
            p_home_model = float(p_home)
            p_source_label = "pin_books_agree" if require_agreement else "pin_books_avg"
            primary_label = ",".join(primary_books) if primary_books else "draftkings"
            p_source_detail = f"books={primary_label};books_used={int(books_used)};pinnacle_books={int(sharp_books_used)}"
        else:
            # ensemble
            assert ensemble_model is not None
            c = cons_by_matchup.get(g.matchup)
            if c is None or c.books_used < int(min_books_used):
                continue
            p_books_home = float(c.p_home)
            books_used = int(c.books_used)
            p_aux: Optional[float] = None
            aux_detail = ""
            if lg == "nhl":
                assert mp_schedule is not None
                gid = find_game_id(schedule=mp_schedule, away=g.away, home=g.home, start_time_utc=g.start_time_utc)
                if gid is None:
                    continue
                pre = fetch_pregame_prediction(game_id=int(gid), cache_dir=Path("data/cache"))
                if pre.moneypuck_home_win is None:
                    continue
                p_moneypuck_home = float(pre.moneypuck_home_win)
                moneypuck_game_id = int(gid)
                p_aux = float(p_moneypuck_home)
                aux_detail = f"game_id={int(gid)}"
            elif lg == "nfl":
                c_sharp = cons_sharp_by_matchup.get(g.matchup)
                if c_sharp is None:
                    continue
                p_sharp_home = float(c_sharp.p_home)
                sharp_books_used = int(c_sharp.books_used)
                p_aux = float(p_sharp_home)
                aux_detail = f"sharp_books={int(c_sharp.books_used)}"
            else:
                pred = fetch_predictor_for_event(s, league=lg, event_id=str(g.event_id), cache_dir=Path("data/cache"))
                if pred.p_home is None:
                    continue
                p_espn_home = float(pred.p_home)
                espn_last_modified = pred.last_modified
                p_aux = float(p_espn_home)
                lm = pred.last_modified or ""
                aux_detail = f"last_modified={lm}" if lm else ""

            if p_aux is None:
                continue
            raw = float(ensemble_model.predict_p_home(p_books=float(p_books_home), p_moneypuck=float(p_aux)))
            p_home_model = raw
            # Safety: keep ensemble outputs within the convex hull of its inputs.
            lo = min(float(p_books_home), float(p_aux))
            hi = max(float(p_books_home), float(p_aux))
            clipped = max(lo, min(hi, raw))
            if abs(clipped - raw) > 1e-12:
                p_home_clamped = True
            p_home = float(clipped)
            p_source_label = "ensemble"
            extra = f";{aux_detail}" if aux_detail else ""
            p_source_detail = f"books_used={int(c.books_used)}{extra}"
        if p_home is None:
            continue
        p_away = 1.0 - float(p_home)

        resolved = resolve_game_markets(league=lg, day=day, away=g.away, home=g.home, session=s)
        if resolved is None:
            # Some slates can have date-token drift; try adjacent dates.
            for alt in (day - dt.timedelta(days=1), day + dt.timedelta(days=1)):
                resolved = resolve_game_markets(league=lg, day=alt, away=g.away, home=g.home, session=s)
                if resolved is not None:
                    break
        if resolved is None:
            continue

        home_mkt = fetch_market(ticker=resolved.market_ticker_home_yes, session=s)
        away_mkt = fetch_market(ticker=resolved.market_ticker_away_yes, session=s)

        def _edge_row(*, side: str) -> Optional[EdgeRow]:
            if side == "home":
                team = g.home
                ticker = resolved.market_ticker_home_yes
                m = home_mkt
                p_true = p_home
            else:
                team = g.away
                ticker = resolved.market_ticker_away_yes
                m = away_mkt
                p_true = p_away

            bid = _safe_int(m.get("yes_bid"))
            ask = _safe_int(m.get("yes_ask"))
            if ask is None or ask <= 0 or ask >= 100:
                return None
            if bid is not None and (bid <= 0 or bid >= 100):
                bid = None
            mid = mid_from_bidask(yes_bid=bid, yes_ask=ask) if bid is not None and ask is not None else None

            ask_price = float(ask) / 100.0
            edge_vs_ask = float(p_true) - float(ask_price)
            ev_at_ask: Optional[float] = None
            try:
                taker_fee_at_ask = taker_fee_dollars(contracts=1, price=float(ask_price), rate=float(taker_fee_rate_f))
                ev_at_ask = expected_value_yes(p_true=float(p_true), price=float(ask_price), fee=float(taker_fee_at_ask))
            except Exception:
                ev_at_ask = None

            limit_cents: Optional[int] = None
            exec_cents: Optional[int] = None
            exec_price: Optional[float] = None
            if exec_mode == "taker":
                exec_cents = int(ask)
                exec_price = float(ask_price)
            else:
                if maker_limit_mode_norm in {"ask_minus_1", "ask-1"}:
                    limit_cents = _ask_minus_1_limit_cents(yes_ask=ask)
                elif maker_limit_mode_norm in {"bid_plus_improve", "desired"}:
                    limit_cents = _desired_maker_limit_cents(yes_bid=bid, yes_ask=ask, improve_cents=int(maker_improve_cents))
                else:
                    limit_cents = _edge_capped_limit_cents(
                        p_yes=float(p_true),
                        edge_threshold=float(edge_threshold),
                        yes_bid=bid,
                        yes_ask=ask,
                        improve_cents=int(maker_improve_cents),
                    )
                if limit_cents is None:
                    return None
                exec_cents = int(limit_cents)
                exec_price = float(exec_cents) / 100.0

            if exec_cents is None or exec_price is None:
                return None

            edge_vs_exec = float(p_true) - float(exec_price)
            edge_vs_limit = None if limit_cents is None else (float(p_true) - float(limit_cents) / 100.0)

            fee: float = 0.0
            if fee_mode_norm not in {"none", "off", "0"}:
                if fee_mode_norm == "taker":
                    fee = float(taker_fee_dollars(contracts=1, price=float(exec_price), rate=float(taker_fee_rate_f)))
                else:
                    fee = float(maker_fee_dollars(contracts=1, price=float(exec_price), rate=float(maker_fee_rate_f)))
            ev = expected_value_yes(p_true=float(p_true), price=float(exec_price), fee=float(fee))
            agree_ok = True
            if require_agreement:
                if p_books_home is None or p_sharp_home is None:
                    return None
                if side == "home":
                    edge_books = float(p_books_home) - float(exec_price)
                    edge_sharp = float(p_sharp_home) - float(exec_price)
                else:
                    edge_books = (1.0 - float(p_books_home)) - float(exec_price)
                    edge_sharp = (1.0 - float(p_sharp_home)) - float(exec_price)
                agree_ok = bool(edge_books >= float(edge_threshold) and edge_sharp >= float(edge_threshold))
            rec = bool(agree_ok and edge_vs_exec >= float(edge_threshold) and ev > 0.0)
            return EdgeRow(
                date=day.isoformat(),
                league=lg,
                matchup=g.matchup,
                start_time_utc=_to_iso_z(g.start_time_utc),
                event_id=g.event_id,
                side=side,
                team=team,
                kalshi_ticker=ticker,
                kalshi_yes_bid_cents=bid,
                kalshi_yes_ask_cents=ask,
                kalshi_yes_limit_cents=limit_cents,
                kalshi_yes_exec_cents=exec_cents,
                kalshi_mid=mid,
                p_home=None if p_home is None else float(p_home),
                p_home_model=None if p_home_model is None else float(p_home_model),
                p_home_clamped=bool(p_home_clamped),
                p_books_home=None if p_books_home is None else float(p_books_home),
                books_used=None if books_used is None else int(books_used),
                p_sharp_home=None if p_sharp_home is None else float(p_sharp_home),
                sharp_books_used=None if sharp_books_used is None else int(sharp_books_used),
                p_moneypuck_home=None if p_moneypuck_home is None else float(p_moneypuck_home),
                moneypuck_game_id=None if moneypuck_game_id is None else int(moneypuck_game_id),
                p_espn_home=None if p_espn_home is None else float(p_espn_home),
                espn_predictor_last_modified=espn_last_modified,
                p_yes=float(p_true),
                p_source=p_source_label,
                p_source_detail=p_source_detail,
                edge_vs_ask=edge_vs_ask,
                edge_vs_limit=edge_vs_limit,
                edge_vs_exec=edge_vs_exec,
                execution=exec_mode,
                fee_mode=fee_mode_norm,
                fee_per_contract=float(fee),
                ev_net_per_contract=float(ev),
                ev_net_at_ask_per_contract=ev_at_ask,
                recommend=rec,
            )

        for side in ("home", "away"):
            r = _edge_row(side=side)
            if r is not None:
                out.append(r)

    # Sort: best EV first
    out.sort(key=lambda r: (-(r.ev_net_per_contract or -999.0), -(r.edge_vs_exec or -999.0), r.matchup, r.side))
    return out


def write_edges_csv(rows: List[EdgeRow], *, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fieldnames = list(rows[0].to_dict().keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r.to_dict())
    return out_path

