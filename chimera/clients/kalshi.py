from __future__ import annotations

import base64
import datetime as dt
import json
import os
import random
import re
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from chimera.clients.http import get_with_retries
from chimera.teams import alias_candidates, normalize_team_code


_DEFAULT_PUBLIC_BASE = "https://api.elections.kalshi.com/trade-api/v2"

SERIES_TICKER_BY_LEAGUE = {"nba": "KXNBAGAME", "nhl": "KXNHLGAME", "nfl": "KXNFLGAME"}
SERIES_TICKER_BY_LEAGUE_MARKET = {
    ("nba", "game"): "KXNBAGAME",
    ("nba", "spread"): "KXNBASPREAD",
    ("nba", "total"): "KXNBATOTAL",
    ("nhl", "game"): "KXNHLGAME",
    ("nhl", "spread"): "KXNHLSPREAD",
    ("nhl", "total"): "KXNHLTOTAL",
    ("nfl", "game"): "KXNFLGAME",
    ("nfl", "spread"): "KXNFLSPREAD",
    ("nfl", "total"): "KXNFLTOTAL",
}

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def _public_base() -> str:
    return (
        str(os.environ.get("KALSHI_PUBLIC_BASE") or "")
        or str(os.environ.get("KALSHI_BASE") or "")
        or str(os.environ.get("KALSHI_API_BASE") or "")
        or _DEFAULT_PUBLIC_BASE
    ).strip()


def _align_ts_to_minute(ts: int) -> int:
    return int(ts) - (int(ts) % 60)


def _kalshi_date_token(day: dt.date) -> str:
    yy = day.strftime("%y")
    mon = _MONTHS[day.month - 1]
    dd = day.strftime("%d")
    return f"{yy}{mon}{dd}"


def _team_tokens(code: str, league: str) -> List[str]:
    if not code:
        return []
    canonical = normalize_team_code(code, league) or str(code).upper()
    candidates = alias_candidates(canonical, league) or [canonical]
    tokens: set[str] = set()
    for a in candidates:
        cleaned = _NON_ALNUM.sub("", str(a).upper())
        if 2 <= len(cleaned) <= 4:
            tokens.add(cleaned)
    canonical_clean = _NON_ALNUM.sub("", str(canonical).upper())
    ordered: List[str] = []
    if canonical_clean in tokens:
        ordered.append(canonical_clean)
    ordered.extend(sorted(tokens - {canonical_clean}, key=lambda x: (-len(x), x)))
    return ordered


def _candlestick_mid_prob(candle: Dict[str, object]) -> Optional[float]:
    yes_bid = (candle.get("yes_bid") or {}) if isinstance(candle.get("yes_bid"), dict) else {}
    yes_ask = (candle.get("yes_ask") or {}) if isinstance(candle.get("yes_ask"), dict) else {}
    bid = yes_bid.get("close")
    ask = yes_ask.get("close")
    if bid is None or ask is None:
        return None
    try:
        bid_i = int(bid)
        ask_i = int(ask)
    except Exception:
        return None
    if bid_i < 0 or bid_i > 100 or ask_i < 0 or ask_i > 100:
        return None
    return ((bid_i + ask_i) / 2.0) / 100.0


def _candlestick_bid_ask_close(candle: Dict[str, object]) -> Tuple[Optional[int], Optional[int]]:
    yes_bid = (candle.get("yes_bid") or {}) if isinstance(candle.get("yes_bid"), dict) else {}
    yes_ask = (candle.get("yes_ask") or {}) if isinstance(candle.get("yes_ask"), dict) else {}
    bid = yes_bid.get("close")
    ask = yes_ask.get("close")
    try:
        bid_i = int(bid) if bid is not None else None
    except Exception:
        bid_i = None
    try:
        ask_i = int(ask) if ask is not None else None
    except Exception:
        ask_i = None
    if bid_i is not None and (bid_i < 0 or bid_i > 100):
        bid_i = None
    if ask_i is not None and (ask_i < 0 or ask_i > 100):
        ask_i = None
    return bid_i, ask_i


@dataclass(frozen=True)
class ResolvedGameMarkets:
    event_ticker: str
    market_ticker_home_yes: str
    market_ticker_away_yes: str


@dataclass(frozen=True)
class ResolvedEvent:
    event_ticker: str
    markets: List[Dict[str, Any]]


def fetch_event(
    *,
    event_ticker: str,
    session: requests.Session,
    kalshi_public_base: Optional[str] = None,
) -> Dict[str, Any]:
    base = (kalshi_public_base or _public_base()).rstrip("/")
    et = str(event_ticker or "").strip().upper()
    if not et:
        raise ValueError("missing event_ticker")
    url = f"{base}/events/{et}"
    resp = get_with_retries(session, url, timeout_s=20.0)
    resp.raise_for_status()
    data = resp.json() or {}
    return data if isinstance(data, dict) else {}


def resolve_matchup_event(
    *,
    league: str,
    market_type: str,
    day: dt.date,
    away: str,
    home: str,
    session: requests.Session,
    kalshi_public_base: Optional[str] = None,
) -> Optional[ResolvedEvent]:
    """
    Resolve a Kalshi event (and its markets list) for a matchup.

    market_type: "game" (moneyline), "spread", or "total".
    """
    lg = str(league or "").strip().lower()
    mt = str(market_type or "").strip().lower()
    series = SERIES_TICKER_BY_LEAGUE_MARKET.get((lg, mt))
    if not series:
        return None

    base = (kalshi_public_base or _public_base()).rstrip("/")
    date_token = _kalshi_date_token(day)
    away_can = normalize_team_code(away, lg) or str(away).upper()
    home_can = normalize_team_code(home, lg) or str(home).upper()
    away_tokens = _team_tokens(away_can, lg)
    home_tokens = _team_tokens(home_can, lg)
    if not away_tokens or not home_tokens:
        return None

    for away_tok in away_tokens:
        for home_tok in home_tokens:
            event_ticker = f"{series}-{date_token}{away_tok}{home_tok}"
            url = f"{base}/events/{event_ticker}"
            resp = get_with_retries(session, url, timeout_s=20.0)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json() or {}
            markets = data.get("markets") if isinstance(data, dict) else None
            if not isinstance(markets, list):
                continue
            return ResolvedEvent(event_ticker=event_ticker, markets=[m for m in markets if isinstance(m, dict)])
    return None


def resolve_game_markets(
    *,
    league: str,
    day: dt.date,
    away: str,
    home: str,
    session: requests.Session,
    kalshi_public_base: Optional[str] = None,
) -> Optional[ResolvedGameMarkets]:
    """
    Resolve (home YES ticker, away YES ticker) for a matchup using the Kalshi public event endpoint.
    """
    lg = str(league or "").strip().lower()
    series = SERIES_TICKER_BY_LEAGUE.get(lg)
    if not series:
        return None

    base = (kalshi_public_base or _public_base()).rstrip("/")
    date_token = _kalshi_date_token(day)
    away_can = normalize_team_code(away, lg) or str(away).upper()
    home_can = normalize_team_code(home, lg) or str(home).upper()
    away_tokens = _team_tokens(away_can, lg)
    home_tokens = _team_tokens(home_can, lg)
    if not away_tokens or not home_tokens:
        return None

    for away_tok in away_tokens:
        for home_tok in home_tokens:
            event_ticker = f"{series}-{date_token}{away_tok}{home_tok}"
            url = f"{base}/events/{event_ticker}"
            resp = get_with_retries(session, url, timeout_s=20.0)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json() or {}
            markets = data.get("markets") or []
            if not isinstance(markets, list):
                continue

            home_mkt = ""
            away_mkt = ""
            for m in markets:
                if not isinstance(m, dict):
                    continue
                ticker = str(m.get("ticker") or "").strip()
                if not ticker:
                    continue
                suffix = ticker.split("-")[-1].strip().upper()
                suffix_can = normalize_team_code(suffix, lg)
                if suffix_can and suffix_can == home_can:
                    home_mkt = ticker
                if suffix_can and suffix_can == away_can:
                    away_mkt = ticker
            if home_mkt and away_mkt:
                return ResolvedGameMarkets(
                    event_ticker=event_ticker,
                    market_ticker_home_yes=home_mkt,
                    market_ticker_away_yes=away_mkt,
                )
    return None


def fetch_market(
    *,
    ticker: str,
    session: requests.Session,
    kalshi_public_base: Optional[str] = None,
) -> Dict[str, Any]:
    base = (kalshi_public_base or _public_base()).rstrip("/")
    url = f"{base}/markets/{str(ticker).strip().upper()}"
    resp = get_with_retries(session, url, timeout_s=20.0)
    resp.raise_for_status()
    data = resp.json() or {}
    m = data.get("market") if isinstance(data, dict) else None
    return m if isinstance(m, dict) else {}


def mid_from_bidask(*, yes_bid: object, yes_ask: object) -> Optional[float]:
    try:
        b = float(yes_bid)
        a = float(yes_ask)
    except Exception:
        return None
    return (b + a) / 200.0


def fetch_candlestick_quotes(
    *,
    market_tickers: Sequence[str],
    target_ts: int,
    period_interval_min: int = 1,
    lookback_min: int = 60,
    session: requests.Session,
    kalshi_public_base: Optional[str] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetch bid/ask/mid at the most recent candlestick end_ts <= target_ts.

    Returns mapping market_ticker -> {
      'yes_bid': int cents (or None),
      'yes_ask': int cents (or None),
      'mid': float in [0,1] (or None),
      'spread': float in [0,1] (or None),
      'candle_end_ts': int seconds (or None),
    }
    """
    base = (kalshi_public_base or _public_base()).rstrip("/")
    url = f"{base}/markets/candlesticks"
    out: Dict[str, Dict[str, Optional[float]]] = {
        t: {"yes_bid": None, "yes_ask": None, "mid": None, "spread": None, "candle_end_ts": None} for t in market_tickers
    }
    if not market_tickers:
        return out

    target_ts_int = int(target_ts)
    start_ts = target_ts_int - int(lookback_min) * 60
    end_ts = target_ts_int + int(period_interval_min) * 60
    params = {
        "market_tickers": ",".join([str(t).strip().upper() for t in market_tickers]),
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "period_interval": int(period_interval_min),
    }
    resp = get_with_retries(session, url, params=params, timeout_s=30.0)
    resp.raise_for_status()
    data = resp.json() or {}
    for entry in data.get("markets") or []:
        if not isinstance(entry, dict):
            continue
        ticker = entry.get("market_ticker")
        if not ticker:
            continue
        candles = entry.get("candlesticks") or []
        if not isinstance(candles, list) or not candles:
            continue
        best_end_ts: Optional[int] = None
        best_bid: Optional[int] = None
        best_ask: Optional[int] = None
        best_mid: Optional[float] = None
        for candle in candles:
            if not isinstance(candle, dict):
                continue
            try:
                candle_end_ts = int(candle.get("end_period_ts", -1))
            except Exception:
                continue
            if candle_end_ts > target_ts_int:
                continue
            bid_i, ask_i = _candlestick_bid_ask_close(candle)
            mid = _candlestick_mid_prob(candle)
            if bid_i is None or ask_i is None or mid is None:
                continue
            if best_end_ts is None or candle_end_ts > best_end_ts:
                best_end_ts = candle_end_ts
                best_bid = bid_i
                best_ask = ask_i
                best_mid = mid
        if best_end_ts is None or best_mid is None:
            continue
        spread = None
        try:
            spread = float(best_ask - best_bid) / 100.0 if best_bid is not None and best_ask is not None else None
        except Exception:
            spread = None
        out[str(ticker)] = {
            "yes_bid": None if best_bid is None else float(best_bid),
            "yes_ask": None if best_ask is None else float(best_ask),
            "mid": float(best_mid),
            "spread": spread,
            "candle_end_ts": None if best_end_ts is None else float(best_end_ts),
        }
    return out


# -----------------------------
# Private API (signed requests)
# -----------------------------


class KalshiPrivateClient:
    """
    Minimal signed client for Kalshi private endpoints.

    Signature details can vary across API versions. This implementation follows the common
    pattern used in Kalshi examples:

      signature = RSA_SHA256( timestamp + method + path + body )
      headers:
        KALSHI-ACCESS-KEY
        KALSHI-ACCESS-SIGNATURE
        KALSHI-ACCESS-TIMESTAMP
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        key_id: Optional[str] = None,
        private_key_pem: Optional[bytes] = None,
        private_key_path: Optional[Path] = None,
        timestamp_unit: str = "ms",
        padding_mode: str = "pss",
        sign_query: bool = False,
    ) -> None:
        self.base_url = (base_url or os.environ.get("KALSHI_API_BASE") or _public_base()).rstrip("/")
        self.key_id = (key_id or os.environ.get("KALSHI_API_KEY_ID") or "").strip()
        self.timestamp_unit = str(timestamp_unit).strip().lower() or "ms"
        self.padding_mode = str(padding_mode).strip().lower() or "pkcs1v15"
        self.sign_query = bool(sign_query)
        parsed = urlparse(self.base_url)
        self._base_path = (parsed.path or "").rstrip("/")
        if not self._base_path:
            # Kalshi docs sign paths like `/trade-api/v2/...`.
            self._base_path = "/trade-api/v2"

        pem: Optional[bytes] = None
        if private_key_pem:
            pem = private_key_pem
        else:
            raw = os.environ.get("KALSHI_API_PRIVATE_KEY") or ""
            if raw.strip():
                pem = raw.encode("utf-8")
        if pem is None and private_key_path is not None:
            pem = Path(private_key_path).read_bytes()
        if pem is None:
            p = os.environ.get("KALSHI_PRIVATE_KEY_PATH") or ""
            if p.strip():
                pem = Path(p.strip()).read_bytes()

        if not self.key_id:
            raise ValueError("missing KALSHI_API_KEY_ID")
        if pem is None:
            raise ValueError("missing Kalshi private key (KALSHI_API_PRIVATE_KEY or KALSHI_PRIVATE_KEY_PATH)")

        self._key = load_pem_private_key(pem, password=None)
        self._session = requests.Session()

    def _timestamp(self) -> str:
        now = time.time()
        if self.timestamp_unit == "s":
            return str(int(now))
        return str(int(now * 1000.0))

    def _pad(self):
        if self.padding_mode == "pss":
            return padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH)
        return padding.PKCS1v15()

    def _sign(self, message: str) -> str:
        sig = self._key.sign(message.encode("utf-8"), self._pad(), hashes.SHA256())
        return base64.b64encode(sig).decode("ascii")

    def _headers(self, method: str, path: str, body: str) -> Dict[str, str]:
        ts = self._timestamp()
        # Kalshi signing per docs:
        #   signature = RSA-PSS-SHA256( timestamp + METHOD + path )
        # Where path includes the `/trade-api/v2/...` prefix and excludes query params.
        # Body is NOT included in the signature string.
        prehash = ts + method.upper() + path
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self._sign(prehash),
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, *, params: Optional[Dict[str, object]] = None, json_body: Any = None) -> Dict[str, Any]:
        p = "/" + str(path).lstrip("/")

        # Build a deterministic query string for signing (avoid ambiguity).
        qs = ""
        if params:
            try:
                items = sorted((str(k), str(v)) for k, v in params.items())
            except Exception:
                items = [(str(k), str(v)) for k, v in (params or {}).items()]
            qs = urlencode(items)

        sign_path = self._base_path + p + (f"?{qs}" if (qs and self.sign_query) else "")
        url = self.base_url + p + (f"?{qs}" if qs else "")

        body = "" if json_body is None else json.dumps(json_body, separators=(",", ":"), sort_keys=True)
        headers = self._headers(method, sign_path, body)
        resp = self._session.request(method.upper(), url, data=(body if body else None), headers=headers, timeout=30.0)
        resp.raise_for_status()
        out = resp.json()
        if not isinstance(out, dict):
            raise TypeError(f"unexpected response from {p} (expected object)")
        return out

    def _request_first_success(
        self,
        method: str,
        paths: Sequence[str],
        *,
        params: Optional[Dict[str, object]] = None,
        json_body: Any = None,
    ) -> Dict[str, Any]:
        last_exc: Optional[BaseException] = None
        for p in paths:
            try:
                return self.request(method, p, params=params, json_body=json_body)
            except requests.HTTPError as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                # Try the next candidate if the route doesn't exist.
                if status == 404:
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("no paths provided")

    def get_portfolio_balance(self) -> Dict[str, Any]:
        return self._request_first_success("GET", ["/portfolio/balance", "/portfolio"], params=None)

    def get_portfolio_positions(self) -> Dict[str, Any]:
        return self._request_first_success("GET", ["/portfolio/positions"], params=None)

    def get_fills(self, *, ticker: Optional[str] = None, limit: int = 500, cursor: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, object] = {"limit": int(limit)}
        if ticker:
            params["ticker"] = str(ticker).strip().upper()
        if cursor:
            params["cursor"] = str(cursor)
        return self._request_first_success("GET", ["/portfolio/fills", "/fills"], params=params)

    def place_order(self, *, ticker: str, side: str, action: str, count: int, price_cents: int) -> Dict[str, Any]:
        side_norm = str(side).strip().lower()
        if side_norm not in {"yes", "no"}:
            raise ValueError("side must be 'yes' or 'no'")
        px = int(price_cents)
        if px < 1 or px > 99:
            raise ValueError("price_cents must be in [1, 99]")

        # Kalshi v2 order schema expects one of:
        #   yes_price / no_price (cents), or yes_price_dollars / no_price_dollars.
        payload: Dict[str, Any] = {
            "ticker": str(ticker).strip().upper(),
            "side": side_norm,
            "action": str(action).strip().lower(),
            "count": int(count),
            "type": "limit",
        }
        if side_norm == "yes":
            payload["yes_price"] = px
        else:
            payload["no_price"] = px
        return self._request_first_success("POST", ["/portfolio/orders", "/orders"], json_body=payload)

    def cancel_order(self, *, order_id: str) -> Dict[str, Any]:
        oid = str(order_id).strip()
        if not oid:
            raise ValueError("order_id required")
        return self._request_first_success("DELETE", [f"/portfolio/orders/{oid}", f"/orders/{oid}"], params=None)
