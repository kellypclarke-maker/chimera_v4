from __future__ import annotations

import datetime as dt
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from zoneinfo import ZoneInfo

from kalshi_core.clients.fees import maker_fee_dollars
from kalshi_core.clients.kalshi_public import (
    discover_viable_crypto_tickers_for_date,
    extract_kalshi_date_token_from_ticker,
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

COINBASE_EXCHANGE = "https://api.exchange.coinbase.com"
KRAKEN_API = "https://api.kraken.com"
BITSTAMP_API = "https://www.bitstamp.net"

# Matches e.g. "Above $100000", "Below $60,000.00", "$64,250 to 64,499.99", "$59,749.99 or below".
RANGE_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*to\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
BELOW_RE = re.compile(r"(?:^|\b)(?:below|or below|or less|under)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
ABOVE_RE = re.compile(r"(?:^|\b)(?:above|or above|or more|over)\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)

# Matches e.g. "$1.63991 or above", "$59,749.99 or below", "49° or below".
TRAILING_BELOW_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:°)?\s*(?:or\s+below|or\s+less|or\s+under)\b", re.IGNORECASE)
TRAILING_ABOVE_RE = re.compile(r"\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:°)?\s*(?:or\s+above|or\s+more|or\s+over)\b", re.IGNORECASE)

# Parse rule time targets like: "before March 1, 2026 at 12:00AM ET"
RULE_BEFORE_RE = re.compile(
    r"\bbefore\s+(?P<mon>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\s+at\s+"
    r"(?P<h>\d{1,2})\s*:\s*(?P<m>\d{2})\s*(?P<ampm>AM|PM)\s*(?P<tz>ET|EST|EDT)\b",
    re.IGNORECASE,
)

# Some range rules include "at 12 AM EST on Feb 18, 2026".
RULE_AT_ON_RE = re.compile(
    r"\bat\s+(?P<h>\d{1,2})\s*(?::\s*(?P<m>\d{2}))?\s*(?P<ampm>AM|PM)\s*(?P<tz>ET|EST|EDT)\s+on\s+"
    r"(?P<mon>[A-Za-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})\b",
    re.IGNORECASE,
)

MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


@dataclass(frozen=True)
class ProductInfo:
    product_id: str
    spot_price: float
    sigma_per_sqrt_s: float
    spot_sources: List[str]


def _parse_iso_dt(x: str) -> Optional[dt.datetime]:
    s = str(x or "").strip()
    if not s:
        return None
    try:
        t = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        return t
    except Exception:
        return None


def _parse_money(x: str) -> Optional[float]:
    s = str(x or "").strip().replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _parse_bounds(yes_sub_title: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    s = str(yes_sub_title or "").strip()
    if not s:
        return None

    m = RANGE_RE.search(s)
    if m:
        lo = _parse_money(m.group(1))
        hi = _parse_money(m.group(2))
        if lo is None or hi is None:
            return None
        return (min(lo, hi), max(lo, hi))

    m = TRAILING_BELOW_RE.search(s)
    if m:
        hi = _parse_money(m.group(1))
        if hi is None:
            return None
        return (None, float(hi))

    m = TRAILING_ABOVE_RE.search(s)
    if m:
        lo = _parse_money(m.group(1))
        if lo is None:
            return None
        return (float(lo), None)

    m = BELOW_RE.search(s)
    if m:
        hi = _parse_money(m.group(1))
        if hi is None:
            return None
        return (None, float(hi))

    m = ABOVE_RE.search(s)
    if m:
        lo = _parse_money(m.group(1))
        if lo is None:
            return None
        return (float(lo), None)

    return None


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def _p_terminal_between(*, s0: float, sigma_total: float, lo: Optional[float], hi: Optional[float]) -> float:
    if s0 <= 0.0:
        return float("nan")
    if sigma_total <= 0.0:
        if lo is not None and s0 < float(lo):
            return 0.0
        if hi is not None and s0 > float(hi):
            return 0.0
        return 1.0

    mu = math.log(float(s0))
    sig = float(sigma_total)

    if lo is not None and float(lo) <= 0.0:
        lo = None

    if lo is not None and hi is not None:
        a = (math.log(float(lo)) - mu) / sig
        b = (math.log(float(hi)) - mu) / sig
        return max(0.0, min(1.0, _normal_cdf(b) - _normal_cdf(a)))
    if hi is not None:
        b = (math.log(float(hi)) - mu) / sig
        return max(0.0, min(1.0, _normal_cdf(b)))
    if lo is not None:
        a = (math.log(float(lo)) - mu) / sig
        return max(0.0, min(1.0, 1.0 - _normal_cdf(a)))
    return float("nan")


def _p_hit_upper(*, s0: float, barrier: float, sigma_total: float) -> float:
    if s0 <= 0.0 or barrier <= 0.0:
        return float("nan")
    if float(s0) >= float(barrier):
        return 1.0
    if sigma_total <= 0.0:
        return 0.0
    x0 = math.log(float(s0))
    a = math.log(float(barrier))
    z = (a - x0) / float(sigma_total)
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(z))))


def _p_hit_lower(*, s0: float, barrier: float, sigma_total: float) -> float:
    if s0 <= 0.0 or barrier <= 0.0:
        return float("nan")
    if float(s0) <= float(barrier):
        return 1.0
    if sigma_total <= 0.0:
        return 0.0
    x0 = math.log(float(s0))
    a = math.log(float(barrier))
    z = (x0 - a) / float(sigma_total)
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(z))))


def _parse_rule_target_dt(rules_primary: str) -> Optional[dt.datetime]:
    s = str(rules_primary or "")
    m = RULE_BEFORE_RE.search(s)
    if m:
        mon = MONTHS.get(str(m.group("mon") or "").strip().upper()[:3], 0)
        if mon <= 0:
            return None
        day = int(m.group("day"))
        year = int(m.group("year"))
        h = int(m.group("h"))
        minute = int(m.group("m"))
        ampm = str(m.group("ampm") or "").strip().upper()
        if ampm == "AM" and h == 12:
            h = 0
        if ampm == "PM" and h != 12:
            h += 12
        local = dt.datetime(year, mon, day, h, minute, tzinfo=ZoneInfo("America/New_York"))
        return local.astimezone(dt.timezone.utc)

    m = RULE_AT_ON_RE.search(s)
    if m:
        mon = MONTHS.get(str(m.group("mon") or "").strip().upper()[:3], 0)
        if mon <= 0:
            return None
        day = int(m.group("day"))
        year = int(m.group("year"))
        h = int(m.group("h"))
        minute = int(m.group("m") or 0)
        ampm = str(m.group("ampm") or "").strip().upper()
        if ampm == "AM" and h == 12:
            h = 0
        if ampm == "PM" and h != 12:
            h += 12
        local = dt.datetime(year, mon, day, h, minute, tzinfo=ZoneInfo("America/New_York"))
        return local.astimezone(dt.timezone.utc)

    return None


def _product_for_event(event_ticker: str) -> Optional[str]:
    ev = str(event_ticker or "").strip().upper()
    if not ev:
        return None

    mapping = {
        "KXBTC15M": "BTC-USD",
        "KXBTCD": "BTC-USD",
        "KXBTC": "BTC-USD",
        "KXETH15M": "ETH-USD",
        "KXETHD": "ETH-USD",
        "KXETH": "ETH-USD",
        "KXDOGED": "DOGE-USD",
        "KXDOGE": "DOGE-USD",
        "KXXRPD": "XRP-USD",
        "KXXRP": "XRP-USD",
        "KXSOLE": "SOL-USD",
        "KXSOLD": "SOL-USD",
        "KXSOL": "SOL-USD",
    }

    best = None
    for k in sorted(mapping.keys(), key=len, reverse=True):
        if ev.startswith(k):
            best = mapping[k]
            break
    return best


def _load_cached_json(path: Path, *, ttl_minutes: float) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    age_min = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)).total_seconds() / 60.0
    if age_min > float(ttl_minutes):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _fetch_candles(
    *,
    product_id: str,
    granularity_s: int,
    start: dt.datetime,
    end: dt.datetime,
    cache_path: Path,
    ttl_minutes: float,
) -> Optional[List[List[float]]]:
    cached = _load_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if cached is not None:
        arr = cached.get("candles")
        return arr if isinstance(arr, list) else None

    # Coinbase public candles endpoint caps responses to ~300 points.
    max_points = 300
    max_range_s = float(max_points) * float(int(granularity_s))
    if (end - start).total_seconds() > max_range_s:
        start = end - dt.timedelta(seconds=max_range_s)

    url = f"{COINBASE_EXCHANGE}/products/{product_id}/candles"
    params = {
        "granularity": int(granularity_s),
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    }
    headers = {"User-Agent": "kalshi-meta-crypto/0.2"}

    try:
        with requests.Session() as s:
            resp = s.get(url, params=params, headers=headers, timeout=20.0)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                return None
            _write_json(cache_path, {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "candles": data})
            return data
    except Exception:
        return None


def _estimate_sigma_and_spot_from_candles(candles: List[List[float]], *, granularity_s: int) -> Optional[Tuple[float, float]]:
    # Coinbase format: [ time, low, high, open, close, volume ]
    if not candles:
        return None
    ordered = sorted(candles, key=lambda x: float(x[0]) if isinstance(x, list) and x else 0.0)
    closes: List[float] = []
    for row in ordered:
        if not (isinstance(row, list) and len(row) >= 5):
            continue
        try:
            closes.append(float(row[4]))
        except Exception:
            continue
    if len(closes) < 6:
        return None

    spot = float(closes[-1])
    rets: List[float] = []
    for i in range(1, len(closes)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev <= 0.0 or cur <= 0.0:
            continue
        rets.append(math.log(cur / prev))
    if len(rets) < 5:
        return None

    mean = sum(rets) / float(len(rets))
    var = sum((x - mean) ** 2 for x in rets) / float(max(1, len(rets) - 1))
    stdev = math.sqrt(max(0.0, var))

    sigma_per_sqrt_s = float(stdev) / math.sqrt(float(granularity_s))
    return (spot, sigma_per_sqrt_s)


def _coinbase_spot(product_id: str) -> Optional[float]:
    url = f"{COINBASE_EXCHANGE}/products/{product_id}/ticker"
    try:
        with requests.Session() as s:
            resp = s.get(url, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        px = float(data.get("price"))
        return px if px > 0.0 else None
    except Exception:
        return None


def _kraken_pair_for_product(product_id: str) -> str:
    # Kraken uses XBT for BTC.
    mapping = {
        "BTC-USD": "XBTUSD",
        "ETH-USD": "ETHUSD",
        "DOGE-USD": "DOGEUSD",
        "XRP-USD": "XRPUSD",
        "SOL-USD": "SOLUSD",
    }
    return mapping.get(str(product_id).strip().upper(), "")


def _kraken_spot(product_id: str) -> Optional[float]:
    pair = _kraken_pair_for_product(product_id)
    if not pair:
        return None
    url = f"{KRAKEN_API}/0/public/Ticker"
    try:
        with requests.Session() as s:
            resp = s.get(url, params={"pair": pair}, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        result = data.get("result") if isinstance(data, dict) else None
        if not isinstance(result, dict) or not result:
            return None
        first_key = next(iter(result.keys()))
        payload = result.get(first_key)
        if not isinstance(payload, dict):
            return None
        # 'c' is last trade closed [price, lot volume]
        c = payload.get("c")
        if isinstance(c, list) and c:
            px = float(c[0])
            return px if px > 0.0 else None
    except Exception:
        return None
    return None


def _bitstamp_pair_for_product(product_id: str) -> str:
    mapping = {
        "BTC-USD": "btcusd",
        "ETH-USD": "ethusd",
        "DOGE-USD": "dogeusd",
        "XRP-USD": "xrpusd",
        "SOL-USD": "solusd",
    }
    return mapping.get(str(product_id).strip().upper(), "")


def _bitstamp_spot(product_id: str) -> Optional[float]:
    pair = _bitstamp_pair_for_product(product_id)
    if not pair:
        return None
    url = f"{BITSTAMP_API}/api/v2/ticker/{pair}/"
    try:
        with requests.Session() as s:
            resp = s.get(url, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        px = float(data.get("last"))
        return px if px > 0.0 else None
    except Exception:
        return None


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return ys[mid]
    return 0.5 * (ys[mid - 1] + ys[mid])


def _spot_proxy(
    *,
    product_id: str,
    sources: List[str],
    cache_path: Path,
    ttl_minutes: float,
) -> Tuple[Optional[float], List[str]]:
    cached = _load_cached_json(cache_path, ttl_minutes=ttl_minutes)
    if cached is not None:
        px = cached.get("spot")
        used = cached.get("sources")
        try:
            spot = float(px)
        except Exception:
            spot = None
        if spot and spot > 0.0 and isinstance(used, list):
            return (spot, [str(x) for x in used])

    spot_vals: List[float] = []
    used: List[str] = []

    for s in sources:
        key = str(s or "").strip().lower()
        if key == "coinbase":
            px = _coinbase_spot(product_id)
        elif key == "kraken":
            px = _kraken_spot(product_id)
        elif key == "bitstamp":
            px = _bitstamp_spot(product_id)
        else:
            px = None
        if px is None or px <= 0.0:
            continue
        spot_vals.append(float(px))
        used.append(key)

    spot = _median(spot_vals)
    if spot is not None and spot > 0.0:
        _write_json(cache_path, {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "spot": float(spot), "sources": used})
    return (spot, used)


def _sigma_floor_per_sqrt_s(annual_vol: float) -> float:
    v = max(0.0, float(annual_vol))
    if v <= 0.0:
        return 0.0
    seconds_per_year = 365.25 * 24.0 * 3600.0
    return v / math.sqrt(seconds_per_year)


class CryptoSpecialist:
    name = "crypto"
    categories = ["Crypto"]

    def _cache_dir(self, context: RunContext) -> Path:
        return context.data_root / "external" / "crypto"

    def propose(self, context: RunContext) -> List[CandidateProposal]:
        out: List[CandidateProposal] = []
        cfg = context.config

        join_ticks = int(cfg.get("crypto_join_ticks", 1))
        slippage_cents = float(cfg.get("crypto_slippage_cents_per_contract", 0.25))
        min_ev = float(cfg.get("crypto_min_ev_dollars", 0.0))
        maker_fee_rate = float(cfg.get("maker_fee_rate", 0.0))
        min_yes_bid_cents = int(cfg.get("min_yes_bid_cents", 1))
        max_yes_spread_cents = int(cfg.get("max_yes_spread_cents", 10))
        shadow_mode = bool(str(cfg.get("shadow_execution_mode", "")).strip())
        shadow_force_top_n = max(0, int(cfg.get("shadow_crypto_force_top_liquid_n", 5)))
        shadow_force_enabled = str(cfg.get("shadow_crypto_force_top_liquid_enabled", "1")).strip().lower() not in {
            "0",
            "false",
            "no",
        }
        shadow_force_liquidity_n = shadow_force_top_n if (shadow_mode and shadow_force_enabled) else 0
        shadow_liquidity_pool: List[Tuple[Tuple[int, int, int], CandidateProposal]] = []

        granularity_s = int(cfg.get("crypto_granularity_s", 60))
        lookback_hours = float(cfg.get("crypto_lookback_hours", 24.0))
        cache_ttl = float(cfg.get("crypto_cache_ttl_minutes", 5.0))
        spot_sources = cfg.get("crypto_spot_sources") or ["coinbase", "kraken", "bitstamp"]
        if not isinstance(spot_sources, list):
            spot_sources = ["coinbase", "kraken", "bitstamp"]

        annual_vol_floor = float(cfg.get("crypto_sigma_floor_annual", 0.35))
        sigma_floor = _sigma_floor_per_sqrt_s(annual_vol_floor)

        # Many Kalshi crypto range markets settle on a 60-second average before the timestamp.
        settle_avg_window_s = float(cfg.get("crypto_settlement_avg_window_s", 60.0))

        as_of_dt = _parse_iso_dt(context.as_of_ts)
        if as_of_dt is None:
            return []

        cache_dir = self._cache_dir(context)
        cache_dir.mkdir(parents=True, exist_ok=True)

        products: Dict[str, ProductInfo] = {}
        needed_products: List[str] = []
        for row in context.markets:
            if category_from_row(row) != "Crypto":
                continue
            product = _product_for_event(str(row.get("event_ticker") or ""))
            if product and product not in products and product not in needed_products:
                needed_products.append(product)

        for product_id in needed_products:
            end = as_of_dt
            start = as_of_dt - dt.timedelta(hours=float(lookback_hours))

            candle_cache = cache_dir / f"candles_{product_id.replace('-', '_')}_{granularity_s}.json"
            candles = _fetch_candles(
                product_id=product_id,
                granularity_s=granularity_s,
                start=start,
                end=end,
                cache_path=candle_cache,
                ttl_minutes=cache_ttl,
            )
            if candles is None:
                continue
            est = _estimate_sigma_and_spot_from_candles(candles, granularity_s=granularity_s)
            if est is None:
                continue
            candle_spot, sigma_per_sqrt_s = est

            spot_cache = cache_dir / f"spot_{product_id.replace('-', '_')}.json"
            proxy_spot, used_sources = _spot_proxy(
                product_id=product_id,
                sources=[str(x) for x in spot_sources],
                cache_path=spot_cache,
                ttl_minutes=cache_ttl,
            )

            spot = float(proxy_spot) if proxy_spot is not None else float(candle_spot)
            sigma = max(float(sigma_per_sqrt_s), float(sigma_floor))

            products[product_id] = ProductInfo(
                product_id=product_id,
                spot_price=float(spot),
                sigma_per_sqrt_s=float(sigma),
                spot_sources=list(used_sources),
            )

        if not products:
            return []

        viable_btc_tickers: Optional[set[str]] = None
        dynamic_enabled = str(cfg.get("shadow_crypto_dynamic_discovery", "1")).strip().lower() not in {"0", "false", "no"}
        if dynamic_enabled:
            btc_rows = [
                row
                for row in context.markets
                if str(row.get("ticker") or "").strip().upper().startswith("KXBTC-")
                or str(row.get("event_ticker") or "").strip().upper().startswith("KXBTC-")
            ]
            date_token: Optional[str] = None
            for row in btc_rows:
                date_token = extract_kalshi_date_token_from_ticker(str(row.get("ticker") or ""))
                if date_token:
                    break
                date_token = extract_kalshi_date_token_from_ticker(str(row.get("event_ticker") or ""))
                if date_token:
                    break
            btc_info = products.get("BTC-USD")
            if btc_info is not None and date_token:
                try:
                    with requests.Session() as s:
                        viable = discover_viable_crypto_tickers_for_date(
                            session=s,
                            current_spot=float(btc_info.spot_price),
                            date_token=str(date_token).strip().upper(),
                            series_ticker="KXBTC",
                            variance_pct=max(0.0, float(cfg.get("shadow_crypto_variance_pct", 0.02))),
                            base_url=str(cfg.get("base_url") or ""),
                        )
                    if viable:
                        viable_btc_tickers = {str(t).strip().upper() for t in viable}
                except Exception:
                    viable_btc_tickers = None

        for row in context.markets:
            if category_from_row(row) != "Crypto":
                continue

            ticker = str(row.get("ticker") or "").strip().upper()
            if (
                viable_btc_tickers is not None
                and str(row.get("event_ticker") or "").strip().upper().startswith("KXBTC-")
                and ticker
                and ticker not in viable_btc_tickers
            ):
                continue

            event_ticker = str(row.get("event_ticker") or "").strip().upper()
            product_id = _product_for_event(event_ticker)
            if not product_id:
                continue
            info = products.get(product_id)
            if info is None:
                continue

            yes_sub = str(row.get("yes_sub_title") or "").strip()
            bounds = _parse_bounds(yes_sub)
            if bounds is None:
                continue
            lower, upper = bounds

            rules_primary = str(row.get("rules_primary") or "")
            target_dt = _parse_rule_target_dt(rules_primary)
            if target_dt is None:
                target_dt = _parse_iso_dt(str(row.get("close_time") or ""))
            if target_dt is None:
                continue

            horizon_s = (target_dt - as_of_dt).total_seconds()
            if horizon_s <= 0.0:
                continue

            # Approximate effect of 60s averaging: effective variance = sigma^2 * (h - 2w/3)
            effective_horizon_s = max(1.0, float(horizon_s) - (2.0 * float(settle_avg_window_s) / 3.0))
            sigma_total = float(info.sigma_per_sqrt_s) * math.sqrt(float(effective_horizon_s))

            blob = " ".join([str(row.get("title") or ""), rules_primary, yes_sub]).lower()
            is_touch = ("crosses" in blob) or ("at any point" in blob) or ("immediately resolves" in blob)

            if is_touch and upper is None and lower is not None:
                p_true = _p_hit_upper(s0=info.spot_price, barrier=float(lower), sigma_total=sigma_total)
                model = "gbm_barrier_up"
            elif is_touch and lower is None and upper is not None:
                p_true = _p_hit_lower(s0=info.spot_price, barrier=float(upper), sigma_total=sigma_total)
                model = "gbm_barrier_down"
            else:
                p_true = _p_terminal_between(s0=info.spot_price, sigma_total=sigma_total, lo=lower, hi=upper)
                model = "gbm_terminal"

            if not (0.0 <= float(p_true) <= 1.0):
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
            slippage = float(slippage_cents) / 100.0
            ev = float(p_fill) * (float(p_true) - price - fees - slippage)

            rules_pointer, rules_missing = rules_pointer_from_row(row)
            liquidity_notes = (
                f"p_true={p_true:.4f};model={model};product={product_id};spot={info.spot_price:.4f};"
                f"sigma_per_sqrt_s={info.sigma_per_sqrt_s:.10f};sigma_total={sigma_total:.6f};"
                f"horizon_s={horizon_s:.0f};effective_horizon_s={effective_horizon_s:.0f};"
                f"target_utc={target_dt.isoformat()};bounds={lower},{upper};"
                f"spot_sources={','.join(info.spot_sources) or 'coinbase_only'};"
                f"vol_source=coinbase_candles_{granularity_s}s"
            )

            proposal = CandidateProposal(
                strategy_id="CRYPTO_gbm_price_bucket_maker" if model == "gbm_terminal" else "CRYPTO_gbm_touch_maker",
                ticker=ticker,
                title=str(row.get("title") or ""),
                category=category_from_row(row),
                close_time=str(row.get("close_time") or ""),
                event_ticker=event_ticker,
                side="yes",
                action="post_yes",
                maker_or_taker="maker",
                yes_price_cents=intended,
                no_price_cents=None,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=safe_int(row.get("no_bid")),
                no_ask=safe_int(row.get("no_ask")),
                sum_prices_cents=None,
                p_fill_assumed=float(p_fill),
                fees_assumed_dollars=float(fees),
                slippage_assumed_dollars=float(slippage),
                ev_dollars=float(ev),
                ev_pct=(float(ev) / price if price > 0 else float("nan")),
                per_contract_notional=1.0,
                size_contracts=1,
                liquidity_notes=liquidity_notes,
                risk_flags=f"model=gbm;kind={model};product={product_id}",
                verification_checklist=(
                    "verify settlement source (CF RTI) vs proxy; verify time window parse; "
                    "verify bounds parse; verify volatility lookback/floor; verify maker fill/fee/slippage assumptions"
                ),
                rules_text_hash=rules_hash_from_row(row),
                rules_missing=rules_missing,
                rules_pointer=rules_pointer,
                resolution_pointer=resolution_pointer_from_row(row),
                market_url=str(row.get("market_url") or ""),
            )

            if ev <= float(min_ev):
                if shadow_force_liquidity_n > 0:
                    spread = (int(yes_ask) - int(yes_bid)) if (yes_bid is not None and yes_ask is not None) else 999
                    bid_size = safe_int(row.get("yes_bid_size"))
                    if bid_size is None:
                        bid_size = safe_int(row.get("yes_bid_size_fp"))
                    liquidity_score = (
                        max(0, int(bid_size or 0)),
                        -max(0, int(spread)),
                        max(0, int(yes_bid or 0)),
                    )
                    shadow_liquidity_pool.append((liquidity_score, proposal))
                continue

            out.append(proposal)

        if shadow_force_liquidity_n > 0 and len(out) < int(shadow_force_liquidity_n):
            seen = {str(p.ticker).strip().upper() for p in out}
            needed = int(shadow_force_liquidity_n) - len(out)
            added = 0
            for _, proposal in sorted(shadow_liquidity_pool, key=lambda x: x[0], reverse=True):
                t = str(proposal.ticker).strip().upper()
                if not t or t in seen:
                    continue
                proposal.liquidity_notes = f"{proposal.liquidity_notes};shadow_liquidity_override=1"
                out.append(proposal)
                seen.add(t)
                added += 1
                if added >= needed:
                    break
            if added > 0:
                print(
                    f"[CRYPTO][SHADOW] liquidity override enabled added={added} "
                    f"target_top_n={int(shadow_force_liquidity_n)} total={len(out)}"
                )

        return out
