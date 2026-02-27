from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import requests

from .http import get_with_retries

DEFAULT_MAIN_BASE = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_MAIN_HOST = "api.elections.kalshi.com"
_DATE_TOKEN_RE = re.compile(r"-(\d{2}[A-Z]{3}\d{2})-")
_DATE_TOKEN_FULL_RE = re.compile(r"^\d{2}[A-Z]{3}\d{2}$")
_SERIES_DATE_PREFIX_RE = re.compile(r"^(?P<series>[A-Z0-9]+)-(?P<date>\d{2}[A-Z]{3}\d{2})(?:$|-)")
_DEBUG_PREFIX = "[KALSHI_DEBUG]"
_DEBUG_PAYLOAD_CHARS = 500
_RATE_LIMIT_BASE_BACKOFF_S = 1.0
_RATE_LIMIT_MAX_BACKOFF_S = 64.0
_rate_limit_until_monotonic = 0.0
_rate_limit_next_backoff_s = _RATE_LIMIT_BASE_BACKOFF_S


def _debug(msg: str) -> None:
    print(f"{_DEBUG_PREFIX} {msg}")


def _payload_preview(payload: Any, *, limit: int = _DEBUG_PAYLOAD_CHARS) -> str:
    try:
        raw = json.dumps(payload, default=str, ensure_ascii=True)
    except Exception:
        raw = str(payload)
    one_line = str(raw).replace("\n", " ").replace("\r", " ")
    return one_line[: max(1, int(limit))]


def _response_payload_preview(resp: requests.Response, *, limit: int = _DEBUG_PAYLOAD_CHARS) -> str:
    try:
        body = resp.text
    except Exception:
        body = ""
    one_line = str(body or "").replace("\n", " ").replace("\r", " ")
    if one_line:
        return one_line[: max(1, int(limit))]
    return "<empty>"


def _parse_retry_after_seconds(raw: object) -> Optional[float]:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    if v <= 0.0:
        return None
    return float(v)


def _extract_http_status_from_exc(exc: BaseException) -> Optional[int]:
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status is not None:
        try:
            return int(status)
        except Exception:
            pass
    msg = str(exc)
    if "429" in msg:
        return 429
    return None


def _extract_retry_after_from_exc(exc: BaseException) -> str:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None)
    if headers is None:
        return ""
    return str(headers.get("Retry-After") or headers.get("retry-after") or "").strip()


def _respect_rate_limit_cooldown() -> None:
    wait_s = float(_rate_limit_until_monotonic - time.monotonic())
    if wait_s > 0.0:
        _debug(f"rate_limit_global_cooldown sleep_s={wait_s:.2f}")
        time.sleep(wait_s)


def _note_rate_limited(*, retry_after_header: str = "") -> float:
    global _rate_limit_until_monotonic, _rate_limit_next_backoff_s
    parsed = _parse_retry_after_seconds(retry_after_header)
    if parsed is None:
        wait_s = float(_rate_limit_next_backoff_s)
        _rate_limit_next_backoff_s = min(
            float(_RATE_LIMIT_MAX_BACKOFF_S),
            max(float(_RATE_LIMIT_BASE_BACKOFF_S), float(_rate_limit_next_backoff_s) * 2.0),
        )
    else:
        wait_s = min(float(_RATE_LIMIT_MAX_BACKOFF_S), max(float(_RATE_LIMIT_BASE_BACKOFF_S), float(parsed)))
        _rate_limit_next_backoff_s = min(float(_RATE_LIMIT_MAX_BACKOFF_S), wait_s * 2.0)
    _rate_limit_until_monotonic = max(float(_rate_limit_until_monotonic), time.monotonic() + wait_s)
    return wait_s


def _reset_rate_limit_backoff() -> None:
    global _rate_limit_next_backoff_s
    _rate_limit_next_backoff_s = float(_RATE_LIMIT_BASE_BACKOFF_S)


def canonical_kalshi_base(base_url: Optional[str] = None) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        raw = (
            str(os.environ.get("KALSHI_API_BASE") or "")
            or str(os.environ.get("KALSHI_PUBLIC_BASE") or "")
            or str(os.environ.get("KALSHI_BASE") or "")
        ).strip()
    if not raw:
        return DEFAULT_MAIN_BASE

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = str(parsed.netloc or "").strip().lower()
    if host != DEFAULT_MAIN_HOST:
        return DEFAULT_MAIN_BASE

    scheme = str(parsed.scheme or "https").strip().lower() or "https"
    return f"{scheme}://{DEFAULT_MAIN_HOST}/trade-api/v2"


def extract_kalshi_date_token_from_ticker(ticker: str) -> Optional[str]:
    m = _DATE_TOKEN_RE.search(str(ticker or "").strip().upper())
    if not m:
        return None
    token = str(m.group(1) or "").strip().upper()
    return token if token else None


def kalshi_date_token_from_iso_date(date_iso: str) -> Optional[str]:
    s = str(date_iso or "").strip()
    if not s:
        return None
    upper = s.upper()
    if _DATE_TOKEN_FULL_RE.fullmatch(upper):
        return upper
    if len(s) < 10:
        return None
    try:
        d = dt.date.fromisoformat(s[:10])
    except Exception:
        return None
    return d.strftime("%y%b%d").upper()


def _date_tokens_with_neighbors(date_token: str) -> Set[str]:
    token = str(date_token or "").strip().upper()
    if not token:
        return set()
    out: Set[str] = {token}
    try:
        d = dt.datetime.strptime(token, "%y%b%d").date()
    except Exception:
        return out
    out.add((d - dt.timedelta(days=1)).strftime("%y%b%d").upper())
    out.add((d + dt.timedelta(days=1)).strftime("%y%b%d").upper())
    return out


def _event_ticker_from_event(event: Dict[str, Any]) -> str:
    for key in ("event_ticker", "ticker", "eventTicker", "eventTickerId"):
        val = str(event.get(key) or "").strip().upper()
        if val:
            return val
    return ""


def _fetch_series_events(
    *,
    session: requests.Session,
    series_ticker: str,
    base_url: Optional[str],
    max_pages: int,
    statuses: Iterable[Optional[str]],
) -> List[Dict[str, Any]]:
    base = canonical_kalshi_base(base_url)
    series = str(series_ticker).strip().upper()
    status_list = [str(s or "").strip().lower() for s in statuses]
    _debug(
        f"_fetch_series_events start base={base} endpoint={base}/events "
        f"series={series} max_pages={max_pages} statuses={status_list}"
    )
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for raw_status in status_list:
        status = str(raw_status or "").strip().lower()
        cursor = ""
        pages = 0
        _debug(f"_fetch_series_events status sweep status={status or '<none>'}")
        while pages < max(1, int(max_pages)):
            params: Dict[str, Any] = {"series_ticker": series, "limit": 200}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            _debug(f"GET {base}/events params={params}")
            try:
                _respect_rate_limit_cooldown()
                resp = get_with_retries(session, f"{base}/events", params=params, timeout_s=20.0)
                resp.raise_for_status()
            except Exception as exc:
                status_code = _extract_http_status_from_exc(exc)
                if status_code == 429:
                    wait_s = _note_rate_limited(retry_after_header=_extract_retry_after_from_exc(exc))
                    _debug(
                        f"request rate_limited url={base}/events status={status or '<none>'} "
                        f"cooldown_s={wait_s:.2f} series={series}"
                    )
                    return out
                _debug(f"request failed url={base}/events status={status or '<none>'} error={exc}")
                break
            _reset_rate_limit_backoff()
            _debug(
                f"response status={resp.status_code} url={resp.url} "
                f"payload={_response_payload_preview(resp)}"
            )
            payload = resp.json() if resp.content else {}
            if not isinstance(payload, dict):
                _debug(f"discard page: payload is {type(payload).__name__}, expected dict")
                break
            events = payload.get("events")
            if not isinstance(events, list):
                _debug(
                    f"discard page: payload['events'] type={type(events).__name__} "
                    f"payload={_payload_preview(payload)}"
                )
                break
            _debug(f"events page accepted count={len(events)} status={status or '<none>'} cursor={cursor or '<none>'}")
            for ev in events:
                if not isinstance(ev, dict):
                    _debug(f"discard event: non-dict entry type={type(ev).__name__}")
                    continue
                et = _event_ticker_from_event(ev)
                if not et:
                    _debug(f"discard event: missing event_ticker payload={_payload_preview(ev)}")
                    continue
                if et in seen:
                    _debug(f"discard event: duplicate event_ticker={et}")
                    continue
                seen.add(et)
                out.append(ev)
                _debug(f"accept event_ticker={et}")
            next_cursor = str(payload.get("cursor") or "").strip()
            pages += 1
            if not next_cursor or next_cursor == cursor:
                _debug(
                    f"stop pagination status={status or '<none>'} "
                    f"next_cursor={next_cursor or '<none>'} pages={pages}"
                )
                break
            cursor = next_cursor
        if out:
            # Keep status priority; if one status yielded rows, do not fan out further.
            _debug(f"status sweep complete status={status or '<none>'} accumulated_events={len(out)}")
            break
    _debug(f"_fetch_series_events done total_events={len(out)}")
    return out


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def _cents_from_dollars(value: object) -> Optional[int]:
    v = _safe_float(value)
    if v is None:
        return None
    return int(round(float(v) * 100.0))


def _size_from_fp(value: object) -> Optional[int]:
    v = _safe_float(value)
    if v is None:
        return None
    return int(float(v) * 100.0)


def _normalize_market_quotes(market: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(market)
    ticker = str(out.get("ticker") or "").strip().upper() or "<UNKNOWN>"

    yes_bid = _safe_int(out.get("yes_bid"))
    if yes_bid is None or yes_bid <= 0:
        fallback_yes_bid = _cents_from_dollars(out.get("yes_bid_dollars"))
        if fallback_yes_bid is not None and fallback_yes_bid > 0:
            out["yes_bid"] = int(fallback_yes_bid)
            _debug(
                f"quote_fallback ticker={ticker} field=yes_bid source=yes_bid_dollars "
                f"raw={out.get('yes_bid_dollars')} cents={out['yes_bid']}"
            )

    yes_ask = _safe_int(out.get("yes_ask"))
    if yes_ask is None or yes_ask <= 0:
        fallback_yes_ask = _cents_from_dollars(out.get("yes_ask_dollars"))
        if fallback_yes_ask is not None and fallback_yes_ask > 0:
            out["yes_ask"] = int(fallback_yes_ask)
            _debug(
                f"quote_fallback ticker={ticker} field=yes_ask source=yes_ask_dollars "
                f"raw={out.get('yes_ask_dollars')} cents={out['yes_ask']}"
            )

    no_bid = _safe_int(out.get("no_bid"))
    if no_bid is None or no_bid <= 0:
        fallback_no_bid = _cents_from_dollars(out.get("no_bid_dollars"))
        if fallback_no_bid is not None and fallback_no_bid > 0:
            out["no_bid"] = int(fallback_no_bid)
            _debug(
                f"quote_fallback ticker={ticker} field=no_bid source=no_bid_dollars "
                f"raw={out.get('no_bid_dollars')} cents={out['no_bid']}"
            )

    no_ask = _safe_int(out.get("no_ask"))
    if no_ask is None or no_ask <= 0:
        fallback_no_ask = _cents_from_dollars(out.get("no_ask_dollars"))
        if fallback_no_ask is not None and fallback_no_ask > 0:
            out["no_ask"] = int(fallback_no_ask)
            _debug(
                f"quote_fallback ticker={ticker} field=no_ask source=no_ask_dollars "
                f"raw={out.get('no_ask_dollars')} cents={out['no_ask']}"
            )

    yes_bid_size = _safe_int(out.get("yes_bid_size"))
    if yes_bid_size is None or yes_bid_size <= 0:
        fallback_size = _size_from_fp(out.get("yes_bid_size_fp"))
        if fallback_size is not None and fallback_size > 0:
            out["yes_bid_size"] = int(fallback_size)
            _debug(
                f"quote_fallback ticker={ticker} field=yes_bid_size source=yes_bid_size_fp "
                f"raw={out.get('yes_bid_size_fp')} normalized={out['yes_bid_size']}"
            )

    return out


def _market_strike_bounds(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    strike_type = str(market.get("strike_type") or "").strip().lower()
    floor = _safe_float(market.get("floor_strike"))
    cap = _safe_float(market.get("cap_strike"))
    if strike_type == "greater":
        return floor, None
    if strike_type == "less":
        return None, cap
    return floor, cap


def _market_strike_hint(market: Dict[str, Any]) -> Optional[float]:
    lo, hi = _market_strike_bounds(market)
    if lo is not None and hi is not None:
        return (float(lo) + float(hi)) / 2.0
    return lo if lo is not None else hi


def _market_in_window(
    market: Dict[str, Any],
    *,
    lower_bound: float,
    upper_bound: float,
) -> bool:
    lo, hi = _market_strike_bounds(market)
    if lo is not None and hi is not None:
        return not (float(hi) < float(lower_bound) or float(lo) > float(upper_bound))
    if lo is not None:
        return float(lo) <= float(upper_bound)
    if hi is not None:
        return float(hi) >= float(lower_bound)
    return False


def fetch_open_event_tickers_for_date(
    *,
    session: requests.Session,
    date_token: str,
    series_ticker: str = "KXBTC",
    base_url: Optional[str] = None,
    max_pages: int = 10,
    status: str = "open",
) -> List[str]:
    series = str(series_ticker).strip().upper()
    token_in = str(date_token).strip().upper()
    token = kalshi_date_token_from_iso_date(token_in) or token_in
    base = canonical_kalshi_base(base_url)
    _debug(
        f"fetch_open_event_tickers_for_date base={base} endpoint={base}/events "
        f"series={series} token_input={token_in} token_filter={token} status_pref={status}"
    )
    status_plan: List[Optional[str]] = []
    preferred = str(status or "").strip().lower()
    if preferred:
        status_plan.append(preferred)
    for st in ("open", "active", "initialized", ""):
        if st not in status_plan:
            status_plan.append(st)
    _debug(f"fetch_open_event_tickers_for_date status_plan={status_plan}")

    events = _fetch_series_events(
        session=session,
        series_ticker=series,
        base_url=base_url,
        max_pages=max_pages,
        statuses=status_plan,
    )
    if not events:
        _debug("fetch_open_event_tickers_for_date no events returned from API")
        return []

    matched: Set[str] = set()
    raw_event_tickers: List[str] = []
    for ev in events:
        et = _event_ticker_from_event(ev)
        if not et:
            _debug(f"discard event row: missing event_ticker payload={_payload_preview(ev)}")
            continue
        raw_event_tickers.append(et)
        if token and token in et.upper():
            matched.add(et)
            _debug(f"accept event_ticker={et} reason=date_token_substring_match token={token}")
            continue
        _debug(
            f"discard event_ticker={et} reason=date_token_substring_missing "
            f"required_token={token}"
        )
    _debug(
        f"fetch_open_event_tickers_for_date raw_event_tickers_first5={raw_event_tickers[:5]} "
        f"raw_count={len(raw_event_tickers)}"
    )
    out = sorted(matched)
    _debug(f"fetch_open_event_tickers_for_date return count={len(out)} tickers={out}")
    return out


def fetch_event(
    *,
    session: requests.Session,
    event_ticker: str,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    base = canonical_kalshi_base(base_url)
    et = str(event_ticker or "").strip().upper()
    if not et:
        _debug("fetch_event skipped: empty event_ticker")
        return {}
    url = f"{base}/events/{et}"
    _debug(f"GET {url}")
    try:
        _respect_rate_limit_cooldown()
        resp = get_with_retries(session, url, timeout_s=20.0)
        resp.raise_for_status()
    except Exception as exc:
        status_code = _extract_http_status_from_exc(exc)
        if status_code == 429:
            wait_s = _note_rate_limited(retry_after_header=_extract_retry_after_from_exc(exc))
            _debug(f"fetch_event rate_limited event_ticker={et} cooldown_s={wait_s:.2f}")
            return {}
        raise
    _reset_rate_limit_backoff()
    _debug(f"response status={resp.status_code} url={resp.url} payload={_response_payload_preview(resp)}")
    payload = resp.json() if resp.content else {}
    if not isinstance(payload, dict):
        _debug(f"fetch_event unexpected payload type={type(payload).__name__} for event_ticker={et}")
        return {}
    markets = payload.get("markets")
    if isinstance(markets, list):
        normalized: List[Dict[str, Any]] = []
        for market in markets:
            if not isinstance(market, dict):
                normalized.append(market)
                continue
            normalized.append(_normalize_market_quotes(market))
        payload["markets"] = normalized
        _debug(
            f"fetch_event normalized markets event_ticker={et} "
            f"count={len(normalized)}"
        )
    return payload if isinstance(payload, dict) else {}


def _event_ticker_from_market_ticker(market_ticker: str) -> str:
    mt = str(market_ticker or "").strip().upper()
    if not mt:
        return ""
    m = _DATE_TOKEN_RE.search(mt)
    if m:
        return mt[: m.end(1)]
    parts = [p for p in mt.split("-") if p]
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1]}"
    return ""


def _series_date_from_input(value: str) -> Optional[Tuple[str, str]]:
    s = str(value or "").strip().upper()
    if not s:
        return None
    m = _SERIES_DATE_PREFIX_RE.match(s)
    if not m:
        return None
    series = str(m.group("series") or "").strip().upper()
    token = str(m.group("date") or "").strip().upper()
    if not series or not token:
        return None
    return (series, token)


def get_markets(
    *,
    session: requests.Session,
    event_tickers: Sequence[str],
    base_url: Optional[str] = None,
    max_events: int = 80,
) -> List[Dict[str, Any]]:
    """Fetch market rows by event ticker using /events/{event_ticker}."""
    normalized_events: List[str] = []
    seen_events: Set[str] = set()
    for raw in event_tickers:
        source = str(raw or "").strip().upper()
        if not source:
            continue
        series_date = _series_date_from_input(source)
        if series_date is not None:
            series, token = series_date
            expanded = fetch_open_event_tickers_for_date(
                session=session,
                date_token=token,
                series_ticker=series,
                base_url=base_url,
                max_pages=max(1, min(int(max_events), 20)),
                status="open",
            )
            if expanded:
                _debug(
                    f"get_markets expanded_prefix input={source} series={series} token={token} "
                    f"events={expanded[:10]} count={len(expanded)}"
                )
                for et in expanded:
                    e = str(et or "").strip().upper()
                    if not e or e in seen_events:
                        continue
                    seen_events.add(e)
                    normalized_events.append(e)
                continue
            _debug(
                f"get_markets prefix expansion empty input={source} "
                f"series={series} token={token}"
            )

        et = source
        inferred = _event_ticker_from_market_ticker(source)
        if inferred:
            if inferred != et:
                _debug(
                    f"get_markets converted_input input={et} "
                    f"event_ticker={inferred}"
                )
            et = inferred
        if et in seen_events:
            continue
        seen_events.add(et)
        normalized_events.append(et)

    out: List[Dict[str, Any]] = []
    for et in normalized_events[: max(1, int(max_events))]:
        try:
            event = fetch_event(session=session, event_ticker=et, base_url=base_url)
        except Exception as exc:
            _debug(f"get_markets discard event_ticker={et} reason=fetch_event_error error={exc}")
            continue
        markets = event.get("markets") if isinstance(event, dict) else None
        if not isinstance(markets, list):
            _debug(
                f"get_markets discard event_ticker={et} reason=markets_not_list "
                f"markets_type={type(markets).__name__}"
            )
            continue

        raw_market_tickers: List[str] = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            mt = str(market.get("ticker") or "").strip().upper()
            if mt:
                raw_market_tickers.append(mt)
        _debug(
            f"get_markets raw_tickers_first10 event_ticker={et} "
            f"total={len(raw_market_tickers)} tickers={raw_market_tickers[:10]}"
        )

        for market in markets:
            if not isinstance(market, dict):
                continue
            normalized = _normalize_market_quotes(market)
            if not str(normalized.get("event_ticker") or "").strip():
                normalized["event_ticker"] = et
            out.append(normalized)

    _debug(f"get_markets return total_markets={len(out)} event_count={len(normalized_events)}")
    return out


def _sample_open_market_tickers_for_series(
    *,
    session: requests.Session,
    series_ticker: str,
    base_url: Optional[str],
    max_pages: int = 1,
    max_events: int = 3,
    max_markets_per_event: int = 50,
) -> List[str]:
    series = str(series_ticker or "").strip().upper()
    events = _fetch_series_events(
        session=session,
        series_ticker=series,
        base_url=base_url,
        max_pages=max_pages,
        statuses=("open", "active", "initialized", ""),
    )
    samples: List[str] = []
    seen: Set[str] = set()
    for ev in events[: max(1, int(max_events))]:
        et = _event_ticker_from_event(ev)
        if not et:
            continue
        try:
            details = fetch_event(session=session, event_ticker=et, base_url=base_url)
        except Exception:
            continue
        markets = details.get("markets") if isinstance(details, dict) else None
        if not isinstance(markets, list):
            continue
        for market in markets[: max(1, int(max_markets_per_event))]:
            if not isinstance(market, dict):
                continue
            status = str(market.get("status") or "").strip().lower()
            if status and status not in {"open", "active", "initialized"}:
                continue
            mt = str(market.get("ticker") or "").strip().upper()
            if not mt or mt in seen:
                continue
            seen.add(mt)
            samples.append(mt)
            if len(samples) >= 5:
                return samples
    return samples


def discover_market_tickers_for_series_date(
    *,
    session: requests.Session,
    date_token: str,
    series_ticker: str,
    base_url: Optional[str] = None,
    max_pages: int = 10,
    max_events: int = 60,
    max_markets_per_event: int = 300,
) -> List[str]:
    base = canonical_kalshi_base(base_url)
    series = str(series_ticker).strip().upper()
    token_in = str(date_token).strip().upper()
    token = kalshi_date_token_from_iso_date(token_in) or token_in
    _debug(
        f"discover_market_tickers_for_series_date base={base} "
        f"events_endpoint={base}/events event_details_endpoint={base}/events/{{event_ticker}} "
        f"series={series} date_token_input={token_in} date_token_filter={token} "
        f"max_pages={max_pages} max_events={max_events} "
        f"max_markets_per_event={max_markets_per_event}"
    )
    event_tickers = fetch_open_event_tickers_for_date(
        session=session,
        date_token=token,
        series_ticker=series,
        base_url=base_url,
        max_pages=max_pages,
        status="open",
    )
    if not event_tickers:
        sample = _sample_open_market_tickers_for_series(
            session=session,
            series_ticker=series,
            base_url=base_url,
            max_pages=1,
            max_events=3,
            max_markets_per_event=max_markets_per_event,
        )
        _debug(
            "discover_market_tickers_for_series_date no event_tickers returned "
            f"sample_open_markets={sample}"
        )
        return []
    _debug(f"discover_market_tickers_for_series_date event_tickers_count={len(event_tickers)} values={event_tickers}")

    out: Set[str] = set()
    open_market_samples: List[str] = []
    open_market_sample_seen: Set[str] = set()
    for event_ticker in event_tickers[: max(1, int(max_events))]:
        _debug(f"discover_market_tickers_for_series_date fetch event_ticker={event_ticker}")
        markets = get_markets(
            session=session,
            event_tickers=[event_ticker],
            base_url=base_url,
            max_events=1,
        )
        if not markets:
            _debug(f"discard event_ticker={event_ticker} reason=no_markets_from_get_markets")
            continue
        _debug(f"event_ticker={event_ticker} markets_count={len(markets)}")
        for market in markets[: max(1, int(max_markets_per_event))]:
            if not isinstance(market, dict):
                _debug(
                    f"discard market in event_ticker={event_ticker} reason=non_dict "
                    f"market_type={type(market).__name__}"
                )
                continue
            m_status = str(market.get("status") or "").strip().lower()
            if m_status and m_status not in {"open", "active", "initialized"}:
                _debug(
                    f"discard market ticker={str(market.get('ticker') or '').strip().upper() or '<missing>'} "
                    f"event_ticker={event_ticker} reason=status_filtered status={m_status}"
                )
                continue
            mt = str(market.get("ticker") or "").strip().upper()
            if not mt:
                _debug(f"discard market event_ticker={event_ticker} reason=missing_market_ticker payload={_payload_preview(market)}")
                continue
            if mt not in open_market_sample_seen and len(open_market_samples) < 5:
                open_market_sample_seen.add(mt)
                open_market_samples.append(mt)
            if token and token not in mt.upper():
                _debug(
                    f"discard market ticker={mt} event_ticker={event_ticker} "
                    f"reason=date_token_substring_missing required_token={token}"
                )
                continue
            if mt in out:
                _debug(f"discard market ticker={mt} reason=duplicate")
                continue
            out.add(mt)
            _debug(f"accept market ticker={mt} event_ticker={event_ticker} status={m_status or '<empty>'}")
    result = sorted(out)
    if not result:
        if not open_market_samples:
            open_market_samples = _sample_open_market_tickers_for_series(
                session=session,
                series_ticker=series,
                base_url=base_url,
                max_pages=1,
                max_events=3,
                max_markets_per_event=max_markets_per_event,
            )
        _debug(
            "discover_market_tickers_for_series_date empty result after filtering; "
            f"first_open_market_samples={open_market_samples[:5]}"
        )
    _debug(f"discover_market_tickers_for_series_date return count={len(result)} tickers={result}")
    return result


def discover_weather_tickers_for_date(
    *,
    session: requests.Session,
    date_token: str,
    base_url: Optional[str] = None,
    series_tickers: Sequence[str] = ("KXHIGHNY", "KXLOWNY"),
    max_pages: int = 10,
    max_events: int = 20,
    max_markets_per_event: int = 80,
) -> List[str]:
    base = canonical_kalshi_base(base_url)
    token_in = str(date_token).strip().upper()
    token = kalshi_date_token_from_iso_date(token_in) or token_in
    _debug(
        f"discover_weather_tickers_for_date base={base} token_input={token_in} token_filter={token} "
        f"series_tickers={[str(x) for x in series_tickers]}"
    )
    out: Set[str] = set()
    for raw_series in series_tickers:
        series = str(raw_series or "").strip().upper()
        if not series:
            _debug("discover_weather_tickers_for_date skip empty series entry")
            continue
        _debug(f"discover_weather_tickers_for_date series={series} start")
        discovered = discover_market_tickers_for_series_date(
            session=session,
            date_token=token,
            series_ticker=series,
            base_url=base_url,
            max_pages=max_pages,
            max_events=max_events,
            max_markets_per_event=max_markets_per_event,
        )
        _debug(f"discover_weather_tickers_for_date series={series} discovered_count={len(discovered)}")
        for mt in discovered:
            if mt in out:
                _debug(f"discover_weather_tickers_for_date dedupe ticker={mt}")
                continue
            out.add(mt)
    result = sorted(out)
    _debug(f"discover_weather_tickers_for_date return count={len(result)} tickers={result}")
    return result


def discover_sports_tickers_for_date(
    *,
    session: requests.Session,
    date_token: str,
    base_url: Optional[str] = None,
    series_tickers: Sequence[str] = ("KXNBAGAME", "KXNHLGAME"),
    max_pages: int = 10,
    max_events: int = 80,
    max_markets_per_event: int = 120,
) -> List[str]:
    base = canonical_kalshi_base(base_url)
    token_in = str(date_token).strip().upper()
    token = kalshi_date_token_from_iso_date(token_in) or token_in
    _debug(
        f"discover_sports_tickers_for_date base={base} token_input={token_in} token_filter={token} "
        f"series_tickers={[str(x) for x in series_tickers]}"
    )
    out: Set[str] = set()
    for raw_series in series_tickers:
        series = str(raw_series or "").strip().upper()
        if not series:
            _debug("discover_sports_tickers_for_date skip empty series entry")
            continue
        _debug(f"discover_sports_tickers_for_date series={series} start")
        discovered = discover_market_tickers_for_series_date(
            session=session,
            date_token=token,
            series_ticker=series,
            base_url=base_url,
            max_pages=max_pages,
            max_events=max_events,
            max_markets_per_event=max_markets_per_event,
        )
        _debug(f"discover_sports_tickers_for_date series={series} discovered_count={len(discovered)}")
        for mt in discovered:
            if mt in out:
                _debug(f"discover_sports_tickers_for_date dedupe ticker={mt}")
                continue
            out.add(mt)
    result = sorted(out)
    _debug(f"discover_sports_tickers_for_date return count={len(result)} tickers={result}")
    return result


def discover_viable_crypto_tickers_for_date(
    *,
    session: requests.Session,
    current_spot: float,
    date_token: str,
    series_ticker: str = "KXBTC",
    variance_pct: float = 0.02,
    base_url: Optional[str] = None,
) -> List[str]:
    spot = float(current_spot)
    var = max(0.0, float(variance_pct))
    lower_bound = spot * (1.0 - var)
    upper_bound = spot * (1.0 + var)

    series = str(series_ticker).strip().upper()
    token_in = str(date_token).strip().upper()
    token = kalshi_date_token_from_iso_date(token_in) or token_in
    base = canonical_kalshi_base(base_url)
    _debug(
        f"discover_viable_crypto_tickers_for_date base={base} "
        f"events_endpoint={base}/events event_details_endpoint={base}/events/{{event_ticker}} "
        f"series={series} token_input={token_in} token_filter={token} spot={spot:.6f} variance_pct={var:.6f} "
        f"window=[{lower_bound:.6f}, {upper_bound:.6f}]"
    )
    event_tickers = fetch_open_event_tickers_for_date(
        session=session,
        date_token=token,
        series_ticker=series,
        base_url=base_url,
        max_pages=10,
        status="open",
    )
    if not event_tickers:
        sample = _sample_open_market_tickers_for_series(
            session=session,
            series_ticker=series,
            base_url=base_url,
            max_pages=1,
            max_events=3,
            max_markets_per_event=50,
        )
        _debug(
            "discover_viable_crypto_tickers_for_date no event_tickers returned "
            f"sample_open_markets={sample}"
        )
        return []
    _debug(f"discover_viable_crypto_tickers_for_date event_tickers_count={len(event_tickers)} values={event_tickers}")

    candidates: Dict[str, Optional[float]] = {}
    fallback_open: Set[str] = set()
    for event_ticker in event_tickers:
        _debug(f"discover_viable_crypto_tickers_for_date fetch event_ticker={event_ticker}")
        markets = get_markets(
            session=session,
            event_tickers=[event_ticker],
            base_url=base_url,
            max_events=1,
        )
        if not markets:
            _debug(f"discard event_ticker={event_ticker} reason=no_markets_from_get_markets")
            continue
        _debug(f"event_ticker={event_ticker} markets_count={len(markets)}")
        for market in markets:
            if not isinstance(market, dict):
                _debug(
                    f"discard market in event_ticker={event_ticker} reason=non_dict "
                    f"market_type={type(market).__name__}"
                )
                continue
            status = str(market.get("status") or "").strip().lower()
            if status and status not in {"open", "active", "initialized"}:
                _debug(
                    f"discard market ticker={str(market.get('ticker') or '').strip().upper() or '<missing>'} "
                    f"event_ticker={event_ticker} reason=status_filtered status={status}"
                )
                continue
            ticker = str(market.get("ticker") or "").strip().upper()
            if not ticker:
                _debug(f"discard market event_ticker={event_ticker} reason=missing_market_ticker payload={_payload_preview(market)}")
                continue
            if token and token not in ticker.upper():
                _debug(
                    f"discard market ticker={ticker} event_ticker={event_ticker} "
                    f"reason=date_token_substring_missing required_token={token}"
                )
                continue
            fallback_open.add(ticker)
            _debug(f"candidate fallback_open ticker={ticker} event_ticker={event_ticker}")
            if not _market_in_window(
                market,
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
            ):
                lo, hi = _market_strike_bounds(market)
                _debug(
                    f"discard market ticker={ticker} event_ticker={event_ticker} "
                    f"reason=outside_variance_window strike_bounds=({lo},{hi}) "
                    f"window=[{lower_bound:.6f},{upper_bound:.6f}]"
                )
                continue
            hint = _market_strike_hint(market)
            candidates[ticker] = hint
            _debug(
                f"accept crypto market ticker={ticker} event_ticker={event_ticker} "
                f"strike_hint={hint}"
            )

    def _sort_key(ticker: str) -> Tuple[int, float, str]:
        hint = candidates.get(ticker)
        if hint is None:
            return (1, 0.0, ticker)
        return (0, abs(float(hint) - float(spot)), ticker)

    if candidates:
        ranked = sorted(candidates.keys(), key=_sort_key)
        _debug(f"discover_viable_crypto_tickers_for_date return filtered count={len(ranked)} tickers={ranked}")
        return ranked
    fallback = sorted(fallback_open)
    _debug(
        f"discover_viable_crypto_tickers_for_date filtered set empty; "
        f"fallback_open_count={len(fallback)} tickers={fallback}"
    )
    return fallback


def fetch_btc_spot_rest_fallback(session: requests.Session) -> Optional[float]:
    def _extract_price(payload: Any) -> Optional[float]:
        if isinstance(payload, dict):
            # Binance / Coinbase / Bitstamp
            for k in ("price", "amount"):
                if k in payload:
                    v = _safe_float(payload.get(k))
                    if v is not None and v > 0.0:
                        return float(v)
            # Kraken
            result = payload.get("result")
            if isinstance(result, dict):
                for pair in result.values():
                    if isinstance(pair, dict):
                        c = pair.get("c")
                        if isinstance(c, list) and c:
                            v = _safe_float(c[0])
                            if v is not None and v > 0.0:
                                return float(v)
        return None

    urls: Sequence[str] = (
        "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSDT",
        "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
        "https://www.bitstamp.net/api/v2/ticker/btcusd/",
    )
    for url in urls:
        try:
            resp = session.get(url, timeout=5.0)
            resp.raise_for_status()
            payload = resp.json() if resp.content else {}
            px = _extract_price(payload)
            if px is not None:
                return float(px)
        except Exception:
            continue
    return None
