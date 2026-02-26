from __future__ import annotations

import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Set

import requests

RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class RequestStats:
    window_size: int
    total_requests: int = 0
    count_429: int = 0
    retry_events: int = 0
    retry_delay_s_total: float = 0.0
    series_retried: Set[str] | None = None
    series_failed: Set[str] | None = None
    _recent_429: Deque[int] | None = None
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self.series_retried = set() if self.series_retried is None else self.series_retried
        self.series_failed = set() if self.series_failed is None else self.series_failed
        self._recent_429 = deque(maxlen=max(1, int(self.window_size))) if self._recent_429 is None else self._recent_429
        self._lock = threading.Lock() if self._lock is None else self._lock

    def record_response(self, *, status_code: int) -> None:
        with self._lock:
            self.total_requests += 1
            is_429 = 1 if int(status_code) == 429 else 0
            self.count_429 += is_429
            self._recent_429.append(is_429)

    def record_retry(self, *, series_ticker: str = "", delay_s: float = 0.0) -> None:
        with self._lock:
            self.retry_events += 1
            self.retry_delay_s_total += float(delay_s)
            st = str(series_ticker).strip().upper()
            if st:
                self.series_retried.add(st)

    def record_series_failed(self, *, series_ticker: str) -> None:
        with self._lock:
            st = str(series_ticker).strip().upper()
            if st:
                self.series_failed.add(st)

    def recent_429_rate(self) -> float:
        with self._lock:
            if not self._recent_429:
                return 0.0
            return float(sum(self._recent_429)) / float(len(self._recent_429))

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            avg_delay_ms = 0.0 if self.retry_events <= 0 else (self.retry_delay_s_total / float(self.retry_events)) * 1000.0
            recent_rate = 0.0 if not self._recent_429 else float(sum(self._recent_429)) / float(len(self._recent_429))
            return {
                "total_requests": int(self.total_requests),
                "count_429": int(self.count_429),
                "retry_events": int(self.retry_events),
                "avg_retry_delay_ms": round(avg_delay_ms, 3),
                "series_retried": len(self.series_retried),
                "series_failed": len(self.series_failed),
                "recent_window_size": int(self._recent_429.maxlen or 0),
                "recent_429_rate": round(recent_rate, 6),
            }


def _retry_delay_seconds(
    *,
    attempt: int,
    retry_after_header: str,
    min_sleep_ms: int,
    max_sleep_ms: int,
) -> float:
    min_s = max(0.001, float(min_sleep_ms) / 1000.0)
    max_s = max(min_s, float(max_sleep_ms) / 1000.0)
    raw = str(retry_after_header or "").strip()
    if raw:
        try:
            v = float(raw)
            return max(min_s, min(max_s, v))
        except Exception:
            pass
    backoff = min(max_s, min_s * (2 ** max(0, int(attempt))))
    jitter = random.uniform(0.0, min_s)
    return max(min_s, min(max_s, backoff + jitter))


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
    min_sleep_ms: int = 200,
    max_sleep_ms: int = 8000,
) -> requests.Response:
    last_exc: Optional[BaseException] = None
    last_resp: Optional[requests.Response] = None
    for attempt in range(int(max_retries) + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=float(timeout_s))
            last_resp = resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= int(max_retries):
                raise
            delay_s = _retry_delay_seconds(
                attempt=attempt,
                retry_after_header="",
                min_sleep_ms=int(min_sleep_ms),
                max_sleep_ms=int(max_sleep_ms),
            )
            time.sleep(delay_s)
            continue

        if int(resp.status_code) not in RETRY_STATUSES or attempt >= int(max_retries):
            return resp

        delay_s = _retry_delay_seconds(
            attempt=attempt,
            retry_after_header=str(resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""),
            min_sleep_ms=int(min_sleep_ms),
            max_sleep_ms=int(max_sleep_ms),
        )
        time.sleep(delay_s)

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise RuntimeError(f"request failed for {url}: {last_exc}") from last_exc
    raise RuntimeError(f"request failed for {url}: unknown error")


def request_json_with_backoff(
    *,
    session: requests.Session,
    url: str,
    params: Dict[str, object],
    max_retries: int,
    min_sleep_ms: int,
    max_sleep_ms: int,
    timeout_s: float,
    stats: Optional[RequestStats] = None,
    series_ticker: str = "",
) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            resp = session.get(url, params=params, timeout=float(timeout_s))
            if stats is not None:
                stats.record_response(status_code=int(resp.status_code))
        except requests.RequestException:
            if attempt >= int(max_retries):
                raise
            delay_s = _retry_delay_seconds(
                attempt=attempt,
                retry_after_header="",
                min_sleep_ms=min_sleep_ms,
                max_sleep_ms=max_sleep_ms,
            )
            if stats is not None:
                stats.record_retry(series_ticker=series_ticker, delay_s=delay_s)
            time.sleep(delay_s)
            attempt += 1
            continue

        if int(resp.status_code) in RETRY_STATUSES:
            if attempt >= int(max_retries):
                resp.raise_for_status()
            delay_s = _retry_delay_seconds(
                attempt=attempt,
                retry_after_header=str(resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""),
                min_sleep_ms=min_sleep_ms,
                max_sleep_ms=max_sleep_ms,
            )
            if stats is not None:
                stats.record_retry(series_ticker=series_ticker, delay_s=delay_s)
            time.sleep(delay_s)
            attempt += 1
            continue

        resp.raise_for_status()
        out = resp.json()
        if not isinstance(out, dict):
            raise TypeError(f"expected JSON object for {url}")
        return out


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
    min_sleep_ms: int = 200,
    max_sleep_ms: int = 8000,
) -> Dict[str, Any]:
    resp = get_with_retries(
        session,
        url,
        params=params,
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
        min_sleep_ms=min_sleep_ms,
        max_sleep_ms=max_sleep_ms,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object from {url}, got {type(data).__name__}")
    return data

