from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

import requests


_RETRY_STATUS = {429, 500, 502, 503, 504}


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
) -> requests.Response:
    last_exc: Optional[BaseException] = None
    last_resp: Optional[requests.Response] = None
    for attempt in range(int(max_retries)):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=float(timeout_s))
            last_resp = resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= max_retries - 1:
                raise
            sleep_s = min(30.0, 0.75 * (2**attempt))
            time.sleep(sleep_s + random.uniform(0.0, 0.25))
            continue

        if resp.status_code not in _RETRY_STATUS:
            return resp

        if attempt >= max_retries - 1:
            return resp

        retry_after = resp.headers.get("Retry-After") or resp.headers.get("retry-after") or ""
        try:
            sleep_s = float(str(retry_after).strip()) if str(retry_after).strip() else min(30.0, 0.75 * (2**attempt))
        except Exception:
            sleep_s = min(30.0, 0.75 * (2**attempt))
        time.sleep(sleep_s + random.uniform(0.0, 0.25))

    if last_resp is not None:
        return last_resp
    if last_exc is not None:
        raise RuntimeError(f"request failed for {url}: {last_exc}") from last_exc
    raise RuntimeError(f"request failed for {url}: unknown error")


def get_json(
    session: requests.Session,
    url: str,
    *,
    params: Optional[Dict[str, object]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout_s: float = 30.0,
    max_retries: int = 8,
) -> Dict[str, Any]:
    resp = get_with_retries(
        session,
        url,
        params=params,
        headers=headers,
        timeout_s=timeout_s,
        max_retries=max_retries,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object from {url}, got {type(data).__name__}")
    return data


